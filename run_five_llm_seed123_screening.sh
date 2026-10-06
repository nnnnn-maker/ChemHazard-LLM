#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="sft_coarse_jsonl_seed123_simple"
RUN_ROOT="seed123_simple_screening"
COARSE_INPUT="${COARSE_INPUT:-data/pubchem/full_dataset_coarse_labels.csv}"

usage() {
  cat <<'EOF'
Usage:
  bash run_five_llm_seed123_screening.sh prepare
  bash run_five_llm_seed123_screening.sh qwen
  bash run_five_llm_seed123_screening.sh mistral
  bash run_five_llm_seed123_screening.sh gemma
  bash run_five_llm_seed123_screening.sh chatglm
  bash run_five_llm_seed123_screening.sh chemllm

Run "prepare" once before launching the five model jobs. Each model job runs:
  1. zero-shot inference on validation and test sets;
  2. simple LoRA training with training seed 123;
  3. adapted-model inference on validation and test sets;
  4. evaluation for all four prediction files.
EOF
}

prepare_data() {
  python build_pubchem_coarse_sft_dataset.py \
    --input "$COARSE_INPUT" \
    --output-root "$DATA_ROOT" \
    --seed 123 \
    --split-method multilabel_stratified \
    --train-oversample-factor 1

  echo "Expected split sizes for the current 11,556-compound dataset:"
  wc -l \
    "$DATA_ROOT/train.jsonl" \
    "$DATA_ROOT/val.jsonl" \
    "$DATA_ROOT/test.jsonl"
}

run_inference_and_evaluation() {
  local model="$1"
  local setting="$2"
  local split="$3"
  local adapter_path="${4:-}"
  local input_file="$DATA_ROOT/$split.jsonl"
  local pred_file="predictions/$RUN_ROOT/${model}_${setting}_split123_${split}_predictions.jsonl"
  local eval_dir="evaluation_llm_coarse/$RUN_ROOT/$model/$setting/$split"
  local prefix="${model}_${setting}_split123_${split}"
  local default_inference_precision="4bit"
  if [[ "$model" == "qwen" ]]; then
    default_inference_precision="4bit-fp16"
  fi
  local inference_args=(
    --model-name "$model"
    --input "$input_file"
    --output "$pred_file"
    --max-input-tokens 2048
    --max-new-tokens 192
  )

  case "${INFERENCE_PRECISION:-$default_inference_precision}" in
    4bit)
      inference_args+=(--load-in-4bit --dtype bfloat16 --bnb-4bit-compute-dtype bfloat16)
      ;;
    4bit-fp16)
      inference_args+=(--load-in-4bit --dtype float16 --bnb-4bit-compute-dtype float16)
      ;;
    8bit)
      inference_args+=(--load-in-8bit --dtype bfloat16)
      ;;
    bf16)
      inference_args+=(--dtype bfloat16)
      ;;
    fp16)
      inference_args+=(--dtype float16)
      ;;
    *)
      echo "Unsupported INFERENCE_PRECISION=${INFERENCE_PRECISION}. Use 4bit, 4bit-fp16, 8bit, bf16, or fp16." >&2
      exit 2
      ;;
  esac

  if [[ -n "$adapter_path" ]]; then
    inference_args+=(--adapter-path "$adapter_path")
  fi
  if [[ "$model" == "qwen" || "$model" == "chatglm" || "$model" == "chemllm" ]]; then
    inference_args+=(--trust-remote-code)
  fi

  python run_pubchem_coarse_inference.py "${inference_args[@]}" \
    2>&1 | tee "logs/$RUN_ROOT/${model}_${setting}_${split}.log"

  python evaluate_pubchem_coarse_predictions.py \
    --gold "$input_file" \
    --pred "$pred_file" \
    --output-dir "$eval_dir" \
    --prefix "$prefix" \
    --leaderboard-path "evaluation_llm_coarse/$RUN_ROOT/${model}_leaderboard.csv"
}

run_model() {
  local model="$1"
  case "$model" in
    qwen|mistral|gemma|chatglm|chemllm) ;;
    *)
      echo "Unsupported model: $model" >&2
      usage
      exit 2
      ;;
  esac

  if [[ "$model" == "mistral" ]]; then
    python verify_reproduction_setup.py --verify-mistral
  fi

  for split in train val test; do
    if [[ ! -s "$DATA_ROOT/$split.jsonl" ]]; then
      echo "Missing $DATA_ROOT/$split.jsonl. Run the prepare command first." >&2
      exit 2
    fi
  done

  mkdir -p \
    "logs/$RUN_ROOT" \
    "predictions/$RUN_ROOT" \
    "evaluation_llm_coarse/$RUN_ROOT/$model" \
    checkpoints_lora_sft

  # Zero-shot has no training seed. "split123" denotes the fixed data split.
  run_inference_and_evaluation "$model" "zeroshot" "val"
  run_inference_and_evaluation "$model" "zeroshot" "test"

  local adapter_path="checkpoints_lora_sft/${model}_simple_lora_split123_trainseed123"
  local train_args=(
    --model-name "$model"
    --train-file "$DATA_ROOT/train.jsonl"
    --val-file "$DATA_ROOT/val.jsonl"
    --output-dir "$adapter_path"
    --seed 123
    --max-length 1024
    --num-train-epochs 2
    --learning-rate 2e-4
    --per-device-train-batch-size 1
    --per-device-eval-batch-size 1
    --gradient-accumulation-steps 16
    --lora-r 16
    --lora-alpha 32
    --lora-dropout 0.05
    --load-in-4bit
    --gradient-checkpointing
    --overwrite-output-dir
  )
  if [[ "$model" == "qwen" ]]; then
    # The validated Qwen2.5-7B-Instruct run used FP16 compute.
    train_args+=(--bnb-4bit-compute-dtype float16 --dtype float16 --fp16)
  else
    train_args+=(--bnb-4bit-compute-dtype bfloat16 --dtype bfloat16 --bf16)
  fi
  if [[ "$model" == "qwen" || "$model" == "chatglm" || "$model" == "chemllm" ]]; then
    train_args+=(--trust-remote-code)
  fi

  python train_pubchem_coarse_lora_sft.py "${train_args[@]}" \
    2>&1 | tee "logs/$RUN_ROOT/${model}_simple_lora_train.log"

  run_inference_and_evaluation "$model" "simple_lora" "val" "$adapter_path"
  run_inference_and_evaluation "$model" "simple_lora" "test" "$adapter_path"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "prepare" ]]; then
  prepare_data
else
  run_model "$1"
fi
