#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"
PRED_ROOT="predictions"
EVAL_ROOT="evaluation_llm_coarse"
SEEDS=(123 777 2025)
COARSE_INPUT="${COARSE_INPUT:-data/pubchem/full_dataset_coarse_labels.csv}"

usage() {
  cat <<'EOF'
Usage:
  bash run_chemhazard_main_reproduction.sh prepare
  bash run_chemhazard_main_reproduction.sh train <123|777|2025>
  bash run_chemhazard_main_reproduction.sh infer <123|777|2025>
  bash run_chemhazard_main_reproduction.sh ensemble
  bash run_chemhazard_main_reproduction.sh evaluate

Thresholds are selected only from validation predictions. Test predictions are
used only after the thresholds have been frozen.
EOF
}

adapter_path() {
  echo "checkpoints_lora_sft/mistral_split123_ghs_pictograms_rare2x_trainseed$1"
}

require_seed() {
  local seed="${1:-}"
  [[ " ${SEEDS[*]} " == *" ${seed} "* ]] || {
    echo "seed must be one of: ${SEEDS[*]}" >&2
    exit 2
  }
}

prepare() {
  if [[ -f full_dataset.csv ]]; then
    python build_pubchem_coarse_labels.py \
      --input full_dataset.csv \
      --output processed_coarse/full_dataset_coarse_labels.csv \
      --summary processed_coarse/full_dataset_coarse_labels_summary.json
    COARSE_INPUT="processed_coarse/full_dataset_coarse_labels.csv"
  elif [[ ! -f "$COARSE_INPUT" ]]; then
    echo "Missing processed PubChem data: $COARSE_INPUT" >&2
    echo "Provide data/pubchem/full_dataset_coarse_labels.csv or full_dataset.csv." >&2
    exit 1
  fi

  python build_pubchem_coarse_sft_dataset.py \
    --input "$COARSE_INPUT" \
    --output-root "$DATA_ROOT" \
    --seed 123 \
    --split-method multilabel_stratified \
    --include-ghs-evidence \
    --ghs-evidence-columns Pictograms \
    --train-oversample-factor 2 \
    --train-oversample-labels oxidizing,gas_under_pressure,cmr,stot,environmental_hazard

  python export_fixed_split_manifest.py \
    --split-root "$DATA_ROOT" \
    --output reproducibility/pubchem_seed123_split_manifest.csv
}

train_member() {
  local seed="$1"
  require_seed "$seed"
  python verify_reproduction_setup.py --verify-mistral
  mkdir -p logs
  python train_pubchem_coarse_lora_sft.py \
    --model-name mistral \
    --train-file "$DATA_ROOT/train.jsonl" \
    --val-file "$DATA_ROOT/val.jsonl" \
    --output-dir "$(adapter_path "$seed")" \
    --seed "$seed" \
    --max-length 1024 \
    --num-train-epochs 2 \
    --learning-rate 2e-4 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --weight-decay 0.01 \
    --optim adamw_torch \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load-in-4bit \
    --bnb-4bit-compute-dtype bfloat16 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --bf16 \
    --overwrite-output-dir \
    2>&1 | tee "logs/mistral_main_trainseed${seed}.log"
}

infer_member() {
  local seed="$1"
  require_seed "$seed"
  python verify_reproduction_setup.py --verify-mistral
  mkdir -p "$PRED_ROOT" logs
  for split in val test; do
    local suffix=""
    [[ "$split" == "val" ]] && suffix="_val"
    python run_pubchem_coarse_inference.py \
      --model-name mistral \
      --input "$DATA_ROOT/${split}.jsonl" \
      --adapter-path "$(adapter_path "$seed")" \
      --output "$PRED_ROOT/mistral_main_trainseed${seed}${suffix}_predictions.jsonl" \
      --load-in-4bit \
      --dtype bfloat16 \
      --bnb-4bit-compute-dtype bfloat16 \
      --max-input-tokens 2048 \
      --max-new-tokens 192 \
      2>&1 | tee "logs/mistral_main_trainseed${seed}_${split}.log"
  done
}

ensemble() {
  python combine_multilabel_votes.py \
    --gold "$DATA_ROOT/test.jsonl" \
    --pred \
      "$PRED_ROOT/mistral_main_trainseed123_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed777_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed2025_predictions.jsonl" \
    --default-threshold 2 \
    --output "$PRED_ROOT/mistral_split123_ghs_pictograms_rare2x_ensemble3_predictions.jsonl"

  python calibrate_multilabel_vote_thresholds.py \
    --val-gold "$DATA_ROOT/val.jsonl" \
    --test-gold "$DATA_ROOT/test.jsonl" \
    --val-pred \
      "$PRED_ROOT/mistral_main_trainseed123_val_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed777_val_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed2025_val_predictions.jsonl" \
    --test-pred \
      "$PRED_ROOT/mistral_main_trainseed123_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed777_predictions.jsonl" \
      "$PRED_ROOT/mistral_main_trainseed2025_predictions.jsonl" \
    --output-dir "$PRED_ROOT" \
    --prefix mistral_split123_ghs_pictograms_rare2x_labelcal
}

evaluate_final() {
  local run="mistral_split123_ghs_pictograms_rare2x_labelcal_constrained_ensemble3"
  python evaluate_pubchem_coarse_predictions.py \
    --gold "$DATA_ROOT/test.jsonl" \
    --pred "$PRED_ROOT/${run}_predictions.jsonl" \
    --output-dir "$EVAL_ROOT/$run" \
    --prefix "$run"
}

command="${1:-}"
case "$command" in
  prepare) prepare ;;
  train) train_member "${2:-}" ;;
  infer) infer_member "${2:-}" ;;
  ensemble) ensemble ;;
  evaluate) evaluate_final ;;
  *) usage; exit 2 ;;
esac
