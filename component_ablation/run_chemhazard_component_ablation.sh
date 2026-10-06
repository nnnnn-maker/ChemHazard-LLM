#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-audit}"
FORCE="${FORCE:-0}"
MAIN_ROOT="sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"
RARE_LABELS="oxidizing=2,gas_under_pressure=2,cmr=2,stot=2,environmental_hazard=2"
mkdir -p logs predictions evaluation_llm_coarse analysis_outputs checkpoints_lora_sft

usage() {
  cat <<'EOF'
Usage:
  bash run_chemhazard_component_ablation.sh audit
  bash run_chemhazard_component_ablation.sh bootstrap
  bash run_chemhazard_component_ablation.sh zeroshot
  bash run_chemhazard_component_ablation.sh noghs_rare2x
  bash run_chemhazard_component_ablation.sh pictogram_norare

Run audit first. Only run a training mode if the audit reports that prediction
as missing. Existing adapters are reused. Set FORCE=1 only when an intentional
rerun is required.
EOF
}

run_inference() {
  local input="$1"
  local output="$2"
  shift 2
  python run_pubchem_coarse_inference.py \
    --model-name mistral \
    --input "$input" \
    --output "$output" \
    "$@" \
    --load-in-4bit \
    --dtype bfloat16 \
    --bnb-4bit-compute-dtype bfloat16 \
    --max-input-tokens 2048 \
    --max-new-tokens 192
}

train_adapter() {
  local train_file="$1"
  local val_file="$2"
  local adapter="$3"
  local log_file="$4"
  if [[ -s "$adapter/adapter_config.json" && "$FORCE" != "1" ]]; then
    echo "Reusing existing adapter: $adapter"
    return
  fi
  python train_pubchem_coarse_lora_sft.py \
    --model-name mistral \
    --train-file "$train_file" \
    --val-file "$val_file" \
    --output-dir "$adapter" \
    --seed 123 \
    --max-length 1024 \
    --num-train-epochs 2 \
    --learning-rate 2e-4 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --load-in-4bit \
    --bnb-4bit-compute-dtype bfloat16 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --bf16 \
    --overwrite-output-dir \
    2>&1 | tee "$log_file"
}

refuse_existing_prediction() {
  local path="$1"
  if [[ -s "$path" && "$FORCE" != "1" ]]; then
    echo "Prediction already exists; refusing to overwrite: $path" >&2
    echo "Run audit, or use FORCE=1 only for an intentional rerun." >&2
    exit 2
  fi
}

if [[ "$MODE" != "audit" && "$MODE" != "bootstrap" ]]; then
  python verify_reproduction_setup.py --verify-mistral
fi

case "$MODE" in
  audit)
    python component_ablation/run_chemhazard_component_ablation_audit.py --project-dir "$PROJECT_DIR"
    ;;
  bootstrap)
    python component_ablation/run_chemhazard_component_ablation_audit.py \
      --project-dir "$PROJECT_DIR" \
      --run-bootstrap \
      --n-bootstrap 10000
    ;;
  zeroshot)
    output="predictions/mistral_base_zeroshot_seed123_ghs_pictograms_rare2x_predictions.jsonl"
    refuse_existing_prediction "$output"
    run_inference "$MAIN_ROOT/test.jsonl" "$output" \
      2>&1 | tee logs/mistral_base_zeroshot_seed123_ghs_pictograms_rare2x.log
    ;;
  noghs_rare2x)
    root="sft_coarse_jsonl_seed123_noghs_rare2x"
    adapter="checkpoints_lora_sft/mistral_seed123_lora_noghs_rare2x"
    output="predictions/mistral_seed123_lora_noghs_rare2x_predictions.jsonl"
    refuse_existing_prediction "$output"
    if [[ ! -s "$root/train.jsonl" || ! -s "$root/val.jsonl" || ! -s "$root/test.jsonl" ]]; then
      python build_pubchem_coarse_sft_dataset.py \
        --seed 123 \
        --split-method multilabel_stratified \
        --output-root "$root" \
        --train-oversample-spec "$RARE_LABELS"
    fi
    train_adapter "$root/train.jsonl" "$root/val.jsonl" "$adapter" \
      logs/mistral_seed123_lora_noghs_rare2x_train.log
    run_inference "$root/test.jsonl" "$output" --adapter-path "$adapter" \
      2>&1 | tee logs/mistral_seed123_lora_noghs_rare2x_inference.log
    ;;
  pictogram_norare)
    root="sft_coarse_jsonl_seed123_ghs_pictograms_norare"
    adapter="checkpoints_lora_sft/mistral_seed123_lora_ghs_pictograms_norare"
    output="predictions/mistral_seed123_lora_ghs_pictograms_norare_predictions.jsonl"
    refuse_existing_prediction "$output"
    if [[ ! -s "$root/train.jsonl" || ! -s "$root/val.jsonl" || ! -s "$root/test.jsonl" ]]; then
      python build_pubchem_coarse_sft_dataset.py \
        --seed 123 \
        --split-method multilabel_stratified \
        --output-root "$root" \
        --include-ghs-evidence \
        --ghs-evidence-columns Pictograms
    fi
    train_adapter "$root/train.jsonl" "$root/val.jsonl" "$adapter" \
      logs/mistral_seed123_lora_ghs_pictograms_norare_train.log
    run_inference "$root/test.jsonl" "$output" --adapter-path "$adapter" \
      2>&1 | tee logs/mistral_seed123_lora_ghs_pictograms_norare_inference.log
    ;;
  -h|--help|help) usage ;;
  *) usage; echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
