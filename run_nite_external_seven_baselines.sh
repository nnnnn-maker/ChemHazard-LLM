#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-sklearn}"
NITE_DIR="${NITE_DIR:-external_validation/nite_japan_ghs}"
NITE_INPUT="${NITE_INPUT:-${NITE_DIR}/nite_external_test_exact_inchikey_disjoint.jsonl}"
NITE_TAG="${NITE_TAG:-exact}"
EVALUATOR="${NITE_DIR}/evaluate_nite_external_predictions.py"
PUBCHEM_INPUT="${PUBCHEM_INPUT:-processed_coarse/full_dataset_coarse_labels.csv}"
SPLIT_ROOT="${SPLIT_ROOT:-sft_coarse_jsonl_seed123_ghs_pictograms_rare2x}"
N_ESTIMATORS="${N_ESTIMATORS:-500}"
MIN_SAMPLES_LEAF="${MIN_SAMPLES_LEAF:-2}"
N_JOBS="${N_JOBS:--1}"
CHEMPROP_MODEL_PATH="${CHEMPROP_MODEL_PATH:-checkpoints_chemprop/chemprop_seed123_multitask}"
CHEMPROP_THRESHOLD="${CHEMPROP_THRESHOLD:-0.5}"

PRED_DIR="predictions/nite_japan_ghs/${NITE_TAG}/baselines"
METRIC_DIR="evaluation_llm_coarse/nite_japan_ghs/${NITE_TAG}/baselines"
ANALYSIS_DIR="analysis_outputs/nite_japan_ghs/${NITE_TAG}/baselines"
LOG_DIR="logs/nite_japan_ghs/${NITE_TAG}/baselines"
mkdir -p "$PRED_DIR" "$METRIC_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash run_nite_external_seven_baselines.sh sklearn
  bash run_nite_external_seven_baselines.sh chemprop
  bash run_nite_external_seven_baselines.sh collect
  bash run_nite_external_seven_baselines.sh all

Recommended environment order:
  1. In the ordinary chem2/sklearn environment, run: sklearn
  2. Activate the validated Chemprop 2.3 environment, run: chemprop
  3. Run: collect

The models are trained/fitted only on the PubChem seed123 training split.
NITE labels are used only by the final mask-aware evaluator. The fixed Chemprop
probability threshold is 0.5, matching the internal Table 4 experiment.

Optional variables:
  NITE_INPUT, NITE_TAG, N_ESTIMATORS, MIN_SAMPLES_LEAF, N_JOBS
  CHEMPROP_MODEL_PATH, CHEMPROP_THRESHOLD
EOF
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 2; }
}

run_sklearn() {
  require_file "$NITE_INPUT"
  require_file "$EVALUATOR"
  require_file "$PUBCHEM_INPUT"
  require_file "$SPLIT_ROOT/train.jsonl"
  require_file "run_nite_external_sklearn_baselines.py"
  python run_nite_external_sklearn_baselines.py \
    --pubchem-input "$PUBCHEM_INPUT" \
    --split-root "$SPLIT_ROOT" \
    --nite-input "$NITE_INPUT" \
    --evaluator "$EVALUATOR" \
    --tag "$NITE_TAG" \
    --seed 123 \
    --n-estimators "$N_ESTIMATORS" \
    --min-samples-leaf "$MIN_SAMPLES_LEAF" \
    --n-jobs "$N_JOBS" \
    2>&1 | tee "$LOG_DIR/six_sklearn_baselines.log"
}

run_chemprop() {
  require_file "$NITE_INPUT"
  require_file "$EVALUATOR"
  require_file "prepare_nite_chemprop_predict.py"
  require_file "convert_nite_chemprop_predictions.py"
  command -v chemprop >/dev/null 2>&1 || {
    echo "chemprop command not found. Activate the validated Chemprop 2.3 environment." >&2
    exit 2
  }
  [[ -d "$CHEMPROP_MODEL_PATH" ]] || {
    echo "Missing frozen PubChem Chemprop checkpoint directory: $CHEMPROP_MODEL_PATH" >&2
    echo "Use the same checkpoint that produced the internal Table 4 Chemprop result." >&2
    exit 2
  }

  CHEMPROP_INPUT="$ANALYSIS_DIR/chemprop_nite_predict_input.csv"
  CHEMPROP_AUDIT="$ANALYSIS_DIR/chemprop_nite_input_audit.json"
  CHEMPROP_RAW="$PRED_DIR/chemprop_raw_predictions.csv"
  CHEMPROP_JSONL="$PRED_DIR/chemprop_predictions.jsonl"
  CHEMPROP_METRIC="$METRIC_DIR/chemprop_metrics.json"

  python prepare_nite_chemprop_predict.py \
    --input "$NITE_INPUT" \
    --output "$CHEMPROP_INPUT" \
    --audit "$CHEMPROP_AUDIT"

  chemprop predict \
    --test-path "$CHEMPROP_INPUT" \
    --smiles-columns smiles \
    --model-paths "$CHEMPROP_MODEL_PATH" \
    --output "$CHEMPROP_RAW" \
    2>&1 | tee "$LOG_DIR/chemprop_predict.log"

  python convert_nite_chemprop_predictions.py \
    --predict-input "$CHEMPROP_INPUT" \
    --raw "$CHEMPROP_RAW" \
    --output "$CHEMPROP_JSONL" \
    --threshold "$CHEMPROP_THRESHOLD"

  python "$EVALUATOR" \
    --gold "$NITE_INPUT" \
    --predictions "$CHEMPROP_JSONL" \
    --output "$CHEMPROP_METRIC"

  python - "$CHEMPROP_METRIC" <<'PY'
import json, sys
from pathlib import Path
metric = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required_zero = ["missing_gold_ids_in_predictions", "extra_prediction_ids", "missing_prediction_cells"]
bad = {key: metric.get(key) for key in required_zero if metric.get(key) != 0}
if metric.get("gold_records") != metric.get("matched_records"):
    bad["matched_records"] = (metric.get("matched_records"), metric.get("gold_records"))
if bad:
    raise SystemExit(f"Incomplete Chemprop NITE result: {bad}")
print("Chemprop integrity audit passed.")
PY
}

collect_results() {
  require_file "collect_nite_external_baseline_results.py"
  python collect_nite_external_baseline_results.py \
    --metric-dir "$METRIC_DIR" \
    --output-csv "$METRIC_DIR/seven_baselines_summary.csv" \
    --output-json "$METRIC_DIR/seven_baselines_summary.json"
}

case "$MODE" in
  sklearn) run_sklearn ;;
  chemprop) run_chemprop ;;
  collect) collect_results ;;
  all)
    run_sklearn
    run_chemprop
    collect_results
    ;;
  -h|--help|help) usage ;;
  *) usage; echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
