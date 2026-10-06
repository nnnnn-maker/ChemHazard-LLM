#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LABEL_COUNT=9
NITE_DIR="${NITE_DIR:-external_validation/nite_japan_ghs}"
NITE_INPUT="${NITE_INPUT:-${NITE_DIR}/nite_external_test_exact_inchikey_disjoint.jsonl}"
NITE_TAG="${NITE_TAG:-exact}"
NITE_LIMIT="${NITE_LIMIT:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash run_nite_external_one_model.sh <model> <setting> [adapter_path]

Models:
  qwen | mistral | gemma | chatglm | chemllm

Settings:
  zeroshot
  task_adapted

Examples:
  bash run_nite_external_one_model.sh mistral zeroshot
  bash run_nite_external_one_model.sh mistral task_adapted
  bash run_nite_external_one_model.sh qwen task_adapted \
    checkpoints_lora_sft/qwen_simple_lora_split123_trainseed123

Optional environment variables:
  INFERENCE_PRECISION=4bit|4bit-fp16|8bit|bf16|fp16
  NITE_INPUT=<jsonl path>
  NITE_TAG=<output subdirectory name>
  NITE_LIMIT=<positive integer for smoke test; 0 means full dataset>

This script performs inference and mask-aware evaluation only. It never trains or
overwrites an adapter. ChatGLM and ChemLLM must be run in their validated environments.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

MODEL="$1"
SETTING="$2"
EXPLICIT_ADAPTER="${3:-}"

case "$MODEL" in
  qwen|mistral|gemma|chatglm|chemllm) ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    usage
    exit 2
    ;;
esac

case "$SETTING" in
  zeroshot|task_adapted) ;;
  *)
    echo "Unsupported setting: $SETTING" >&2
    usage
    exit 2
    ;;
esac

if [[ "$MODEL" == "mistral" ]]; then
  python verify_reproduction_setup.py --verify-mistral
fi

if [[ ! -s "$NITE_INPUT" ]]; then
  echo "Missing or empty NITE input: $NITE_INPUT" >&2
  exit 2
fi

EVALUATOR="${NITE_DIR}/evaluate_nite_external_predictions.py"
if [[ ! -s "$EVALUATOR" ]]; then
  echo "Missing evaluator: $EVALUATOR" >&2
  exit 2
fi

ADAPTER_PATH=""
if [[ "$SETTING" == "task_adapted" ]]; then
  if [[ -n "$EXPLICIT_ADAPTER" ]]; then
    ADAPTER_PATH="$EXPLICIT_ADAPTER"
  else
    ADAPTER_PATH="checkpoints_lora_sft/${MODEL}_simple_lora_split123_trainseed123"
  fi
  if [[ ! -s "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json: ${ADAPTER_PATH}/adapter_config.json" >&2
    exit 2
  fi
elif [[ -n "$EXPLICIT_ADAPTER" ]]; then
  echo "An adapter path is not allowed for zeroshot." >&2
  exit 2
fi

if ! [[ "$NITE_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "NITE_LIMIT must be a non-negative integer: $NITE_LIMIT" >&2
  exit 2
fi

RUN_NAME="${MODEL}_${SETTING}"
if (( NITE_LIMIT > 0 )); then
  RUN_NAME="${RUN_NAME}_smoke${NITE_LIMIT}"
fi

PRED_DIR="predictions/nite_japan_ghs/${NITE_TAG}"
EVAL_DIR="evaluation_llm_coarse/nite_japan_ghs/${NITE_TAG}"
LOG_DIR="logs/nite_japan_ghs/${NITE_TAG}"
PRED_FILE="${PRED_DIR}/${RUN_NAME}_predictions.jsonl"
METRIC_FILE="${EVAL_DIR}/${RUN_NAME}_metrics.json"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

DEFAULT_PRECISION="4bit"
if [[ "$MODEL" == "qwen" ]]; then
  DEFAULT_PRECISION="4bit-fp16"
fi
PRECISION="${INFERENCE_PRECISION:-$DEFAULT_PRECISION}"

INFERENCE_ARGS=(
  --model-name "$MODEL"
  --input "$NITE_INPUT"
  --output "$PRED_FILE"
  --max-input-tokens 2048
  --max-new-tokens 192
)

case "$PRECISION" in
  4bit)
    INFERENCE_ARGS+=(
      --load-in-4bit
      --dtype bfloat16
      --bnb-4bit-compute-dtype bfloat16
    )
    ;;
  4bit-fp16)
    INFERENCE_ARGS+=(
      --load-in-4bit
      --dtype float16
      --bnb-4bit-compute-dtype float16
    )
    ;;
  8bit)
    INFERENCE_ARGS+=(--load-in-8bit --dtype bfloat16)
    ;;
  bf16)
    INFERENCE_ARGS+=(--dtype bfloat16)
    ;;
  fp16)
    INFERENCE_ARGS+=(--dtype float16)
    ;;
  *)
    echo "Unsupported INFERENCE_PRECISION: $PRECISION" >&2
    exit 2
    ;;
esac

if [[ -n "$ADAPTER_PATH" ]]; then
  INFERENCE_ARGS+=(--adapter-path "$ADAPTER_PATH")
fi

if [[ "$MODEL" == "qwen" || "$MODEL" == "chatglm" || "$MODEL" == "chemllm" ]]; then
  INFERENCE_ARGS+=(--trust-remote-code)
fi

if (( NITE_LIMIT > 0 )); then
  INFERENCE_ARGS+=(--limit "$NITE_LIMIT")
fi

echo "Project:    $PROJECT_DIR"
echo "Model:      $MODEL"
echo "Setting:    $SETTING"
echo "Input:      $NITE_INPUT"
echo "Precision:  $PRECISION"
echo "Adapter:    ${ADAPTER_PATH:-<none>}"
echo "Prediction: $PRED_FILE"
echo "Metrics:    $METRIC_FILE"

python run_pubchem_coarse_inference.py "${INFERENCE_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

python - "$PRED_FILE" "$NITE_INPUT" "$NITE_LIMIT" "$LABEL_COUNT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

pred_path = Path(sys.argv[1])
gold_path = Path(sys.argv[2])
limit = int(sys.argv[3])
label_count = int(sys.argv[4])

pred_rows = [json.loads(line) for line in pred_path.open(encoding="utf-8") if line.strip()]
gold_count = sum(1 for line in gold_path.open(encoding="utf-8") if line.strip())
expected = min(limit, gold_count) if limit > 0 else gold_count
statuses = Counter(str(row.get("parse_status", "missing")) for row in pred_rows)
valid = sum(isinstance(row.get("labels"), dict) and len(row["labels"]) == label_count for row in pred_rows)

print(f"Prediction audit: rows={len(pred_rows)}, expected={expected}, valid={valid}, status={dict(statuses)}")
if len(pred_rows) != expected:
    raise SystemExit(f"Prediction row count mismatch: {len(pred_rows)} != {expected}")
if valid == 0:
    raise SystemExit("All predictions are invalid. Evaluation is blocked; do not use an all-zero fallback.")
if valid != expected:
    print(f"WARNING: {expected - valid} predictions are invalid; evaluator will report missing prediction cells.")
PY

python "$EVALUATOR" \
  --gold "$NITE_INPUT" \
  --predictions "$PRED_FILE" \
  --output "$METRIC_FILE"

echo "Completed: $RUN_NAME"
echo "Prediction file: $PRED_FILE"
echo "Metric file:     $METRIC_FILE"
