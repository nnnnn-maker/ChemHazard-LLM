#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

NITE_DIR="${NITE_DIR:-external_validation/nite_japan_ghs}"
NITE_INPUT="${NITE_INPUT:-${NITE_DIR}/nite_external_test_exact_inchikey_disjoint.jsonl}"
NITE_TAG="${NITE_TAG:-exact}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-4bit}"
SKIP_MEMBERS="${SKIP_MEMBERS:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash run_nite_external_chemhazard.sh <adapter_1> <adapter_2> <adapter_3>

The three paths must be the exact Mistral adapters used in the final internal
ChemHazard-LLM ensemble. Do not substitute adapters from the invalidated early
five-model screen or newly selected adapters, and do not tune vote thresholds on
the NITE external test set.

Example:
  bash run_nite_external_chemhazard.sh \
    checkpoints_lora_sft/<final_member_1> \
    checkpoints_lora_sft/<final_member_2> \
    checkpoints_lora_sft/<final_member_3>

Optional environment variables:
  INFERENCE_PRECISION=4bit|4bit-fp16|8bit|bf16|fp16
  SKIP_MEMBERS=1        Reuse three already completed member prediction files.
  NITE_INPUT=<path>     Override the NITE JSONL input.
  NITE_TAG=<name>       Output subdirectory name; default: exact.

Fixed final vote rule:
  oxidizing, gas_under_pressure, cmr: positive with at least 1/3 votes
  all other labels: positive with at least 2/3 votes
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  echo >&2
  echo "Candidate Mistral adapters:" >&2
  find checkpoints_lora_sft -maxdepth 2 -name adapter_config.json -printf '%h\n' 2>/dev/null \
    | sort \
    | grep -E 'mistral.*ghs_pictograms_rare2x' >&2 || true
  exit 2
fi

ADAPTERS=("$1" "$2" "$3")

if [[ "$SKIP_MEMBERS" != "1" ]]; then
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

for adapter in "${ADAPTERS[@]}"; do
  if [[ ! -s "${adapter}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json: ${adapter}/adapter_config.json" >&2
    exit 2
  fi
done

PRED_DIR="predictions/nite_japan_ghs/${NITE_TAG}"
EVAL_DIR="evaluation_llm_coarse/nite_japan_ghs/${NITE_TAG}"
LOG_DIR="logs/nite_japan_ghs/${NITE_TAG}"
mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

precision_args=()
case "$INFERENCE_PRECISION" in
  4bit)
    precision_args=(--load-in-4bit --dtype bfloat16 --bnb-4bit-compute-dtype bfloat16)
    ;;
  4bit-fp16)
    precision_args=(--load-in-4bit --dtype float16 --bnb-4bit-compute-dtype float16)
    ;;
  8bit)
    precision_args=(--load-in-8bit --dtype bfloat16)
    ;;
  bf16)
    precision_args=(--dtype bfloat16)
    ;;
  fp16)
    precision_args=(--dtype float16)
    ;;
  *)
    echo "Unsupported INFERENCE_PRECISION: $INFERENCE_PRECISION" >&2
    exit 2
    ;;
esac

if [[ "$SKIP_MEMBERS" != "1" ]]; then
  for i in 0 1 2; do
    member=$((i + 1))
    adapter="${ADAPTERS[$i]}"
    pred_file="${PRED_DIR}/chemhazard_member${member}_predictions.jsonl"
    log_file="${LOG_DIR}/chemhazard_member${member}.log"

    echo "Running ChemHazard member ${member}: ${adapter}"
    python run_pubchem_coarse_inference.py \
      --model-name mistral \
      --input "$NITE_INPUT" \
      --adapter-path "$adapter" \
      --output "$pred_file" \
      "${precision_args[@]}" \
      --max-input-tokens 2048 \
      --max-new-tokens 192 \
      2>&1 | tee "$log_file"
  done
else
  echo "SKIP_MEMBERS=1: reusing existing member predictions in ${PRED_DIR}"
fi

python - "$NITE_INPUT" "$PRED_DIR" <<'PY'
import json
import sys
from pathlib import Path

LABELS = [
    "flammable",
    "oxidizing",
    "gas_under_pressure",
    "corrosive",
    "acute_toxicity",
    "irritant_harmful",
    "cmr",
    "stot",
    "environmental_hazard",
]

gold_path = Path(sys.argv[1])
pred_dir = Path(sys.argv[2])
gold_rows = [json.loads(line) for line in gold_path.open(encoding="utf-8") if line.strip()]
gold_ids = [str(row["id"]) for row in gold_rows]
gold_id_set = set(gold_ids)

members = []
for member in range(1, 4):
    path = pred_dir / f"chemhazard_member{member}_predictions.jsonl"
    if not path.is_file():
        raise SystemExit(f"Missing member prediction file: {path}")
    records = {}
    invalid = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = str(row.get("id", ""))
        labels = row.get("labels")
        if (
            record_id in records
            or not isinstance(labels, dict)
            or set(labels) != set(LABELS)
            or any(labels[label] not in (0, 1, False, True) for label in LABELS)
        ):
            invalid.append(record_id)
            continue
        records[record_id] = {label: int(labels[label]) for label in LABELS}

    missing = gold_id_set - set(records)
    extra = set(records) - gold_id_set
    print(
        f"member{member}: valid={len(records)}, invalid={len(invalid)}, "
        f"missing={len(missing)}, extra={len(extra)}"
    )
    if invalid or missing or extra or len(records) != len(gold_rows):
        raise SystemExit(
            f"Member {member} is incomplete. Ensemble is blocked; "
            "do not substitute missing predictions with zeros."
        )
    members.append(records)

thresholds = {
    label: 1 if label in {"oxidizing", "gas_under_pressure", "cmr"} else 2
    for label in LABELS
}

out_path = pred_dir / "chemhazard_llm_constrained_ensemble3_predictions.jsonl"
with out_path.open("w", encoding="utf-8") as handle:
    for record_id in gold_ids:
        votes = {
            label: sum(member[record_id][label] for member in members)
            for label in LABELS
        }
        labels = {
            label: int(votes[label] >= thresholds[label])
            for label in LABELS
        }
        handle.write(json.dumps({
            "id": record_id,
            "labels": labels,
            "votes": votes,
            "vote_thresholds": thresholds,
            "parse_status": "ok",
        }, ensure_ascii=False) + "\n")

metadata_path = pred_dir / "chemhazard_llm_constrained_ensemble3_metadata.json"
metadata_path.write_text(json.dumps({
    "gold": str(gold_path),
    "members": [
        str(pred_dir / f"chemhazard_member{member}_predictions.jsonl")
        for member in range(1, 4)
    ],
    "thresholds": thresholds,
    "num_records": len(gold_rows),
    "note": "Thresholds are frozen from the internal experiment and were not tuned on NITE.",
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"saved: {out_path}")
print(f"saved: {metadata_path}")
PY

FINAL_PRED="${PRED_DIR}/chemhazard_llm_constrained_ensemble3_predictions.jsonl"
FINAL_METRIC="${EVAL_DIR}/chemhazard_llm_metrics.json"

python "$EVALUATOR" \
  --gold "$NITE_INPUT" \
  --predictions "$FINAL_PRED" \
  --output "$FINAL_METRIC"

echo "ChemHazard-LLM external evaluation completed."
echo "Prediction file: $FINAL_PRED"
echo "Metric file:     $FINAL_METRIC"
