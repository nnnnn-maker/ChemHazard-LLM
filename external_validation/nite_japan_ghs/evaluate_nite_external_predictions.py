from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = [
    "flammable", "oxidizing", "gas_under_pressure", "corrosive", "acute_toxicity",
    "irritant_harmful", "cmr", "stot", "environmental_hazard",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def div(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ChemHazard predictions against masked NITE external labels.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold = {str(row["id"]): row for row in read_jsonl(args.gold)}
    pred = {str(row["id"]): row for row in read_jsonl(args.predictions)}
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "scored": 0, "missing_prediction": 0} for label in LABELS}

    matched_ids = sorted(set(gold) & set(pred))
    for record_id in matched_ids:
        g = gold[record_id]
        p_labels = pred[record_id].get("labels", {})
        for label in LABELS:
            if not int(g.get("label_mask", {}).get(label, 0)):
                continue
            truth = g.get("labels", {}).get(label)
            value = p_labels.get(label)
            if value not in (0, 1, False, True):
                counts[label]["missing_prediction"] += 1
                continue
            truth, value = int(truth), int(value)
            counts[label]["scored"] += 1
            if truth == 1 and value == 1: counts[label]["tp"] += 1
            elif truth == 0 and value == 1: counts[label]["fp"] += 1
            elif truth == 1 and value == 0: counts[label]["fn"] += 1
            else: counts[label]["tn"] += 1

    per_label = {}
    for label, c in counts.items():
        precision = div(c["tp"], c["tp"] + c["fp"])
        recall = div(c["tp"], c["tp"] + c["fn"])
        per_label[label] = {
            **c,
            "precision": precision,
            "recall": recall,
            "f1": div(2 * precision * recall, precision + recall),
            "accuracy": div(c["tp"] + c["tn"], c["scored"]),
        }

    total = {key: sum(c[key] for c in counts.values()) for key in ["tp", "fp", "fn", "tn", "scored", "missing_prediction"]}
    micro_precision = div(total["tp"], total["tp"] + total["fp"])
    micro_recall = div(total["tp"], total["tp"] + total["fn"])
    active = [value for value in per_label.values() if value["scored"]]
    result = {
        "gold_records": len(gold),
        "prediction_records": len(pred),
        "matched_records": len(matched_ids),
        "missing_gold_ids_in_predictions": len(set(gold) - set(pred)),
        "extra_prediction_ids": len(set(pred) - set(gold)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "micro_accuracy": div(total["tp"] + total["tn"], total["scored"]),
        "macro_f1": div(sum(value["f1"] for value in active), len(active)),
        "scored_label_cells": total["scored"],
        "missing_prediction_cells": total["missing_prediction"],
        "per_label": per_label,
        "warning": "Scores use only label_mask=1 cells; unknown NITE labels are excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
