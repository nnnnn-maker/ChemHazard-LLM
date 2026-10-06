from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = [
    "morgan_physchem_rf",
    "physchem_rf",
    "morgan_rf",
    "physchem_logreg",
    "morgan_logreg",
    "morgan_physchem_classifier_chain_rf",
    "chemprop",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect seven NITE external baseline metrics.")
    parser.add_argument("--metric-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    expected_integrity: dict[str, int] | None = None
    for model in MODELS:
        path = args.metric_dir / f"{model}_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metric file: {path}")
        metric = json.loads(path.read_text(encoding="utf-8"))
        integrity = {
            key: int(metric.get(key, -1))
            for key in [
                "gold_records",
                "prediction_records",
                "matched_records",
                "missing_gold_ids_in_predictions",
                "extra_prediction_ids",
                "scored_label_cells",
                "missing_prediction_cells",
            ]
        }
        if expected_integrity is None:
            expected_integrity = integrity
        elif integrity != expected_integrity:
            raise ValueError(f"Metric integrity differs for {model}: {integrity}")
        rows.append(
            {
                "model": model,
                "micro_precision": metric["micro_precision"],
                "micro_recall": metric["micro_recall"],
                "micro_f1": metric["micro_f1"],
                "macro_f1": metric["macro_f1"],
                "micro_accuracy": metric["micro_accuracy"],
                **integrity,
            }
        )

    if expected_integrity is None:
        raise RuntimeError("No metrics found")
    if (
        expected_integrity["gold_records"] != expected_integrity["matched_records"]
        or expected_integrity["prediction_records"] != expected_integrity["gold_records"]
        or expected_integrity["missing_gold_ids_in_predictions"] != 0
        or expected_integrity["extra_prediction_ids"] != 0
        or expected_integrity["missing_prediction_cells"] != 0
    ):
        raise ValueError(f"Incomplete external comparison: {expected_integrity}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
