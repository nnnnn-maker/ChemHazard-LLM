from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_prediction(path: Path, expected_ids: list[str]) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        record_id = str(row.get("id", ""))
        labels = row.get("labels")
        if not record_id or record_id in records:
            raise ValueError(f"Missing or duplicate id in {path}: {record_id!r}")
        if not isinstance(labels, dict) or set(labels) != set(LABELS):
            raise ValueError(f"Invalid nine-label prediction for id={record_id} in {path}")
        if any(labels[label] not in (0, 1, False, True) for label in LABELS):
            raise ValueError(f"Non-binary prediction for id={record_id} in {path}")
        records[record_id] = {label: int(labels[label]) for label in LABELS}

    expected = set(expected_ids)
    missing = expected - set(records)
    extra = set(records) - expected
    if missing or extra:
        raise ValueError(
            f"ID mismatch for {path}: missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={sorted(missing)[:5]}, extra_examples={sorted(extra)[:5]}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multilabel prediction files by label-wise voting.")
    parser.add_argument("--gold", required=True, help="Gold JSONL used only to fix and validate sample IDs.")
    parser.add_argument("--pred", nargs="+", required=True, help="Member prediction JSONL files.")
    parser.add_argument("--output", required=True, help="Output ensemble prediction JSONL.")
    parser.add_argument("--default-threshold", type=int, default=2)
    parser.add_argument("--relaxed-labels", default="")
    parser.add_argument("--relaxed-threshold", type=int, default=1)
    parser.add_argument("--metadata", default="", help="Optional metadata JSON path.")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_paths = [Path(path) for path in args.pred]
    gold_rows = read_jsonl(gold_path)
    gold_ids = [str(row.get("id", "")) for row in gold_rows]
    if not all(gold_ids) or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Gold file contains missing or duplicate IDs.")

    num_members = len(pred_paths)
    if not 1 <= args.default_threshold <= num_members:
        raise ValueError("--default-threshold must be between 1 and the number of members.")
    if not 1 <= args.relaxed_threshold <= num_members:
        raise ValueError("--relaxed-threshold must be between 1 and the number of members.")

    relaxed = {label.strip() for label in args.relaxed_labels.split(",") if label.strip()}
    unknown = relaxed - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown relaxed labels: {sorted(unknown)}")

    members = [validate_prediction(path, gold_ids) for path in pred_paths]
    thresholds = {
        label: args.relaxed_threshold if label in relaxed else args.default_threshold
        for label in LABELS
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record_id in gold_ids:
            votes = {
                label: sum(member[record_id][label] for member in members)
                for label in LABELS
            }
            labels = {label: int(votes[label] >= thresholds[label]) for label in LABELS}
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "labels": labels,
                        "votes": votes,
                        "vote_thresholds": thresholds,
                        "parse_status": "label_wise_vote",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "gold": str(gold_path),
                "members": [str(path) for path in pred_paths],
                "num_members": num_members,
                "num_records": len(gold_ids),
                "thresholds": thresholds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {output_path}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
