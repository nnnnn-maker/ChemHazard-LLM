from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score, precision_score, recall_score


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

RARE_FOCUS_LABELS = {"oxidizing", "gas_under_pressure", "cmr"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_gold(path: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    ids: list[str] = []
    records: dict[str, dict[str, int]] = {}
    for item in read_jsonl(path):
        record_id = str(item.get("id", ""))
        labels = item.get("labels")
        if not record_id or record_id in records:
            raise ValueError(f"Missing or duplicate id in {path}: {record_id!r}")
        if not isinstance(labels, dict):
            raise ValueError(f"Missing labels in {path}: id={record_id}")
        ids.append(record_id)
        records[record_id] = {label: int(labels.get(label, 0)) for label in LABELS}
    return ids, records


def load_predictions(
    path: Path, expected_ids: list[str]
) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    for item in read_jsonl(path):
        record_id = str(item.get("id", ""))
        labels = item.get("labels")
        if not record_id or record_id in records:
            raise ValueError(f"Missing or duplicate id in {path}: {record_id!r}")
        if not isinstance(labels, dict):
            raise ValueError(f"Missing labels in {path}: id={record_id}")
        values = {label: int(labels.get(label, 0)) for label in LABELS}
        if any(value not in (0, 1) for value in values.values()):
            raise ValueError(f"Non-binary labels in {path}: id={record_id}")
        records[record_id] = values

    expected = set(expected_ids)
    missing = expected - set(records)
    extra = set(records) - expected
    if missing or extra:
        raise ValueError(
            f"ID mismatch for {path}: missing={len(missing)}, extra={len(extra)}"
        )
    return records


def label_metrics(
    ids: list[str],
    gold: dict[str, dict[str, int]],
    members: list[dict[str, dict[str, int]]],
    label: str,
    threshold: int,
) -> dict[str, float | int]:
    y_true = [gold[record_id][label] for record_id in ids]
    y_pred = [
        int(sum(member[record_id][label] for member in members) >= threshold)
        for record_id in ids
    ]
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "support": int(sum(y_true)),
    }


def best_by_f1(scored: list[tuple[int, dict[str, float | int]]]) -> int:
    # Preserve the historical tie-break: F1 first, then recall, then first threshold.
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"]))[0]


def select_thresholds(
    ids: list[str],
    gold: dict[str, dict[str, int]],
    members: list[dict[str, dict[str, int]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    rare_only: dict[str, int] = {}
    constrained: dict[str, int] = {}
    audit: dict[str, Any] = {}

    for label in LABELS:
        scored = [
            (threshold, label_metrics(ids, gold, members, label, threshold))
            for threshold in (1, 2, 3)
        ]
        rare_only[label] = best_by_f1(scored) if label in RARE_FOCUS_LABELS else 2

        base_precision = next(
            metrics["precision"] for threshold, metrics in scored if threshold == 2
        )
        max_precision_drop = 0.03
        if label in {"oxidizing", "gas_under_pressure"}:
            max_precision_drop = 0.15
        elif label == "cmr":
            max_precision_drop = 0.12

        candidates = [
            item
            for item in scored
            if item[1]["precision"] >= base_precision - max_precision_drop
        ]
        constrained[label] = best_by_f1(candidates)
        audit[label] = {
            "majority_vote_precision": base_precision,
            "maximum_allowed_precision_drop": max_precision_drop,
            "threshold_metrics": {
                str(threshold): metrics for threshold, metrics in scored
            },
            "rare_only_threshold": rare_only[label],
            "constrained_threshold": constrained[label],
        }

    return rare_only, constrained, audit


def write_ensemble(
    path: Path,
    ids: list[str],
    members: list[dict[str, dict[str, int]]],
    thresholds: dict[str, int],
    tag: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id in ids:
            votes = {
                label: sum(member[record_id][label] for member in members)
                for label in LABELS
            }
            labels = {
                label: int(votes[label] >= thresholds[label]) for label in LABELS
            }
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "labels": labels,
                        "votes": votes,
                        "parse_status": tag,
                        "thresholds": thresholds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select ensemble vote thresholds on validation data and apply them to test data."
    )
    parser.add_argument("--val-gold", required=True)
    parser.add_argument("--test-gold", required=True)
    parser.add_argument("--val-pred", nargs=3, required=True)
    parser.add_argument("--test-pred", nargs=3, required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument(
        "--prefix", default="mistral_split123_ghs_pictograms_rare2x_labelcal"
    )
    args = parser.parse_args()

    val_ids, val_gold = load_gold(Path(args.val_gold))
    test_ids, _ = load_gold(Path(args.test_gold))
    val_members = [load_predictions(Path(path), val_ids) for path in args.val_pred]
    test_members = [load_predictions(Path(path), test_ids) for path in args.test_pred]

    rare_only, constrained, audit = select_thresholds(
        val_ids, val_gold, val_members
    )
    output_dir = Path(args.output_dir)
    outputs = {
        "rareonly": output_dir / f"{args.prefix}_rareonly_ensemble3_predictions.jsonl",
        "constrained": output_dir
        / f"{args.prefix}_constrained_ensemble3_predictions.jsonl",
    }
    for name, thresholds in (
        ("rareonly", rare_only),
        ("constrained", constrained),
    ):
        write_ensemble(
            outputs[name],
            test_ids,
            test_members,
            thresholds,
            f"label_calibrated_{name}_ensemble_vote",
        )
        print(f"{name}: {thresholds}")
        print(f"saved: {outputs[name]}")

    metadata_path = output_dir / f"{args.prefix}_threshold_selection.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selection_split": str(args.val_gold),
                "evaluation_split": str(args.test_gold),
                "validation_members": args.val_pred,
                "test_members": args.test_pred,
                "rare_focus_labels": sorted(RARE_FOCUS_LABELS),
                "rareonly_thresholds": rare_only,
                "constrained_thresholds": constrained,
                "validation_audit": audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
