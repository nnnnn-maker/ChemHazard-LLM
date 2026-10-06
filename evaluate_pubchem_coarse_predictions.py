from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import hamming_loss, precision_recall_fscore_support

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


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_binary(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    return 0


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError("No JSON object found")


def last_assistant_content(messages: Iterable[dict]) -> str:
    assistant_messages = [message.get("content", "") for message in messages if message.get("role") == "assistant"]
    return assistant_messages[-1] if assistant_messages else ""


def extract_labels(record: dict) -> dict[str, int]:
    candidates: list[object] = []
    for key in ["labels", "prediction", "predictions", "output", "outputs"]:
        if key in record:
            candidates.append(record[key])
    for key in ["assistant", "response", "completion", "prediction_text", "output_text"]:
        if key in record:
            candidates.append(record[key])
    if "messages" in record:
        candidates.append(last_assistant_content(record["messages"]))

    for candidate in candidates:
        if isinstance(candidate, dict):
            if all(label in candidate for label in LABELS):
                return {label: normalize_binary(candidate.get(label, 0)) for label in LABELS}
        elif isinstance(candidate, str) and candidate.strip():
            try:
                parsed = parse_json_object(candidate)
            except Exception:
                continue
            if all(label in parsed for label in LABELS):
                return {label: normalize_binary(parsed.get(label, 0)) for label in LABELS}

    raise ValueError("Could not extract label JSON from record")


def subset_accuracy(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    if not y_true:
        return 0.0
    matches = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    return matches / len(y_true)


def align_records(gold_records: list[dict], pred_records: list[dict]) -> tuple[list[dict], list[dict]]:
    gold_by_id = {str(record.get("id", idx)): record for idx, record in enumerate(gold_records)}
    pred_by_id = {str(record.get("id", idx)): record for idx, record in enumerate(pred_records)}
    shared_ids = [record_id for record_id in gold_by_id if record_id in pred_by_id]
    missing_pred = [record_id for record_id in gold_by_id if record_id not in pred_by_id]
    if missing_pred:
        raise SystemExit(f"Prediction file is missing {len(missing_pred)} ids. Example: {missing_pred[:5]}")
    return [gold_by_id[record_id] for record_id in shared_ids], [pred_by_id[record_id] for record_id in shared_ids]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_leaderboard(path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "run_name",
        "num_samples",
        "parse_failures",
        "subset_accuracy",
        "micro_f1",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "macro_precision",
        "macro_recall",
        "hamming_loss",
        "pred_file",
    ]
    existing_rows: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        existing_rows = [existing for existing in existing_rows if existing.get("run_name") != row["run_name"]]
    existing_rows.append(row)
    existing_rows.sort(key=lambda item: float(item["micro_f1"]), reverse=True)
    write_csv(path, existing_rows, fieldnames)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions for PubChem coarse hazard labels.")
    parser.add_argument(
        "--gold",
        default=str(project_dir / "sft_coarse_jsonl" / "test.jsonl"),
        help="Gold JSONL file.",
    )
    parser.add_argument("--pred", required=True, help="Prediction JSONL file.")
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "evaluation_llm_coarse"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="run",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--leaderboard-path",
        default=str(project_dir / "evaluation_llm_coarse" / "leaderboard.csv"),
        help="CSV file for accumulating run summaries.",
    )
    args = parser.parse_args()

    gold_records = read_jsonl(Path(args.gold))
    pred_records = read_jsonl(Path(args.pred))
    gold_records, pred_records = align_records(gold_records, pred_records)

    y_true: list[list[int]] = []
    y_pred: list[list[int]] = []
    error_rows: list[dict[str, object]] = []
    parse_failures = 0

    for gold_record, pred_record in zip(gold_records, pred_records):
        gold = extract_labels(gold_record)
        try:
            pred = extract_labels(pred_record)
            parse_status = pred_record.get("parse_status", "ok")
        except Exception as exc:
            pred = {label: 0 for label in LABELS}
            parse_status = f"parse_failed: {type(exc).__name__}"
            parse_failures += 1

        gold_vector = [gold[label] for label in LABELS]
        pred_vector = [pred[label] for label in LABELS]
        y_true.append(gold_vector)
        y_pred.append(pred_vector)

        if gold_vector != pred_vector:
            error_rows.append(
                {
                    "id": gold_record.get("id", ""),
                    "parse_status": parse_status,
                    "gold": json.dumps(gold, ensure_ascii=False, sort_keys=True),
                    "pred": json.dumps(pred, ensure_ascii=False, sort_keys=True),
                }
            )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        "num_samples": len(y_true),
        "parse_failures": parse_failures,
        "subset_accuracy": subset_accuracy(y_true, y_pred),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "hamming_loss": hamming_loss(y_true, y_pred),
        "pred_file": str(Path(args.pred)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    per_label_rows = []
    for idx, label in enumerate(LABELS):
        per_label_rows.append(
            {
                "label": label,
                "precision": per_label_precision[idx],
                "recall": per_label_recall[idx],
                "f1": per_label_f1[idx],
                "support": int(per_label_support[idx]),
            }
        )
    write_csv(output_dir / f"{args.prefix}_per_label.csv", per_label_rows, ["label", "precision", "recall", "f1", "support"])
    write_csv(output_dir / f"{args.prefix}_errors.csv", error_rows, ["id", "parse_status", "gold", "pred"])

    leaderboard_row = {"run_name": args.prefix, **metrics}
    append_leaderboard(Path(args.leaderboard_path), leaderboard_row)

    print(f"Metrics written to: {metrics_path}")
    print(f"Per-label metrics written to: {output_dir / f'{args.prefix}_per_label.csv'}")
    print(f"Error cases written to: {output_dir / f'{args.prefix}_errors.csv'}")
    print(f"Leaderboard updated: {args.leaderboard_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
