from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DISPLAY_METRICS = [
    ("Precision", "micro_precision"),
    ("Recall", "micro_recall"),
    ("F1-score", "micro_f1"),
    ("Macro-F1", "macro_f1"),
    ("subset_accuracy", "subset_accuracy"),
    ("hamming_loss", "hamming_loss"),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_metrics_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.leaderboard:
        rows = read_csv_rows(Path(args.leaderboard))
        if not rows:
            raise SystemExit(f"No rows found in leaderboard: {args.leaderboard}")
        normalized_rows: list[dict[str, object]] = []
        for row in rows:
            normalized = {key: value for key, value in row.items()}
            if "run_name" not in normalized and "model" in normalized:
                normalized["run_name"] = str(normalized["model"])
            normalized_rows.append(normalized)
        return normalized_rows

    input_dir = Path(args.input_dir)
    metric_paths = sorted(input_dir.glob("*_metrics.json"))
    if not metric_paths:
        metric_paths = sorted(input_dir.glob("*/*_metrics.json"))
    if not metric_paths:
        metric_paths = sorted(input_dir.glob("*/metrics.json"))
    if not metric_paths:
        raise SystemExit(f"No metric JSON files found under: {input_dir}")

    rows: list[dict[str, object]] = []
    for metric_path in metric_paths:
        payload = read_metrics_json(metric_path)
        if "test" in payload and isinstance(payload["test"], dict):
            payload = payload["test"].get("metrics", {})
            run_name = metric_path.parent.name
        else:
            run_name = metric_path.name.removesuffix("_metrics.json")
        rows.append(
            {
                "run_name": run_name,
                **payload,
            }
        )
    return rows


def to_float(row: dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    if value in (None, ""):
        return 0.0
    return float(value)


def select_rows(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    if args.runs:
        order = [item.strip() for item in args.runs.split(",") if item.strip()]
        rows_by_name = {str(row.get("run_name", "")): row for row in rows}
        missing = [name for name in order if name not in rows_by_name]
        if missing:
            raise SystemExit(f"Requested runs not found: {missing}")
        return [rows_by_name[name] for name in order]

    return sorted(rows, key=lambda row: str(row.get("run_name", "")))


def build_table_rows(rows: list[dict[str, object]]) -> tuple[list[str], list[list[str]]]:
    headers = [""] + [str(row.get("run_name", "")) for row in rows]
    table_rows: list[list[str]] = []
    for display_name, metric_key in DISPLAY_METRICS:
        table_rows.append(
            [display_name] + [f"{to_float(row, metric_key):.4f}" for row in rows]
        )
    return headers, table_rows


def write_wide_csv(path: Path, headers: list[str], table_rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table_rows)


def write_markdown(path: Path, headers: list[str], table_rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a wide comparison table for PubChem coarse LLM evaluation runs."
    )
    parser.add_argument(
        "--leaderboard",
        default=str(project_dir / "evaluation_llm_coarse" / "leaderboard.csv"),
        help="Leaderboard CSV written by evaluate_pubchem_coarse_predictions.py.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(project_dir / "evaluation_llm_coarse"),
        help="Fallback directory for *_metrics.json files when --leaderboard is unavailable.",
    )
    parser.add_argument(
        "--runs",
        default="",
        help="Comma-separated run_name order for columns, e.g. qwen,mistral,gemma,chatglm,chemllm.",
    )
    parser.add_argument(
        "--csv-output",
        default=str(project_dir / "evaluation_llm_coarse" / "comparison_table.csv"),
        help="Wide CSV output path.",
    )
    parser.add_argument(
        "--md-output",
        default=str(project_dir / "evaluation_llm_coarse" / "comparison_table.md"),
        help="Markdown table output path.",
    )
    args = parser.parse_args()

    leaderboard_path = Path(args.leaderboard)
    if leaderboard_path.exists():
        rows = load_rows(args)
    else:
        rows = load_rows(
            argparse.Namespace(
                leaderboard="",
                input_dir=args.input_dir,
            )
        )

    selected_rows = select_rows(rows, args)
    if not selected_rows:
        raise SystemExit("No runs available to summarize.")

    headers, table_rows = build_table_rows(selected_rows)
    write_wide_csv(Path(args.csv_output), headers, table_rows)
    write_markdown(Path(args.md_output), headers, table_rows)
    print(f"CSV written to: {args.csv_output}")
    print(f"Markdown written to: {args.md_output}")


if __name__ == "__main__":
    main()
