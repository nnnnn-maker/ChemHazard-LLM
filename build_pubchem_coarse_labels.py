from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_ENCODING = "utf-8-sig"
READ_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"]

COARSE_LABEL_MAP = {
    "flammable": ["H220", "H224", "H225", "H226", "H227", "H228"],
    "oxidizing": ["H272"],
    "gas_under_pressure": ["H280"],
    "corrosive": ["H290", "H314", "H318"],
    "acute_toxicity": [
        "H300",
        "H301",
        "H302",
        "H303",
        "H304",
        "H310",
        "H311",
        "H312",
        "H313",
        "H330",
        "H331",
        "H332",
    ],
    "irritant_harmful": ["H315", "H316", "H317", "H319", "H320", "H334", "H335", "H336"],
    "cmr": ["H340", "H341", "H350", "H351", "H360", "H361", "H362"],
    "stot": ["H370", "H371", "H372", "H373"],
    "environmental_hazard": ["H400", "H401", "H402", "H410", "H411", "H412", "H413"],
}


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [{key: normalize_text(value) for key, value in row.items()} for row in reader]
                return rows, list(reader.fieldnames or [])
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return [], []


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=DEFAULT_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u3000", " ")).strip()


def extract_h_codes(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(r"H\d{3}", text.upper())
    seen: set[str] = set()
    ordered: list[str] = []
    for code in matches:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def build_reverse_map() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for coarse_label, h_codes in COARSE_LABEL_MAP.items():
        for h_code in h_codes:
            reverse.setdefault(h_code, []).append(coarse_label)
    return reverse


def map_to_coarse_labels(h_codes: list[str], reverse_map: dict[str, list[str]]) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for h_code in h_codes:
        for coarse_label in reverse_map.get(h_code, []):
            if coarse_label not in seen:
                seen.add(coarse_label)
                matched.append(coarse_label)
    return matched


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Map PubChem GHS H codes to 9 coarse-grained hazard categories."
    )
    parser.add_argument(
        "--input",
        default=str(project_dir / "full_dataset.csv"),
        help="Input PubChem CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv"),
        help="Output CSV with coarse labels.",
    )
    parser.add_argument(
        "--summary",
        default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels_summary.json"),
        help="Output JSON summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    rows, fieldnames = read_csv_rows(input_path)
    reverse_map = build_reverse_map()

    coarse_label_columns = [f"label_{label}" for label in COARSE_LABEL_MAP]
    output_rows: list[dict[str, str]] = []
    h_code_counter: Counter[str] = Counter()
    coarse_label_counter: Counter[str] = Counter()
    label_cardinality_counter: Counter[int] = Counter()
    rows_with_h_codes = 0
    rows_with_coarse_labels = 0

    for row in rows:
        h_codes = extract_h_codes(row.get("H_Statements", ""))
        coarse_labels = map_to_coarse_labels(h_codes, reverse_map)
        if h_codes:
            rows_with_h_codes += 1
        if coarse_labels:
            rows_with_coarse_labels += 1

        h_code_counter.update(h_codes)
        coarse_label_counter.update(coarse_labels)
        label_cardinality_counter[len(coarse_labels)] += 1

        output_row = dict(row)
        output_row["h_codes_extracted"] = "|".join(h_codes)
        output_row["coarse_labels"] = "|".join(coarse_labels)
        output_row["num_h_codes_extracted"] = str(len(h_codes))
        output_row["num_coarse_labels"] = str(len(coarse_labels))
        for label in COARSE_LABEL_MAP:
            output_row[f"label_{label}"] = "1" if label in coarse_labels else "0"
        output_rows.append(output_row)

    output_fieldnames = list(fieldnames)
    for field in ["h_codes_extracted", "coarse_labels", "num_h_codes_extracted", "num_coarse_labels", *coarse_label_columns]:
        if field not in output_fieldnames:
            output_fieldnames.append(field)
    write_csv_rows(output_path, output_rows, output_fieldnames)

    summary = {
        "input_file": str(input_path),
        "num_rows": len(rows),
        "rows_with_h_codes": rows_with_h_codes,
        "rows_with_coarse_labels": rows_with_coarse_labels,
        "coarse_label_map": COARSE_LABEL_MAP,
        "coarse_label_distribution": dict(coarse_label_counter),
        "top_20_h_codes": h_code_counter.most_common(20),
        "label_cardinality_distribution": {str(k): v for k, v in sorted(label_cardinality_counter.items())},
    }
    write_json(summary_path, summary)

    print(f"Rows processed: {len(rows)}")
    print(f"Rows with H codes: {rows_with_h_codes}")
    print(f"Rows with coarse labels: {rows_with_coarse_labels}")
    print(f"Output CSV: {output_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()


