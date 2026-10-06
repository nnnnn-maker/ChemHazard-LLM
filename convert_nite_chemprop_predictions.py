from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Chemprop NITE probabilities to complete nine-label JSONL."
    )
    parser.add_argument("--predict-input", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    input_rows = read_csv(args.predict_input)
    raw_rows = read_csv(args.raw)
    if len(input_rows) != len(raw_rows):
        raise ValueError(f"Chemprop row mismatch: input={len(input_rows)} raw={len(raw_rows)}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")

    input_ids = [str(row.get("id", "")) for row in input_rows]
    raw_ids = [str(row.get("id", "")) for row in raw_rows]
    raw_has_ids = all(raw_ids)
    if raw_has_ids and raw_ids != input_ids:
        raise ValueError("Chemprop output ids/order do not match the NITE prediction input")

    missing_columns = [label for label in LABELS if raw_rows and label not in raw_rows[0]]
    if missing_columns:
        raise ValueError(f"Missing Chemprop probability columns: {missing_columns}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record_id, row in zip(input_ids, raw_rows):
            scores = {label: float(row[label]) for label in LABELS}
            if any(not math.isfinite(value) for value in scores.values()):
                raise ValueError(f"Non-finite Chemprop probability for id={record_id}")
            labels = {label: int(score >= args.threshold) for label, score in scores.items()}
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "labels": labels,
                        "scores": scores,
                        "parse_status": (
                            f"ok:chemprop_frozen_pubchem_seed123:threshold_{args.threshold:g}"
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
