from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = [
    "flammable", "oxidizing", "gas_under_pressure", "corrosive",
    "acute_toxicity", "irritant_harmful", "cmr", "stot", "environmental_hazard",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Chemprop probability CSV to PubChem prediction JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        missing = [label for label in LABELS if label not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing Chemprop columns: {missing}; columns={reader.fieldnames}")
        for row in reader:
            scores = {label: float(row[label]) for label in LABELS}
            labels = {label: int(scores[label] >= args.threshold) for label in LABELS}
            target.write(
                json.dumps(
                    {
                        "id": str(row.get("id", "")),
                        "labels": labels,
                        "scores": scores,
                        "parse_status": f"chemprop_threshold_{args.threshold:g}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
