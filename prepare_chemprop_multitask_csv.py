from __future__ import annotations

import argparse
import csv
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


def extract_smiles(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") != "user":
            continue
        for line in message.get("content", "").splitlines():
            if line.startswith("- ConnectivitySMILES:"):
                return line.split(":", 1)[1].strip()
    return ""


def extract_labels(record: dict[str, Any]) -> dict[str, int]:
    if "labels" in record:
        return {label: int(record["labels"].get(label, 0)) for label in LABELS}
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            payload = json.loads(message.get("content", "{}"))
            return {label: int(payload.get(label, 0)) for label in LABELS}
    raise ValueError(f"Missing labels for record id={record.get('id', '')}")


def is_valid_rdkit_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem, RDLogger
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "RDKit is required for Chemprop SMILES validation. "
            "Run this script in the Chemprop/RDKit environment or pass --skip-rdkit-validation."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles) is not None


def convert_split(
    split_root: Path,
    output_dir: Path,
    split: str,
    drop_missing_smiles: bool,
    validate_rdkit: bool,
) -> dict[str, int | str]:
    rows: list[dict[str, object]] = []
    kept_records: list[dict[str, Any]] = []
    skipped_missing_smiles = 0
    skipped_invalid_smiles = 0
    invalid_examples: list[str] = []
    records = read_jsonl(split_root / f"{split}.jsonl")
    for record in records:
        smiles = extract_smiles(record)
        if not smiles:
            skipped_missing_smiles += 1
            if drop_missing_smiles:
                continue
        if smiles and validate_rdkit and not is_valid_rdkit_smiles(smiles):
            skipped_invalid_smiles += 1
            if len(invalid_examples) < 20:
                invalid_examples.append(f"{record.get('id', '')}:{smiles}")
            continue
        labels = extract_labels(record)
        rows.append({"id": str(record.get("id", "")), "smiles": smiles, **labels})
        kept_records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", *LABELS])
        writer.writeheader()
        writer.writerows(rows)

    predict_path = output_dir / f"{split}_predict_input.csv"
    with predict_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "smiles": row["smiles"]} for row in rows)

    filtered_gold_path = output_dir / f"{split}_gold.jsonl"
    with filtered_gold_path.open("w", encoding="utf-8") as handle:
        for record in kept_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "split": split,
        "input_records": len(records),
        "written_records": len(rows),
        "skipped_missing_smiles": skipped_missing_smiles,
        "skipped_invalid_smiles": skipped_invalid_smiles,
        "invalid_examples": invalid_examples,
        "csv_path": str(csv_path),
        "predict_path": str(predict_path),
        "filtered_gold_path": str(filtered_gold_path),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert PubChem SFT JSONL splits to Chemprop multitask CSV files.")
    parser.add_argument(
        "--split-root",
        default=str(project_dir / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"),
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "analysis_outputs" / "chemprop_seed123"),
        help="Directory for Chemprop train/val/test CSV files.",
    )
    parser.add_argument(
        "--keep-missing-smiles",
        action="store_true",
        help="Keep rows with empty SMILES instead of dropping them.",
    )
    parser.add_argument(
        "--skip-rdkit-validation",
        action="store_true",
        help="Do not drop SMILES that RDKit cannot parse. Not recommended for Chemprop.",
    )
    args = parser.parse_args()

    split_root = Path(args.split_root)
    output_dir = Path(args.output_dir)
    summary = {
        "split_root": str(split_root),
        "output_dir": str(output_dir),
        "labels": LABELS,
        "splits": [],
    }

    for split in ["train", "val", "test"]:
        result = convert_split(
            split_root=split_root,
            output_dir=output_dir,
            split=split,
            drop_missing_smiles=not args.keep_missing_smiles,
            validate_rdkit=not args.skip_rdkit_validation,
        )
        summary["splits"].append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
