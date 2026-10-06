from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

STRUCTURE_FIELDS = {
    "CID",
    "ConnectivitySMILES",
    "IUPACName",
    "MolecularFormula",
}

PHYSCHEM_FIELDS = {
    "MolecularWeight",
    "Complexity",
    "HBondAcceptorCount",
    "HBondDonorCount",
    "HeavyAtomCount",
    "RotatableBondCount",
    "TPSA",
    "XLogP",
    "BoilingPoint",
    "MeltingPoint",
    "FlashPoint",
    "VaporPressure",
    "Density",
    "WaterSolubility",
    "LogP",
    "PhysicalState",
    "Has_Physical_Data",
}

TOXICITY_FIELDS = {
    "LD50_oral",
    "LD50_dermal",
    "LC50_inhalation",
    "Toxicity_Class",
    "IDLH",
    "Has_Toxicity_Data",
}

VARIANTS = {
    "structure_only": {
        "fields": STRUCTURE_FIELDS,
        "pictogram": False,
        "shuffle_pictogram": False,
    },
    "structure_physchem": {
        "fields": STRUCTURE_FIELDS | PHYSCHEM_FIELDS,
        "pictogram": False,
        "shuffle_pictogram": False,
    },
    "structure_physchem_toxicity": {
        "fields": STRUCTURE_FIELDS | PHYSCHEM_FIELDS | TOXICITY_FIELDS,
        "pictogram": False,
        "shuffle_pictogram": False,
    },
    "full_nonleaking": {
        "fields": STRUCTURE_FIELDS | PHYSCHEM_FIELDS | TOXICITY_FIELDS,
        "pictogram": False,
        "shuffle_pictogram": False,
    },
    "pictogram_only": {
        "fields": {"CID"},
        "pictogram": True,
        "shuffle_pictogram": False,
    },
    "full_nonleaking_pictogram": {
        "fields": STRUCTURE_FIELDS | PHYSCHEM_FIELDS | TOXICITY_FIELDS,
        "pictogram": True,
        "shuffle_pictogram": False,
    },
    "full_nonleaking_shuffled_pictogram": {
        "fields": STRUCTURE_FIELDS | PHYSCHEM_FIELDS | TOXICITY_FIELDS,
        "pictogram": True,
        "shuffle_pictogram": True,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def line_field(line: str) -> str | None:
    if not line.startswith("- "):
        return None
    body = line[2:]
    if ":" not in body:
        return None
    return body.split(":", 1)[0].strip()


def extract_user_content(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def extract_pictogram(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("- Pictograms:"):
            return line.split(":", 1)[1].strip()
    return ""


def make_user_content(content: str, keep_fields: set[str], include_pictogram: bool, pictogram_value: str) -> str:
    lines = [
        "Classify the chemical into the predefined coarse-grained hazard categories.",
        "Chemical information:",
    ]
    for line in content.splitlines():
        field = line_field(line)
        if field in keep_fields:
            lines.append(line)
    if include_pictogram and pictogram_value:
        lines.append("Weak GHS pictogram evidence:")
        lines.append(f"- Pictograms: {pictogram_value}")
    lines.append("Return only JSON.")
    return "\n".join(lines)


def transform_record(record: dict[str, Any], variant: dict[str, Any], pictogram_value: str) -> dict[str, Any]:
    transformed = json.loads(json.dumps(record, ensure_ascii=False))
    for message in transformed.get("messages", []):
        if message.get("role") == "user":
            message["content"] = make_user_content(
                str(message.get("content", "")),
                set(variant["fields"]),
                bool(variant["pictogram"]),
                pictogram_value,
            )
    return transformed


def parse_variants(text: str) -> list[str]:
    if not text.strip() or text.strip().lower() == "all":
        return list(VARIANTS.keys())
    variants = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [variant for variant in variants if variant not in VARIANTS]
    if invalid:
        raise ValueError(f"Invalid variants: {invalid}. Valid variants: {sorted(VARIANTS)}")
    return variants


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build clean evidence-ablation SFT JSONL variants from one base split.")
    parser.add_argument(
        "--base-root",
        default=str(project_dir / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"),
        help="Base SFT split root containing train/val/test JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "sft_clean_ablation_seed123_rare2x"),
        help="Output root. Each variant is written to output-root/<variant>/.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=f"Comma-separated variants or 'all'. Valid: {','.join(sorted(VARIANTS))}",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    selected_variants = parse_variants(args.variants)
    summary: dict[str, Any] = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "variants": selected_variants,
        "splits": [],
    }

    for split in ["train", "val", "test"]:
        records = read_jsonl(base_root / f"{split}.jsonl")
        true_pictograms = [extract_pictogram(extract_user_content(record)) for record in records]
        shuffled_pictograms = list(true_pictograms)
        random.Random(args.seed + {"train": 1, "val": 2, "test": 3}[split]).shuffle(shuffled_pictograms)

        for name in selected_variants:
            variant = VARIANTS[name]
            pictograms = shuffled_pictograms if variant["shuffle_pictogram"] else true_pictograms
            transformed = [
                transform_record(record, variant, pictogram)
                for record, pictogram in zip(records, pictograms)
            ]
            out_path = output_root / name / f"{split}.jsonl"
            write_jsonl(out_path, transformed)
            item = {
                "split": split,
                "variant": name,
                "num_records": len(transformed),
                "path": str(out_path),
            }
            summary["splits"].append(item)
            print(json.dumps(item, ensure_ascii=False))

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
