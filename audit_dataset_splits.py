from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:
    raise SystemExit(
        "RDKit is required for scaffold auditing. Install the core requirements first."
    ) from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_smiles(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") != "user":
            continue
        for line in str(message.get("content", "")).splitlines():
            if line.startswith("- ConnectivitySMILES:"):
                return line.split(":", 1)[1].strip()
    return ""


def scaffold_from_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    ) or ""


def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    values = left & right
    return {"count": len(values), "examples": sorted(values)[:20]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit CID, ConnectivitySMILES, and Bemis-Murcko scaffold overlap."
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {
        split: read_jsonl(args.split_root / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    ids = {
        split: {str(record["id"]) for record in split_records}
        for split, split_records in records.items()
    }
    smiles = {
        split: {value for record in split_records if (value := extract_smiles(record))}
        for split, split_records in records.items()
    }
    scaffolds: dict[str, set[str]] = {}
    parsing: dict[str, dict[str, int]] = {}
    for split, split_records in records.items():
        parsed = [scaffold_from_smiles(extract_smiles(record)) for record in split_records]
        nonempty = [value for value in parsed if value]
        scaffolds[split] = set(nonempty)
        parsing[split] = {
            "records": len(split_records),
            "nonempty_or_parsed": len(nonempty),
            "unique_nonempty_scaffolds": len(set(nonempty)),
            "empty_or_failed": len(parsed) - len(nonempty),
        }

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    result = {
        "split_root": str(args.split_root),
        "record_counts": {split: len(value) for split, value in records.items()},
        "id_overlap": {f"{a}_{b}": overlap(ids[a], ids[b]) for a, b in pairs},
        "smiles_overlap": {
            f"{a}_{b}": overlap(smiles[a], smiles[b]) for a, b in pairs
        },
        "scaffold_parsing": parsing,
        "scaffold_overlap": {
            f"{a}_{b}": overlap(scaffolds[a], scaffolds[b]) for a, b in pairs
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
