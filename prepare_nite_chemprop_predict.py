from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare id+SMILES input for frozen Chemprop prediction on NITE."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--skip-rdkit-validation", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    missing: list[str] = []
    invalid: list[str] = []

    chem = None
    if not args.skip_rdkit_validation:
        try:
            from rdkit import Chem, RDLogger
        except Exception as exc:
            raise RuntimeError(f"RDKit is required for Chemprop input validation: {exc}") from exc
        RDLogger.DisableLog("rdApp.*")
        chem = Chem

    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id or record_id in seen:
            raise ValueError(f"Missing or duplicate NITE id: {record_id!r}")
        seen.add(record_id)
        smiles = extract_smiles(record)
        if not smiles:
            missing.append(record_id)
            continue
        if chem is not None and chem.MolFromSmiles(smiles) is None:
            invalid.append(f"{record_id}:{smiles}")
            continue
        rows.append({"id": record_id, "smiles": smiles})

    audit = {
        "input_records": len(records),
        "written_records": len(rows),
        "missing_smiles": len(missing),
        "invalid_smiles": len(invalid),
        "missing_examples": missing[:20],
        "invalid_examples": invalid[:20],
        "external_labels_used": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing or invalid or len(rows) != len(records):
        raise SystemExit(
            "Chemprop external input is incomplete; evaluation is blocked. "
            f"See {args.audit}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
