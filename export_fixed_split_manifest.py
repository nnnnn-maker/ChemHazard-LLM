from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fixed split manifest. Repeated training rows caused by "
            "target-label oversampling are represented by exposure_count."
        )
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str | int]] = []
    ids_by_split: dict[str, set[str]] = {}
    file_hashes: dict[str, str] = {}
    raw_counts: dict[str, int] = {}
    unique_counts: dict[str, int] = {}

    for split in SPLITS:
        path = args.split_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        records = read_jsonl(path)
        identifiers = [str(record.get("id", "")) for record in records]
        if any(not identifier for identifier in identifiers):
            raise ValueError(f"Missing id in {path}")
        counts = Counter(identifiers)
        ids_by_split[split] = set(counts)
        raw_counts[split] = len(records)
        unique_counts[split] = len(counts)
        file_hashes[f"{split}.jsonl"] = file_sha256(path)
        rows.extend(
            {
                "id": identifier,
                "split": "validation" if split == "val" else split,
                "exposure_count": exposure_count,
            }
            for identifier, exposure_count in sorted(counts.items())
        )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids_by_split[left] & ids_by_split[right]
        if overlap:
            raise ValueError(
                f"Cross-split ID overlap for {left}/{right}: "
                f"count={len(overlap)}, examples={sorted(overlap)[:5]}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "split", "exposure_count"]
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "split_root": str(args.split_root),
                "raw_row_counts": raw_counts,
                "unique_id_counts": unique_counts,
                "jsonl_sha256": file_hashes,
                "manifest_sha256": file_sha256(args.output),
                "note": (
                    "exposure_count is greater than one only when the training "
                    "JSONL contains replayed rows from target-label oversampling"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {args.output}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
