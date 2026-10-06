from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from build_pubchem_coarse_sft_dataset import (
    GHS_EVIDENCE_COLUMNS,
    LABEL_COLUMNS,
    LABEL_NAMES,
    allocate_counts,
    build_record,
    build_uniform_oversample_spec,
    label_counts,
    oversample_train_records,
    parse_column_names,
    parse_keywords,
    parse_label_names,
    parse_oversample_spec,
    read_rows,
    row_has_any_label,
    write_jsonl,
)


def scaffold_from_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:
        raise SystemExit(f"RDKit is required for scaffold split: {exc}") from exc

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return "INVALID_SMILES"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or "NO_SCAFFOLD"


def row_labels(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    for column in LABEL_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value in {"1", "1.0", "True", "true"}:
            labels.append(column.removeprefix("label_"))
    return labels


def group_rows_by_scaffold(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[scaffold_from_smiles(row.get("ConnectivitySMILES", ""))].append(row)
    return dict(groups)


def group_label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for row in rows:
        for label in row_labels(row):
            counts[label] += 1
    return counts


def scaffold_split_rows(
    rows: list[dict[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict]:
    groups = group_rows_by_scaffold(rows)
    split_names = ["train", "val", "test"]
    targets = dict(zip(split_names, allocate_counts(len(rows), [1.0 - val_ratio - test_ratio, val_ratio, test_ratio])))

    total_label_counts = {label: 0 for label in LABEL_NAMES}
    for row in rows:
        for label in row_labels(row):
            total_label_counts[label] += 1

    label_targets = {split: {} for split in split_names}
    for label, total in total_label_counts.items():
        train_count, val_count, test_count = allocate_counts(total, [1.0 - val_ratio - test_ratio, val_ratio, test_ratio])
        label_targets["train"][label] = train_count
        label_targets["val"][label] = val_count
        label_targets["test"][label] = test_count

    rng = random.Random(seed)
    scaffold_items = list(groups.items())
    rng.shuffle(scaffold_items)
    scaffold_items.sort(
        key=lambda item: (
            len(item[1]),
            sum(1 for value in group_label_counts(item[1]).values() if value > 0),
        ),
        reverse=True,
    )

    assigned_rows = {split: [] for split in split_names}
    assigned_label_counts = {split: {label: 0 for label in LABEL_NAMES} for split in split_names}

    for scaffold, group in scaffold_items:
        group_counts = group_label_counts(group)
        group_size = len(group)

        def score(split: str) -> tuple[float, float, float]:
            size_deficit = targets[split] - len(assigned_rows[split])
            label_deficit_score = 0.0
            for label, count in group_counts.items():
                if count <= 0:
                    continue
                deficit = label_targets[split][label] - assigned_label_counts[split][label]
                label_deficit_score += min(deficit, count)
            oversize_penalty = min(0, size_deficit - group_size)
            return (label_deficit_score, size_deficit + oversize_penalty * 2, rng.random())

        best_split = max(split_names, key=score)
        assigned_rows[best_split].extend(group)
        for label, count in group_counts.items():
            assigned_label_counts[best_split][label] += count

    metadata = {
        "num_scaffolds": len(groups),
        "largest_scaffold_size": max((len(group) for group in groups.values()), default=0),
        "target_sizes": targets,
        "actual_sizes": {split: len(assigned_rows[split]) for split in split_names},
        "actual_label_counts": assigned_label_counts,
    }
    return assigned_rows["train"], assigned_rows["val"], assigned_rows["test"], metadata


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build PubChem coarse SFT splits grouped by Bemis-Murcko scaffold.")
    parser.add_argument("--input", default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv"))
    parser.add_argument("--output-root", default=str(project_dir / "sft_coarse_jsonl_scaffold_seed123"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-empty-labels", action="store_true")
    parser.add_argument("--include-derived-environmental-cues", action="store_true")
    parser.add_argument("--include-ghs-evidence", action="store_true")
    parser.add_argument("--ghs-evidence-columns", default=",".join(GHS_EVIDENCE_COLUMNS))
    parser.add_argument("--include-non-hcode-evidence", action="store_true")
    parser.add_argument("--include-hazard-rubric", action="store_true")
    parser.add_argument("--train-oversample-factor", type=int, default=1)
    parser.add_argument("--train-oversample-labels", default=",".join(["oxidizing", "gas_under_pressure", "cmr", "stot", "environmental_hazard"]))
    parser.add_argument("--train-oversample-spec", default="")
    parser.add_argument("--hard-positive-label", default="", choices=[""] + LABEL_NAMES)
    parser.add_argument("--hard-positive-keywords", default="")
    parser.add_argument("--hard-positive-factor", type=float, default=1.0)
    args = parser.parse_args()

    if args.train_oversample_factor < 1:
        raise SystemExit("--train-oversample-factor must be >= 1.")
    if args.hard_positive_factor < 1.0:
        raise SystemExit("--hard-positive-factor must be >= 1.0.")

    rows = read_rows(Path(args.input))
    if not args.include_empty_labels:
        rows = [row for row in rows if row_has_any_label(row)]

    train_rows, val_rows, test_rows, scaffold_metadata = scaffold_split_rows(
        rows,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    ghs_evidence_columns = parse_column_names(
        args.ghs_evidence_columns,
        GHS_EVIDENCE_COLUMNS,
        "--ghs-evidence-columns",
    )

    def records_from_rows(split_rows: list[dict[str, str]]) -> list[dict]:
        return [
            build_record(
                row,
                args.include_derived_environmental_cues,
                args.include_ghs_evidence,
                ghs_evidence_columns,
                args.include_non_hcode_evidence,
                args.include_hazard_rubric,
            )
            for row in split_rows
        ]

    train_records = records_from_rows(train_rows)
    val_records = records_from_rows(val_rows)
    test_records = records_from_rows(test_rows)

    oversample_labels = parse_label_names(args.train_oversample_labels)
    oversample_spec = parse_oversample_spec(args.train_oversample_spec)
    if not oversample_spec:
        oversample_spec = build_uniform_oversample_spec(oversample_labels, args.train_oversample_factor)
    hard_positive_keywords = parse_keywords(args.hard_positive_keywords)
    train_records_before_oversampling = list(train_records)
    train_records, hard_positive_matches = oversample_train_records(
        train_records,
        oversample_spec,
        args.hard_positive_label,
        hard_positive_keywords,
        args.hard_positive_factor,
        args.seed,
    )

    output_root = Path(args.output_root)
    write_jsonl(output_root / "train.jsonl", train_records)
    write_jsonl(output_root / "val.jsonl", val_records)
    write_jsonl(output_root / "test.jsonl", test_records)

    summary = {
        "input": str(Path(args.input)),
        "output_root": str(output_root),
        "seed": args.seed,
        "split_method": "bemis_murcko_scaffold",
        "scaffold_metadata": scaffold_metadata,
        "num_records": len(train_records_before_oversampling) + len(val_records) + len(test_records),
        "num_train_before_oversampling": len(train_records_before_oversampling),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_test": len(test_records),
        "train_oversample_factor": args.train_oversample_factor,
        "train_oversample_labels": oversample_labels,
        "train_oversample_spec": oversample_spec,
        "hard_positive_label": args.hard_positive_label,
        "hard_positive_keywords": hard_positive_keywords,
        "hard_positive_factor": args.hard_positive_factor,
        "hard_positive_matches_train": hard_positive_matches,
        "include_ghs_evidence": args.include_ghs_evidence,
        "ghs_evidence_columns": ghs_evidence_columns if args.include_ghs_evidence else [],
        "include_non_hcode_evidence": args.include_non_hcode_evidence,
        "include_derived_environmental_cues": args.include_derived_environmental_cues,
        "include_hazard_rubric": args.include_hazard_rubric,
        "label_counts_train_before_oversampling": label_counts(train_records_before_oversampling),
        "label_counts_train": label_counts(train_records),
        "label_counts_val": label_counts(val_records),
        "label_counts_test": label_counts(test_records),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_records)} train records to: {output_root / 'train.jsonl'}")
    print(f"Wrote {len(val_records)} val records to: {output_root / 'val.jsonl'}")
    print(f"Wrote {len(test_records)} test records to: {output_root / 'test.jsonl'}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
