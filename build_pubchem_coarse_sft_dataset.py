from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable

LABEL_COLUMNS = [
    "label_flammable",
    "label_oxidizing",
    "label_gas_under_pressure",
    "label_corrosive",
    "label_acute_toxicity",
    "label_irritant_harmful",
    "label_cmr",
    "label_stot",
    "label_environmental_hazard",
]

LABEL_NAMES = [label.removeprefix("label_") for label in LABEL_COLUMNS]
DEFAULT_RARE_OVERSAMPLE_LABELS = [
    "oxidizing",
    "gas_under_pressure",
    "cmr",
    "stot",
    "environmental_hazard",
]

FEATURE_COLUMNS = [
    "CID",
    "ConnectivitySMILES",
    "IUPACName",
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "BoilingPoint",
    "MeltingPoint",
    "FlashPoint",
    "VaporPressure",
    "Density",
    "WaterSolubility",
    "PhysicalState",
    "Signal_Word",
    "LD50_oral",
    "LD50_dermal",
    "LC50_inhalation",
    "Toxicity_Class",
    "IDLH",
]

GHS_EVIDENCE_COLUMNS = [
    "Has_GHS_Data",
    "GHS_Classifications",
    "H_Statements",
    "Pictograms",
]

NON_HCODE_EVIDENCE_COLUMNS = [
    "NonHCode_Ecotoxicity_Text",
    "NonHCode_Environmental_Fate_Text",
    "NonHCode_Bioaccumulation_Text",
    "NonHCode_Evidence_Summary",
]

SYSTEM_PROMPT = (
    "You are a chemical hazard classification assistant. "
    "Given structured chemical information, predict the nine coarse-grained hazard labels. "
    "Return only one JSON object with exactly these keys: "
    "flammable, oxidizing, gas_under_pressure, corrosive, acute_toxicity, "
    "irritant_harmful, cmr, stot, environmental_hazard. "
    "Each value must be 0 or 1."
)

HAZARD_RUBRIC_LINES = [
    "- flammable: consider low flash point, volatile liquids/gases, flammable physical state, and flame pictogram evidence.",
    "- oxidizing: consider oxidizer pictogram/evidence and oxidizer-like structural or formula cues such as peroxide, nitrate, chlorate, or oxygen-rich inorganic salts.",
    "- gas_under_pressure: consider gas/aerosol evidence, gas physical state, very high vapor pressure, and low boiling point.",
    "- corrosive: consider corrosion pictogram/evidence, strong acid/base functional groups, and severe skin/eye damage cues.",
    "- acute_toxicity: consider skull/toxicity pictogram evidence, LD50/LC50 fields, toxicity class, and strong acute oral/dermal/inhalation toxicity cues.",
    "- irritant_harmful: consider exclamation pictogram/evidence, skin/eye irritation, sensitization, harmful exposure, and Warning signal word cues.",
    "- cmr: consider health hazard pictogram/evidence and carcinogenicity, mutagenicity, reproductive toxicity, or suspicious chronic toxicity cues.",
    "- stot: consider organ toxicity evidence, repeated/single exposure toxicity cues, health hazard pictogram, and IDLH/toxicity fields.",
    "- environmental_hazard: consider environment pictogram/evidence, aquatic/ecotoxicity cues, high hydrophobicity, low water solubility, halogenated/aromatic structures, and organometal/metalloid elements.",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def is_positive(value: str) -> bool:
    text = as_text(value)
    return text in {"1", "1.0", "True", "true"}


def parse_float(value: object) -> float | None:
    text = as_text(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def formula_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def build_derived_environmental_cues(row: dict[str, str]) -> list[str]:
    smiles = as_text(row.get("ConnectivitySMILES", ""))
    iupac = as_text(row.get("IUPACName", ""))
    formula = as_text(row.get("MolecularFormula", ""))
    combined_text = f"{smiles} {iupac} {formula}".lower()
    elements = formula_elements(formula)

    halogen_elements = {"F", "Cl", "Br", "I"}
    contains_halogen = bool(elements & halogen_elements) or any(
        keyword in combined_text
        for keyword in ["chloro", "chlor", "bromo", "brom", "iodo", "fluoro"]
    )
    aromatic_proxy = any(
        keyword in combined_text
        for keyword in [
            "benz",
            "phenyl",
            "aniline",
            "anthrac",
            "naphth",
            "pyridine",
            "C1=CC".lower(),
            "C=CC".lower(),
        ]
    )
    organometal_metalloid_elements = {
        "Sn",
        "Hg",
        "Pb",
        "Cd",
        "As",
        "Se",
        "Cr",
        "Ni",
        "Cu",
        "Zn",
    }
    xlogp = parse_float(row.get("XLogP", ""))
    water_solubility = parse_float(row.get("WaterSolubility", ""))

    cues = [
        ("Contains halogenated organic structure", contains_halogen),
        ("Contains halogenated aromatic structure", contains_halogen and aromatic_proxy),
        ("High hydrophobicity proxy XLogP >= 3", xlogp is not None and xlogp >= 3.0),
        ("Very high hydrophobicity proxy XLogP >= 5", xlogp is not None and xlogp >= 5.0),
        ("Low water solubility proxy <= 1", water_solubility is not None and water_solubility <= 1.0),
        ("Contains organometal/metalloid element", bool(elements & organometal_metalloid_elements)),
    ]
    return [f"- {name}: {yes_no(value)}" for name, value in cues]


def build_user_prompt(
    row: dict[str, str],
    include_derived_environmental_cues: bool = False,
    include_ghs_evidence: bool = False,
    ghs_evidence_columns: list[str] | None = None,
    include_non_hcode_evidence: bool = False,
    include_hazard_rubric: bool = False,
) -> str:
    lines = [
        "Classify the chemical into the predefined coarse-grained hazard categories.",
        "Chemical information:",
    ]
    for column in FEATURE_COLUMNS:
        value = as_text(row.get(column, ""))
        if value:
            lines.append(f"- {column}: {value}")
    if include_ghs_evidence:
        lines.append("Direct GHS evidence:")
        for column in (ghs_evidence_columns or GHS_EVIDENCE_COLUMNS):
            value = as_text(row.get(column, ""))
            if value:
                lines.append(f"- {column}: {value}")
    if include_non_hcode_evidence:
        lines.append("Non-H-code environmental evidence:")
        for column in NON_HCODE_EVIDENCE_COLUMNS:
            value = as_text(row.get(column, ""))
            if value:
                lines.append(f"- {column}: {value}")
    if include_derived_environmental_cues:
        lines.append("Derived environmental cues (heuristic, use only as supporting evidence):")
        lines.extend(build_derived_environmental_cues(row))
    if include_hazard_rubric:
        lines.append("Hazard review rubric (supporting guidance, do not treat any single cue as definitive):")
        lines.extend(HAZARD_RUBRIC_LINES)
    lines.append("Return only JSON.")
    return "\n".join(lines)


def build_assistant_json(row: dict[str, str]) -> str:
    payload = {
        label.removeprefix("label_"): 1 if is_positive(row.get(label, "0")) else 0
        for label in LABEL_COLUMNS
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def row_has_any_label(row: dict[str, str]) -> bool:
    return any(is_positive(row.get(label, "0")) for label in LABEL_COLUMNS)


def build_record(
    row: dict[str, str],
    include_derived_environmental_cues: bool = False,
    include_ghs_evidence: bool = False,
    ghs_evidence_columns: list[str] | None = None,
    include_non_hcode_evidence: bool = False,
    include_hazard_rubric: bool = False,
) -> dict:
    assistant_json = build_assistant_json(row)
    return {
        "id": as_text(row.get("CID", "")),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    row,
                    include_derived_environmental_cues,
                    include_ghs_evidence,
                    ghs_evidence_columns,
                    include_non_hcode_evidence,
                    include_hazard_rubric,
                ),
            },
            {"role": "assistant", "content": assistant_json},
        ],
        "labels": json.loads(assistant_json),
    }


def allocate_counts(total: int, ratios: list[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    ranked = sorted(
        range(len(ratios)),
        key=lambda idx: (raw_counts[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )
    for idx in ranked[:remainder]:
        counts[idx] += 1
    return counts


def active_label_names(record: dict) -> list[str]:
    return [label for label, value in record["labels"].items() if int(value) == 1]


def random_split_records(records: list[dict], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - val_count - test_count
    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]
    return train, val, test


def iterative_two_way_split(records: list[dict], holdout_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    total = len(records)
    if total == 0:
        return [], []
    if holdout_ratio <= 0:
        return list(records), []
    if holdout_ratio >= 1:
        return [], list(records)

    train_target, holdout_target = allocate_counts(total, [1.0 - holdout_ratio, holdout_ratio])
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    record_labels = {idx: active_label_names(records[idx]) for idx in indices}
    label_totals = {label.removeprefix("label_"): 0 for label in LABEL_COLUMNS}
    for labels in record_labels.values():
        for label in labels:
            label_totals[label] += 1

    label_targets = {
        "train": {},
        "holdout": {},
    }
    for label, total_count in label_totals.items():
        train_count, holdout_count = allocate_counts(total_count, [1.0 - holdout_ratio, holdout_ratio])
        label_targets["train"][label] = train_count
        label_targets["holdout"][label] = holdout_count

    assignments: dict[int, str] = {}
    remaining_sizes = {
        "train": train_target,
        "holdout": holdout_target,
    }
    unassigned = set(indices)

    def assign(idx: int, split_name: str) -> None:
        assignments[idx] = split_name
        remaining_sizes[split_name] -= 1
        for label in record_labels[idx]:
            label_targets[split_name][label] -= 1
        unassigned.discard(idx)

    while True:
        label_candidates: list[tuple[int, str]] = []
        for label in label_totals:
            count = sum(1 for idx in unassigned if label in record_labels[idx])
            if count > 0:
                label_candidates.append((count, label))
        if not label_candidates:
            break

        _count, rarest_label = min(label_candidates, key=lambda item: (item[0], item[1]))
        candidate_indices = [idx for idx in unassigned if rarest_label in record_labels[idx]]
        rng.shuffle(candidate_indices)

        for idx in candidate_indices:
            if idx not in unassigned:
                continue
            eligible_splits = [name for name, size in remaining_sizes.items() if size > 0]
            if not eligible_splits:
                break

            labels = record_labels[idx]
            ranked_splits = sorted(
                eligible_splits,
                key=lambda name: (
                    label_targets[name][rarest_label],
                    sum(label_targets[name][label] for label in labels),
                    remaining_sizes[name],
                    rng.random(),
                ),
                reverse=True,
            )
            assign(idx, ranked_splits[0])

    leftover_indices = list(unassigned)
    rng.shuffle(leftover_indices)
    for idx in leftover_indices:
        eligible_splits = [name for name, size in remaining_sizes.items() if size > 0]
        if not eligible_splits:
            break
        ranked_splits = sorted(
            eligible_splits,
            key=lambda name: (remaining_sizes[name], rng.random()),
            reverse=True,
        )
        assign(idx, ranked_splits[0])

    train_records = [records[idx] for idx in indices if assignments.get(idx) == "train"]
    holdout_records = [records[idx] for idx in indices if assignments.get(idx) == "holdout"]
    return train_records, holdout_records


def multilabel_stratified_split_records(
    records: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_val_records, test_records = iterative_two_way_split(records, test_ratio, seed)
    remaining_ratio = 1.0 - test_ratio
    if remaining_ratio <= 0:
        return [], [], test_records
    val_ratio_within_remaining = val_ratio / remaining_ratio
    train_records, val_records = iterative_two_way_split(train_val_records, val_ratio_within_remaining, seed + 1)
    return train_records, val_records, test_records


def split_records(
    records: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_method: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    if split_method == "random":
        return random_split_records(records, val_ratio, test_ratio, seed)
    return multilabel_stratified_split_records(records, val_ratio, test_ratio, seed)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def label_counts(records: Iterable[dict]) -> dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for record in records:
        labels = record["labels"]
        for key, value in labels.items():
            counts[key] += int(value)
    return counts


def parse_label_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [name for name in names if name not in LABEL_NAMES]
    if invalid:
        raise SystemExit(
            f"Unknown oversample labels: {invalid}. Valid labels: {LABEL_NAMES}"
        )
    return names


def parse_column_names(text: str, valid_columns: list[str], option_name: str) -> list[str]:
    columns = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [column for column in columns if column not in valid_columns]
    if invalid:
        raise SystemExit(
            f"Unknown columns for {option_name}: {invalid}. Valid columns: {valid_columns}"
        )
    return columns


def parse_oversample_spec(text: str) -> dict[str, float]:
    spec: dict[str, float] = {}
    if not text.strip():
        return spec
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "--train-oversample-spec must use label=factor entries, e.g. cmr=2,stot=2."
            )
        label, factor_text = [part.strip() for part in item.split("=", 1)]
        if label not in LABEL_NAMES:
            raise SystemExit(
                f"Unknown oversample label: {label}. Valid labels: {LABEL_NAMES}"
            )
        try:
            factor = float(factor_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid oversample factor for {label}: {factor_text}") from exc
        if factor < 1.0:
            raise SystemExit(f"Oversample factor for {label} must be >= 1.0.")
        spec[label] = factor
    return spec


def build_uniform_oversample_spec(target_labels: list[str], oversample_factor: int) -> dict[str, float]:
    if oversample_factor <= 1:
        return {}
    return {label: float(oversample_factor) for label in target_labels}


def parse_keywords(text: str) -> list[str]:
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def record_text(record: dict) -> str:
    chunks: list[str] = []
    for message in record.get("messages", []):
        if message.get("role") == "user":
            chunks.append(str(message.get("content", "")))
    return "\n".join(chunks).lower()


def matches_hard_positive(
    record: dict,
    label: str,
    keywords: list[str],
) -> bool:
    if not label or not keywords:
        return False
    if int(record["labels"].get(label, 0)) != 1:
        return False
    text = record_text(record)
    return any(keyword in text for keyword in keywords)


def oversample_train_records(
    records: list[dict],
    oversample_spec: dict[str, float],
    hard_positive_label: str,
    hard_positive_keywords: list[str],
    hard_positive_factor: float,
    seed: int,
) -> tuple[list[dict], int]:
    if not oversample_spec and not hard_positive_label:
        return list(records), 0

    rng = random.Random(seed)
    expanded_records = list(records)
    hard_positive_matches = 0
    for record in records:
        labels = record["labels"]
        matched_factors = [
            factor
            for label, factor in oversample_spec.items()
            if int(labels.get(label, 0)) == 1
        ]
        if matches_hard_positive(record, hard_positive_label, hard_positive_keywords):
            matched_factors.append(hard_positive_factor)
            hard_positive_matches += 1
        if not matched_factors:
            continue
        factor = max(matched_factors)
        whole_extra = int(factor) - 1
        fractional_extra = factor - int(factor)
        expanded_records.extend(record.copy() for _ in range(whole_extra))
        if fractional_extra > 0 and rng.random() < fractional_extra:
            expanded_records.append(record.copy())

    rng.shuffle(expanded_records)
    return expanded_records, hard_positive_matches


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert PubChem coarse labels CSV into SFT/LoRA JSONL splits.")
    parser.add_argument(
        "--input",
        default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv"),
        help="Input CSV generated by build_pubchem_coarse_labels.py",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "sft_coarse_jsonl"),
        help="Directory for train/val/test JSONL outputs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-method",
        default="multilabel_stratified",
        choices=["multilabel_stratified", "random"],
        help="Dataset split strategy. multilabel_stratified is recommended for long-tail multilabel tasks.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Keep rows without any coarse label. Default drops them.",
    )
    parser.add_argument(
        "--train-oversample-factor",
        type=int,
        default=1,
        help="Repeat matching training samples by this factor. 1 disables oversampling.",
    )
    parser.add_argument(
        "--train-oversample-labels",
        default=",".join(DEFAULT_RARE_OVERSAMPLE_LABELS),
        help="Comma-separated train labels to oversample when --train-oversample-factor > 1.",
    )
    parser.add_argument(
        "--train-oversample-spec",
        default="",
        help="Optional comma-separated label=factor spec. Overrides --train-oversample-factor/labels.",
    )
    parser.add_argument(
        "--hard-positive-label",
        default="",
        choices=[""] + LABEL_NAMES,
        help="Optional label for keyword-matched positive replay within the training split.",
    )
    parser.add_argument(
        "--hard-positive-keywords",
        default="",
        help="Comma-separated keywords used to select hard positive replay samples.",
    )
    parser.add_argument(
        "--hard-positive-factor",
        type=float,
        default=1.0,
        help="Replay factor for keyword-matched hard positives. 1 disables replay.",
    )
    parser.add_argument(
        "--include-derived-environmental-cues",
        action="store_true",
        help="Add heuristic environmental cue lines derived from structure and physicochemical fields.",
    )
    parser.add_argument(
        "--include-ghs-evidence",
        action="store_true",
        help="Add direct GHS evidence fields such as H_Statements and GHS_Classifications.",
    )
    parser.add_argument(
        "--ghs-evidence-columns",
        default=",".join(GHS_EVIDENCE_COLUMNS),
        help="Comma-separated GHS evidence columns to include when --include-ghs-evidence is set.",
    )
    parser.add_argument(
        "--include-non-hcode-evidence",
        action="store_true",
        help="Add non-H-code environmental evidence columns generated by fetch_pubchem_non_hcode_evidence.py.",
    )
    parser.add_argument(
        "--include-hazard-rubric",
        action="store_true",
        help="Add concise hazard-label decision criteria to the user prompt.",
    )
    args = parser.parse_args()

    if args.train_oversample_factor < 1:
        raise SystemExit("--train-oversample-factor must be >= 1.")
    if args.hard_positive_factor < 1.0:
        raise SystemExit("--hard-positive-factor must be >= 1.0.")

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    rows = read_rows(input_path)
    if not args.include_empty_labels:
        rows = [row for row in rows if row_has_any_label(row)]

    ghs_evidence_columns = parse_column_names(
        args.ghs_evidence_columns,
        GHS_EVIDENCE_COLUMNS,
        "--ghs-evidence-columns",
    )
    records = [
        build_record(
            row,
            args.include_derived_environmental_cues,
            args.include_ghs_evidence,
            ghs_evidence_columns,
            args.include_non_hcode_evidence,
            args.include_hazard_rubric,
        )
        for row in rows
    ]
    train_records, val_records, test_records = split_records(
        records,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.split_method,
    )
    oversample_labels = parse_label_names(args.train_oversample_labels)
    oversample_spec = parse_oversample_spec(args.train_oversample_spec)
    if not oversample_spec:
        oversample_spec = build_uniform_oversample_spec(
            oversample_labels,
            args.train_oversample_factor,
        )
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

    write_jsonl(output_root / "train.jsonl", train_records)
    write_jsonl(output_root / "val.jsonl", val_records)
    write_jsonl(output_root / "test.jsonl", test_records)

    summary = {
        "input": str(input_path),
        "output_root": str(output_root),
        "num_records": len(records),
        "num_train_before_oversampling": len(train_records_before_oversampling),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_test": len(test_records),
        "seed": args.seed,
        "split_method": args.split_method,
        "train_oversample_factor": args.train_oversample_factor,
        "train_oversample_labels": oversample_labels,
        "train_oversample_spec": oversample_spec,
        "hard_positive_label": args.hard_positive_label,
        "hard_positive_keywords": hard_positive_keywords,
        "hard_positive_factor": args.hard_positive_factor,
        "hard_positive_matches_train": hard_positive_matches,
        "include_derived_environmental_cues": args.include_derived_environmental_cues,
        "include_ghs_evidence": args.include_ghs_evidence,
        "ghs_evidence_columns": ghs_evidence_columns if args.include_ghs_evidence else [],
        "include_non_hcode_evidence": args.include_non_hcode_evidence,
        "include_hazard_rubric": args.include_hazard_rubric,
        "label_counts_all": label_counts(records),
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
