from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class RunSpec:
    key: str
    display_name: str
    gold: str
    prediction: str
    lora: int
    pictogram: int
    rare2x: int
    ensemble3: int
    constrained_threshold: int


RUNS = [
    RunSpec(
        "A0_base_zeroshot_pictogram_input",
        "Mistral-Instruct zero-shot (pictogram input)",
        "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl",
        "predictions/mistral_base_zeroshot_seed123_ghs_pictograms_rare2x_predictions.jsonl",
        0,
        1,
        0,
        0,
        0,
    ),
    RunSpec(
        "A1_lora_no_pictogram_rare2x",
        "LoRA + rare2x (no pictogram)",
        "sft_coarse_jsonl_seed123_noghs_rare2x/test.jsonl",
        "predictions/mistral_seed123_lora_noghs_rare2x_predictions.jsonl",
        1,
        0,
        1,
        0,
        0,
    ),
    RunSpec(
        "A2_lora_pictogram_no_rare",
        "LoRA + pictogram (no rare2x)",
        "sft_coarse_jsonl_seed123_ghs_pictograms_norare/test.jsonl",
        "predictions/mistral_seed123_lora_ghs_pictograms_norare_predictions.jsonl",
        1,
        1,
        0,
        0,
        0,
    ),
    RunSpec(
        "A3_lora_pictogram_rare2x_single",
        "LoRA + pictogram + rare2x (single seed123)",
        "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl",
        "predictions/mistral_seed123_lora_ghs_pictograms_rare2x_predictions.jsonl",
        1,
        1,
        1,
        0,
        0,
    ),
    RunSpec(
        "A4_ensemble3_majority",
        "A3 + three-seed majority vote",
        "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl",
        "predictions/mistral_split123_ghs_pictograms_rare2x_ensemble3_predictions.jsonl",
        1,
        1,
        1,
        1,
        0,
    ),
    RunSpec(
        "A5_ensemble3_constrained_threshold",
        "A4 + constrained label-wise vote threshold",
        "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl",
        "predictions/mistral_split123_ghs_pictograms_rare2x_labelcal_constrained_ensemble3_predictions.jsonl",
        1,
        1,
        1,
        1,
        1,
    ),
]


CONTRASTS = [
    ("LoRA task adaptation", "A0_base_zeroshot_pictogram_input", "A2_lora_pictogram_no_rare"),
    ("Pictogram under rare2x", "A1_lora_no_pictogram_rare2x", "A3_lora_pictogram_rare2x_single"),
    ("Rare2x with pictogram", "A2_lora_pictogram_no_rare", "A3_lora_pictogram_rare2x_single"),
    ("Three-seed majority ensemble", "A3_lora_pictogram_rare2x_single", "A4_ensemble3_majority"),
    (
        "Constrained vote threshold",
        "A4_ensemble3_majority",
        "A5_ensemble3_constrained_threshold",
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_gold_labels(record: dict[str, Any]) -> dict[str, int]:
    if isinstance(record.get("labels"), dict):
        labels = record["labels"]
    else:
        labels = None
        for message in record.get("messages", []):
            if message.get("role") == "assistant":
                labels = json.loads(message.get("content", "{}"))
                break
    if not isinstance(labels, dict) or any(label not in labels for label in LABELS):
        raise ValueError(f"Cannot extract complete gold labels for id={record.get('id')}")
    return {label: int(labels[label]) for label in LABELS}


def prediction_has_all_labels(record: dict[str, Any]) -> bool:
    labels = record.get("labels")
    return isinstance(labels, dict) and all(labels.get(label) in (0, 1, False, True) for label in LABELS)


def prompt_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        messages = [
            {"role": message.get("role", ""), "content": message.get("content", "")}
            for message in row.get("messages", [])
            if message.get("role") != "assistant"
        ]
        digest.update(str(row.get("id", "")).encode("utf-8"))
        digest.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(
    project: Path,
    evaluator: Path,
    output_root: Path,
    leaderboard: Path,
    spec: RunSpec,
) -> dict[str, Any]:
    gold_path = project / spec.gold
    pred_path = project / spec.prediction
    status: dict[str, Any] = {**asdict(spec)}
    if not gold_path.is_file():
        return {**status, "status": "missing_gold"}
    gold_rows = read_jsonl(gold_path)
    gold_ids = [str(row.get("id", "")) for row in gold_rows]
    status.update(
        {
            "gold_records": len(gold_rows),
            "gold_unique_ids": len(set(gold_ids)),
            "gold_prompt_sha256": prompt_digest(gold_rows),
        }
    )
    if not pred_path.is_file():
        return {**status, "status": "missing_prediction"}
    pred_rows = read_jsonl(pred_path)
    pred_ids = [str(row.get("id", "")) for row in pred_rows]
    status.update(
        {
            "prediction_records": len(pred_rows),
            "prediction_unique_ids": len(set(pred_ids)),
            "prediction_valid_label_records": sum(prediction_has_all_labels(row) for row in pred_rows),
            "missing_prediction_ids": len(set(gold_ids) - set(pred_ids)),
            "extra_prediction_ids": len(set(pred_ids) - set(gold_ids)),
        }
    )
    if len(set(gold_ids)) != len(gold_rows) or len(set(pred_ids)) != len(pred_rows):
        return {**status, "status": "duplicate_ids"}
    if set(gold_ids) != set(pred_ids):
        return {**status, "status": "id_mismatch"}

    run_dir = output_root / spec.key
    prefix = spec.key
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--gold",
            str(gold_path),
            "--pred",
            str(pred_path),
            "--output-dir",
            str(run_dir),
            "--prefix",
            prefix,
            "--leaderboard-path",
            str(leaderboard),
        ],
        check=True,
    )
    metric_path = run_dir / f"{prefix}_metrics.json"
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    return {**status, "status": "evaluated", **metric}


def verify_gold_equivalence(project: Path) -> dict[str, Any]:
    available: dict[str, list[dict[str, Any]]] = {}
    for spec in RUNS:
        path = project / spec.gold
        if path.is_file() and spec.gold not in available:
            available[spec.gold] = read_jsonl(path)
    if not available:
        return {"status": "no_gold_files"}
    canonical_name = "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl"
    canonical_rows = available.get(canonical_name) or next(iter(available.values()))
    canonical_ids = [str(row.get("id", "")) for row in canonical_rows]
    canonical_labels = {
        str(row.get("id", "")): extract_gold_labels(row) for row in canonical_rows
    }
    comparisons = {}
    for name, rows in available.items():
        ids = [str(row.get("id", "")) for row in rows]
        labels = {str(row.get("id", "")): extract_gold_labels(row) for row in rows}
        comparisons[name] = {
            "records": len(rows),
            "ids_identical_and_ordered": ids == canonical_ids,
            "id_set_identical": set(ids) == set(canonical_ids),
            "labels_identical": labels == canonical_labels,
            "prompt_sha256": prompt_digest(rows),
            "prompt_identical_to_canonical": prompt_digest(rows) == prompt_digest(canonical_rows),
        }
    return {
        "status": "ok"
        if all(item["id_set_identical"] and item["labels_identical"] for item in comparisons.values())
        else "failed",
        "canonical": canonical_name,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and assemble the ChemHazard-LLM component ablation table."
    )
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    project = args.project_dir.resolve()
    evaluator = project / "evaluate_pubchem_coarse_predictions.py"
    paired = project / "paired_bootstrap_delta.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)

    analysis = project / "analysis_outputs" / "chemhazard_component_ablation"
    evaluation = project / "evaluation_llm_coarse" / "component_ablation_seed123"
    analysis.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    leaderboard = analysis / "component_ablation_leaderboard.csv"

    gold_audit = verify_gold_equivalence(project)
    run_rows = [evaluate_run(project, evaluator, evaluation, leaderboard, spec) for spec in RUNS]
    by_key = {row["key"]: row for row in run_rows}

    metric_fields = [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "subset_accuracy",
        "hamming_loss",
    ]
    table_fields = [
        "key",
        "display_name",
        "lora",
        "pictogram",
        "rare2x",
        "ensemble3",
        "constrained_threshold",
        "status",
        "num_samples",
        "parse_failures",
        *metric_fields,
        "prediction",
    ]
    write_csv(
        analysis / "component_ablation_table.csv",
        [{key: row.get(key, "") for key in table_fields} for row in run_rows],
        table_fields,
    )

    delta_rows: list[dict[str, Any]] = []
    bootstrap_outputs: list[str] = []
    canonical_gold = project / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x" / "test.jsonl"
    for effect, base_key, new_key in CONTRASTS:
        base = by_key[base_key]
        new = by_key[new_key]
        row: dict[str, Any] = {
            "effect": effect,
            "base_key": base_key,
            "new_key": new_key,
            "status": "available"
            if base.get("status") == "evaluated" and new.get("status") == "evaluated"
            else "missing_run",
        }
        if row["status"] == "available":
            for metric in metric_fields:
                row[f"base_{metric}"] = base.get(metric)
                row[f"new_{metric}"] = new.get(metric)
                row[f"delta_{metric}"] = new.get(metric) - base.get(metric)
            if args.run_bootstrap:
                if not paired.is_file():
                    raise FileNotFoundError(paired)
                if (
                    base.get("prediction_valid_label_records") == base.get("prediction_records")
                    and new.get("prediction_valid_label_records") == new.get("prediction_records")
                ):
                    output = evaluation / "paired_bootstrap" / f"{base_key}_vs_{new_key}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            sys.executable,
                            str(paired),
                            "--gold",
                            str(canonical_gold),
                            "--base",
                            str(project / base["prediction"]),
                            "--new",
                            str(project / new["prediction"]),
                            "--out",
                            str(output),
                            "--n-bootstrap",
                            str(args.n_bootstrap),
                            "--seed",
                            "123",
                        ],
                        check=True,
                    )
                    bootstrap_outputs.append(str(output))
                else:
                    row["bootstrap_note"] = "Skipped because at least one prediction has incomplete labels."
        delta_rows.append(row)

    delta_fields = ["effect", "base_key", "new_key", "status"]
    for metric in metric_fields:
        delta_fields.extend([f"base_{metric}", f"new_{metric}", f"delta_{metric}"])
    delta_fields.append("bootstrap_note")
    write_csv(
        analysis / "component_effect_deltas.csv",
        [{key: row.get(key, "") for key in delta_fields} for row in delta_rows],
        delta_fields,
    )

    missing = [
        {"key": row["key"], "status": row["status"], "gold": row["gold"], "prediction": row["prediction"]}
        for row in run_rows
        if row.get("status") != "evaluated"
    ]
    payload = {
        "project_dir": str(project),
        "gold_equivalence_audit": gold_audit,
        "runs": run_rows,
        "controlled_contrasts": delta_rows,
        "missing_or_invalid_runs": missing,
        "bootstrap_outputs": bootstrap_outputs,
        "ensemble_member_train_seeds": [123, 777, 2025],
        "threshold_rule": {
            "oxidizing": "1/3",
            "gas_under_pressure": "1/3",
            "cmr": "1/3",
            "all_other_labels": "2/3",
        },
    }
    audit_path = analysis / "component_ablation_audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"missing_or_invalid_runs": missing, "gold_audit_status": gold_audit.get("status")}, ensure_ascii=False, indent=2))
    print(f"saved: {analysis / 'component_ablation_table.csv'}")
    print(f"saved: {analysis / 'component_effect_deltas.csv'}")
    print(f"saved: {audit_path}")


if __name__ == "__main__":
    main()
