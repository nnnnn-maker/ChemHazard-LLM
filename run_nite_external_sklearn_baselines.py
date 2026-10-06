from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline

from run_pubchem_classifier_chain_baseline import (
    CATEGORICAL_COLUMNS,
    LABEL_COLUMNS,
    LABELS,
    NUMERIC_COLUMNS,
    make_preprocessor,
    parse_boolish,
    split_ids,
)
from run_pubchem_split_baselines import make_estimator, make_features


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_user_fields(record: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for message in record.get("messages", []):
        if message.get("role") != "user":
            continue
        for line in str(message.get("content", "")).splitlines():
            if not line.startswith("- ") or ":" not in line:
                continue
            key, value = line[2:].split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def build_external_frame(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    observed_fields: dict[str, int] = {}
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            raise ValueError("NITE record without an id")
        if record_id in seen_ids:
            duplicate_ids.append(record_id)
        seen_ids.add(record_id)
        fields = parse_user_fields(record)
        for key in fields:
            observed_fields[key] = observed_fields.get(key, 0) + 1
        rows.append({"id": record_id, **fields})
    if duplicate_ids:
        raise ValueError(f"Duplicate NITE ids: {duplicate_ids[:10]}")

    frame = pd.DataFrame(rows)
    if "ConnectivitySMILES" not in frame.columns:
        raise ValueError("NITE prompts do not contain ConnectivitySMILES")
    frame["ConnectivitySMILES"] = frame["ConnectivitySMILES"].fillna("").astype(str)
    if (frame["ConnectivitySMILES"].str.strip() == "").any():
        bad = frame.loc[frame["ConnectivitySMILES"].str.strip() == "", "id"].tolist()
        raise ValueError(f"Missing NITE SMILES: {bad[:10]}")

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        if column.startswith("Has_"):
            frame[column] = frame[column].map(parse_boolish)
        else:
            frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()

    audit = {
        "records": len(frame),
        "unique_ids": int(frame["id"].nunique()),
        "missing_smiles": int((frame["ConnectivitySMILES"].str.strip() == "").sum()),
        "observed_prompt_fields": observed_fields,
        "available_numeric_counts": {
            column: int(frame[column].notna().sum()) for column in NUMERIC_COLUMNS
        },
        "available_categorical_counts": {
            column: int((frame[column].astype(str).str.strip() != "").sum())
            for column in CATEGORICAL_COLUMNS
        },
        "note": (
            "Only fields present in the leakage-controlled NITE prompts are used. "
            "Missing external physicochemical fields remain missing and are transformed "
            "by preprocessing fitted on PubChem training data only."
        ),
    }
    return frame, audit


def ensure_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "CID" not in work.columns:
        raise ValueError("PubChem input CSV must contain CID")
    work["CID"] = work["CID"].astype(str)
    if "ConnectivitySMILES" not in work.columns:
        raise ValueError("PubChem input CSV must contain ConnectivitySMILES")
    for column in NUMERIC_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column not in work.columns:
            work[column] = ""
        if column.startswith("Has_"):
            work[column] = work[column].map(parse_boolish)
        else:
            work[column] = work[column].fillna("").astype(str).str.strip().str.lower()
    for column in LABEL_COLUMNS:
        if column not in work.columns:
            raise ValueError(f"PubChem input CSV must contain {column}")
        work[column] = pd.to_numeric(work[column], errors="raise").astype(int)
    return work.set_index("CID", drop=False)


def write_predictions(
    path: Path,
    ids: list[str],
    predictions: np.ndarray,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, values in zip(ids, predictions):
            labels = {label: int(value) for label, value in zip(LABELS, values)}
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "labels": labels,
                        "parse_status": f"ok:{model_name}:frozen_pubchem_seed123",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def evaluate(
    evaluator: Path,
    gold: Path,
    predictions: Path,
    output: Path,
    expected_records: int,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    integrity = {
        "gold_records": expected_records,
        "prediction_records": expected_records,
        "matched_records": expected_records,
        "missing_gold_ids_in_predictions": 0,
        "extra_prediction_ids": 0,
        "missing_prediction_cells": 0,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in integrity.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"External evaluation integrity failure: {mismatches}")
    return result


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "micro_accuracy",
        "gold_records",
        "matched_records",
        "scored_label_cells",
        "missing_prediction_cells",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Train six frozen PubChem seed123 sklearn baselines and evaluate them "
            "on the masked NITE external test set."
        )
    )
    parser.add_argument(
        "--pubchem-input",
        type=Path,
        default=project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_dir / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x",
    )
    parser.add_argument(
        "--nite-input",
        type=Path,
        default=(
            project_dir
            / "external_validation"
            / "nite_japan_ghs"
            / "nite_external_test_exact_inchikey_disjoint.jsonl"
        ),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=(
            project_dir
            / "external_validation"
            / "nite_japan_ghs"
            / "evaluate_nite_external_predictions.py"
        ),
    )
    parser.add_argument("--tag", default="exact")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    for required in [args.pubchem_input, args.split_root / "train.jsonl", args.nite_input, args.evaluator]:
        if not required.is_file():
            raise FileNotFoundError(required)

    pubchem = ensure_training_frame(pd.read_csv(args.pubchem_input, encoding="utf-8-sig"))
    train_ids = split_ids(args.split_root / "train.jsonl")
    missing_train = [cid for cid in train_ids if cid not in pubchem.index]
    if missing_train:
        raise ValueError(f"PubChem train IDs missing from CSV: {missing_train[:10]}")
    train_df = pubchem.loc[train_ids].copy()
    y_train = train_df[LABEL_COLUMNS].to_numpy(dtype=int)

    nite_records = read_jsonl(args.nite_input)
    nite_df, audit = build_external_frame(nite_records)
    nite_ids = nite_df["id"].astype(str).tolist()
    audit.update(
        {
            "pubchem_train_records": len(train_df),
            "pubchem_unique_train_ids": len(train_ids),
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "external_labels_used_for_fit": False,
            "external_labels_used_for_threshold_selection": False,
        }
    )

    prediction_dir = project_dir / "predictions" / "nite_japan_ghs" / args.tag / "baselines"
    evaluation_dir = project_dir / "evaluation_llm_coarse" / "nite_japan_ghs" / args.tag / "baselines"
    analysis_dir = project_dir / "analysis_outputs" / "nite_japan_ghs" / args.tag / "baselines"
    for directory in [prediction_dir, evaluation_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "sklearn_external_input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    specifications = {
        "morgan_physchem_rf": ("morgan_physchem", "rf"),
        "physchem_rf": ("physchem", "rf"),
        "morgan_rf": ("morgan", "rf"),
        "physchem_logreg": ("physchem", "logreg"),
        "morgan_logreg": ("morgan", "logreg"),
    }
    results: list[dict[str, Any]] = []
    for model_name, (feature_kind, estimator_kind) in specifications.items():
        print(f"\n=== Training on PubChem only: {model_name} ===", flush=True)
        model = Pipeline(
            [
                ("features", make_features(feature_kind)),
                (
                    "model",
                    make_estimator(
                        estimator_kind,
                        args.seed,
                        args.n_estimators,
                        args.min_samples_leaf,
                        args.n_jobs,
                    ),
                ),
            ]
        )
        model.fit(train_df, y_train)
        y_pred = np.asarray(model.predict(nite_df), dtype=int)
        pred_path = prediction_dir / f"{model_name}_predictions.jsonl"
        metric_path = evaluation_dir / f"{model_name}_metrics.json"
        write_predictions(pred_path, nite_ids, y_pred, model_name)
        metric = evaluate(args.evaluator, args.nite_input, pred_path, metric_path, len(nite_ids))
        results.append({"model": model_name, **metric})

    print("\n=== Training on PubChem only: morgan_physchem_classifier_chain_rf ===", flush=True)
    chain_base = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        class_weight="balanced_subsample",
    )
    chain = Pipeline(
        [
            ("features", make_preprocessor()),
            ("chain", ClassifierChain(chain_base, order="random", random_state=args.seed)),
        ]
    )
    chain.fit(train_df, y_train)
    chain_pred = np.asarray(chain.predict(nite_df), dtype=int)
    chain_name = "morgan_physchem_classifier_chain_rf"
    chain_pred_path = prediction_dir / f"{chain_name}_predictions.jsonl"
    chain_metric_path = evaluation_dir / f"{chain_name}_metrics.json"
    write_predictions(chain_pred_path, nite_ids, chain_pred, chain_name)
    chain_metric = evaluate(
        args.evaluator,
        args.nite_input,
        chain_pred_path,
        chain_metric_path,
        len(nite_ids),
    )
    results.append({"model": chain_name, **chain_metric})

    summary_json = evaluation_dir / "six_sklearn_baselines_summary.json"
    summary_csv = evaluation_dir / "six_sklearn_baselines_summary.csv"
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(summary_csv, results)
    print(f"\nSaved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
