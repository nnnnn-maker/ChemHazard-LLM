from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_pubchem_classifier_chain_baseline import (
    CATEGORICAL_COLUMNS,
    LABELS,
    LABEL_COLUMNS,
    NUMERIC_COLUMNS,
    MorganTransformer,
    compute_metrics,
    ensure_columns,
    per_label_metrics,
    split_ids,
    write_csv,
    write_json,
    write_predictions,
)


def make_physchem() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def make_features(kind: str):
    if kind == "physchem":
        return make_physchem()
    if kind == "morgan":
        return MorganTransformer()
    if kind == "morgan_physchem":
        return FeatureUnion([("physchem", make_physchem()), ("morgan", MorganTransformer())])
    raise ValueError(kind)


def make_estimator(kind: str, seed: int, n_estimators: int, min_samples_leaf: int, n_jobs: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    if kind == "logreg":
        return OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            ),
            n_jobs=n_jobs,
        )
    raise ValueError(kind)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run five conventional baselines on the exact IDs of an existing SFT split."
    )
    parser.add_argument(
        "--input",
        default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv"),
    )
    parser.add_argument(
        "--split-root",
        default=str(project_dir / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"),
    )
    parser.add_argument("--run-tag", default="seed123_split")
    parser.add_argument("--evaluation-root", default=str(project_dir / "evaluation_llm_coarse"))
    parser.add_argument("--prediction-root", default=str(project_dir / "predictions"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input)
    split_root = Path(args.split_root)
    evaluation_root = Path(args.evaluation_root)
    prediction_root = Path(args.prediction_root)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "CID" not in df.columns:
        raise ValueError("Input CSV must contain CID.")
    df["CID"] = df["CID"].astype(str)
    df = ensure_columns(df).set_index("CID", drop=False)

    train_ids = split_ids(split_root / "train.jsonl")
    test_ids = split_ids(split_root / "test.jsonl")
    missing = [cid for cid in train_ids + test_ids if cid not in df.index]
    if missing:
        raise ValueError(f"{len(missing)} split IDs are absent from the input CSV: {missing[:5]}")

    train_df = df.loc[train_ids].copy()
    test_df = df.loc[test_ids].copy()
    y_train = train_df[LABEL_COLUMNS].to_numpy(dtype=int)
    y_test = test_df[LABEL_COLUMNS].to_numpy(dtype=int)

    specifications = {
        "morgan_physchem_rf": ("morgan_physchem", "rf"),
        "physchem_rf": ("physchem", "rf"),
        "morgan_rf": ("morgan", "rf"),
        "physchem_logreg": ("physchem", "logreg"),
        "morgan_logreg": ("morgan", "logreg"),
    }

    summary = []
    for model_name, (feature_kind, estimator_kind) in specifications.items():
        run_name = f"{model_name}_{args.run_tag}"
        print(f"running: {run_name}")
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
        y_pred = np.asarray(model.predict(test_df), dtype=int)
        values = compute_metrics(y_test, y_pred)
        pred_path = prediction_root / f"{run_name}_predictions.jsonl"
        output_dir = evaluation_root / run_name
        write_predictions(pred_path, test_ids, y_pred)
        write_json(
            output_dir / "metrics.json",
            {
                "model": model_name,
                "run_tag": args.run_tag,
                "input": str(input_path),
                "split_root": str(split_root),
                "seed": args.seed,
                "n_estimators": args.n_estimators if estimator_kind == "rf" else None,
                "min_samples_leaf": args.min_samples_leaf if estimator_kind == "rf" else None,
                "pred_file": str(pred_path),
                **values,
            },
        )
        write_csv(output_dir / "per_label.csv", per_label_metrics(y_test, y_pred))
        summary.append({"model": model_name, **values})

    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / f"conventional_baselines_{args.run_tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
