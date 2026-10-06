from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

LABEL_COLUMNS = [f"label_{label}" for label in LABELS]

NUMERIC_COLUMNS = [
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
    "LD50_oral",
    "LD50_dermal",
    "LC50_inhalation",
    "Toxicity_Class",
    "IDLH",
]

CATEGORICAL_COLUMNS = [
    "PhysicalState",
    "Signal_Word",
    "Has_Physical_Data",
    "Has_Toxicity_Data",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_ids(path: Path) -> list[str]:
    # SFT train files can include rare-label replay duplicates; keep first occurrence.
    ids = [str(record.get("id", "")) for record in read_jsonl(path)]
    return list(dict.fromkeys(ids))


def parse_boolish(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return "true"
    if text in {"0", "false", "no"}:
        return "false"
    return ""


class MorganTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col: str = "ConnectivitySMILES", radius: int = 2, n_bits: int = 2048):
        self.smiles_col = smiles_col
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "MorganTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
        except Exception as exc:  # pragma: no cover - depends on server env
            raise RuntimeError(f"RDKit is required for Morgan fingerprints: {exc}") from exc

        fingerprints = np.zeros((len(X), self.n_bits), dtype=np.float32)
        smiles_values = X[self.smiles_col].fillna("").astype(str).tolist()
        for index, smiles in enumerate(smiles_values):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((self.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints[index, :] = arr
        return fingerprints


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
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
    if "ConnectivitySMILES" not in work.columns:
        raise ValueError("Input CSV must contain ConnectivitySMILES.")
    for column in LABEL_COLUMNS:
        if column not in work.columns:
            raise ValueError(f"Input CSV must contain {column}.")
        work[column] = work[column].fillna(0).astype(int)
    return work


def make_preprocessor() -> FeatureUnion:
    physchem = ColumnTransformer(
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
    return FeatureUnion(
        [
            ("physchem", physchem),
            ("morgan", MorganTransformer()),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(LABELS):
        rows.append(
            {
                "label": label,
                "support": int(y_true[:, index].sum()),
                "precision": float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "recall": float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "f1": float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, ids: list[str], y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cid, pred in zip(ids, y_pred):
            labels = {label: int(value) for label, value in zip(LABELS, pred)}
            handle.write(
                json.dumps(
                    {
                        "id": cid,
                        "labels": labels,
                        "parse_status": "classifier_chain_rf",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Morgan + PhysChem RandomForest classifier-chain baseline on an existing SFT split."
    )
    parser.add_argument(
        "--input",
        default=str(project_dir / "processed_coarse" / "full_dataset_coarse_labels.csv"),
        help="Processed coarse-label CSV with CID, SMILES, features, and labels.",
    )
    parser.add_argument(
        "--split-root",
        default=str(project_dir / "sft_coarse_jsonl_seed123_ghs_pictograms_rare2x"),
        help="SFT split root containing train.jsonl, val.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "evaluation_llm_coarse" / "morgan_physchem_classifier_chain_rf_seed123"),
        help="Directory for metrics and per-label outputs.",
    )
    parser.add_argument(
        "--pred",
        default=str(project_dir / "predictions" / "morgan_physchem_classifier_chain_rf_seed123_predictions.jsonl"),
        help="Output prediction JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input)
    split_root = Path(args.split_root)
    output_dir = Path(args.output_dir)
    pred_path = Path(args.pred)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "CID" not in df.columns:
        raise ValueError("Input CSV must contain CID.")
    df["CID"] = df["CID"].astype(str)
    df = ensure_columns(df).set_index("CID", drop=False)

    train_ids = split_ids(split_root / "train.jsonl")
    test_ids = split_ids(split_root / "test.jsonl")
    missing_train = [cid for cid in train_ids if cid not in df.index]
    missing_test = [cid for cid in test_ids if cid not in df.index]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing IDs in CSV: train={len(missing_train)} test={len(missing_test)} "
            f"examples={missing_train[:3] + missing_test[:3]}"
        )

    train_df = df.loc[train_ids].copy()
    test_df = df.loc[test_ids].copy()
    y_train = train_df[LABEL_COLUMNS].to_numpy(dtype=int)
    y_test = test_df[LABEL_COLUMNS].to_numpy(dtype=int)

    base_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        class_weight="balanced_subsample",
    )
    model = Pipeline(
        [
            ("features", make_preprocessor()),
            ("chain", ClassifierChain(base_model, order="random", random_state=args.seed)),
        ]
    )

    print(f"train_samples={len(train_df)} test_samples={len(test_df)}")
    print(f"n_estimators={args.n_estimators} seed={args.seed}")
    model.fit(train_df, y_train)
    y_pred = model.predict(test_df).astype(int)

    metrics = compute_metrics(y_test, y_pred)
    payload = {
        "model": "morgan_physchem_classifier_chain_rf",
        "input": str(input_path),
        "split_root": str(split_root),
        "pred_file": str(pred_path),
        **metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_label.csv", per_label_metrics(y_test, y_pred))
    write_predictions(pred_path, test_ids, y_pred)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"pred: {pred_path}")
    print(f"metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
