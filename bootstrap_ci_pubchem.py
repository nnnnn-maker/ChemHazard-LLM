from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss


LABELS = [
    "flammable", "oxidizing", "gas_under_pressure", "corrosive",
    "acute_toxicity", "irritant_harmful", "cmr", "stot", "environmental_hazard",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_labels(record: dict[str, Any]) -> dict[str, int]:
    if isinstance(record.get("labels"), dict):
        labels = record["labels"]
    else:
        labels = None
        for message in record.get("messages", []):
            if message.get("role") == "assistant":
                labels = json.loads(message.get("content", "{}"))
                break
    if not isinstance(labels, dict) or any(label not in labels for label in LABELS):
        raise ValueError(f"Cannot extract all labels for id={record.get('id')}")
    return {label: int(labels[label]) for label in LABELS}


def aligned_arrays(gold_path: Path, pred_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)
    gold = {str(row["id"]): extract_labels(row) for row in gold_rows}
    pred = {str(row["id"]): extract_labels(row) for row in pred_rows}
    if set(gold) != set(pred):
        raise ValueError(f"Gold/prediction ID mismatch: gold={len(gold)}, pred={len(pred)}")
    ids = [str(row["id"]) for row in gold_rows]
    y_true = np.asarray([[gold[cid][label] for label in LABELS] for cid in ids], dtype=int)
    y_pred = np.asarray([[pred[cid][label] for label in LABELS] for cid in ids], dtype=int)
    return y_true, y_pred


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-level bootstrap confidence intervals for PubChem metrics.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    y_true, y_pred = aligned_arrays(Path(args.gold), Path(args.pred))
    point = metrics(y_true, y_pred)
    rng = np.random.default_rng(args.seed)
    draws = {metric: np.empty(args.n_bootstrap, dtype=float) for metric in point}
    for index in range(args.n_bootstrap):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        values = metrics(y_true[sample], y_pred[sample])
        for metric, value in values.items():
            draws[metric][index] = value

    lower = 100 * args.alpha / 2
    upper = 100 * (1 - args.alpha / 2)
    payload = {
        "gold": args.gold,
        "pred": args.pred,
        "num_samples": int(len(y_true)),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "metrics": {
            metric: {
                "point": point[metric],
                "ci95_low": float(np.percentile(draws[metric], lower)),
                "ci95_high": float(np.percentile(draws[metric], upper)),
            }
            for metric in point
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
