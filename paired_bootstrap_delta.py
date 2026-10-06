from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_ci_pubchem import aligned_arrays, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired sample-level bootstrap metric differences.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    y_true, y_base = aligned_arrays(Path(args.gold), Path(args.base))
    y_true_new, y_new = aligned_arrays(Path(args.gold), Path(args.new))
    if not np.array_equal(y_true, y_true_new):
        raise ValueError("Gold arrays differ after alignment.")

    base_point = metrics(y_true, y_base)
    new_point = metrics(y_true, y_new)
    point_delta = {metric: new_point[metric] - base_point[metric] for metric in base_point}
    draws = {metric: np.empty(args.n_bootstrap, dtype=float) for metric in point_delta}
    rng = np.random.default_rng(args.seed)
    for index in range(args.n_bootstrap):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        base_values = metrics(y_true[sample], y_base[sample])
        new_values = metrics(y_true[sample], y_new[sample])
        for metric in draws:
            draws[metric][index] = new_values[metric] - base_values[metric]

    lower = 100 * args.alpha / 2
    upper = 100 * (1 - args.alpha / 2)
    results = {}
    for metric, samples in draws.items():
        # Lower values are better only for Hamming loss.
        adverse = samples >= 0 if metric == "hamming_loss" else samples <= 0
        results[metric] = {
            "base": base_point[metric],
            "new": new_point[metric],
            "delta_new_minus_base": point_delta[metric],
            "ci95_low": float(np.percentile(samples, lower)),
            "ci95_high": float(np.percentile(samples, upper)),
            "probability_delta_le_0": float(np.mean(samples <= 0)),
            "probability_not_better": float(np.mean(adverse)),
        }

    payload = {
        "gold": args.gold,
        "base": args.base,
        "new": args.new,
        "num_samples": int(len(y_true)),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "metrics": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
