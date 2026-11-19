"""Confusion matrix pada data uji untuk skenario baseline, A, A2, dan C (P5).

Memakai pembersihan, split kronologis, fitur, dan model yang sama dengan run_experiments.py.
Logistic Regression (deterministik) dan Random Forest seed 0.

Contoh:
    python src/confusion_matrices.py --data data/KaggleV2-May-2016.csv --out results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import best_f1_threshold, top_k_indices  # noqa: E402
from preprocessing import chronological_split, clean, feature_columns, load_raw  # noqa: E402
from run_experiments import make_model  # noqa: E402


def counts(y: np.ndarray, flag: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    return {
        "TP": int(((y == 1) & flag).sum()),
        "FP": int(((y == 0) & flag).sum()),
        "FN": int(((y == 1) & ~flag).sum()),
        "TN": int(((y == 0) & ~flag).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, _ = clean(load_raw(args.data))
    train, val, test = chronological_split(df, 0.6, 0.2)
    feats = feature_columns(False)
    y_tr, y_val, y_te = train["target"].values, val["target"].values, test["target"].values

    rows = []
    for model_name in ["logreg", "random_forest"]:
        pipe = make_model(model_name, args.seed, False).fit(train[feats], y_tr)
        s_val = pipe.predict_proba(val[feats])[:, 1]
        s_te = pipe.predict_proba(test[feats])[:, 1]
        thr_a = best_f1_threshold(y_val, s_val)
        thr_a2 = float(np.quantile(s_val, 0.90))
        top = np.zeros(len(s_te), dtype=bool)
        top[top_k_indices(s_te, 0.10, args.seed)] = True
        flags = {
            "baseline_thr0.5": s_te >= 0.5,
            "A_tuned_threshold": s_te >= thr_a,
            "A2_capacity_threshold": s_te >= thr_a2,
            "C_top10pct": top,
        }
        for scenario, flag in flags.items():
            rows.append({"scenario": scenario, "model": model_name, "seed": args.seed,
                         "n_test": len(y_te), **counts(y_te, flag)})

    out = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out) / "confusion_matrices.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
