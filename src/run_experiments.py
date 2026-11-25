"""Jalankan baseline dan skenario A/B/C untuk prediksi no-show (P5).

Contoh:
    python src/run_experiments.py --data data/KaggleV2-May-2016.csv --out results

Skenario (lihat README):
    baseline   : LR dan RF, threshold 0,5
    A          : threshold dipilih pada validation set (maksimum F1)
    B          : kalibrasi probabilitas (sigmoid dan isotonic) pada validation set
    C          : evaluasi pada kapasitas 10% teratas (precision/recall@10%) untuk semua skor
    reference  : pemilihan acak dan heuristik lead_days sebagai pembanding
Test set hanya dipakai sekali untuk pelaporan; tidak ada tuning yang melihatnya.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (  # noqa: E402
    best_f1_threshold,
    bootstrap_ci,
    compute_metrics,
    precision_recall_at_k,
)
from preprocessing import (  # noqa: E402
    NUMERIC,
    build_preprocessor,
    chronological_split,
    clean,
    drop_seen_patients,
    feature_columns,
    load_raw,
)

LABEL_MODEL = {"logreg": "Logistic Regression", "random_forest": "Random Forest"}
def comma_ticks(*axes) -> None:
    """Koma desimal pada sumbu, sesuai penulisan angka di paper."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in axes:
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)


def fmt3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


LABEL_SCORE = {"raw": "Skor mentah", "sigmoid": "Kalibrasi sigmoid", "isotonic": "Kalibrasi isotonic"}


def make_model(name: str, seed: int, exclude_sms: bool) -> Pipeline:
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", build_preprocessor(exclude_sms)), ("clf", clf)])


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_splits(train, val, test, val_full, test_full) -> pd.DataFrame:
    train_ids = set(train["patient_id"])
    seen_val = set(train["patient_id"])
    seen_test = train_ids | set(val_full["patient_id"])
    rows = []
    for name, part, seen in [("train", train, set()), ("validation", val, seen_val), ("test", test, seen_test)]:
        rows.append(
            {
                "split": name,
                "n_rows": len(part),
                "start_date": part["appointment_day"].min().date(),
                "end_date": part["appointment_day"].max().date(),
                "no_show_rate": round(float(part["target"].mean()), 4),
                "n_patients": int(part["patient_id"].nunique()),
                "pct_rows_patient_seen_in_earlier_split": round(float(part["patient_id"].isin(seen).mean()) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def describe_features(prep) -> dict:
    """Jumlah fitur akhir setelah transformasi (hanya dari data latih yang sudah di-fit)."""
    enc = prep.named_transformers_["cat"]
    cats = enc.categories_[0]
    infrequent = enc.infrequent_categories_[0]
    n_infreq = 0 if infrequent is None else len(infrequent)
    return {
        "n_features_total": int(len(prep.get_feature_names_out())),
        "n_numeric_scaled": len(NUMERIC),
        "numeric_scaled_columns": list(NUMERIC),
        "n_binary_passthrough": len([c for c in prep.transformers_[1][2]]),
        "neighbourhood_categories_in_train": int(len(cats)),
        "neighbourhood_infrequent_merged": int(n_infreq),
        "neighbourhood_onehot_columns": int(len(cats) - n_infreq + (1 if n_infreq else 0)),
        "class_weight": "None (default scikit-learn; tanpa resampling)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path CSV Kaggle (joniarroba/noshowappointments)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--seeds", type=int, default=5, help="Jumlah seed untuk Random Forest")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--exclude-sms", action="store_true", help="Sensitivitas: buang fitur sms_received")
    ap.add_argument("--unseen-patients", action="store_true", help="Sensitivitas: val/test hanya pasien baru")
    args = ap.parse_args()

    out = Path(args.out)
    fig_dir = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, report = clean(load_raw(args.data))
    train, val_full, test_full = chronological_split(df, args.train_frac, args.val_frac)
    val, test = (val_full, test_full)
    if args.unseen_patients:
        val, test = drop_seen_patients(train, val_full, test_full)
    split_table = describe_splits(train, val, test, val_full, test_full)
    split_table.to_csv(out / "dataset_characteristics.csv", index=False)
    print(split_table.to_string(index=False))

    feats = feature_columns(args.exclude_sms)
    y_tr, y_val, y_te = train["target"].values, val["target"].values, test["target"].values

    records: list[dict] = []
    val_records: list[dict] = []  # metrik pada validation set (Lampiran A.1: validation vs test)
    keep_scores: dict[tuple[str, str], np.ndarray] = {}  # skor test seed 0 untuk gambar dan CI
    feature_info: dict = {}

    for model_name in ["logreg", "random_forest"]:
        seeds = [0] if model_name == "logreg" else list(range(args.seeds))
        for seed in seeds:
            pipe = make_model(model_name, seed, args.exclude_sms).fit(train[feats], y_tr)
            s_val = pipe.predict_proba(val[feats])[:, 1]
            s_te = pipe.predict_proba(test[feats])[:, 1]
            if not feature_info:
                feature_info = describe_features(pipe.named_steps["prep"])

            def add(scenario: str, m: dict, val_m: dict) -> None:
                records.append({"scenario": scenario, "model": model_name, "seed": seed, **m})
                val_records.append({"scenario": scenario, "model": model_name, "seed": seed, **val_m})

            add("baseline_thr0.5", compute_metrics(y_te, s_te, threshold=0.5, seed=seed),
                compute_metrics(y_val, s_val, threshold=0.5, seed=seed))
            thr = best_f1_threshold(y_val, s_val)
            add("A_tuned_threshold", compute_metrics(y_te, s_te, threshold=thr, seed=seed),
                compute_metrics(y_val, s_val, threshold=thr, seed=seed))
            # A2 (eksploratif, ditambahkan setelah run pertama): threshold = persentil ke-90 skor
            # validation, yaitu cara menyetel threshold untuk kapasitas 10% tanpa melihat test.
            thr_cap = float(np.quantile(s_val, 0.90))
            add("A2_capacity_threshold", compute_metrics(y_te, s_te, threshold=thr_cap, seed=seed),
                compute_metrics(y_val, s_val, threshold=thr_cap, seed=seed))

            variants = {"raw": s_te}
            for method in ["sigmoid", "isotonic"]:
                # Kalibrator di-fit pada validation, jadi metrik validation B_* bersifat in-sample
                # untuk kalibrator (optimistis) dan hanya dipakai sebagai pembanding terhadap test.
                cal = CalibratedClassifierCV(FrozenEstimator(pipe), method=method).fit(val[feats], y_val)
                s_cal = cal.predict_proba(test[feats])[:, 1]
                variants[method] = s_cal
                add(f"B_{method}", compute_metrics(y_te, s_cal, seed=seed),
                    compute_metrics(y_val, cal.predict_proba(val[feats])[:, 1], seed=seed))
            if seed == 0:
                for v, s in variants.items():
                    keep_scores[(model_name, v)] = s

    # Pembanding tanpa model.
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        records.append(
            {"scenario": "reference_random", "model": "none", "seed": seed,
             **compute_metrics(y_te, rng.random(len(y_te)), seed=seed, probabilistic=False)}
        )
    records.append(
        {"scenario": "reference_lead_days", "model": "none", "seed": 0,
         **compute_metrics(y_te, test["lead_days"].values.astype(float), seed=0, probabilistic=False)}
    )
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        val_records.append(
            {"scenario": "reference_random", "model": "none", "seed": seed,
             **compute_metrics(y_val, rng.random(len(y_val)), seed=seed, probabilistic=False)}
        )
    val_records.append(
        {"scenario": "reference_lead_days", "model": "none", "seed": 0,
         **compute_metrics(y_val, val["lead_days"].values.astype(float), seed=0, probabilistic=False)}
    )

    metrics = pd.DataFrame(records)
    metrics.to_csv(out / "metrics.csv", index=False)
    val_metrics = pd.DataFrame(val_records)
    val_metrics.to_csv(out / "metrics_validation.csv", index=False)
    vcols = [c for c in val_metrics.columns if c not in ("scenario", "model", "seed")]
    vg = val_metrics.groupby(["scenario", "model"])[vcols]
    (vg.mean().round(4).add_suffix("_mean").join(vg.std().round(4).add_suffix("_sd"))
       .reset_index().to_csv(out / "summary_validation.csv", index=False))
    num_cols = [c for c in metrics.columns if c not in ("scenario", "model", "seed")]
    grouped = metrics.groupby(["scenario", "model"])[num_cols]
    summary = grouped.mean().round(4).add_suffix("_mean").join(grouped.std().round(4).add_suffix("_sd"))
    summary.reset_index().to_csv(out / "summary.csv", index=False)

    # Interval bootstrap (seed 0).
    ci_rows = []
    metric_fns = {
        "pr_auc": average_precision_score,
        "precision_at_10pct": lambda y, s: precision_recall_at_k(y, s, 0.10, 0)[0],
        "brier": brier_score_loss,
    }
    for (model_name, variant), s in keep_scores.items():
        for mname, fn in metric_fns.items():
            lo, hi = bootstrap_ci(y_te, s, fn, n_boot=args.n_boot, seed=0)
            ci_rows.append({"model": model_name, "scores": variant, "metric": mname,
                            "point": round(float(fn(y_te, s)), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci.csv", index=False)

    # Gambar.
    base = float(y_te.mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in ["logreg", "random_forest"]:
        p, r, _ = precision_recall_curve(y_te, keep_scores[(m, "raw")])
        ax.plot(r[:-1], p[:-1], label=f"{LABEL_MODEL[m]} (PR-AUC = {fmt3(average_precision_score(y_te, keep_scores[(m, 'raw')]))})")
    ax.axhline(base, ls="--", c="gray", label=f"Proporsi no-show uji = {fmt3(base)}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Kurva precision-recall pada data uji", ylim=(0, 1.0))
    ax.legend(fontsize=8, loc="upper right")
    comma_ticks(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, m in zip(axes, ["logreg", "random_forest"]):
        for v in ["raw", "sigmoid", "isotonic"]:
            frac_pos, mean_pred = calibration_curve(y_te, keep_scores[(m, v)], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", ms=3, label=LABEL_SCORE[v])
        ax.plot([0, 1], [0, 1], "--", c="gray", label="Kalibrasi sempurna")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
        ax.set(xlabel="Rata-rata probabilitas prediksi per bin", title=LABEL_MODEL[m])
    axes[0].set_ylabel("Proporsi no-show aktual per bin")
    axes[0].legend(fontsize=8, loc="upper left")
    comma_ticks(*axes)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=300)
    plt.close(fig)

    fracs = np.arange(0.01, 0.31, 0.01)
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in ["logreg", "random_forest"]:
        ax.plot(fracs * 100, [precision_recall_at_k(y_te, keep_scores[(m, "raw")], f, 0)[0] for f in fracs], label=LABEL_MODEL[m])
    ax.plot(fracs * 100, [precision_recall_at_k(y_te, test["lead_days"].values.astype(float), f, 0)[0] for f in fracs],
            label="Heuristik jarak hari penjadwalan (lead_days)")
    ax.axhline(base, ls="--", c="gray", label="Pemilihan acak (proporsi no-show uji)")
    ax.set(xlabel="Kapasitas pengingat (% janji temu teratas)", ylabel="Precision", title="Precision@k pada data uji")
    ax.legend(fontsize=8, loc="upper right")
    comma_ticks(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "precision_at_k.png", dpi=300)
    plt.close(fig)

    config = {
        "args": vars(args),
        "data_sha256": sha256_of(args.data),
        "cleaning_report": report,
        "features": feats,
        "feature_info": feature_info,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    print("\nRingkasan (mean) tersimpan di", out / "summary.csv")


if __name__ == "__main__":
    main()
