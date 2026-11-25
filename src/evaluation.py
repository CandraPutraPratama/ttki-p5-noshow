"""Metrik evaluasi untuk prioritisasi no-show pada kapasitas pengingat terbatas."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_k_indices(score: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Indeks `ceil(frac*n)` skor tertinggi; seri (ties) dipecah acak dengan seed tetap."""
    n = len(score)
    k = max(1, math.ceil(frac * n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    order = perm[np.argsort(-np.asarray(score)[perm], kind="stable")]
    return order[:k]


def precision_recall_at_k(y: np.ndarray, score: np.ndarray, frac: float = 0.10, seed: int = 0):
    y = np.asarray(y)
    idx = top_k_indices(score, frac, seed)
    hits = float(y[idx].sum())
    precision = hits / len(idx)
    recall = hits / max(1.0, float(y.sum()))
    lift = precision / max(1e-12, float(y.mean()))
    return precision, recall, lift


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE dengan bin lebar sama pada [0, 1]."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def expected_calibration_error_quantile(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE dengan bin kuantil (isi tiap bin hampir sama), binning yang sama dengan diagram reliabilitas.

    Tepi bin dihitung seperti `sklearn.calibration.calibration_curve(strategy="quantile")`;
    bin kosong dilewati. Dilaporkan sebagai pelengkap ECE bin lebar sama.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.percentile(p, np.linspace(0.0, 100.0, bins + 1))
    ids = np.searchsorted(edges[1:-1], p)
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def best_f1_threshold(y_val: np.ndarray, score_val: np.ndarray) -> float:
    """Threshold yang memaksimalkan F1 pada validation set (bukan test)."""
    p, r, t = precision_recall_curve(y_val, score_val)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def compute_metrics(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float | None = None,
    k_frac: float = 0.10,
    seed: int = 0,
    probabilistic: bool = True,
) -> dict:
    y = np.asarray(y)
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    p_k, r_k, lift = precision_recall_at_k(y, score, k_frac, seed)
    out.update({"precision_at_10pct": p_k, "recall_at_10pct": r_k, "lift_at_10pct": lift})
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, score))
        out["ece"] = expected_calibration_error(y, score)
        out["ece_quantile"] = expected_calibration_error_quantile(y, score)
    if threshold is not None:
        pred = (np.asarray(score) >= threshold).astype(int)
        out.update(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "flagged_fraction": float(pred.mean()),
            }
        )
    return out


def bootstrap_ci(y, score, metric_fn, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Interval persentil bootstrap (resampling baris)."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric_fn(y[idx], score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
