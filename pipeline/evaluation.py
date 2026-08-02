"""Stage 6 - evaluation on OOT: metric computation, scorers, threshold tuning, lift table.

No dependency on training.py (so training can import scorers from here without a cycle).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlkit import stats


def pos_score(model, X) -> np.ndarray:
    """Probability/score of the positive class, robust to estimator type."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 1:
            return p[:, 0]
        classes = list(getattr(model, "classes_", [0, 1]))
        return p[:, classes.index(1)] if 1 in classes else p[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def oot_metrics(model, X, y) -> dict[str, float]:
    y = np.asarray(y)
    score = pos_score(model, X)
    out: dict[str, float] = {}

    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
        out["ks"] = float(stats.ks_statistic(y, score))
        out["gini"] = float(stats.gini_coefficient(y, score))
        # precision/recall/f1 at the F1-optimal operating threshold. The default 0.5 cut from
        # model.predict() classifies almost everything as negative on imbalanced data (here ~5%
        # positives), which would report precision/recall/f1 = 0 even for a good ranker. We pick
        # the threshold that maximises F1 on the PR curve (data-derived, not hardcoded).
        prec_c, rec_c, thr = precision_recall_curve(y, score)
        if len(thr):
            f1s = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)
            threshold = float(thr[int(np.argmax(f1s))])
        else:
            threshold = 0.5
        pred = (score >= threshold).astype(int)
    else:
        # single-class target (e.g. a degenerate CV fold): fall back to the model's own decision
        pred = model.predict(X)

    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    return out


def primary_scorer(primary: str):
    """Return a callable (estimator, X, y) -> float for CV scoring on the primary metric."""
    key = {"PR_AUC": "pr_auc", "ROC_AUC": "roc_auc", "KS": "ks", "F1": "f1",
           "Recall": "recall", "Precision": "precision"}.get(primary, "pr_auc")

    def _score(est, X, y):
        return oot_metrics(est, X, y).get(key, 0.0)

    return _score


def tune_thresholds(model, X, y) -> dict:
    """F1-optimal threshold + business-anchored thresholds (precision>=0.5, recall>=0.8)."""
    y = np.asarray(y)
    score = pos_score(model, X)
    prec, rec, thr = precision_recall_curve(y, score)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1])) if len(thr) else 0
    f1_optimal = float(thr[best_idx]) if len(thr) else 0.5
    p50 = np.where(prec[:-1] >= 0.5)[0]
    r80 = np.where(rec[:-1] >= 0.8)[0]
    return {
        "f1_optimal": f1_optimal,
        "precision_50": float(thr[p50[0]]) if len(p50) else None,
        "recall_80": float(thr[r80[-1]]) if len(r80) else None,
    }


def lift_table(y, score, n_bins: int = 10):
    import pandas as pd

    y = np.asarray(y)
    score = np.asarray(score)
    order = np.argsort(-score)
    y_sorted = y[order]
    base = y.mean() if y.mean() > 0 else 1e-9
    n = len(y)
    rows = []
    for d in range(n_bins):
        lo, hi = int(d * n / n_bins), int((d + 1) * n / n_bins)
        seg = y_sorted[lo:hi]
        rate = seg.mean() if len(seg) else 0.0
        rows.append({
            "decile": d + 1,
            "n": len(seg),
            "positives": int(seg.sum()),
            "rate": round(float(rate), 4),
            "lift": round(float(rate / base), 3),
            "cum_capture": round(float(y_sorted[:hi].sum() / max(y.sum(), 1)), 4),
        })
    return pd.DataFrame(rows)
