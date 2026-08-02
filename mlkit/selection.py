"""Scalable, leakage-safe feature selection primitives.

Designed for 5k-feature feature stores where v1's O(n^2) Python correlation loop hangs.
Everything here operates on TRAIN ONLY (callers must pass train rows). The expensive steps
sample rows so they stay tractable at 500k x 5k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Hygiene filters
# ---------------------------------------------------------------------------
def hygiene_filter(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_missing_pct: float = 0.70,
    max_cardinality: int = 50,
) -> dict:
    """Drop constant cols, >max_missing cols, and ID-like high-cardinality categoricals.

    Returns dict(kept, dropped_constant, dropped_high_missing, dropped_high_cardinality).
    """
    dropped_constant, dropped_high_missing, dropped_high_card = [], [], []
    miss = df[feature_cols].isna().mean()
    for c in feature_cols:
        if df[c].nunique(dropna=True) <= 1:
            dropped_constant.append(c)
        elif miss[c] > max_missing_pct:
            dropped_high_missing.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > max_cardinality:
            dropped_high_card.append(c)
    dropped = set(dropped_constant) | set(dropped_high_missing) | set(dropped_high_card)
    kept = [c for c in feature_cols if c not in dropped]
    return {
        "kept": kept,
        "dropped_constant": dropped_constant,
        "dropped_high_missing": dropped_high_missing,
        "dropped_high_cardinality": dropped_high_card,
    }


# ---------------------------------------------------------------------------
# 2. Redundancy pruning via correlation CLUSTERING (no O(n^2) Python loop)
# ---------------------------------------------------------------------------
def cluster_prune(
    df: pd.DataFrame,
    feature_cols: list[str],
    rank: dict[str, float],
    threshold: float = 0.90,
    sample_rows: int = 50_000,
    seed: int = 42,
) -> dict:
    """Cluster numeric features by |correlation| and keep the highest-`rank` feature per cluster.

    Uses scipy hierarchical clustering on a sampled correlation matrix. O(k^2) in features for
    the corr matrix but vectorised in C, and the cluster assignment is a single linkage call —
    no Python double loop over feature pairs.
    """
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        return {"kept": list(feature_cols), "dropped_redundant": []}

    sample = df[numeric]
    if len(sample) > sample_rows:
        sample = sample.sample(sample_rows, random_state=seed)
    corr = sample.corr().abs().fillna(0.0)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry against fp drift
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # features in the same cluster have |corr| >= threshold (distance <= 1 - threshold)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    keep: list[str] = []
    by_cluster: dict[int, list[str]] = {}
    for col, lab in zip(numeric, labels):
        by_cluster.setdefault(lab, []).append(col)
    for cols in by_cluster.values():
        keep.append(max(cols, key=lambda c: rank.get(c, 0.0)))

    keep_set = set(keep)
    # non-numeric features pass through untouched
    kept = [c for c in feature_cols if c in keep_set or not pd.api.types.is_numeric_dtype(df[c])]
    dropped = [c for c in numeric if c not in keep_set]
    return {"kept": kept, "dropped_redundant": dropped}


# ---------------------------------------------------------------------------
# 3. Model-based importance (null-importance / gain)
# ---------------------------------------------------------------------------
def null_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_k: int = 200,
    sample_rows: int = 50_000,
    n_runs: int = 1,
    seed: int = 42,
) -> dict:
    """Rank features by (real gain - shuffled-target gain) using a fast gradient booster.

    Null-importance removes spurious importance that noisy/high-cardinality features get by
    chance. Uses sklearn HistGradientBoosting (no libomp dependency) so it runs anywhere.
    Returns dict(ranked: list[(feat, score)], selected: top_k feature names).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    work = df[feature_cols + [target_col]]
    if len(work) > sample_rows:
        work = work.sample(sample_rows, random_state=seed)

    # encode: HGB handles numerics + NaNs natively; cast object cols to category codes
    X = work[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    X = X.to_numpy(dtype=np.float32)
    y = work[target_col].to_numpy()

    rng = np.random.default_rng(seed)

    def _importance(y_vec) -> np.ndarray:
        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
        model.fit(X, y_vec)
        # permutation importance on the training sample is fine here: we only need a ranking
        r = permutation_importance(model, X, y_vec, n_repeats=3, random_state=seed, scoring="average_precision")
        return r.importances_mean

    real = _importance(y)
    null = np.zeros_like(real)
    for _ in range(n_runs):
        null += _importance(rng.permutation(y))
    null /= max(n_runs, 1)

    score = real - null
    ranked = sorted(zip(feature_cols, score.tolist()), key=lambda t: t[1], reverse=True)
    selected = [f for f, s in ranked[:top_k]]
    return {"ranked": ranked, "selected": selected}
