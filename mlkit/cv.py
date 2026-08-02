"""Cross-validation splitters that respect customer grouping and time.

Random K-fold on customer-month panel data leaks: the same customer appears in multiple
rows, so a random split puts the same customer in both train and validation folds and
inflates the score. Use GroupKFold by customer when a group column exists; otherwise fall
back to StratifiedKFold.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold


def make_cv(y, groups=None, n_splits: int = 5, seed: int = 42):
    """Return (splitter, split_kwargs) appropriate for the data.

    - groups present -> GroupKFold (no within-customer leakage). GroupKFold is deterministic
      and ignores shuffle/seed by construction.
    - no groups       -> StratifiedKFold(shuffle=True, seed) to preserve class balance.

    Returns a list of (train_idx, val_idx) so callers don't worry about the kwargs difference.
    """
    y = np.asarray(y)
    n = len(y)
    # never ask for more splits than the rarest class / number of groups can support
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(n_splits, n_groups))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(n), y, groups))

    # stratified fallback
    min_class = int(np.bincount(y.astype(int)).min()) if y.size else 0
    n_splits = max(2, min(n_splits, max(2, min_class)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n), y))


def cv_score(estimator, X, y, splits, scorer) -> tuple[float, float]:
    """Run CV over precomputed splits and return (mean, std) of the scorer.

    `scorer` is a callable (estimator, X_val, y_val) -> float. Cloning keeps folds independent.
    """
    from sklearn.base import clone

    scores = []
    for tr_idx, val_idx in splits:
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        scores.append(scorer(est, X[val_idx], y[val_idx]))
    arr = np.asarray(scores, dtype=float)
    return float(arr.mean()), float(arr.std())
