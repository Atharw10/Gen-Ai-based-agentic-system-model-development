"""Stage 4c - scalable feature selection funnel. TRAIN ONLY (caller passes train rows).

Funnel: hygiene -> univariate IV -> redundancy clustering -> model importance (top_k).
Replaces v1's hand-coded 13-concept FE + O(n^2) correlation loop.
"""
from __future__ import annotations

import pandas as pd

from contracts import FeatureSelection, SelectionReport
from mlkit import selection, stats


def select_features(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: FeatureSelection,
    seed: int = 42,
) -> SelectionReport:
    n_in = len(feature_cols)

    # 1. hygiene
    hy = selection.hygiene_filter(df_train, feature_cols, cfg.max_missing_pct)
    survivors = hy["kept"]

    # 2. univariate IV (vectorised table); keep ranking for later tie-breaks
    iv_df = stats.compute_iv_table(df_train, target_col, feature_cols=survivors)
    iv_rank = dict(zip(iv_df["feature"], iv_df["IV"]))
    low_iv = iv_df[iv_df["IV"] < cfg.min_iv]["feature"].tolist()
    survivors = [c for c in survivors if c not in set(low_iv)]
    # Suspiciously high IV almost always means target leakage — flag for human review.
    suspicious = iv_df[iv_df["IV"] > 0.5]["feature"].tolist()

    # 3. redundancy pruning via correlation clustering (keep highest-IV per cluster)
    cl = selection.cluster_prune(
        df_train, survivors, rank=iv_rank, threshold=cfg.corr_threshold,
        sample_rows=cfg.sample_rows_for_selection, seed=seed,
    )
    survivors = cl["kept"]

    # 4. model-based importance, cap at top_k
    if len(survivors) > cfg.top_k_by_importance:
        ni = selection.null_importance(
            df_train, survivors, target_col, top_k=cfg.top_k_by_importance,
            sample_rows=cfg.sample_rows_for_selection, seed=seed,
        )
        selected = ni["selected"]
    else:
        selected = survivors

    return SelectionReport(
        n_in=n_in,
        n_after_hygiene=len(hy["kept"]),
        n_after_univariate=n_in - len(hy["dropped_constant"]) - len(hy["dropped_high_missing"])
        - len(hy["dropped_high_cardinality"]) - len(low_iv),
        n_after_redundancy=len(cl["kept"]),
        n_selected=len(selected),
        selected_features=selected,
        dropped_constant=hy["dropped_constant"],
        dropped_high_missing=hy["dropped_high_missing"],
        dropped_high_cardinality=hy["dropped_high_cardinality"],
        dropped_low_iv=low_iv,
        dropped_redundant=cl["dropped_redundant"],
        suspicious_high_iv=suspicious,
        iv_table=iv_df.head(100).to_dict("records"),
    )
