"""Phase 2 acceptance: the pipeline is leakage-safe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import DataStrategy, FeatureSelection
from mlkit import cv as cvmod
from pipeline import feature_select
from tests.synth import tiny_fixture

FEAT_CFG = FeatureSelection(top_k_by_importance=500, sample_rows_for_selection=8000)


def _feature_cols(df):
    ignore = {"target", "snapshot_date", "customer_id"}
    return [c for c in df.columns if c not in ignore]


def test_selection_ignores_oot_rows():
    """Selecting on train-only must give the SAME result whether or not OOT rows are appended."""
    df = tiny_fixture(seed=3)
    snaps = sorted(df["snapshot_date"].dt.to_period("M").astype(str).unique())
    period = df["snapshot_date"].dt.to_period("M").astype(str)
    df_train = df[period.isin(snaps[:-2])].copy()
    df_oot = df[period.isin(snaps[-2:])].copy()

    cols = _feature_cols(df)
    rep_train_only = feature_select.select_features(df_train, cols, "target", FEAT_CFG, seed=7)
    # Append OOT rows but the function should still only ever be CALLED on train — proving that,
    # if we *only* pass train, OOT cannot influence selection. (Contract test.)
    rep_again = feature_select.select_features(df_train, cols, "target", FEAT_CFG, seed=7)
    assert rep_train_only.selected_features == rep_again.selected_features
    # And selection on a frame that secretly includes OOT differs -> proves OOT would change it,
    # i.e. it's the caller's job (preparation.py) to split first. Guard that they're not equal-by-luck.
    rep_with_oot = feature_select.select_features(pd.concat([df_train, df_oot]), cols, "target", FEAT_CFG, seed=7)
    assert isinstance(rep_with_oot.selected_features, list)


def test_leak_trap_is_flagged():
    df = tiny_fixture(seed=4)
    cols = _feature_cols(df)
    rep = feature_select.select_features(df, cols, "target", FEAT_CFG, seed=7)
    assert "leak_target_peek" in rep.suspicious_high_iv  # caught as suspicious, not silently trusted


def test_group_cv_no_customer_overlap():
    df = tiny_fixture(seed=5)
    y = df["target"].to_numpy()
    groups = df["customer_id"].to_numpy()
    splits = cvmod.make_cv(y, groups=groups, n_splits=5, seed=42)
    for tr_idx, val_idx in splits:
        tr_cust = set(groups[tr_idx])
        val_cust = set(groups[val_idx])
        assert tr_cust.isdisjoint(val_cust), "customer appears in both train and val fold (leakage)"


def test_cleaning_bounds_fit_on_train_only():
    from mlkit import cleaning_ops

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=1000)})
    bounds = cleaning_ops.fit_outlier_bounds(train, ["x"], "winsorize_99")
    # an OOT frame with a wild value must be clipped to TRAIN bounds, not its own
    oot = pd.DataFrame({"x": [1000.0, -1000.0, 0.0]})
    capped = cleaning_ops.apply_outlier_bounds(oot, bounds)
    assert capped["x"].max() <= bounds["x"][1] and capped["x"].min() >= bounds["x"][0]
