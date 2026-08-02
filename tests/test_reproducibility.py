"""Phase 2/3 acceptance (MRM-critical): same config + data + seed => identical results.

This is the property that makes the pipeline auditable. No LLM is involved in a re-run.
"""
from __future__ import annotations

from contracts import DataStrategy, FeatureSelection, RunConfig, SearchSpace, Windows
from pipeline.runner import run_from_config
from tests.synth import tiny_fixture


def _cfg(df):
    snaps = sorted(df["snapshot_date"].dt.to_period("M").astype(str).unique())
    return RunConfig(
        business_objective="cc propensity", target_col="target", date_col="snapshot_date",
        primary_metric="PR_AUC",
        windows=Windows(observation_window_months=3, performance_window_months=1,
                        snapshots_for_training=snaps[:-2], oot_snapshots=snaps[-2:]),
        data_strategy=DataStrategy(group_col="customer_id", imbalance_tier="moderate"),
        feature_selection=FeatureSelection(top_k_by_importance=15, sample_rows_for_selection=6000),
        search=SearchSpace(algorithms=["LogisticRegression", "HistGradientBoosting"], n_trials=4),
        random_seed=42,
    )


def test_same_config_same_results():
    df = tiny_fixture(seed=0)
    cfg = _cfg(df)
    r1, _, p1 = run_from_config(df, cfg)
    r2, _, p2 = run_from_config(df, cfg)

    assert p1.selected_raw_features == p2.selected_raw_features
    assert r1.best_algorithm == r2.best_algorithm
    assert r1.best_trial_id == r2.best_trial_id
    # leaderboard OOT metrics identical to 6 dp
    for row1, row2 in zip(r1.leaderboard, r2.leaderboard):
        assert row1["algorithm"] == row2["algorithm"]
        for k in row1["oot_metrics"]:
            assert abs(row1["oot_metrics"][k] - row2["oot_metrics"][k]) < 1e-9
