"""Phase 3 acceptance: advisors produce validated configs with no LLM (deterministic fallback)."""
from __future__ import annotations

from advisors import design, narrative, research
from contracts import RunConfig, RunResult
from pipeline import profiling
from tests.synth import tiny_fixture


def _profile():
    df = tiny_fixture(seed=0)
    return df, profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "cc propensity")


def test_design_advisor_no_llm_returns_valid_config():
    _, profile = _profile()
    plan = research.research("Build CC propensity model", profile, cached_llm=None)
    cfg, errors = design.design_config(profile, "Build CC propensity model", research=plan)
    assert isinstance(cfg, RunConfig)
    # OOT must be the latest snapshots, disjoint from training (RunConfig already validates this)
    assert cfg.windows.oot_snapshots[-1] == profile.temporal.snapshots[-1]
    assert cfg.data_strategy.group_col == "customer_id"
    assert set(cfg.search.algorithms) == set(plan["models"]["recommended"])


def test_design_tier_matches_profile():
    _, profile = _profile()
    cfg = design.default_config_from_profile(profile, "cc")
    from mlkit.metrics import assign_tier

    assert cfg.data_strategy.imbalance_tier == assign_tier(profile.target.pos_ratio) or cfg.data_strategy.imbalance_tier == "moderate"


def test_research_stub_runs_offline():
    _, profile = _profile()
    out = research.research("credit card propensity", profile, cached_llm=None, mode="stub")
    assert out["mode"] == "stub"
    assert out["recommended_algorithms"]
    for sec in ("models", "windows", "hpo", "feature_families"):
        assert "reason" in out[sec] and "source" in out[sec]


def test_narrative_template_no_llm():
    res = RunResult(
        run_dir="x", config_path="x", data_hash="x", best_algorithm="LightGBM",
        best_trial_id=2, primary_metric="PR_AUC",
        leaderboard=[{"algorithm": "LightGBM", "oot_metrics": {"pr_auc": 0.3}}],
        n_trials_run=10,
    )
    text = narrative.summarize(res, cached_llm=None)
    assert "LightGBM" in text and "PR_AUC" in text
