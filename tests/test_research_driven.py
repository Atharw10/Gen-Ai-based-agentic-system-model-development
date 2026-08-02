"""Research-driven decisions: models/windows/HPO from the plan, with reasoning+source logged."""
from __future__ import annotations

import json
from pathlib import Path

from advisors import design, research
from contracts import RunConfig
from pipeline import profiling, training
from runtime.run_logger import RunLogger
from tests.synth import tiny_fixture


def _profile():
    df = tiny_fixture(seed=0)
    return df, profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "credit card propensity")


def test_research_plan_has_all_sections_with_sources():
    _, profile = _profile()
    plan = research.research("credit card propensity", profile, cached_llm=None)
    assert plan["models"]["recommended"]
    assert plan["windows"]["observation_window_months"] >= 1 and plan["windows"]["oot_months"] >= 1
    assert "spaces" in plan["hpo"]
    assert plan["feature_families"]["families"]
    for sec in ("models", "windows", "hpo", "feature_families"):
        assert plan[sec]["source"]  # provenance present for every decision


def test_design_uses_research_windows_and_models():
    _, profile = _profile()
    plan = research.research("credit card propensity", profile, cached_llm=None)
    cfg, _ = design.design_config(profile, "credit card propensity", research=plan)
    assert set(cfg.search.algorithms) == set(plan["models"]["recommended"])
    assert len(cfg.windows.oot_snapshots) == plan["windows"]["oot_months"]


def test_research_hpo_override_is_used_by_training():
    # a research plan that proposes a narrow LightGBM learning_rate range
    _, profile = _profile()
    plan = research.research("credit card propensity", profile, cached_llm=None)
    plan["hpo"]["spaces"] = {"LightGBM": {"learning_rate": [0.111, 0.113]}}
    cfg, _ = design.design_config(profile, "credit card propensity", research=plan)
    assert cfg.search.hpo_overrides == {"LightGBM": {"learning_rate": [0.111, 0.113]}}

    # suggest_params must draw learning_rate from the overridden range
    import optuna

    study = optuna.create_study()
    seen = []
    def obj(trial):
        p = training.suggest_params(trial, "LightGBM", cfg.search.hpo_overrides)
        seen.append(p["learning_rate"])
        return 0.0
    study.optimize(obj, n_trials=5)
    assert all(0.111 <= lr <= 0.113 for lr in seen)


def test_invalid_hpo_override_falls_back_to_default():
    import optuna

    study = optuna.create_study()
    bad = {"LightGBM": {"learning_rate": [0.2, 0.01]}}  # lo > hi -> invalid -> default range used
    seen = []
    def obj(trial):
        p = training.suggest_params(trial, "LightGBM", bad)
        seen.append(p["learning_rate"])
        return 0.0
    study.optimize(obj, n_trials=5)
    assert all(0.02 <= lr <= 0.2 for lr in seen)  # default LightGBM range


class _FakeLLM:
    def __init__(self, resp):
        self.resp = resp

    def invoke(self, prompt, label="llm"):
        return self.resp


def test_offline_reasons_are_sourced_deterministic_not_gemini():
    _, profile = _profile()
    plan = research.research("credit card propensity", profile, cached_llm=None)
    for sec in ("models", "windows", "hpo", "feature_families"):
        assert "deterministic" in plan[sec]["source"]
        assert "Gemini" not in plan[sec]["source"]


def test_partial_llm_does_not_fake_gemini_source_for_missing_reasons():
    """A section is Source=Gemini ONLY if the LLM gave it a real reason; otherwise it stays
    deterministic. We must never present a hardcoded reason as the model's reasoning."""
    _, profile = _profile()
    resp = json.dumps({
        "summary": "x",
        "models": {"recommended": ["XGBoost"], "reason": "GBM handles severe skew best"},
        "windows": {"observation_window_months": 3, "oot_months": 2},          # no reason
        "hpo": {"spaces": {}},                                                  # no reason
        "feature_families": {"families": [{"name": "rfm", "reason": "r"}], "reason": "rfm matters"},
    })
    plan = research.research("cc", profile, cached_llm=_FakeLLM(resp))

    # models + families had genuine LLM reasons -> Gemini, reason verbatim from the model
    assert "Gemini" in plan["models"]["source"] and plan["models"]["reason"] == "GBM handles severe skew best"
    assert "Gemini" in plan["feature_families"]["source"]
    # windows + hpo had NO reason -> stay deterministic, NOT stamped Gemini
    assert "deterministic" in plan["windows"]["source"] and "Gemini" not in plan["windows"]["source"]
    assert "deterministic" in plan["hpo"]["source"] and "Gemini" not in plan["hpo"]["source"]


def test_run_logger_writes_reasons_and_sources(tmp_path):
    log = RunLogger(run_dir=str(tmp_path), verbose=False)
    with log.step("research"):
        log.reason("research/models", "LightGBM, XGBoost", "GBMs handle skew", "Gemini + web")
    md = (tmp_path / "log1_cot.md").read_text()
    assert "research/models" in md and "**Why:**" in md and "**Source:** Gemini + web" in md
    reasons = json.loads((tmp_path / "log1_reasons.json").read_text())
    assert reasons[0]["source"] == "Gemini + web"
