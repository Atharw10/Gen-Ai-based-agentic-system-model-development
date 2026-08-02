"""Validation agent: deterministic checks flag known bugs; full report saved to artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from app import orchestrator
from app.validation_agent import validate_run
from pipeline import validation
from tests.synth import tiny_fixture


class _FakeResult:
    """Minimal result-like object for unit-testing the deterministic checks."""
    primary_metric = "PR_AUC"
    best_algorithm = "X"
    column_alias_map = None
    selection_report = None
    split_report = None
    def __init__(self, oot):
        self.leaderboard = [{"algorithm": "X", "oot_metrics": oot}]


def test_checks_flag_zero_metric_and_below_random():
    det = validation.run_checks(_FakeResult({"pr_auc": 0.0, "roc_auc": 0.4}))
    ids = {f["id"]: f["level"] for f in det["findings"]}
    assert det["verdict"] == "RED"
    assert ids.get("primary_metric_zero") == "FAIL"
    assert ids.get("auc_below_random") == "FAIL"


def test_checks_flag_leakage_too_perfect():
    det = validation.run_checks(_FakeResult({"pr_auc": 0.9999, "roc_auc": 1.0}))
    ids = {f["id"] for f in det["findings"]}
    assert "auc_too_perfect" in ids
    assert det["verdict"] in ("AMBER", "RED")


def test_checks_green_when_healthy():
    det = validation.run_checks(_FakeResult({"pr_auc": 0.42, "roc_auc": 0.81,
                                             "precision": 0.5, "recall": 0.6, "f1": 0.55}))
    # no split/selection/run_dir provided -> only metric checks, all healthy
    assert det["verdict"] == "GREEN"


def test_validate_run_saves_report(tmp_path):
    df = tiny_fixture(seed=0)
    res, model, prepared, summary = orchestrator.run(
        df, business_objective="x", target_col="target", date_col="snapshot_date",
        group_col="customer_id", cached_llm=None, search_rounds=1,
        artifacts_root=str(tmp_path), source_path=None, model_type="propensity", verbose=False,
    )
    report = validate_run(res.run_dir, res, prepared, model, summary=summary,
                          cached_llm=None, context={"use_llm": False, "tavily_key_set": False},
                          verbose=False)
    assert report["verdict"] in ("GREEN", "AMBER", "RED")
    rj = Path(res.run_dir) / "validation_report.json"
    rm = Path(res.run_dir) / "validation_report.md"
    assert rj.exists() and rm.exists()
    saved = json.loads(rj.read_text(encoding="utf-8"))
    assert "deterministic" in saved and "llm_review" in saved
