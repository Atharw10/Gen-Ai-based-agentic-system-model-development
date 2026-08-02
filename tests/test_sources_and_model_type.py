"""Phase-1 additions: data-source registry + explicit model_type hint.

Covers:
  * FileSource loads CSV/Parquet and prunes columns (the only REAL source in Phase 1).
  * CDP/CML sources are registered (appear in the dropdown) but raise NotWiredError until Phase 2.
  * model_type reliably drives the use-case template (research) and the primary metric (design),
    overriding fuzzy text — while remaining fully optional/backward compatible.
"""
from __future__ import annotations

import pandas as pd
import pytest

from advisors import design, research
from contracts import RunConfig
from dataio import sources
from pipeline import profiling
from tests.synth import tiny_fixture


# ----------------------------- data sources -------------------------------------------------
def test_registry_lists_all_sources():
    keys = set(sources.SOURCES)
    assert {"file", "hive", "impala", "feature_store", "datalake", "custom_sql"} <= keys
    # menu order is stable and FileSource is real
    assert any(s.key == "file" for s in sources.list_sources())


def test_file_source_loads_csv_and_prunes_columns(tmp_path):
    df = tiny_fixture(seed=1)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)

    spec = sources.SourceSpec(path=str(p))
    out = sources.load_dataframe("file", spec)
    assert isinstance(out, pd.DataFrame) and len(out) == len(df)

    # column pruning
    cols = ["target", "snapshot_date"]
    pruned = sources.load_dataframe("file", sources.SourceSpec(path=str(p), columns=cols))
    assert list(pruned.columns) == cols


def test_file_source_loads_parquet(tmp_path):
    df = tiny_fixture(seed=2)
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    out = sources.load_dataframe("file", sources.SourceSpec(path=str(p)))
    assert len(out) == len(df)


@pytest.mark.parametrize("key", ["hive", "impala", "feature_store", "datalake", "custom_sql"])
def test_cdp_sources_are_stubs_until_phase2(key):
    spec = sources.SourceSpec(path="s3a://bucket/features/", table="risk_db.cust_features",
                              snapshot_col="snapshot_month", snapshot_range=("2025-01", "2025-06"),
                              sql="SELECT 1")
    with pytest.raises(sources.NotWiredError):
        sources.load_dataframe(key, spec)


def test_pushdown_sql_includes_columns_and_partition_filter():
    spec = sources.SourceSpec(table="risk_db.cust_features", snapshot_col="snapshot_month",
                              snapshot_range=("2025-01", "2025-06"), columns=["a", "b"])
    q = sources._push_down_sql(spec)
    assert "SELECT a, b FROM risk_db.cust_features" in q
    assert "snapshot_month BETWEEN '2025-01' AND '2025-06'" in q


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        sources.get_source("nope")


# ----------------------------- model_type hint ----------------------------------------------
def _profile():
    df = tiny_fixture(seed=0)
    return profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "anything")


def test_model_type_selects_use_case_in_research():
    profile = _profile()
    # objective text says nothing about fraud, but model_type=fraud must steer the use case
    plan = research.research("anything", profile, cached_llm=None, model_type="fraud")
    fams = " ".join(f["name"] for f in plan["feature_families"]["families"]).lower()
    assert "velocity" in fams or "anomal" in fams  # fraud families, not generic


def test_credit_risk_model_type_sets_ks_metric_and_is_stored():
    profile = _profile()
    plan = research.research("loan book", profile, cached_llm=None, model_type="credit_risk")
    cfg, _ = design.design_config(profile, "loan book", research=plan, model_type="credit_risk")
    assert isinstance(cfg, RunConfig)
    assert cfg.model_type == "credit_risk"
    assert cfg.primary_metric == "KS"


def test_model_type_optional_keeps_legacy_behaviour():
    profile = _profile()
    # no model_type -> falls back to text matching, config still valid, model_type is None
    plan = research.research("Build CC propensity model", profile, cached_llm=None)
    cfg, _ = design.design_config(profile, "Build CC propensity model", research=plan)
    assert cfg.model_type is None
    assert isinstance(cfg, RunConfig)
