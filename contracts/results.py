"""Typed result objects passed between pipeline stages and back to the orchestrator.

Pure schema. No pandas/sklearn/LLM imports — keeps the layering test happy and makes
results trivially serialisable for logging and HITL checkpoints.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TargetProfile(BaseModel):
    name: str
    n_unique: int
    is_binary: bool
    pos_ratio: float | None = None
    null_pct: float = 0.0
    distribution: dict[str, int] = Field(default_factory=dict)


class TemporalProfile(BaseModel):
    date_col: str | None = None
    min_date: str | None = None
    max_date: str | None = None
    n_snapshots: int = 0
    snapshots: list[str] = Field(default_factory=list)
    rows_per_snapshot: dict[str, int] = Field(default_factory=dict)
    # positive (target==1) rate per snapshot month — used to spot drift before choosing windows/OOT
    pos_ratio_per_snapshot: dict[str, float] = Field(default_factory=dict)


class ProfileResult(BaseModel):
    business_objective: str = ""
    n_rows: int
    n_cols: int
    memory_mb: float = 0.0
    target: TargetProfile
    temporal: TemporalProfile
    numeric_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    high_cardinality_cols: list[str] = Field(default_factory=list)
    constant_cols: list[str] = Field(default_factory=list)
    group_col: str | None = None
    overall_missing_pct: float = 0.0


class SelectionReport(BaseModel):
    n_in: int
    n_after_hygiene: int
    n_after_univariate: int
    n_after_redundancy: int
    n_selected: int
    selected_features: list[str]
    dropped_constant: list[str] = Field(default_factory=list)
    dropped_high_missing: list[str] = Field(default_factory=list)
    dropped_high_cardinality: list[str] = Field(default_factory=list)
    dropped_low_iv: list[str] = Field(default_factory=list)
    dropped_redundant: list[str] = Field(default_factory=list)
    suspicious_high_iv: list[str] = Field(default_factory=list)  # likely target leakage — flagged, not trusted
    iv_table: list[dict[str, Any]] = Field(default_factory=list)


class SplitReport(BaseModel):
    train_rows: int
    oot_rows: int
    train_range: list[str]
    oot_range: list[str]
    train_pos_ratio: float | None = None
    oot_pos_ratio: float | None = None


class TrialResult(BaseModel):
    trial_id: int
    algorithm: str
    params: dict[str, Any]
    cv_score: float
    cv_std: float
    oot_metrics: dict[str, float] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_dir: str
    config_path: str
    data_hash: str
    best_algorithm: str
    best_trial_id: int
    primary_metric: str
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    optimal_threshold: float = 0.5
    business_thresholds: dict[str, float | None] = Field(default_factory=dict)
    selection_report: SelectionReport | None = None
    split_report: SplitReport | None = None
    n_trials_run: int = 0
    column_alias_map: dict[str, Any] | None = None  # present when input columns were aliased
