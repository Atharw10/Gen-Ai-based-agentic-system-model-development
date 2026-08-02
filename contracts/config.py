"""Typed run contracts (v2).

The RunConfig is the reproducible heart of the pipeline: the design advisor emits it,
the deterministic pipeline consumes it, and it is persisted with every run. Re-running a
saved RunConfig (with the same data hash + seed) reproduces the model byte-for-byte.

No LLM / pandas imports here on purpose — this module is a pure schema shared by all planes.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Algo = Literal[
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "GradientBoosting",
    "HistGradientBoosting",
]
Tier = Literal["balanced", "mild", "moderate", "severe", "extreme"]
PrimaryMetric = Literal["PR_AUC", "ROC_AUC", "KS", "F1", "Recall", "Precision"]
# Explicit model-type hint (replaces fuzzy text matching in research/design). Optional/back-compat.
ModelType = Literal[
    "propensity", "cross_sell", "up_sell", "churn", "fraud", "credit_risk", "logistic", "generic"
]

TREE_ALGOS: frozenset[str] = frozenset(
    {"RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "HistGradientBoosting"}
)


class Windows(BaseModel):
    observation_window_months: int = Field(ge=1, le=24)
    performance_window_months: int = Field(ge=1, le=12)
    snapshots_for_training: list[str] = Field(min_length=1)  # ["2025-01", ...]
    oot_snapshots: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint_and_ordered(self) -> "Windows":
        tr, oot = set(self.snapshots_for_training), set(self.oot_snapshots)
        overlap = tr & oot
        if overlap:
            raise ValueError(f"train/OOT snapshot overlap: {sorted(overlap)}")
        # YYYY-MM strings sort lexicographically == chronologically
        if min(oot) <= max(tr):
            raise ValueError(
                "OOT snapshots must come strictly after all training snapshots "
                f"(max train={max(tr)}, min oot={min(oot)})"
            )
        return self


class DataStrategy(BaseModel):
    group_col: str | None = None  # customer id; REQUIRED for correct CV if it exists
    imputation_numeric: Literal["median", "mean", "zero"] = "median"
    imputation_categorical: Literal["mode", "constant"] = "mode"
    outlier_strategy: Literal["winsorize_99", "iqr_safe", "none"] = "winsorize_99"
    scaling_required: bool = False
    woe_transformation: bool = False
    correlation_threshold: float = Field(0.85, ge=0.80, le=0.99)
    psi_threshold: float = Field(0.25, ge=0.05, le=1.0)
    imbalance_tier: Tier


class FeatureSelection(BaseModel):
    max_missing_pct: float = Field(0.70, ge=0.0, le=1.0)
    min_iv: float = Field(0.02, ge=0.0)
    corr_threshold: float = Field(0.90, ge=0.80, le=0.99)
    top_k_by_importance: int = Field(200, ge=1)
    importance_method: Literal["null_importance", "gbm_gain", "mutual_info"] = "null_importance"
    sample_rows_for_selection: int = Field(50_000, ge=1_000)


class SearchSpace(BaseModel):
    """What the iteration/HPO loop is allowed to vary."""

    algorithms: list[Algo] = Field(min_length=1, max_length=5)
    n_trials: int = Field(20, ge=1, le=200)
    tuner: Literal["optuna", "random", "none"] = "optuna"
    early_stop_no_improve: int = Field(8, ge=1)  # stop after K trials w/o OOT improvement
    # Research-proposed per-algorithm HPO ranges, e.g.
    #   {"LightGBM": {"learning_rate": [0.01, 0.2], "num_leaves": [15, 255]}}
    # Used by training if valid; otherwise the hardcoded default ranges apply.
    hpo_overrides: dict | None = None


class RunConfig(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    business_objective: str
    model_type: ModelType | None = None  # optional hint; None => legacy text-matching behaviour
    target_col: str
    date_col: str
    primary_metric: PrimaryMetric
    windows: Windows
    data_strategy: DataStrategy
    feature_selection: FeatureSelection = Field(default_factory=lambda: FeatureSelection())
    search: SearchSpace
    random_seed: int = 42
    rationale: str = ""

    @model_validator(mode="after")
    def _algo_strategy_consistency(self) -> "RunConfig":
        # Scaling is pointless for a tree-only shortlist — auto-correct rather than crash,
        # so an imperfect LLM design can't poison the run.
        if self.data_strategy.scaling_required and set(self.search.algorithms) <= TREE_ALGOS:
            self.data_strategy.scaling_required = False
        return self
