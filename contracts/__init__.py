"""Typed contracts shared by all planes (orchestration / pipeline / advisors)."""
from .config import (
    Algo,
    DataStrategy,
    FeatureSelection,
    ModelType,
    PrimaryMetric,
    RunConfig,
    SearchSpace,
    Tier,
    TREE_ALGOS,
    Windows,
)
from .results import (
    ProfileResult,
    RunResult,
    SelectionReport,
    SplitReport,
    TargetProfile,
    TemporalProfile,
    TrialResult,
)

__all__ = [
    "Algo",
    "DataStrategy",
    "FeatureSelection",
    "ModelType",
    "PrimaryMetric",
    "RunConfig",
    "SearchSpace",
    "Tier",
    "TREE_ALGOS",
    "Windows",
    "ProfileResult",
    "RunResult",
    "SelectionReport",
    "SplitReport",
    "TargetProfile",
    "TemporalProfile",
    "TrialResult",
]
