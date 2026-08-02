"""Stage 4a - cleaning. Learns impute values + outlier bounds on TRAIN, applies to BOTH.

Returns the fitted artifacts so they can be persisted and re-applied at scoring time.
"""
from __future__ import annotations

import pandas as pd

from contracts import DataStrategy
from mlkit import cleaning_ops


class CleaningArtifacts:
    """Fitted cleaning state, reusable at scoring time."""

    def __init__(self, impute_values: dict, outlier_bounds: dict, dropped_cols: list[str]):
        self.impute_values = impute_values
        self.outlier_bounds = outlier_bounds
        self.dropped_cols = dropped_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=[c for c in self.dropped_cols if c in df.columns])
        out = cleaning_ops.apply_outlier_bounds(out, self.outlier_bounds)
        out = cleaning_ops.apply_impute_values(out, self.impute_values)
        return out

    def to_dict(self) -> dict:
        return {
            "impute_values": self.impute_values,
            "outlier_bounds": {k: list(v) for k, v in self.outlier_bounds.items()},
            "dropped_cols": self.dropped_cols,
        }


def fit_clean(
    df_train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    strategy: DataStrategy,
    max_missing_pct: float = 0.70,
) -> CleaningArtifacts:
    """Fit cleaning on train only."""
    miss = df_train[numeric_cols + categorical_cols].isna().mean()
    dropped = miss[miss > max_missing_pct].index.tolist()
    num = [c for c in numeric_cols if c not in dropped]
    cat = [c for c in categorical_cols if c not in dropped]

    bounds = cleaning_ops.fit_outlier_bounds(df_train, num, strategy.outlier_strategy)
    fills = cleaning_ops.fit_impute_values(
        df_train, num, cat, strategy.imputation_numeric, strategy.imputation_categorical
    )
    return CleaningArtifacts(impute_values=fills, outlier_bounds=bounds, dropped_cols=dropped)
