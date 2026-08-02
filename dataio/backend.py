"""DataBackend interface — contains the future Spark cutover.

The deterministic pipeline operates on pandas frames. Heavy IO/aggregation (loading from CDP,
profiling, snapshot splitting, sampling-to-pandas) goes through a DataBackend so that swapping
in Spark later touches ONE module, not all of mlkit/pipeline. Decision: Spark is a data-access
backend, not a modelling engine (500k x 5k fits single-node after selection).

(Package is named `dataio`, not `io`, to avoid shadowing Python's stdlib `io` module.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataBackend(ABC):
    @abstractmethod
    def load(self, source: str) -> pd.DataFrame:
        """Load a source into a (pandas) frame the pipeline can consume."""

    @abstractmethod
    def sample(self, df, n: int, seed: int) -> pd.DataFrame:
        """Return up to n rows as pandas (for selection/importance on big data)."""

    @abstractmethod
    def split_by_snapshot(self, df, date_col: str, train_snaps: list[str], oot_snaps: list[str]):
        """Return (df_train, df_oot) by YYYY-MM snapshot membership."""
