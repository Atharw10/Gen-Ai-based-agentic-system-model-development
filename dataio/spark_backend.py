"""Spark implementation of DataBackend — stub. Implement when CDP integration lands (Phase 6).

The intent: Spark does the heavy data pull from the feature store, profiling, snapshot
splitting, and `sample()`-to-pandas. Modelling stays single-node on the sampled/selected
columns. Most of `pipeline/` never learns which backend produced its frames.
"""
from __future__ import annotations

from dataio.backend import DataBackend


class SparkBackend(DataBackend):  # pragma: no cover - not implemented in the POC
    def __init__(self, spark=None):
        self.spark = spark

    def load(self, source: str):
        raise NotImplementedError("SparkBackend.load — implement against CDP/feature store")

    def sample(self, df, n: int, seed: int):
        raise NotImplementedError("SparkBackend.sample — df.sample(...).toPandas()")

    def split_by_snapshot(self, df, date_col, train_snaps, oot_snaps):
        raise NotImplementedError("SparkBackend.split_by_snapshot — filter by snapshot month")
