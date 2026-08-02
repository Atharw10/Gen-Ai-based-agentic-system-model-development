"""Data-source registry — the seam between "where the data lives" and the pandas pipeline.

A DataSource resolves a user's dropdown choice (+ a few follow-up fields) into a single pandas
DataFrame that the rest of the pipeline consumes UNCHANGED. The deterministic plane never knows or
cares which source produced the frame.

Design rules:
  * No LLM imports (deterministic plane — the layering test guards this).
  * CML-safe imports: heavy/cluster libs (pyspark, impyla) are imported LAZILY inside .load(), so
    importing this module never fails on a machine without those libraries. Only the source you
    actually pick needs its library installed.
  * "Optimized query" lives here: a source pushes column-pruning / snapshot(partition) filtering /
    sampling DOWN to Spark/Impala/SQL and returns only the reduced result to pandas. (Phase 2 for
    the CDP sources; the interface already accepts those args so the pipeline never changes.)

Phase 1 status:
  * FileSource          — REAL (wraps the existing CSV/Parquet read).
  * Hive/Impala/FeatureStore/DataLake/CustomSQL — STUBS that raise a clear NotWiredError until the
    CDP/CML connection is provided. They appear in the menu so the full shape is visible now.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


class NotWiredError(NotImplementedError):
    """Raised by a source that is registered (visible in the menu) but not yet connected."""


@dataclass
class SourceSpec:
    """The small set of follow-up fields a source may need (only the relevant ones are used)."""
    path: Optional[str] = None              # FileSource / DataLake
    table: Optional[str] = None             # Hive / Impala / FeatureStore  (e.g. "risk_db.cust_features")
    snapshot_col: Optional[str] = None      # partition/date column -> enables partition pruning
    snapshot_range: Optional[tuple] = None  # (start, end) inclusive, e.g. ("2025-01", "2025-06")
    sql: Optional[str] = None               # CustomSQL / Impala
    columns: Optional[list] = None          # column pruning (None = all)
    sample_fraction: Optional[float] = None # negative/row sampling fraction (None = no sampling)
    options: dict = field(default_factory=dict)  # backend-specific extras (connection, etc.)


class DataSource:
    """Common interface. Every source returns ONE pandas DataFrame the pipeline can use as-is."""
    key: str = "base"
    label: str = "Data source"
    needs: tuple = ()  # which SourceSpec fields to prompt for in the UI

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> str:
        return self.label


class FileSource(DataSource):
    key = "file"
    label = "Local file (CSV/Parquet)"
    needs = ("path",)

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        if not spec.path:
            raise ValueError("FileSource needs a file path (spec.path).")
        ext = Path(spec.path).suffix.lower()
        df = pd.read_parquet(spec.path) if ext in (".parquet", ".pq") \
            else pd.read_csv(spec.path, low_memory=False)
        # column pruning works for files too (optional)
        if spec.columns:
            keep = [c for c in spec.columns if c in df.columns]
            if keep:
                df = df[keep]
        return df


# --------------------------------------------------------------------------------------------
# CDP / CML sources — STUBS for Phase 1. Each documents EXACTLY how Phase 2 will fetch + reduce.
# Heavy libraries are imported lazily inside load() (never at module import) so this file loads
# fine on any machine, including ones without pyspark/impyla.
# --------------------------------------------------------------------------------------------
def _get_spark(options: dict):
    """Return a SparkSession. In CML this is usually pre-wired via a Data Connection; otherwise we
    fall back to SparkSession.builder. Imported lazily so non-Spark machines are unaffected."""
    sess = options.get("spark")
    if sess is not None:
        return sess
    from pyspark.sql import SparkSession  # lazy: only when a Spark source is actually used
    return SparkSession.builder.appName("amd_v2").getOrCreate()


def _push_down_sql(spec: SourceSpec, default_cols: str = "*") -> str:
    """Build a column/partition/sample-pruned SQL string from a SourceSpec (shared by Hive/Impala)."""
    cols = ", ".join(spec.columns) if spec.columns else default_cols
    q = f"SELECT {cols} FROM {spec.table}"
    clauses = []
    if spec.snapshot_col and spec.snapshot_range:
        lo, hi = spec.snapshot_range
        clauses.append(f"{spec.snapshot_col} BETWEEN '{lo}' AND '{hi}'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    if spec.sample_fraction:
        q += f" TABLESAMPLE ({int(spec.sample_fraction * 100)} PERCENT)"
    return q


class HiveSource(DataSource):
    key = "hive"
    label = "CDP Hive table (Spark)"
    needs = ("table", "snapshot_col", "snapshot_range")

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        if not spec.table:
            raise ValueError("HiveSource needs spec.table, e.g. 'risk_db.cust_features'.")
        # ---- Phase 2 (wire when a CML SparkSession is available) ----
        #   spark = _get_spark(spec.options)
        #   sdf = spark.sql(_push_down_sql(spec))   # column + partition + sample pushdown on cluster
        #   return sdf.toPandas()                    # collect ONLY the reduced result into pandas
        raise NotWiredError(
            "CDP Hive source not wired yet. Phase 2: pass a CML SparkSession via "
            "spec.options['spark']; it will run an optimized spark.sql() (column + snapshot-"
            "partition pruning + sampling) and return the reduced frame as pandas. "
            f"Planned query: {_push_down_sql(spec)}"
        )


class ImpalaSource(DataSource):
    key = "impala"
    label = "CDP Impala query (SQL)"
    needs = ("sql",)

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        query = spec.sql or (_push_down_sql(spec) if spec.table else None)
        if not query:
            raise ValueError("ImpalaSource needs spec.sql (or spec.table to build one).")
        # ---- Phase 2 ----
        #   from impala.dbapi import connect            # lazy import
        #   conn = connect(**spec.options["impala"])    # host/port/auth from CML
        #   return pd.read_sql(query, conn)
        raise NotWiredError(
            "CDP Impala source not wired yet. Phase 2: provide connection details in "
            f"spec.options['impala']; query will run via impyla. Planned query: {query}"
        )


class FeatureStoreSource(HiveSource):
    key = "feature_store"
    label = "Feature Store table"
    needs = ("table", "snapshot_col", "snapshot_range")
    # Behaves like HiveSource; the menu pre-labels it as the curated feature table.


class DataLakeSource(DataSource):
    key = "datalake"
    label = "Data Lake file (Parquet on S3/ADLS)"
    needs = ("path",)

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        if not spec.path:
            raise ValueError("DataLakeSource needs spec.path, e.g. 's3a://bucket/features/'.")
        # ---- Phase 2 ----
        #   spark = _get_spark(spec.options)
        #   sdf = spark.read.parquet(spec.path)
        #   if spec.columns: sdf = sdf.select(*spec.columns)
        #   return sdf.toPandas()
        raise NotWiredError(
            "Data Lake source not wired yet. Phase 2: reads Parquet from object storage via Spark "
            f"and collects the reduced frame. Planned path: {spec.path}"
        )


class CustomSQLSource(DataSource):
    key = "custom_sql"
    label = "Custom SQL query"
    needs = ("sql",)

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        if not spec.sql:
            raise ValueError("CustomSQLSource needs spec.sql.")
        # ---- Phase 2 (Spark path; an Impala/JDBC path can be added the same way) ----
        #   spark = _get_spark(spec.options)
        #   return spark.sql(spec.sql).toPandas()
        raise NotWiredError(
            "Custom SQL source not wired yet. Phase 2: runs your SQL via Spark/Impala and returns "
            "the reduced frame as pandas."
        )


# Registry in display order (drives the dropdown). Add new sources here only.
_REGISTRY: list[DataSource] = [
    HiveSource(),
    ImpalaSource(),
    FeatureStoreSource(),
    DataLakeSource(),
    FileSource(),
    CustomSQLSource(),
]
SOURCES: dict[str, DataSource] = {s.key: s for s in _REGISTRY}


def list_sources() -> list[DataSource]:
    """Ordered list for building the dropdown menu."""
    return list(_REGISTRY)


def get_source(key: str) -> DataSource:
    if key not in SOURCES:
        raise KeyError(f"unknown data source '{key}'. Available: {list(SOURCES)}")
    return SOURCES[key]


def load_dataframe(key: str, spec: SourceSpec) -> pd.DataFrame:
    """Convenience: resolve a source key + spec to a pandas DataFrame for the pipeline."""
    return get_source(key).load(spec)
