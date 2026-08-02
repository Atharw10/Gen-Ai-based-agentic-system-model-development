"""Single-node pandas implementation of DataBackend (today's default)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataio.backend import DataBackend
from pipeline.splitting import temporal_split


class PandasBackend(DataBackend):
    def load(self, source: str) -> pd.DataFrame:
        ext = Path(source).suffix.lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(source)
        return pd.read_csv(source, low_memory=False)

    def sample(self, df, n: int, seed: int) -> pd.DataFrame:
        return df.sample(n, random_state=seed) if len(df) > n else df

    def split_by_snapshot(self, df, date_col, train_snaps, oot_snaps):
        df_train, df_oot, _ = temporal_split(df, date_col, train_snaps, oot_snaps)
        return df_train, df_oot
