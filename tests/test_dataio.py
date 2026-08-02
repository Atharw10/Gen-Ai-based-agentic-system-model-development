"""dataio backend: pandas impl loads, samples, and snapshot-splits correctly."""
from __future__ import annotations

from dataio.pandas_backend import PandasBackend
from tests.synth import tiny_fixture


def test_pandas_backend_roundtrip(tmp_path):
    df = tiny_fixture(seed=0)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    be = PandasBackend()
    loaded = be.load(str(p))
    assert loaded.shape[0] == df.shape[0]

    s = be.sample(loaded, 1000, seed=1)
    assert len(s) == 1000

    snaps = sorted(loaded["snapshot_date"].astype(str).str.slice(0, 7).unique())
    tr, oot = be.split_by_snapshot(loaded, "snapshot_date", snaps[:-2], snaps[-2:])
    assert len(tr) > 0 and len(oot) > 0 and len(tr) + len(oot) == len(loaded)
