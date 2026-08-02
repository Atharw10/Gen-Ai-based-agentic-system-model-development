"""Synthetic banking panel-data generator for tests and demos.

Produces customer-month snapshots with:
  - customer_id (group), snapshot_date (N monthly snapshots)
  - correlated numeric features, some with injected temporal drift (for PSI tests)
  - a configurable positive rate (default 5%)
  - a LEAKAGE-TRAP column (`leak_target_peek`) that is the target with noise -> very high IV
  - high-missing and constant columns (hygiene-filter targets)

Two sizes via `make_panel(...)`: tiny fixtures for unit tests, big frames for scale tests.
The signal is intentionally linear+interaction so a real model beats base rate, but not trivially.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(
    n_customers: int = 2000,
    n_snapshots: int = 6,
    n_informative: int = 8,
    n_noise: int = 30,
    pos_rate: float = 0.05,
    drift_features: int = 3,
    seed: int = 0,
    add_leak: bool = True,
) -> pd.DataFrame:
    """Generate a customer-month panel DataFrame.

    Total rows ~= n_customers * n_snapshots. Each customer appears in every snapshot, so
    a correct CV MUST group by customer_id (random CV would leak).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    snapshots = [start + pd.DateOffset(months=i) for i in range(n_snapshots)]

    # Per-customer latent propensity (stable trait) -> drives the target across months.
    cust_latent = rng.normal(size=n_customers)

    frames = []
    informative_w = rng.normal(size=n_informative)

    for m_idx, snap in enumerate(snapshots):
        n = n_customers
        data: dict[str, np.ndarray] = {
            "customer_id": np.arange(n),
            "snapshot_date": np.repeat(snap, n),
        }

        # Informative features: partly driven by the customer latent + month effect.
        X_inf = np.zeros((n, n_informative))
        for j in range(n_informative):
            drift = (0.15 * m_idx) if j < drift_features else 0.0  # temporal drift on a few
            X_inf[:, j] = 0.6 * cust_latent + drift + rng.normal(scale=0.8, size=n)
            data[f"feat_inf_{j}"] = X_inf[:, j]

        # Pure noise features (selection should drop most of these).
        for j in range(n_noise):
            data[f"feat_noise_{j}"] = rng.normal(size=n)

        # A constant column and a high-missing column (hygiene targets).
        data["const_col"] = np.ones(n)
        hm = rng.normal(size=n)
        mask = rng.random(n) < 0.85  # 85% missing
        hm[mask] = np.nan
        data["high_missing_col"] = hm

        # A high-cardinality string id-like column (should be dropped).
        data["txn_ref"] = np.array([f"REF{rng.integers(0, 10**9)}" for _ in range(n)])

        # Target: logistic on informative signal + latent, calibrated to pos_rate.
        logit = X_inf @ informative_w + 0.8 * cust_latent
        # add a mild interaction
        logit += 0.5 * (X_inf[:, 0] * X_inf[:, 1])
        # calibrate intercept to hit the desired positive rate
        target_logit_offset = np.quantile(logit, 1 - pos_rate)
        prob = 1 / (1 + np.exp(-(logit - target_logit_offset)))
        y = (rng.random(n) < prob).astype(int)
        data["target"] = y

        if add_leak:
            # Leakage trap: target with a little noise -> IV will be enormous; selection/leakage
            # tests assert this is flagged, not silently used.
            leak = y.astype(float).copy()
            flip = rng.random(n) < 0.05
            leak[flip] = 1 - leak[flip]
            data["leak_target_peek"] = leak

        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True)
    # shuffle rows so order doesn't encode anything
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def tiny_fixture(seed: int = 0) -> pd.DataFrame:
    """Small, fast frame for unit tests (~12k rows, ~40 feature cols)."""
    return make_panel(n_customers=2000, n_snapshots=6, seed=seed)


if __name__ == "__main__":
    df = tiny_fixture()
    print(df.shape)
    print("pos_rate:", round(df["target"].mean(), 4))
    print("snapshots:", sorted(df["snapshot_date"].dt.to_period("M").astype(str).unique()))
    print("cols:", len(df.columns))
