"""Stage 4b (optional) - domain feature engineering for banking propensity.

Creates a handful of derived ratio/velocity features when their source columns resolve.
This is a BOLT-ON, not the core of feature handling (selection funnel is the core).
Deterministic + LLM-free: the concept->column map is either supplied explicitly (e.g. by an
advisor, validated upstream) or inferred by regex here. No LLM import in this plane.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# concept -> regex hints (used only for the no-map fallback)
FEATURE_CONCEPTS: dict[str, list[str]] = {
    "credit_3m": [r"(credit|cr|inflow).*3.?m", r"3.?m.*(credit|cr|inflow)"],
    "credit_6m": [r"(credit|cr|inflow).*6.?m", r"6.?m.*(credit|cr|inflow)"],
    "debit_3m": [r"(debit|dr|outflow|spend).*3.?m", r"3.?m.*(debit|dr|outflow|spend)"],
    "debit_6m": [r"(debit|dr|outflow|spend).*6.?m", r"6.?m.*(debit|dr|outflow|spend)"],
    "txn_count_3m": [r"(txn|trans).*(count|cnt|num).*3.?m"],
    "total_amount_3m": [r"(total|sum).*(amt|amount).*3.?m"],
    "bounce_count_3m": [r"bounc.*3.?m", r"3.?m.*bounc"],
    "mandate_pay_amount_3m": [r"mandate.*3.?m", r"standing.*instr.*3.?m"],
    "mcc_dining_3m": [r"(dining|restaurant|food).*3.?m"],
    "mcc_travel_3m": [r"(travel|hotel|airline).*3.?m"],
    "mcc_ecommerce_3m": [r"(ecom|ecommerce|online).*3.?m"],
}


def regex_column_map(df: pd.DataFrame, name_lookup: dict[str, str] | None = None) -> dict[str, str | None]:
    """Match feature concepts to columns by regex.

    `name_lookup` (alias -> real name) lets matching run on REAL names while the returned column
    refs stay as the actual (possibly aliased) df columns — so it still works after aliasing.
    """
    mapping: dict[str, str | None] = {}
    taken: set[str] = set()
    cols = df.columns.tolist()
    for concept, pats in FEATURE_CONCEPTS.items():
        found = None
        for col in cols:
            if col in taken:
                continue
            text = (name_lookup or {}).get(col, col)  # match on the real name when available
            if any(re.search(p, text, re.IGNORECASE) for p in pats):
                found = col
                break
        mapping[concept] = found
        if found:
            taken.add(found)
    return mapping


def engineer(
    df: pd.DataFrame,
    column_map: dict[str, str | None] | None = None,
    name_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str | None]]:
    """Add derived features in-place on a copy. Returns (df, created_names, resolved_map)."""
    cmap = column_map if column_map is not None else regex_column_map(df, name_lookup=name_lookup)
    # validate any supplied map against actual columns
    cmap = {k: (v if (v in df.columns) else None) for k, v in cmap.items()}
    out = df.copy()
    created: list[str] = []

    def col(c):  # concept -> column or None
        return cmap.get(c)

    def has(*cs):
        return all(col(c) is not None for c in cs)

    def ratio(name, num, den):
        out[name] = out[col(num)] / out[col(den)].replace(0, np.nan)
        created.append(name)

    if has("credit_3m", "credit_6m"):
        ratio("credit_velocity_3m_6m", "credit_3m", "credit_6m")
    if has("debit_3m", "debit_6m"):
        ratio("debit_velocity_3m_6m", "debit_3m", "debit_6m")
    if has("credit_3m", "debit_3m"):
        ratio("credit_debit_ratio_3m", "credit_3m", "debit_3m")
    if has("total_amount_3m", "txn_count_3m"):
        ratio("avg_txn_size_3m", "total_amount_3m", "txn_count_3m")
    if has("mandate_pay_amount_3m", "debit_3m"):
        ratio("mandate_share_3m", "mandate_pay_amount_3m", "debit_3m")
    if has("bounce_count_3m", "txn_count_3m"):
        ratio("bounce_rate_3m", "bounce_count_3m", "txn_count_3m")
    if has("mcc_ecommerce_3m", "total_amount_3m"):
        ratio("ecommerce_share_3m", "mcc_ecommerce_3m", "total_amount_3m")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, created, cmap
