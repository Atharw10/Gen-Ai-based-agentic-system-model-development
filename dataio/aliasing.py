"""Column-name aliasing — privacy shield applied at data input.

Renames every column (including target / date / group) to neutral aliases BEFORE anything
runs, so the LLM advisor plane only ever sees opaque names (col_0001, target, snapshot_date,
entity_id) — never your real schema. The real<->alias mapping is written to its own artifact
file so a human can translate results back; the LLM never receives it.

No LLM import here (data plane). The DataFrame is renamed; values are untouched and never leave.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# neutral role aliases — generic enough to leak no business meaning
TARGET_ALIAS = "target"
DATE_ALIAS = "snapshot_date"
GROUP_ALIAS = "entity_id"


@dataclass
class AliasMap:
    forward: dict[str, str]   # real  -> alias
    reverse: dict[str, str]   # alias -> real
    target: str               # aliased target column
    date: str | None          # aliased date column (or None)
    group: str | None         # aliased group column (or None)

    def to_dict(self) -> dict:
        return {"forward": self.forward, "reverse": self.reverse,
                "target": self.target, "date": self.date, "group": self.group}

    def unalias(self, names) -> list[str]:
        return [self.reverse.get(n, n) for n in names]


def build_alias_map(
    columns: list[str], target_col: str, date_col: str | None = None, group_col: str | None = None
) -> AliasMap:
    """Deterministic alias assignment in column order. Roles get neutral names, features col_NNNN."""
    for role, col in [("target_col", target_col), ("date_col", date_col), ("group_col", group_col)]:
        if col is not None and col not in columns:
            raise ValueError(f"aliasing: {role}={col!r} not found in columns {columns[:8]}...")
    forward: dict[str, str] = {}
    feat_i = 1
    for c in columns:
        if c == target_col:
            forward[c] = TARGET_ALIAS
        elif date_col is not None and c == date_col:
            forward[c] = DATE_ALIAS
        elif group_col is not None and c == group_col:
            forward[c] = GROUP_ALIAS
        else:
            forward[c] = f"col_{feat_i:04d}"
            feat_i += 1
    reverse = {v: k for k, v in forward.items()}
    return AliasMap(
        forward=forward,
        reverse=reverse,
        target=TARGET_ALIAS,
        date=DATE_ALIAS if date_col is not None else None,
        group=GROUP_ALIAS if group_col is not None else None,
    )


def apply_aliases(df: pd.DataFrame, amap: AliasMap) -> pd.DataFrame:
    return df.rename(columns=amap.forward)


def real_names(aliases, reverse_map: dict[str, str]) -> list[str]:
    """Translate a list of alias names back to real names using a reverse map ({alias: real})."""
    return [reverse_map.get(a, a) for a in aliases]


def real_selection_report(selection_report, reverse_map: dict[str, str]) -> dict:
    """Full real-name view of a SelectionReport for human-facing artifacts."""
    sr = selection_report

    def pairs(names):
        return [{"alias": a, "real_name": reverse_map.get(a, a)} for a in names]

    return {
        "n_in": sr.n_in,
        "n_selected": sr.n_selected,
        "selected": pairs(sr.selected_features),
        "suspicious_high_iv": pairs(sr.suspicious_high_iv),
        "dropped_redundant": pairs(sr.dropped_redundant),
        "dropped_low_iv": pairs(sr.dropped_low_iv),
        "dropped_constant": pairs(sr.dropped_constant),
        "dropped_high_missing": pairs(sr.dropped_high_missing),
        "dropped_high_cardinality": pairs(sr.dropped_high_cardinality),
        "iv_table": [
            {"real_name": reverse_map.get(r.get("feature"), r.get("feature")),
             "alias": r.get("feature"), "IV": r.get("IV"), "strength": r.get("strength")}
            for r in sr.iv_table
        ],
    }


def save_alias_map(amap: AliasMap, run_dir: str) -> str:
    """Write the mapping to its own artifact files (JSON + human-readable CSV). Returns json path."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "column_alias_map.json"
    json_path.write_text(json.dumps(amap.to_dict(), indent=2, default=str), encoding="utf-8")

    roles = {amap.target: "target"}
    if amap.date:
        roles[amap.date] = "date"
    if amap.group:
        roles[amap.group] = "group"
    rows = [{"real_name": real, "alias": alias, "role": roles.get(alias, "feature")}
            for real, alias in amap.forward.items()]
    pd.DataFrame(rows).to_csv(root / "column_alias_map.csv", index=False)
    return str(json_path)
