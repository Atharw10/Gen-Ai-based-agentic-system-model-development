"""Stage 7 - persistence. Writes the reproducible artifact bundle + a human model card.

The bundle (config + data_hash + seed + fitted transformers + model) is sufficient to score
new data and to reproduce the run, which is the MRM/audit story.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from contracts import RunConfig, RunResult


def data_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def persist_run(
    run_dir: str,
    config: RunConfig,
    prepared,
    best_model,
    result: RunResult,
    leaderboard_df,
    lift_df,
    alias_map: dict | None = None,
) -> None:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)

    # If columns were aliased, write a full real-name view so a human can read the result; the
    # LLM only ever saw aliases.
    if alias_map:
        from dataio.aliasing import real_selection_report

        real_report = real_selection_report(prepared.selection_report, alias_map.get("reverse", {}))
        with open(root / "selection_report_real.json", "w") as f:
            json.dump(real_report, f, indent=2, default=str)

    (root / "run_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (root / "data_hash.txt").write_text(result.data_hash, encoding="utf-8")
    (root / "run_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    with open(root / "selection_report.json", "w") as f:
        json.dump(prepared.selection_report.model_dump(), f, indent=2, default=str)
    with open(root / "cleaning_artifacts.json", "w") as f:
        json.dump(prepared.cleaning_artifacts.to_dict(), f, indent=2, default=str)

    leaderboard_df.to_csv(root / "leaderboard.csv", index=False)
    if lift_df is not None:
        lift_df.to_csv(root / "lift_table.csv", index=False)

    # deployable bundle: everything needed to score new raw data
    joblib.dump(
        {
            "config": config.model_dump(),
            "cleaning_artifacts": prepared.cleaning_artifacts,
            "preprocessor": prepared.preprocessor,
            "model": best_model,
            "selected_raw_features": prepared.selected_raw_features,
            "feature_names": prepared.feature_names,
            "column_map": prepared.column_map,
            "optimal_threshold": result.optimal_threshold,
            "alias_map": alias_map,  # rename incoming real-name data to aliases before scoring
        },
        root / "model_bundle.joblib",
    )

    _write_model_card(root, config, result, prepared, alias_map)


def _write_model_card(root: Path, config: RunConfig, result: RunResult, prepared, alias_map=None) -> None:
    sr = prepared.split_report
    sel = prepared.selection_report
    reverse = (alias_map or {}).get("reverse", {})

    def real(names):
        return [reverse.get(n, n) for n in names]
    best_row = next((r for r in result.leaderboard if r.get("algorithm") == result.best_algorithm), {})
    lines = [
        f"# Model Card — {config.business_objective}",
        "",
        f"- **Primary metric:** {result.primary_metric}",
        f"- **Best algorithm:** {result.best_algorithm} (trial {result.best_trial_id})",
        f"- **Trials run:** {result.n_trials_run}",
        f"- **Data hash:** `{result.data_hash[:16]}…`",
        f"- **Seed:** {config.random_seed}",
        "",
        "## Data",
        f"- Train rows: {sr.train_rows:,} ({sr.train_range}); pos_ratio={sr.train_pos_ratio}",
        f"- OOT rows:   {sr.oot_rows:,} ({sr.oot_range}); pos_ratio={sr.oot_pos_ratio}",
        f"- Imbalance tier: {config.data_strategy.imbalance_tier}",
        "",
        "## Feature selection",
        f"- In: {sel.n_in} → hygiene: {sel.n_after_hygiene} → univariate: {sel.n_after_univariate}"
        f" → redundancy: {sel.n_after_redundancy} → **selected: {sel.n_selected}**",
        "- Top selected features"
        + (" (real names)" if reverse else "")
        + ": "
        + ", ".join(
            real([r["feature"] for r in sel.iv_table if r["feature"] in set(sel.selected_features)][:10])
        ),
    ]
    if sel.suspicious_high_iv:
        lines.append(
            "- ⚠️ **Leakage suspects (high IV)"
            + (" — real names" if reverse else "")
            + "**: "
            + ", ".join(real(sel.suspicious_high_iv))
        )
    lines += [
        "",
        "## OOT performance (best model)",
        *[f"- {k}: {v:.4f}" for k, v in best_row.get("oot_metrics", {}).items()],
        "",
        "## Thresholds",
        f"- F1-optimal: {result.optimal_threshold:.4f}",
        *[f"- {k}: {v}" for k, v in result.business_thresholds.items()],
        "",
        "## Reproducibility",
        "Re-run `run_config.json` against data matching `data_hash.txt` with the same seed to "
        "reproduce this model byte-for-byte. No LLM is involved in reproduction.",
    ]
    (root / "model_card.md").write_text("\n".join(lines), encoding="utf-8")
