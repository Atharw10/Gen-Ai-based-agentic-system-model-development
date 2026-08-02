"""Deterministic end-to-end run from a validated RunConfig. NO LLM in this path.

This is what `test_reproducibility` exercises and what the orchestrator calls after the
design advisor has produced a config. Same (config + data + seed) -> identical result.
"""
from __future__ import annotations

import pandas as pd

from contracts import RunConfig, RunResult
from pipeline import evaluation, persistence, profiling, training
from pipeline.preparation import prepare_data


def run_from_config(
    df: pd.DataFrame,
    config: RunConfig,
    run_dir: str | None = None,
    source_path: str | None = None,
    on_trial=None,
    enable_domain_features: bool = True,
) -> tuple[RunResult, object, object]:
    """Run prepare -> search -> evaluate -> persist. Returns (result, best_model, prepared)."""
    profile = profiling.profile_dataset(
        df, config.target_col, config.date_col, config.data_strategy.group_col, config.business_objective
    )
    prepared = prepare_data(df, config, profile, enable_domain_features=enable_domain_features)

    all_trials, best = training.run_search(prepared, config, on_trial=on_trial)
    return finalize_run(prepared, config, all_trials, best, run_dir, source_path)


def finalize_run(prepared, config, all_trials, best, run_dir=None, source_path=None, alias_map=None):
    """Refit best, evaluate on OOT, build leaderboard, persist. Shared by runner + orchestrator."""
    best_model = training.refit_best(prepared, config, best)

    thresholds = evaluation.tune_thresholds(best_model, prepared.X_oot, prepared.y_oot)
    score = evaluation.pos_score(best_model, prepared.X_oot)
    lift_df = evaluation.lift_table(prepared.y_oot, score)

    leaderboard = [
        {"trial_id": t.trial_id, "algorithm": t.algorithm, "cv_score": round(t.cv_score, 5),
         "params": t.params, "oot_metrics": {k: round(v, 5) for k, v in t.oot_metrics.items()}}
        for t in sorted(all_trials, key=lambda t: t.oot_metrics.get(training._oot_key(config.primary_metric), 0),
                        reverse=True)
    ]
    leaderboard_df = pd.DataFrame(
        [{"trial_id": r["trial_id"], "algorithm": r["algorithm"], "cv_score": r["cv_score"],
          **{f"oot_{k}": v for k, v in r["oot_metrics"].items()}} for r in leaderboard]
    )

    dhash = persistence.data_hash(source_path) if source_path else "no-source-path"
    result = RunResult(
        run_dir=run_dir or "",
        config_path=f"{run_dir}/run_config.json" if run_dir else "",
        data_hash=dhash,
        best_algorithm=best.algorithm,
        best_trial_id=best.trial_id,
        primary_metric=config.primary_metric,
        leaderboard=leaderboard,
        optimal_threshold=thresholds["f1_optimal"],
        business_thresholds=thresholds,
        selection_report=prepared.selection_report,
        split_report=prepared.split_report,
        n_trials_run=len(all_trials),
        column_alias_map=alias_map,
    )

    if run_dir:
        persistence.persist_run(run_dir, config, prepared, best_model, result, leaderboard_df, lift_df,
                                alias_map=alias_map)
        # save validation curves as PNGs into <run_dir>/charts/ (best-effort; never breaks a run)
        try:
            from pipeline import charts
            charts.save_all_charts(run_dir, prepared, best_model, result)
        except Exception:
            pass
    return result, best_model, prepared
