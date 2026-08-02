"""Stage 5 - training + hyperparameter search (deterministic, group/time-aware CV).

Model registry is robust to optional boosters: if lightgbm/xgboost aren't installed, those
algorithms fall back to sklearn's libomp-free HistGradientBoosting so the pipeline runs anywhere.
The Optuna study is seeded, so a given RunConfig reproduces the same trials.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone

from contracts import RunConfig, TrialResult
from mlkit import cv as cvmod
from pipeline.evaluation import oot_metrics, primary_scorer


# ---------------------------------------------------------------------------
# Model registry — builds an estimator for (algo, params), with safe fallbacks
# ---------------------------------------------------------------------------
def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def build_estimator(algo: str, params: dict, scale_pos_weight: float, use_class_weight: bool):
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    cw = "balanced" if use_class_weight else None

    if algo == "LogisticRegression":
        return LogisticRegression(max_iter=2000, class_weight=cw, n_jobs=-1, **params)
    if algo == "RandomForest":
        return RandomForestClassifier(n_jobs=-1, random_state=42, class_weight=cw, **params)
    if algo == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, **params)
    if algo == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42, **params)
    if algo == "LightGBM":
        if _has("lightgbm"):
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_jobs=-1, random_state=42, class_weight=cw, verbose=-1,
                scale_pos_weight=scale_pos_weight if not use_class_weight else None, **params,
            )
        return HistGradientBoostingClassifier(random_state=42, **_hgb_from(params))
    if algo == "XGBoost":
        if _has("xgboost"):
            import xgboost as xgb

            return xgb.XGBClassifier(
                tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42,
                scale_pos_weight=scale_pos_weight if use_class_weight else 1.0, **params,
            )
        return HistGradientBoostingClassifier(random_state=42, **_hgb_from(params))
    raise ValueError(f"unknown algorithm: {algo}")


def _hgb_from(params: dict) -> dict:
    """Translate boosting params to HistGradientBoosting when falling back."""
    out = {}
    if "n_estimators" in params:
        out["max_iter"] = params["n_estimators"]
    if "learning_rate" in params:
        out["learning_rate"] = params["learning_rate"]
    if "max_depth" in params and params["max_depth"]:
        out["max_depth"] = params["max_depth"]
    return out


# ---------------------------------------------------------------------------
# Per-algorithm search spaces (Optuna)
# ---------------------------------------------------------------------------
def _valid_range(o):
    """A research override is a usable [lo, hi] pair of numbers with lo < hi."""
    return isinstance(o, (list, tuple)) and len(o) == 2 and \
        all(isinstance(x, (int, float)) for x in o) and o[0] < o[1]


def _rng(trial, name, lo, hi, ov, kind="float", log=False, step=None):
    """Suggest a value, using a research override range if it's valid, else the default range."""
    o = (ov or {}).get(name)
    if _valid_range(o):
        lo, hi = o
    if kind == "int":
        return trial.suggest_int(name, int(lo), int(hi), step=step or 1)
    return trial.suggest_float(name, float(lo), float(hi), log=log)


def suggest_params(trial, algo: str, hpo_overrides: dict | None = None) -> dict:
    """Default HPO ranges, overridden per-parameter by a validated research plan if present.

    hpo_overrides is keyed by algorithm, e.g. {"LightGBM": {"learning_rate": [0.01, 0.2]}}.
    Any invalid/absent range silently falls back to the hardcoded default.
    """
    ov = (hpo_overrides or {}).get(algo, {})
    if algo == "LogisticRegression":
        return {"C": _rng(trial, "C", 1e-3, 10.0, ov, "float", log=True)}
    if algo == "RandomForest":
        return {
            "n_estimators": _rng(trial, "n_estimators", 100, 500, ov, "int", step=100),
            "max_depth": _rng(trial, "max_depth", 4, 16, ov, "int"),
            "min_samples_leaf": _rng(trial, "min_samples_leaf", 1, 50, ov, "int"),
        }
    if algo in ("LightGBM", "XGBoost"):
        return {
            "n_estimators": _rng(trial, "n_estimators", 200, 600, ov, "int", step=100),
            "learning_rate": _rng(trial, "learning_rate", 0.02, 0.2, ov, "float", log=True),
            "max_depth": _rng(trial, "max_depth", 3, 10, ov, "int"),
        }
    if algo in ("GradientBoosting", "HistGradientBoosting"):
        return {
            "learning_rate": _rng(trial, "learning_rate", 0.02, 0.2, ov, "float", log=True),
            "max_depth": _rng(trial, "max_depth", 2, 8, ov, "int"),
        }
    return {}


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------
def run_search(prepared, config: RunConfig, on_trial=None) -> tuple[list[TrialResult], TrialResult]:
    """Run the HPO loop across the configured algorithms. Returns (all_trials, best_trial).

    `on_trial(TrialResult)` is an optional callback (used by the orchestrator/LLM strategist to
    observe progress and steer). Ranking is by OOT primary metric — the only honest signal.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, X_oot, y_oot = prepared.X_train, prepared.y_train, prepared.X_oot, prepared.y_oot
    groups = prepared.groups_train
    primary = config.primary_metric
    tier = config.data_strategy.imbalance_tier
    use_cw = tier != "balanced"
    pos = max(int((y_train == 1).sum()), 1)
    spw = float((y_train == 0).sum()) / pos

    n_splits = 3 if tier in ("severe", "extreme") else 5
    splits = cvmod.make_cv(y_train, groups=groups, n_splits=n_splits, seed=config.random_seed)
    scorer = primary_scorer(primary)

    all_trials: list[TrialResult] = []
    counter = {"i": 0}

    hpo_overrides = config.search.hpo_overrides
    def objective(trial, algo):
        params = suggest_params(trial, algo, hpo_overrides)
        est = build_estimator(algo, params, spw, use_cw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_mean, cv_std = cvmod.cv_score(est, X_train, y_train, splits, scorer)
            fitted = clone(est).fit(X_train, y_train)
            ootm = oot_metrics(fitted, X_oot, y_oot)
        tid = counter["i"]
        counter["i"] += 1
        tr = TrialResult(
            trial_id=tid, algorithm=algo, params=params, cv_score=cv_mean, cv_std=cv_std,
            oot_metrics=ootm,
        )
        all_trials.append(tr)
        if on_trial:
            on_trial(tr)
        # Optuna maximises CV score; OOT is the final ranking key (kept on the trial object).
        return cv_mean

    # Split the trial budget across algorithms. Distribute the remainder so the TOTAL number of
    # trials equals config.search.n_trials exactly (integer division alone drops the remainder —
    # e.g. 20 // 6 = 3 would run only 18). The first `rem` algorithms each get one extra trial.
    algos = config.search.algorithms
    base, rem = divmod(config.search.n_trials, len(algos))
    per_algo_counts = [max(1, base + (1 if i < rem else 0)) for i in range(len(algos))]
    for algo, n_algo in zip(algos, per_algo_counts):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.random_seed)
        )
        study.optimize(lambda t, a=algo: objective(t, a), n_trials=n_algo, show_progress_bar=False)

    key = _oot_key(primary)
    best = max(all_trials, key=lambda t: t.oot_metrics.get(key, float("-inf")))
    return all_trials, best


def _oot_key(primary: str) -> str:
    return {"PR_AUC": "pr_auc", "ROC_AUC": "roc_auc", "KS": "ks", "F1": "f1",
            "Recall": "recall", "Precision": "precision"}.get(primary, "pr_auc")


def refit_best(prepared, config: RunConfig, best: TrialResult):
    """Refit the winning estimator on full train for persistence/scoring."""
    use_cw = config.data_strategy.imbalance_tier != "balanced"
    pos = max(int((prepared.y_train == 1).sum()), 1)
    spw = float((prepared.y_train == 0).sum()) / pos
    est = build_estimator(best.algorithm, best.params, spw, use_cw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(prepared.X_train, prepared.y_train)
    return est
