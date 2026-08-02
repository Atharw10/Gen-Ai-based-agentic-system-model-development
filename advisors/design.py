"""Design advisor: turn the research PLAN (+ profile) into a validated RunConfig.

Research now carries the thinking (models, windows, HPO, families); design's job is to assemble
those into a typed, validated RunConfig and choose the primary metric. If the plan is missing or
produces an invalid config, it falls back to deterministic defaults. The result is always a valid
RunConfig — the pipeline never sees a half-baked plan.
"""
from __future__ import annotations

from contracts import (
    DataStrategy,
    FeatureSelection,
    RunConfig,
    SearchSpace,
    Windows,
)
from contracts.config import Algo
from mlkit.metrics import assign_tier

_VALID_ALGOS = set(Algo.__args__)  # type: ignore[attr-defined]
_VALID_METRICS = {"ROC_AUC", "PR_AUC", "F1", "Recall", "Precision", "KS"}
_METRIC_CANON = {m.upper(): m for m in _VALID_METRICS}  # case-insensitive lookup

# tier -> primary metric (banking convention; objective keywords can override)
_TIER_METRIC = {"balanced": "ROC_AUC", "mild": "F1", "moderate": "PR_AUC",
                "severe": "PR_AUC", "extreme": "Recall", "unknown": "PR_AUC"}

# ── design SKILL: refine primary metric + algorithm shortlist (guarded) ───────
_DESIGN_SYSTEM = (
    "You are a senior banking ML model-design lead. Given the data profile and the research plan, "
    "decide the PRIMARY METRIC and finalise the algorithm shortlist. Use imbalance_tier if useful. "
    "Choose the metric that best matches the business risk and the class imbalance. "
    "Return ONLY this JSON: "
    '{"primary_metric": "ROC_AUC|PR_AUC|F1|Recall|Precision|KS", '
    '"algorithms": ["...valid names..."], "reason": "..."}'
)


def _design_skill():
    from advisors.skills import Skill

    return Skill("model_design", _DESIGN_SYSTEM, tool_names=("imbalance_tier",))


def _design_task(profile, tier, model_type, algos, plan, det_metric) -> str:
    rec = (plan.get("models") or {}).get("recommended") or []
    wreason = (plan.get("windows") or {}).get("reason", "")
    return (
        f"Positive rate: {profile.target.pos_ratio} (imbalance tier: {tier}). "
        f"Monthly snapshots: {profile.temporal.n_snapshots}. Model type: {model_type}. "
        f"Research recommended algorithms: {rec}. Research window reasoning: {wreason}. "
        f"Deterministic default metric (fallback if you have no strong reason): {det_metric}. "
        f"Valid algorithm names: {sorted(_VALID_ALGOS)}."
    )

# Explicit model_type (dropdown) -> primary metric. A value of None means "defer to tier/text".
_MODEL_TYPE_METRIC = {
    "fraud": "Recall",
    "churn": "Recall",
    "credit_risk": "KS",
    "logistic": "KS",   # interpretable logistic/scorecard -> KS (classic scorecard metric)
    "propensity": None, "cross_sell": None, "up_sell": None, "generic": None,
}


def _primary_metric(objective: str, tier: str, model_type: str | None = None) -> str:
    # Explicit model_type wins when it pins a metric; otherwise fall back to text + tier policy.
    if model_type and _MODEL_TYPE_METRIC.get(model_type):
        return _MODEL_TYPE_METRIC[model_type]
    o = (objective or "").lower()
    if any(k in o for k in ("fraud", "default", "churn")):
        return "Recall"
    if any(k in o for k in ("credit scoring", "risk scoring")):
        return "KS"
    return _TIER_METRIC.get(tier, "PR_AUC")


def _windows_from(profile, oot_months: int, obs_months: int) -> Windows:
    snaps = profile.temporal.snapshots
    oot_n = min(max(int(oot_months), 1), len(snaps) - 1)
    return Windows(
        observation_window_months=max(1, int(obs_months)),
        performance_window_months=1,
        snapshots_for_training=snaps[:-oot_n],
        oot_snapshots=snaps[-oot_n:],
    )


def default_config_from_profile(profile: ProfileResult, business_objective: str) -> RunConfig:  # noqa: F821
    """Deterministic design straight from the profile (used as the ultimate fallback)."""
    from advisors.research import research as _research

    return design_config(profile, business_objective, research=_research(business_objective, profile))[0]


def design_config(profile, business_objective, cached_llm=None, research=None, max_retries=3,
                  model_type=None):
    """Build a validated RunConfig from the research plan. Returns (config, errors).

    `model_type` (optional dropdown hint) reliably pins the primary metric for known types and is
    stored on the config; when None, the legacy text/tier policy applies (backward compatible).
    """
    tier = assign_tier(profile.target.pos_ratio)
    plan = research or {}
    errors: list[str] = []

    # ---- models (from research; filter to valid; fall back to tier defaults) ----
    rec = (plan.get("models") or {}).get("recommended") or plan.get("recommended_algorithms") or []
    algos = [a for a in rec if a in _VALID_ALGOS]
    if not algos:
        from advisors.research import _TIER_ALGO

        algos = [a for a in _TIER_ALGO.get(tier, _TIER_ALGO["unknown"]) if a in _VALID_ALGOS]
        errors.append("models:fell_back_to_tier_defaults")

    # ---- windows (from research) ----
    w = plan.get("windows") or {}
    obs = w.get("observation_window_months", 3)
    oot = w.get("oot_months", 2 if profile.temporal.n_snapshots >= 6 else 1)

    # ---- HPO overrides (from research; training validates each range) ----
    hpo = (plan.get("hpo") or {}).get("spaces") or None

    metric = _primary_metric(business_objective, tier, model_type)

    # ---- design SKILL: LLM refines metric + algorithm shortlist, GUARDED ----
    # Hard business rules win: when model_type pins a metric (fraud/churn/credit_risk) the LLM may
    # not override it. The LLM only decides the metric for open types (propensity/generic/None).
    # Any LLM value is validated against the allowed sets; invalid/unreasoned -> deterministic stands.
    metric_pinned = bool(model_type and _MODEL_TYPE_METRIC.get(model_type))
    design_source = "deterministic policy"
    if cached_llm is not None:
        try:
            refined = _design_skill().run(
                _design_task(profile, tier, model_type, algos, plan, metric), cached_llm)
            if isinstance(refined, dict) and str(refined.get("reason", "")).strip():
                m = _METRIC_CANON.get(str(refined.get("primary_metric", "")).upper().replace("-", "_"))
                if m and not metric_pinned:
                    metric = m
                    design_source = "Gemini (skill)"
                keep = [a for a in (refined.get("algorithms") or []) if a in _VALID_ALGOS]
                if keep:
                    algos = keep
                    design_source = "Gemini (skill)"
        except Exception as e:
            errors.append(f"design_skill:{e}")

    # model_type == "logistic": user explicitly wants an interpretable logistic/scorecard model, so
    # guarantee LogisticRegression is in the shortlist (kept as champion; others act as challengers).
    # This also auto-enables scaling below (scaling_required = "LogisticRegression" in algos).
    if model_type == "logistic" and "LogisticRegression" not in algos:
        algos = ["LogisticRegression"] + algos

    for attempt in range(1, max_retries + 1):
        try:
            cfg = RunConfig(
                business_objective=business_objective,
                model_type=model_type,
                target_col=profile.target.name,
                date_col=profile.temporal.date_col or "snapshot_date",
                primary_metric=metric,
                windows=_windows_from(profile, oot, obs),
                data_strategy=DataStrategy(
                    group_col=profile.group_col,
                    imbalance_tier=tier if tier != "unknown" else "moderate",
                    scaling_required="LogisticRegression" in algos,
                ),
                feature_selection=FeatureSelection(),
                search=SearchSpace(algorithms=algos, n_trials=20, hpo_overrides=hpo),
                rationale=(plan.get("models") or {}).get("reason", f"Design for tier '{tier}'."),
            )
            return cfg, errors
        except Exception as e:  # shrink/repair on validation error, then retry
            errors.append(f"attempt {attempt}: {e}")
            oot = max(1, oot - 1)
            algos = algos[:3] or ["LightGBM"]
            hpo = None
    # ultimate guard: minimal valid config
    snaps = profile.temporal.snapshots
    cfg = RunConfig(
        business_objective=business_objective, model_type=model_type, target_col=profile.target.name,
        date_col=profile.temporal.date_col or "snapshot_date", primary_metric="PR_AUC",
        windows=Windows(observation_window_months=1, performance_window_months=1,
                        snapshots_for_training=snaps[:-1], oot_snapshots=snaps[-1:]),
        data_strategy=DataStrategy(group_col=profile.group_col, imbalance_tier="moderate"),
        feature_selection=FeatureSelection(),
        search=SearchSpace(algorithms=["LightGBM"], n_trials=20),
        rationale="fallback minimal config",
    )
    errors.append("FALLBACK_MINIMAL")
    return cfg, errors


# import placed at end to avoid a cycle at module load
from contracts import ProfileResult  # noqa: E402
