"""Stage 7 - chart generation. Saves model-validation curves as PNGs into the run artifacts.

Deterministic plane (no LLM). Uses matplotlib's object-oriented `Figure` API directly (NOT pyplot),
so it never touches the global backend and never tries to open a window — safe headless / on CML and
it won't interfere with a notebook's inline charts.

Large-data-safe:
  * Probability-based curves (ROC/PR/KS/gains/calibration) run on the OOT prediction vector, which is
    already the reduced in-memory modelling frame; we additionally cap to `MAX_POINTS` rows so cost is
    bounded no matter how big OOT is. The curves are statistical aggregates, so a large sample is
    representative.
  * Per-feature charts (PSI, WoE) sample rows to `SAMPLE_ROWS`.

Every chart is best-effort: a failure in one chart is logged to charts/_errors.txt and skipped — it
never breaks the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_POINTS = 200_000    # cap rows for probability-curve plotting
SAMPLE_ROWS = 50_000    # cap rows for per-feature PSI/WoE

_PRIMARY_OOT_KEY = {"PR_AUC": "pr_auc", "ROC_AUC": "roc_auc", "KS": "ks", "F1": "f1",
                    "Recall": "recall", "Precision": "precision"}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _new_fig(w=7.5, h=4.0):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h))
    fig.set_facecolor("white")
    return fig


def _save(fig, charts_dir: Path, name: str, caption: str | None = None) -> str:
    # one-line plain-English explanation printed under the chart so anyone understands it
    if caption:
        fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8.5,
                 style="italic", color="#555555", wrap=True)
    path = charts_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return str(path)


def _subsample(y, score, n=MAX_POINTS, seed=42):
    y = np.asarray(y); score = np.asarray(score)
    if len(y) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=n, replace=False)
        return y[idx], score[idx]
    return y, score


def _real(name, reverse):
    return (reverse or {}).get(name, name)


# --------------------------------------------------------------------------------------------
# probability-based curves (OOT)
# --------------------------------------------------------------------------------------------
def chart_gains_lift(y, score, charts_dir, lift_df=None):
    from pipeline.evaluation import lift_table
    if lift_df is None:
        lift_df = lift_table(y, score, n_bins=10)
    deciles = lift_df["decile"].tolist()
    cum = (lift_df["cum_capture"] * 100).tolist()
    lift = lift_df["lift"].tolist()
    x = list(range(1, len(deciles) + 1))

    fig = _new_fig(9, 3.8)
    ax1, ax2 = fig.subplots(1, 2)
    # cumulative gains
    ax1.plot([0] + [d * 10 for d in x], [0] + cum, marker="o", color="#1565C0", label="model")
    ax1.plot([0, 100], [0, 100], ls="--", color="#9E9E9E", label="random")
    ax1.set_xlabel("% of population (top scores first)"); ax1.set_ylabel("% positives captured")
    ax1.set_title("Cumulative Gains"); ax1.legend(fontsize=8)
    # lift per decile
    bars = ax2.bar(x, lift, color="#2E7D32")
    ax2.axhline(1.0, ls="--", color="#9E9E9E")
    ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax2.set_xlabel("Decile (1 = highest score)"); ax2.set_ylabel("Lift vs base rate")
    ax2.set_title("Lift by Decile")
    fig.suptitle("Gains / Lift (OOT)", fontweight="bold")
    return _save(fig, charts_dir, "gains_lift.png",
                 "How many of the actual positives you capture by targeting the top-scored customers. "
                 "Steeper/higher than the diagonal = better targeting; e.g. top 20% catching most positives.")


def chart_roc_pr(y, score, charts_dir):
    from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                                  roc_auc_score, roc_curve)
    ys, ss = _subsample(y, score)
    fpr, tpr, _ = roc_curve(ys, ss)
    roc_auc = roc_auc_score(ys, ss)
    prec, rec, _ = precision_recall_curve(ys, ss)
    ap = average_precision_score(ys, ss)
    base = float(np.mean(ys))

    fig = _new_fig(9, 3.8)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(fpr, tpr, color="#1565C0", label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="#9E9E9E")
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve"); ax1.legend(fontsize=8)
    ax2.plot(rec, prec, color="#6A1B9A", label=f"PR (AP={ap:.3f})")
    ax2.axhline(base, ls="--", color="#9E9E9E", label=f"base rate={base:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curve"); ax2.legend(fontsize=8)
    fig.suptitle("ROC & Precision-Recall (OOT)", fontweight="bold")
    return _save(fig, charts_dir, "roc_pr.png",
                 "ROC = how well the model ranks positives above negatives (AUC, 1.0 is perfect). "
                 "PR = precision vs recall trade-off — the honest view when positives are rare.")


def chart_ks(y, score, charts_dir):
    ys, ss = _subsample(y, score)
    order = np.argsort(ss)
    ys_sorted = ys[order]
    pos_total = max(ys_sorted.sum(), 1)
    neg_total = max((1 - ys_sorted).sum(), 1)
    cum_pos = np.cumsum(ys_sorted) / pos_total
    cum_neg = np.cumsum(1 - ys_sorted) / neg_total
    gap = np.abs(cum_pos - cum_neg)
    ks_idx = int(np.argmax(gap)); ks = float(gap[ks_idx])
    xs = np.linspace(0, 1, len(ys_sorted))

    fig = _new_fig(7.5, 4.0)
    ax = fig.subplots()
    ax.plot(xs, cum_neg, color="#1565C0", label="cum % negatives (y=0)")
    ax.plot(xs, cum_pos, color="#E53935", label="cum % positives (y=1)")
    ax.vlines(xs[ks_idx], cum_pos[ks_idx], cum_neg[ks_idx], color="#2E7D32", lw=2,
              label=f"KS = {ks:.3f}")
    ax.set_xlabel("Population sorted by score (low → high)"); ax.set_ylabel("Cumulative proportion")
    ax.set_title("KS Curve (OOT)"); ax.legend(fontsize=8)
    return _save(fig, charts_dir, "ks_curve.png",
                 "The largest gap between the score distributions of positives and negatives. "
                 "Higher KS = the model separates the two classes more cleanly.")


def chart_calibration(y, score, charts_dir, n_bins=10):
    ys, ss = _subsample(y, score)
    df = pd.DataFrame({"y": ys, "p": ss})
    try:
        df["b"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    except Exception:
        df["b"] = pd.cut(df["p"], min(5, df["p"].nunique()))
    g = df.groupby("b", observed=True).agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"))

    fig = _new_fig(7.5, 4.0)
    ax = fig.subplots()
    ax.plot(g["mean_pred"], g["frac_pos"], marker="o", color="#1565C0", label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="#9E9E9E", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration (Reliability) Curve — OOT"); ax.legend(fontsize=8)
    return _save(fig, charts_dir, "calibration.png",
                 "Are the predicted probabilities trustworthy? On the diagonal means a predicted 30% "
                 "really converts ~30% of the time; below the line = over-confident.")


# --------------------------------------------------------------------------------------------
# feature-level charts
# --------------------------------------------------------------------------------------------
def chart_csi(X_train, X_oot, feature_names, charts_dir, reverse=None, top=20):
    """CSI = Characteristic Stability Index: per-FEATURE distribution shift train -> OOT.
    (Same formula as PSI, but applied to each input characteristic — the standard banking term.)"""
    from mlkit.stats import compute_psi
    n = min(len(X_train), SAMPLE_ROWS)
    m = min(len(X_oot), SAMPLE_ROWS)
    rng = np.random.default_rng(42)
    tr_idx = rng.choice(len(X_train), n, replace=False) if len(X_train) > n else slice(None)
    ot_idx = rng.choice(len(X_oot), m, replace=False) if len(X_oot) > m else slice(None)
    Xt, Xo = X_train[tr_idx], X_oot[ot_idx]

    rows = []
    for j, name in enumerate(feature_names):
        try:
            csi = compute_psi(pd.Series(Xt[:, j]), pd.Series(Xo[:, j]))
        except Exception:
            csi = float("nan")
        rows.append((_real(name, reverse), csi))
    csi_df = (pd.DataFrame(rows, columns=["feature", "CSI"]).dropna()
              .sort_values("CSI", ascending=False).head(top))
    if csi_df.empty:
        return None
    csi_df.to_csv(charts_dir / "csi_per_feature.csv", index=False)

    colors = ["#E53935" if v > 0.25 else "#FB8C00" if v > 0.10 else "#2E7D32"
              for v in csi_df["CSI"]]
    fig = _new_fig(9, max(3, len(csi_df) * 0.34 + 1))
    ax = fig.subplots()
    ax.barh(csi_df["feature"][::-1], csi_df["CSI"][::-1], color=colors[::-1])
    ax.axvline(0.10, ls="--", color="#FB8C00", lw=1, label="0.10 (minor shift)")
    ax.axvline(0.25, ls="--", color="#E53935", lw=1, label="0.25 (unstable)")
    ax.set_xlabel("CSI (train -> OOT)")
    ax.set_title("Characteristic Stability — CSI per feature (red = unstable, train vs OOT)")
    ax.legend(fontsize=8)
    return _save(fig, charts_dir, "csi_per_feature.png",
                 "CSI = how much each FEATURE's distribution shifted from train to the OOT period. "
                 ">0.25 (red) = unstable and risky to rely on; <0.10 (green) = stable.")


def chart_score_psi(score_train, score_oot, charts_dir):
    """PSI = Population Stability Index on the model SCORE distribution (train vs OOT)."""
    from mlkit.stats import compute_psi
    if score_train is None or score_oot is None:
        return None
    psi = float(compute_psi(pd.Series(score_train), pd.Series(score_oot)))
    flag = "STABLE" if psi < 0.10 else ("minor shift" if psi < 0.25 else "UNSTABLE")
    bins = np.linspace(0, 1, 21)
    fig = _new_fig(8, 3.6)
    ax = fig.subplots()
    ax.hist(score_train, bins=bins, density=True, alpha=0.55, color="#1565C0", label="train")
    ax.hist(score_oot, bins=bins, density=True, alpha=0.55, color="#E53935", label="OOT")
    ax.set_xlabel("model score"); ax.set_ylabel("density")
    ax.set_title(f"Score PSI (train vs OOT) = {psi:.3f}  [{flag}]")
    ax.legend(fontsize=8)
    return _save(fig, charts_dir, "score_psi.png",
                 "PSI = how much the model's SCORE distribution shifted from train to OOT. "
                 ">0.25 = unstable (model sees a different population); <0.10 = stable.")


def _woe_by_bin(x, y, n_bins=10):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d["y"].nunique() < 2 or d["x"].nunique() < 2:
        return None
    try:
        d["b"] = pd.qcut(d["x"], n_bins, duplicates="drop")
    except Exception:
        d["b"] = pd.cut(d["x"], min(5, d["x"].nunique()))
    g = d.groupby("b", observed=True)["y"].agg(["count", "sum"])
    n = len(g)
    good = g["count"] - g["sum"]; bad = g["sum"]
    tg = max(int((d["y"] == 0).sum()), 1); tb = max(int((d["y"] == 1).sum()), 1)
    pg = (good + 0.5) / (tg + 0.5 * n); pb = (bad + 0.5) / (tb + 0.5 * n)
    woe = np.log(pg / pb)
    return list(woe.values)


def chart_woe_trends(X_train, y_train, feature_names, importances, charts_dir,
                     reverse=None, top_k=6):
    if importances is None:
        return None
    n = min(len(X_train), SAMPLE_ROWS)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), n, replace=False) if len(X_train) > n else slice(None)
    Xt = X_train[idx]; yt = np.asarray(y_train)[idx]

    top = np.argsort(importances)[::-1][:top_k]
    panels = []
    for j in top:
        woe = _woe_by_bin(Xt[:, j], yt)
        if woe and len(woe) >= 2:
            panels.append((_real(feature_names[j], reverse), woe))
    if not panels:
        return None

    ncol = 2; nrow = (len(panels) + 1) // 2
    fig = _new_fig(9, nrow * 2.4 + 0.5)
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for k, (name, woe) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        mono = "monotonic ✓" if (all(np.diff(woe) >= 0) or all(np.diff(woe) <= 0)) else "non-monotonic"
        ax.plot(range(1, len(woe) + 1), woe, marker="o", color="#1565C0")
        ax.axhline(0, ls="--", lw=0.8, color="#9E9E9E")
        ax.set_title(f"{name[:28]}  ({mono})", fontsize=8)
        ax.set_xlabel("bin (low→high)", fontsize=7); ax.set_ylabel("WoE", fontsize=7)
        ax.tick_params(labelsize=7)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("WoE trend for top features (monotonic = well-behaved)", fontweight="bold")
    fig.subplots_adjust(hspace=0.6, wspace=0.25, top=0.90)
    return _save(fig, charts_dir, "woe_trends.png",
                 "How a feature's odds of the positive change across its value bins (low→high). "
                 "A smooth one-directional (monotonic) line means the feature behaves sensibly.")


# --------------------------------------------------------------------------------------------
# existing dashboard charts (also logged to artifacts)
# --------------------------------------------------------------------------------------------
def _importances(model):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_)
        return np.abs(c[0] if c.ndim > 1 else c)
    return None


def chart_iv(iv_table, charts_dir, reverse=None, top=20):
    rows = [r for r in (iv_table or []) if r.get("IV") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.get("IV", 0), reverse=True)[:top]
    feats = [_real(r["feature"], reverse) for r in rows]
    ivs = [r.get("IV", 0) for r in rows]
    fig = _new_fig(9, max(3, len(rows) * 0.34 + 1))
    ax = fig.subplots()
    ax.barh(feats[::-1], ivs[::-1], color="#1565C0")
    ax.axvline(0.02, ls="--", color="#FB8C00", lw=1, label="weak (0.02)")
    ax.axvline(0.10, ls="--", color="#2E7D32", lw=1, label="medium (0.10)")
    ax.set_xlabel("Information Value"); ax.set_title("IV — selected features"); ax.legend(fontsize=8)
    return _save(fig, charts_dir, "iv_selected.png",
                 "Predictive strength of each selected feature (Information Value). "
                 "Higher = more signal about the target; >0.10 is a useful predictor.")


def chart_feature_importance(model, feature_names, charts_dir, reverse=None, top=20):
    fi = _importances(model)
    if fi is None or not feature_names:
        return None
    fi = fi / (fi.sum() or 1)
    order = np.argsort(fi)[::-1][:top]
    feats = [_real(feature_names[i], reverse) for i in order]
    vals = [fi[i] for i in order]
    fig = _new_fig(9, max(3, len(order) * 0.34 + 1))
    ax = fig.subplots()
    ax.barh(feats[::-1], vals[::-1], color="#1565C0")
    ax.set_xlabel("Relative importance"); ax.set_title("Feature importance — best model")
    return _save(fig, charts_dir, "feature_importance.png",
                 "How much the final model actually relies on each feature when making predictions.")


def chart_leaderboard(leaderboard, primary_metric, charts_dir):
    key = _PRIMARY_OOT_KEY.get(primary_metric, "pr_auc")
    best = {}
    for r in (leaderboard or []):
        v = r.get("oot_metrics", {}).get(key, 0)
        best[r["algorithm"]] = max(best.get(r["algorithm"], 0), v)
    if not best:
        return None
    algos = sorted(best, key=best.get, reverse=True)
    vals = [best[a] for a in algos]
    fig = _new_fig(8, max(3, len(algos) * 0.5 + 1))
    ax = fig.subplots()
    bars = ax.barh(algos[::-1], vals[::-1], color="#2E7D32")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xlabel(f"OOT {key}"); ax.set_title(f"Best OOT {key} per algorithm")
    return _save(fig, charts_dir, "leaderboard.png",
                 "Best out-of-time score reached by each algorithm tried — the longest bar is the winner.")


def chart_oot_metrics(leaderboard, best_algorithm, primary_metric, charts_dir):
    key = _PRIMARY_OOT_KEY.get(primary_metric, "pr_auc")
    row = next((r for r in (leaderboard or []) if r.get("algorithm") == best_algorithm), None)
    oot = (row or {}).get("oot_metrics", {})
    if not oot:
        return None
    fig = _new_fig(7, 3.4)
    ax = fig.subplots()
    colors = ["#1976D2" if k == key else "#64B5F6" for k in oot]
    bars = ax.bar(list(oot.keys()), list(oot.values()), color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.1); ax.set_title(f"OOT metrics — {best_algorithm} (primary={key})")
    return _save(fig, charts_dir, "oot_metrics.png",
                 "The winning model's performance on the held-out future (OOT) period across all metrics; "
                 "the primary metric is highlighted.")


def chart_posrate_per_snapshot(run_dir, charts_dir, oot_range=None):
    import json
    pp = Path(run_dir) / "data_profile.json"
    if not pp.exists():
        return None
    prof = json.loads(pp.read_text(encoding="utf-8"))
    pmr = (prof.get("temporal", {}) or {}).get("pos_ratio_per_snapshot", {}) or {}
    if len(pmr) < 2:
        return None
    import statistics
    months = list(pmr); rates = [pmr[m] * 100 for m in months]; med = statistics.median(rates)
    oot_set = set(oot_range or [])
    fig = _new_fig(9, 3.4)
    ax = fig.subplots()
    ax.plot(range(len(months)), rates, marker="o", lw=2, color="#1565C0")
    ax.axhline(med, ls="--", lw=1, color="#90A4AE", label=f"median {med:.2f}%")
    for i, m in enumerate(months):
        if m in oot_set:
            ax.axvspan(i - 0.5, i + 0.5, color="#FFE0B2", alpha=0.6)
    ax.set_xticks(range(len(months))); ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Positive rate (%)"); ax.set_title("Positive rate per snapshot (OOT shaded)")
    ax.legend(fontsize=8)
    return _save(fig, charts_dir, "posrate_per_snapshot.png",
                 "The event (positive) rate in each month — flat = stable; a spike/dip = drift to watch; "
                 "the orange band marks the OOT test months.")


# --------------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------------
def save_all_charts(run_dir, prepared, model, result) -> list[str]:
    """Generate every chart into <run_dir>/charts/. Best-effort: per-chart failures are logged
    to charts/_errors.txt and skipped, never raised."""
    from pipeline.evaluation import pos_score

    charts_dir = Path(run_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    reverse = (result.column_alias_map or {}).get("reverse", {}) if result.column_alias_map else {}
    saved, errors = [], []

    y = prepared.y_oot
    try:
        score = pos_score(model, prepared.X_oot)
    except Exception as e:
        score = None
        errors.append(f"pos_score: {type(e).__name__}: {e}")
    # train scores (sampled) for the Score-PSI chart
    try:
        Xtr = prepared.X_train
        if len(Xtr) > MAX_POINTS:
            _rng = np.random.default_rng(42)
            Xtr = Xtr[_rng.choice(len(Xtr), MAX_POINTS, replace=False)]
        score_train = pos_score(model, Xtr)
    except Exception as e:
        score_train = None
        errors.append(f"pos_score_train: {type(e).__name__}: {e}")
    oot_range = result.split_report.oot_range if result.split_report else None
    iv_table = result.selection_report.iv_table if result.selection_report else []

    jobs = [
        ("gains_lift",        lambda: chart_gains_lift(y, score, charts_dir)),
        ("roc_pr",            lambda: chart_roc_pr(y, score, charts_dir)),
        ("ks",                lambda: chart_ks(y, score, charts_dir)),
        ("calibration",       lambda: chart_calibration(y, score, charts_dir)),
        ("score_psi",         lambda: chart_score_psi(score_train, score, charts_dir)),
        ("csi",               lambda: chart_csi(prepared.X_train, prepared.X_oot,
                                                prepared.feature_names, charts_dir, reverse)),
        ("woe_trends",        lambda: chart_woe_trends(prepared.X_train, prepared.y_train,
                                                       prepared.feature_names, _importances(model),
                                                       charts_dir, reverse)),
        ("iv",                lambda: chart_iv(iv_table, charts_dir, reverse)),
        ("feature_importance", lambda: chart_feature_importance(model, prepared.feature_names,
                                                                charts_dir, reverse)),
        ("leaderboard",       lambda: chart_leaderboard(result.leaderboard, result.primary_metric,
                                                        charts_dir)),
        ("oot_metrics",       lambda: chart_oot_metrics(result.leaderboard, result.best_algorithm,
                                                        result.primary_metric, charts_dir)),
        ("posrate",           lambda: chart_posrate_per_snapshot(run_dir, charts_dir, oot_range)),
    ]
    for name, fn in jobs:
        try:
            p = fn()
            if p:
                saved.append(p)
        except Exception as e:  # never let a chart break the run
            errors.append(f"{name}: {type(e).__name__}: {e}")
    if errors:
        (charts_dir / "_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    return saved
