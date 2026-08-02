"""Deterministic post-run validation checks (no LLM).

A battery of fixed rules that inspect a finished run's results + artifacts and flag anything that
looks broken — IV all zero, metrics = 0, AUC worse than random, near-perfect AUC (leakage), no
Tavily web links, LLM enabled but never called, chart errors, empty/ drifted split, etc.

Each check returns a finding: {id, level (FAIL|WARN|INFO|OK), message, evidence}. Reproducible and
hallucination-proof (pure rules). The overall verdict is RED (any FAIL) / AMBER (any WARN) / GREEN.

No LLM import here (deterministic plane — the layering test guards this). The optional LLM "second
opinion" lives in advisors/reviewer.py and is combined by app/validation_agent.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_PRIMARY_OOT_KEY = {"PR_AUC": "pr_auc", "ROC_AUC": "roc_auc", "KS": "ks", "F1": "f1",
                    "Recall": "recall", "Precision": "precision"}


def _best_oot(result) -> dict:
    # finalize_run sorts the leaderboard by OOT primary metric desc -> row 0 is the best model
    return (result.leaderboard[0].get("oot_metrics", {}) if result.leaderboard else {})


def run_checks(result, prepared=None, model=None, run_dir=None, context=None) -> dict:
    """Run all deterministic checks. Returns {verdict, findings, counts}."""
    context = context or {}
    findings: list[dict] = []

    def add(level, cid, message, evidence=None):
        findings.append({"id": cid, "level": level, "message": message, "evidence": evidence})

    sr = getattr(result, "split_report", None)
    sel = getattr(result, "selection_report", None)
    reverse = ((result.column_alias_map or {}).get("reverse", {})
               if getattr(result, "column_alias_map", None) else {})
    oot = _best_oot(result)
    pkey = _PRIMARY_OOT_KEY.get(result.primary_metric, str(result.primary_metric).lower())

    # ── metrics ──────────────────────────────────────────────────────────────
    pm = oot.get(pkey)
    if not oot:
        add("FAIL", "no_oot_metrics", "No OOT metrics found on the best model.")
    elif pm is None:
        add("WARN", "primary_metric_missing",
            f"Primary metric '{pkey}' not present in OOT metrics.", {"oot_keys": list(oot)})
    elif pm == 0 or (isinstance(pm, float) and math.isnan(pm)):
        add("FAIL", "primary_metric_zero", f"Primary OOT metric {pkey} = {pm}.", {pkey: pm})

    roc = oot.get("roc_auc")
    if roc is not None and roc < 0.5:
        add("FAIL", "auc_below_random", f"OOT ROC-AUC {roc:.3f} < 0.5 (worse than random).", {"roc_auc": roc})
    if (roc is not None and roc >= 0.999) or (oot.get("pr_auc", 0) >= 0.999):
        add("WARN", "auc_too_perfect",
            "Near-perfect OOT score — strongly suggests target leakage.",
            {"roc_auc": roc, "pr_auc": oot.get("pr_auc")})
    if oot and all((oot.get(k, 0) or 0) == 0 for k in ("precision", "recall", "f1")):
        add("WARN", "prf_all_zero",
            "Precision, recall and F1 are all 0 at the chosen threshold.",
            {k: oot.get(k) for k in ("precision", "recall", "f1")})

    # ── feature selection / IV ────────────────────────────────────────────────
    if sel is not None:
        if sel.n_selected == 0:
            add("FAIL", "no_features_selected", "0 features were selected.", {"n_in": sel.n_in})
        elif sel.n_in and sel.n_selected == sel.n_in:
            add("WARN", "no_features_dropped",
                "Selection dropped nothing (all input features kept) — filter may be misconfigured.",
                {"n_in": sel.n_in})
        sel_set = set(sel.selected_features or [])
        ivs = [(r.get("IV", r.get("iv", 0)) or 0) for r in (sel.iv_table or [])
               if r.get("feature") in sel_set]
        if ivs:
            zero = sum(1 for v in ivs if v == 0)
            if zero / len(ivs) > 0.9:
                add("WARN", "iv_all_zero",
                    f"{zero}/{len(ivs)} selected features have IV = 0 — IV computation/keys look broken.")
        if getattr(sel, "suspicious_high_iv", None):
            add("WARN", "leakage_suspects",
                f"{len(sel.suspicious_high_iv)} high-IV features flagged as possible leakage.",
                {"features": sel.suspicious_high_iv[:5]})

    # ── split sanity ──────────────────────────────────────────────────────────
    if sr is not None:
        if sr.train_rows == 0 or sr.oot_rows == 0:
            add("FAIL", "empty_split", f"Empty split: train={sr.train_rows}, oot={sr.oot_rows}.")
        if sr.train_pos_ratio and sr.oot_pos_ratio:
            rel = abs(sr.oot_pos_ratio - sr.train_pos_ratio) / max(sr.train_pos_ratio, 1e-9)
            if rel > 0.5 and abs(sr.oot_pos_ratio - sr.train_pos_ratio) > 0.01:
                add("WARN", "split_posrate_drift",
                    f"OOT positive rate {sr.oot_pos_ratio:.2%} differs sharply from train "
                    f"{sr.train_pos_ratio:.2%}.")

    # ── artifacts (LLM usage, web links, chart errors) ────────────────────────
    if run_dir:
        rd = Path(run_dir)
        cs = rd / "cost_summary.json"
        if cs.exists():
            try:
                c = json.loads(cs.read_text(encoding="utf-8"))
                if context.get("use_llm") and c.get("api_calls", 0) == 0:
                    add("WARN", "llm_not_used",
                        "LLM was enabled but 0 API calls were recorded (did it silently run offline?).",
                        {"api_calls": c.get("api_calls", 0)})
            except Exception:
                pass
        if context.get("use_llm") and context.get("tavily_key_set"):
            lr = rd / "log1_reasons.json"
            if lr.exists():
                try:
                    reasons = json.loads(lr.read_text(encoding="utf-8"))
                    if not any("web" in str(r.get("source", "")).lower() for r in reasons):
                        add("WARN", "no_web_links",
                            "Tavily key is set but research produced no web links "
                            "(check SSL/proxy or that tavily is installed).")
                except Exception:
                    pass
        ce = rd / "charts" / "_errors.txt"
        if ce.exists():
            add("WARN", "chart_errors", "Some charts failed to render.",
                {"detail": ce.read_text(encoding="utf-8")[:300]})

    # ── model-level stability (Score PSI) + feature stability (CSI) + risk ranking ────────────
    # These need the fitted model + the prepared matrices. Computed deterministically (no LLM),
    # row-sampled for large data. Standard thresholds: <0.10 stable, 0.10-0.25 minor, >0.25 unstable.
    if prepared is not None and model is not None:
        try:
            import numpy as _np
            import pandas as _pd
            from mlkit.stats import compute_psi
            from pipeline.evaluation import lift_table, pos_score

            _rng = _np.random.default_rng(42)

            def _smp(X, n=50_000):
                return X[_rng.choice(len(X), n, replace=False)] if len(X) > n else X

            # 1) Score PSI — model score distribution train vs OOT
            s_tr = pos_score(model, _smp(prepared.X_train))
            s_oot = pos_score(model, _smp(prepared.X_oot))
            spsi = float(compute_psi(_pd.Series(s_tr), _pd.Series(s_oot)))
            if spsi > 0.25:
                add("FAIL", "score_psi_unstable",
                    f"Score PSI train->OOT = {spsi:.3f} (>0.25: model scores a different population).",
                    {"psi": round(spsi, 3)})
            elif spsi > 0.10:
                add("WARN", "score_psi_minor_shift",
                    f"Score PSI train->OOT = {spsi:.3f} (0.10-0.25: minor shift).", {"psi": round(spsi, 3)})

            # 2) CSI gate — per-feature distribution shift train vs OOT
            Xt, Xo = _smp(prepared.X_train), _smp(prepared.X_oot)
            unstable = []
            for j, nm in enumerate(prepared.feature_names or []):
                try:
                    csi = compute_psi(_pd.Series(Xt[:, j]), _pd.Series(Xo[:, j]))
                    if csi is not None and csi > 0.25:
                        unstable.append([reverse.get(nm, nm), round(float(csi), 3)])
                except Exception:
                    pass
            if unstable:
                unstable.sort(key=lambda t: t[1], reverse=True)
                add("WARN", "csi_unstable_features",
                    f"{len(unstable)} feature(s) with CSI > 0.25 (unstable train->OOT).",
                    {"features": unstable[:8]})

            # 3) Risk ranking — OOT event rate should fall monotonically from top score decile down
            lift = lift_table(_np.asarray(prepared.y_oot), pos_score(model, prepared.X_oot))
            rates = lift["rate"].tolist()  # decile 1 (index 0) = highest scores
            reversals = sum(1 for i in range(len(rates) - 1) if rates[i] < rates[i + 1] - 1e-9)
            if reversals >= 2:
                add("WARN", "rank_order_breaks",
                    f"Risk ranking has {reversals} decile reversals — event rate not monotonic "
                    "across score bands.", {"decile_event_rates": [round(r, 4) for r in rates]})
        except Exception as e:
            add("INFO", "stability_checks_skipped",
                f"Score-PSI / CSI / rank checks skipped: {type(e).__name__}: {e}")

    if not findings:
        add("OK", "all_clear", "All deterministic checks passed.")

    verdict = ("RED" if any(f["level"] == "FAIL" for f in findings)
               else "AMBER" if any(f["level"] == "WARN" for f in findings)
               else "GREEN")
    counts = {lvl: sum(1 for f in findings if f["level"] == lvl) for lvl in ("FAIL", "WARN", "INFO", "OK")}
    return {"verdict": verdict, "findings": findings, "counts": counts}
