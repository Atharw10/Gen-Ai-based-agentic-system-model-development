# Decision Log

## v2 architecture decisions (refactor — supersedes v1 where they conflict)

| # | Decision | Rationale |
|---|---|---|
| V1 | LLM reasons, Python executes — NO `exec()` of generated modelling code | Reproducibility/MRM: re-running a config gives byte-identical results; auditors review code, not LLM output |
| V2 | `RunConfig` (pydantic) is the reproducible contract | Design advisor emits it; pipeline consumes it; config + data hash + seed reproduces the model |
| V3 | Three planes: `pipeline/` (deterministic), `advisors/` (LLM-only), `contracts/` (typed) | Clean separation; `tests/test_layering.py` mechanically forbids LLM imports in the deterministic plane |
| V4 | Split temporally FIRST, then fit cleaning/selection/encoding on train only | Removes the v1 OOT-into-preprocessing leakage |
| V5 | Group-aware CV (`GroupKFold` by customer) when a group col exists | v1's random `StratifiedKFold` leaked customers across folds → optimistic CV |
| V6 | Scalable selection funnel (hygiene → IV → corr-clustering → null-importance) | Replaces hand-coded 13-concept FE + O(n²) corr loop that hangs at 5k features |
| V7 | Real iteration loop: Optuna HPO + optional LLM strategist | The v1 "iterate & pick best" was unimplemented (fixed hyperparameters) |
| V8 | Single tier function `mlkit.metrics.assign_tier` | Killed the 3 duplicate tier functions (the "5% boundary" inconsistency) |
| V9 | LLM client: temperature=0 + disk response cache | Determinism + cost; cached prompt/response is an audit record |
| V10 | Optional boosters: LightGBM/XGBoost fall back to sklearn HistGradientBoosting | Pipeline runs on any machine (no libomp/SDK hard dependency) |
| V11 | Resumable HITL (checkpoint files / callable), not blocking `input()` | Headless + auditable; supports CI and widget UIs |
| V12 | Spark is a data-access backend (`dataio/`), not a modelling engine | 500k×5k fits single-node post-selection; avoid rewriting mlkit in Spark |
| V13 | Two drivers over shared nodes (`app/nodes.py`): linear (`orchestrator.py`) + LangGraph (`graph.py`) | Keep the offline one-shot path; add a real graph with conditional-edge loops (design-regenerate, search-rounds) |
| V14 | LangGraph HITL = `interrupt()` + checkpointer (pause/resume) | Service/UI-ready, resumable; `run_graph()` auto-resolves gates for headless/test use |
| V15 | Heavy objects (df, models, LLM) live in a process-local registry, NOT checkpointed state | The checkpointer serialises state every step; never checkpoint a 500k-row frame |
| V16 | Column aliasing at input (privacy shield): all columns incl. target/date/group → neutral aliases before anything runs | LLM advisor plane only ever sees `col_0001`/`target`/`snapshot_date`/`entity_id`, never real schema; real↔alias mapping written to `column_alias_map.json`/`.csv`; data values never leave. Human-facing artifacts auto-translate back to real names: `selection_report_real.json`, real-name highlights in `model_card.md`, CLI/notebook printouts, and `RunResult.column_alias_map` for downstream translation |
| V17 | Agentic stop-decision (feature b): agent decides continue-vs-stop each round | Hard cap on rounds + offline=single-round + LLM judges leaderboard with deterministic improvement fallback; logged to `search_log.json` |
| V18 | Live step logging + reasoning audit log (`runtime/run_logger.py`) | `▶ step … ✓ (Xs)` printed live (notebook/CLI) + `log1_cot.md`/`log1_reasons.json` record WHY + SOURCE for every decision |
| V19 | Research advisor produces a structured PLAN (models / windows / HPO ranges / feature families), each with reason+source | Decisions are research-driven, not hardcoded; design assembles the plan into a validated RunConfig; deterministic plan when no LLM (source='deterministic default') |
| V20 | Research-proposed HPO ranges used by training if valid, else hardcoded defaults | `SearchSpace.hpo_overrides`; each range validated ([lo<hi] numeric) per-parameter |
| V21 | Research-suggested feature families matched against the data ('feature store') pre-alias | `node_domain_fe` builds derived features for resolved families (matching on real names via the alias map), else normal FE/selection; logged |
| V22 | Reasoning integrity: a decision is Source=Gemini only if the LLM actually supplied that section's reason | No hardcoded reason is ever stamped as model reasoning; offline reasons honestly sourced 'deterministic default'; design echoes the research plan's reason, never invents one |
| V23 | Per-run LLM token + cost accounting (`runtime/cost.py`) | Every call metered via Gemini `usage_metadata` (length-estimate fallback) through `CachedLLM`; per-step rows in `cost_log.jsonl` + roll-up `cost_summary.json`; cache hits = $0; configurable pricing table (placeholders — verify); offline writes a $0 summary |
| V24 | Upfront column validation before the run starts (`_validate_columns` in `app/orchestrator.py`, reused by `app/graph.py`; guard also in `dataio.aliasing.build_alias_map`) | A wrong/typo target/date/group name used to surface as a cryptic `KeyError: 'target'` deep in profiling (aliasing silently skipped the unmatched name but state still pointed at the alias). Now both engines raise a clear, case-insensitive "did you mean …?" error immediately; aliasing raises explicitly as a second net |
| V25 | OOT precision/recall/F1 reported at the **F1-optimal operating threshold**, not the default 0.5 (`pipeline/evaluation.py:oot_metrics`) | On imbalanced data (~5% positives) `model.predict()`'s 0.5 cut classifies almost everything negative, so precision/recall/F1 collapsed to 0 even for a strong ranker. Threshold is now derived from the PR curve (data-driven, not hardcoded); AUC/PR-AUC/KS are threshold-free and unchanged. Single-class folds fall back to `model.predict()` |
| V26 | Exact HPO trial budget across algorithms (`pipeline/training.py:run_search`) | `n_trials // len(algos)` dropped the remainder (e.g. 20 trials over 6 algos ran only 18). Remainder is now distributed (`divmod`) so the run executes exactly `n_trials`; each algo still runs ≥1 trial |
| V27 | Extended token accounting: `cached_tokens` + `thinking_tokens` captured alongside input/output (`advisors/llm.py:_usage`, `runtime/cost.py`) | Gemini 2.5 reports context-cache reads and reasoning tokens separately; both are extracted (from `usage_metadata.input_token_details.cache_read` / `output_token_details.reasoning_tokens`, with raw-API fallbacks), summed in `cost_summary.json`, and restored on disk-cache hits. Both are legitimately 0 on `flash-lite` (no thinking by default; local disk cache, not server-side context cache) |
| V28 | Notebook exports a copy-paste cost row to a shared `artifacts/runs/cost_log.csv` | One appended row per run in a fixed format (Use Case · Model · Iteration ID · Prompt/Cached/Output/Thinking/Total Tokens) for the team's central usage sheet; reads `cost_summary.json`, works offline ($0). A standalone shareable `token_tracker.py` mirrors this for other projects |
| V29 | Explicit `model_type` dropdown (hint only) replaces fuzzy text matching for use-case detection | `ModelType` enum (propensity/cross_sell/up_sell/churn/fraud/credit_risk/generic) on `RunConfig` (optional, default None → legacy behaviour). Threaded through both engines into `research()` (use-case template + prompt context line, **same LLM call count**) and `design()` (deterministic metric policy). User still types target/date. Additive: pipeline, graph structure, and LLM calls unchanged |
| V30 | Data-source registry (`dataio/sources.py`) — the seam between "where data lives" and the pandas pipeline | `DataSource.load(spec) -> pandas.DataFrame`; the pipeline is unchanged and only ever sees a pandas frame. `FileSource` is live; CDP/CML sources (Hive/Impala/FeatureStore/DataLake/CustomSQL) are stubs raising `NotWiredError` until Phase 2, with column/snapshot-partition/sampling **pushdown** designed in. CML-safe: pyspark/impyla imported lazily inside `load()`, never at module import. No LLM import (layering test stays green) |
| V31 | Per-snapshot positive-rate + drift check feeds window/OOT selection | Profiling computes `TemporalProfile.pos_ratio_per_snapshot`; `mlkit.stats.pos_rate_drift` flags months deviating sharply from the median. Saved to `data_profile.json`, shown at the profiling gate, and passed into research (LLM prompt + deterministic windows reason) so the observation/OOT window favours a stable recent period and warns on anomalous OOT. **OOT remains the most recent month(s)** — the rate informs window length and warns on drift, never moves which months are OOT. Additive; no notebook change required |
| V32 | Validation curves auto-logged to `<run>/charts/` (`pipeline/charts.py`) | On every run, finalize saves a model-validation pack: gains/lift, ROC+PR, KS, calibration, PSI-per-feature, WoE trends, plus IV / feature-importance / leaderboard / OOT-metrics / posrate. Uses matplotlib `Figure` API (no pyplot/backend → headless/CML-safe, won't clash with notebook inline). Best-effort: per-chart failures go to `charts/_errors.txt`, never break the run. Large-data-safe: probability curves run on the OOT vector capped to 200k rows; per-feature charts sample 50k. Notebook displays the validation set inline; profiling gate also shows the posrate chart |
| V33 | Double-validation agent (`pipeline/validation.py` + `advisors/reviewer.py` + `app/validation_agent.py`) | Runs after everything completes. Deterministic checks (no LLM, reproducible, hallucination-proof) are the source of truth — IV-all-zero, metric 0 / AUC<0.5, near-perfect AUC (leakage), P/R/F1 all 0, no Tavily links, LLM-enabled-but-0-calls, chart errors, empty/drifted split → GREEN/AMBER/RED. Optional LLM reviewer adds a grounded 2nd opinion and cross-checks the narrative for hallucinated numbers (verdict stays deterministic-driven). Saves `validation_report.json/.md`; prints verdict. WARN-only (never blocks). Layering preserved: deterministic checks in `pipeline/`, LLM in `advisors/`, combiner in `app/` |
| V34 | Stability + rank-ordering checks/charts: Score PSI, CSI, risk ranking | (a) Renamed the per-feature stability chart PSI→**CSI** (`csi_per_feature.png/.csv`) to match banking vocabulary (CSI = per-characteristic; PSI = model score). (b) Added a **Score-PSI** chart (`score_psi.png`, train vs OOT score distribution). (c) Validation agent now gates three more deterministic checks: Score PSI (>0.25 FAIL, 0.10–0.25 WARN), CSI per feature (any selected feature >0.25 → WARN), and risk-ranking (≥2 decile reversals in OOT event rate → WARN). All sampled for large data; layering preserved |

Entrypoints: `run_v2.py --engine linear|graph` (headless CLI) and `notebook_v2.ipynb` (thin
driver). Flows documented in `ORCHESTRATION.md`. v1 (`notebook.ipynb`, `prompts/`, `tools/`)
is retained read-only as legacy.

## Architecture decisions (v1 — historical)

| # | Decision | Rationale |
|---|---|---|
| 1 | LangGraph over CrewAI | Graph-of-nodes model is cleaner for deterministic ML pipeline with retries |
| 2 | Gemini 2.5 Flash Lite | Free tier, fast, sufficient quality; Pro not available to user |
| 3 | Specified codegen (Level-3) over goal-driven (Level-5) | Reproducibility required for banking; auditability over autonomy |
| 4 | 3-layer structure: mlkit + tools + prompts | mlkit = pure functions, tools = orchestration, prompts = LLM instructions |
| 5 | Shared NS dict for data flow | Single source of truth, easy to override via HITL |
| 6 | Profiling before Model Design | Model Design needs data context to be informed (not pure-LLM guess) |
| 7 | Plan+Code mode NOT implemented | (Stage 5 prototype only) \| Too complex for POC; documented as future enhancement |
| 8 | Adaptive column mapping (LLM $\rightarrow$ regex $\rightarrow$ user override) | Handles any column naming convention |
| 9 | Tier-driven imbalance handling | Auto-adapts: class_weight, SMOTE, scoring, CV folds, threshold tuning |
| 10 | Single CSV with temporal OOT carve-out | Standard banking practice; no separate test file needed |

## Feature decisions

| # | Choice | Why |
|---|---|---|
| F1 | IV computed but NOT used for auto-drop | Failed multiple times on binary features; correlation pruning more reliable |
| F2 | PSI auto-drops features > 0.25 (with HITL override) | Bank-standard stability check |
| F3 | WoE conditional (only if Model Design picks LogReg) | Useful for credit scoring; unnecessary for tree models |
| F4 | Correlation threshold default 0.85 | Conservative for banking (vs 0.95 common in DS) |
| F5 | Skip IQR for nunique <= 10 | Prevents binary-feature collapse |

## Logging decisions

| # | Choice | Why |
|---|---|---|
| L1 | 3 streams (cot/code/tools) | Different audiences: humans, engineers, auditors |
| L2 | Per-run timestamped folder | Isolation, no overwrites |
| L3 | log1 markdown, log2 python, log3 jsonl | Native format for each consumer |

## HITL decisions

| # | Choice | Why |
|---|---|---|
| H1 | Synchronous input() (Option A) | Simplest; works in Jupyter without widgets |
| H2 | 3 actions: accept / override / regenerate | Covers most user needs |
| H3 | Override shows current value + options | Not pure free-form; reduces input errors |
| H4 | Regenerate appends free-form to prompt | Powerful escape hatch |

---