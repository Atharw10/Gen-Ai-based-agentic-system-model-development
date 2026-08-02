# Project: Agentic Banking Propensity ML Pipeline (v2)

> **v2 is the active system.** Core principle: **the LLM reasons and proposes; deterministic
> Python executes.** There is **no `exec()` of generated modelling code** (that was the v1 flaw).
> A run is defined by a validated `RunConfig` + data hash + seed, and reproduces byte-for-byte
> with no LLM in the loop. v1 (`notebook.ipynb`, `prompts/`, `tools/`) is retained read-only as
> legacy and is NOT used by v2.

## Context
* **Who**: Atharaw Patle, Intern at Axis Bank
* **Mentor**: Siddharth Rajpriye
* **Project name**: GenAI Model AI Development – Atharaw Intern Project
* **Goal**: End-to-end agentic ML system that automates propensity model building for banking
  (specifically CC upselling).

---

## 1. Architecture — three planes + support

The system is split so the LLM only ever touches judgement, never the data math.

| Plane | Packages | Role |
|---|---|---|
| **Typed contracts** | `contracts/` | `RunConfig` (the reproducible contract) + result types |
| **Deterministic pipeline** (no LLM — enforced by `tests/test_layering.py`) | `pipeline/`, `mlkit/` | split → clean → select → train → eval → persist |
| **LLM advisors** (the only LLM plane) | `advisors/` | research, design, search-strategist, narrative |
| **Orchestration** | `app/`, `runtime/` | two engines + HITL + the model-search loop + logging/cost |
| **Data access** | `dataio/` | `DataBackend` interface (pandas now; Spark/CML later) |

```text
project/
├── run_v2.py              # headless CLI entrypoint  (--engine linear|graph, --llm, --hitl)
├── notebook_v2.ipynb      # thin notebook driver (ENGINE/INTERACTIVE/USE_LLM toggles)
│
├── contracts/
│   ├── config.py          # RunConfig, Windows, DataStrategy, FeatureSelection, SearchSpace(+hpo_overrides)
│   └── results.py         # ProfileResult, SelectionReport, SplitReport, TrialResult, RunResult
│
├── pipeline/              # DETERMINISTIC plane (pure python, tested, no LLM imports)
│   ├── profiling.py       # observe data -> ProfileResult
│   ├── splitting.py       # temporal_split FIRST (anti-leakage)
│   ├── cleaning.py        # impute/outliers fit on TRAIN only
│   ├── feature_engineer.py# optional domain ratios; matches on real names via alias map
│   ├── feature_select.py  # funnel: hygiene -> IV -> corr-clustering -> null-importance; flags leakage
│   ├── preparation.py     # leakage-safe orchestration -> PreparedData (X/y)
│   ├── training.py        # model registry (+sklearn fallbacks) + Optuna HPO + group-aware CV
│   ├── evaluation.py      # OOT metrics, threshold tuning, lift table
│   ├── persistence.py     # artifact bundle + model card + real-name reports
│   └── runner.py          # run_from_config(), finalize_run()
│
├── mlkit/                 # reusable ML functions
│   ├── selection.py  stats.py  cv.py  cleaning_ops.py  metrics.py   # <- LIVE in v2
│   └── features.py  cleaning.py  splits.py  reporting.py  schema_validator.py  # <- v1 legacy
│
├── advisors/             # ONLY LLM plane
│   ├── llm.py             # CachedLLM (temp=0, disk cache, token metering), build_llm, extract_json
│   ├── research.py        # structured research PLAN: models/windows/HPO/feature-families (+reason+source)
│   ├── design.py          # assembles plan -> validated RunConfig (deterministic)
│   ├── search.py          # strategist (revise algos) + should_continue (stop-decision)
│   └── narrative.py       # human model summary
│
├── app/
│   ├── nodes.py           # shared node functions (used by BOTH engines)
│   ├── orchestrator.py    # LINEAR engine (one-shot; HITLGate)
│   └── graph.py           # GRAPH engine (LangGraph; interrupt/resume HITL + checkpointer)
│
├── runtime/
│   ├── hitl.py            # resumable HITL gate (auto / approval-file / callable)
│   ├── run_logger.py      # live step log + log1_cot.md reasoning audit (why + source)
│   └── cost.py            # per-run LLM token + cost accounting
│
├── dataio/               # named dataio (NOT io) to avoid shadowing stdlib
│   ├── backend.py  pandas_backend.py  spark_backend.py(stub)
│   └── aliasing.py        # column-name privacy shield + real-name translation
│
├── tests/                # 40+ tests incl. layering, leakage, reproducibility, cost, research-driven
└── artifacts/runs/<ts>/  # per-run outputs (see section 5)
```

---

## 2. End-to-end flow (what runs, in order)

Both engines run the **same** shared nodes (`app/nodes.py`); the only difference is HITL style.

```
alias columns (privacy)  → profile → [HITL] → research → domain FE (families)
   → design → [HITL] → prepare (split/clean/select/encode) → [HITL]
   → SEARCH LOOP { search round (Optuna HPO) → assess (continue/stop?) → strategist }
   → finalize (refit + OOT eval + persist) → [HITL] → narrate → done
```

- **LLM is used only in:** research, design (assembling the plan; the heavy reasoning is research),
  search-strategist, stop-decision, narrative. Everything else is deterministic.
- Offline (no `--llm`): those steps use deterministic fallbacks — the run still completes.

---

## 3. Key features (what makes it agentic, safe, auditable)

- **No exec / reproducible.** Pipeline stages are tested functions. Same `RunConfig`+data+seed →
  identical results (`tests/test_reproducibility.py`). `model_bundle.joblib` is deployable.
- **Two engines.** `--engine linear` (one-shot, notebook/CLI/CI) or `--engine graph` (LangGraph;
  HITL via `interrupt()`/resume — service/UI-ready). Verified to produce identical results.
- **Leakage-safe.** Split happens FIRST; cleaning/selection/encoding fit on train only; CV is
  **GroupKFold by customer**; suspicious-high-IV features are flagged as likely leakage.
- **Research-driven decisions (not hardcoded).** `advisors/research.py` produces a structured plan:
  **models** (per problem type + imbalance tier), **observation/OOT windows**, **HPO ranges**,
  and **feature families** — each with its own **reason + source**. Design assembles it into a
  validated `RunConfig`; training uses the HPO ranges **if valid**, else built-in defaults.
- **Reasoning integrity.** A decision is `Source: Gemini` ONLY if the LLM actually supplied that
  section's reason; otherwise it's honestly `Source: deterministic default`. No hardcoded sentence
  is ever passed off as model reasoning.
- **Agentic stop-decision.** After each search round the agent decides continue vs stop (hard cap
  on rounds; offline = single round; LLM judges the leaderboard with a deterministic fallback).
- **Live logging + reasoning audit.** Steps print live (`▶ step … ✓ (Xs)`); `log1_cot.md` /
  `log1_reasons.json` record **why + source** for every decision.
- **Cost accounting.** Every LLM call is metered (Gemini `usage_metadata`, length-estimate
  fallback) → `cost_log.jsonl` + `cost_summary.json` (input/cached/output/thinking tokens +
  estimated $; cache hits = $0). The notebook also appends a copy-paste row per run to a shared
  `artifacts/runs/cost_log.csv` for the team usage sheet (a standalone `token_tracker.py` mirrors
  this for other projects). Cached/thinking tokens are legitimately 0 on `flash-lite`.
- **Input validation.** Target/date/group column names are validated up front; a wrong name raises
  a clear case-insensitive "did you mean …?" error instead of a cryptic `KeyError` mid-run.
- **Honest operating-point metrics.** OOT precision/recall/F1 are reported at the F1-optimal
  threshold (data-derived), so they stay meaningful on imbalanced data instead of collapsing to 0
  at the default 0.5 cut. AUC/PR-AUC/KS are threshold-free.
- **Privacy (column aliasing).** All columns (incl. target/date/group) are aliased at input, so the
  LLM never sees real schema; real↔alias map is written to artifacts and results are translated
  back to real names for humans. (Note: design is now deterministic and the research prompt sends
  only aggregates, so the LLM receives no column names regardless — aliasing is defense-in-depth.)
- **HITL** at profiling / model_design / feature_selection / evaluation: accept / override / regenerate.
- **Model Type & Data Source dropdowns.** An explicit `model_type` (propensity/cross_sell/up_sell/
  churn/fraud/credit_risk/generic) replaces fuzzy text matching for use-case detection — it's a
  *hint* (you still set target/date) threaded into research/design without changing LLM call count.
  A `dataio/sources.py` registry resolves the data-source choice into a pandas frame: `FileSource`
  is live; CDP/CML sources (Hive/Impala/Feature Store/Data Lake/Custom SQL) are stubs (Phase 2) with
  column/partition/sampling pushdown designed in. CML-safe (pyspark/impyla imported lazily).
- **Per-snapshot positive rate + drift check.** Profiling computes the positive (target=1) rate for
  **each snapshot month** (`TemporalProfile.pos_ratio_per_snapshot`), saved to `data_profile.json`
  and shown at the profiling gate. `mlkit.stats.pos_rate_drift` flags months that deviate sharply
  from the median; this is fed into research so the observation/OOT window favours a stable recent
  period and warns when the OOT period is anomalous. (OOT is always the most recent month(s) by
  definition — the rate informs window length and warns on drift, it never moves which months are OOT.)

---

## 4. How to run

**Install** (Python 3.11+):
```
pip install -r requirements-v2.txt          # mac also: brew install libomp (for lightgbm)
python -m pytest -q                          # 40+ tests (fast ones instant; model tests ~minutes)
```

**CLI** (headless):
```
python run_v2.py --data data.csv --target cc_converted_next_month \
   --date snapshot_date --group customer_id --objective "Build CC propensity model" \
   [--engine linear|graph] [--llm] [--rounds 2] [--hitl auto|approval_file]
```
- `--llm` enables the advisor plane (needs `GOOGLE_API_KEY`). Without it → fully offline, $0.
- `--engine graph` = LangGraph (pause/resume HITL). Default `linear`.

**Notebook** (`notebook_v2.ipynb`): the first cell shows numbered dropdowns — pick the **Model Type**
and **Data Source** (then its follow-up fields), and enter target/date columns. Choose `ENGINE`,
`INTERACTIVE=True` to approve gates inline (Accept / Override / Regenerate), `USE_LLM=True` for
Gemini. Run cells top to bottom; a per-run token-usage row is appended to `cost_log.csv`.

**Windows + LLM:** install Python 3.11 (add to PATH); `py -3.11 -m venv .venv`; `.venv\Scripts\Activate.ps1`;
`pip install -r requirements-v2.txt`; `$env:GOOGLE_API_KEY="AIza..."` (or `setx ...` + new shell);
then the CLI above with `--llm`.

---

## 5. Artifacts produced per run (`artifacts/runs/<timestamp>/`)

| File | Contents |
|---|---|
| `model_bundle.joblib` | deployable: preprocessor + model + threshold + cleaning + alias map |
| `run_config.json` + `data_hash.txt` | the reproducible contract + data fingerprint |
| `run_result.json`, `leaderboard.csv`, `lift_table.csv` | results |
| `model_card.md` | human summary incl. real-name top features + leakage suspects |
| `data_profile.json` | full data profile incl. **per-snapshot positive rate** (drift check) |
| `log1_cot.md` / `log1_reasons.json` | **reasoning audit: why + source for every decision** |
| `search_log.json` | per-round best metric + continue/stop decisions |
| `cost_log.jsonl` / `cost_summary.json` | **LLM tokens (input/cached/output/thinking) + estimated cost per step** |
| `cost_log.csv` (in `artifacts/runs/`) | **copy-paste token-usage row per run for the team sheet** |
| `column_alias_map.json` / `.csv`, `selection_report_real.json` | alias ↔ real-name mapping |
| `checkpoints/` | HITL gate snapshots |

---

## 6. Data assumptions
- Tabular, monthly customer grain; a **snapshot date** column; a **binary target** (0/1); ideally a
  **customer-id** column (enables group-aware CV).
- Typical POC: 100K–500K rows · 5–6 months · tens of features · 1–10% positives.

## 7. Scaling & PySpark (current honest status)
- **Input today:** the whole CSV/Parquet is read into one in-memory pandas DataFrame
  (`run_v2.load()` / notebook). The `dataio` backend is the seam for a future SQL/CML reader (not
  yet wired into the run path).
- **Crore (10M) rows:** fine for modest column counts; **will OOM at load** for crore × thousands
  of features (single-node pandas). Selection samples internally, but the raw frame is loaded first.
  Mitigation: filter/sample in **SQL/Spark before loading**, and downsample the majority class.
- **PySpark conversion:** the **architecture/contracts/orchestration survive unchanged**. Hybrid
  (Spark reads/reduces in `dataio`, single-node modelling on a reduced frame) is clean and
  recommended. Full distributed training = reimplementing `pipeline/`+`mlkit/` internals on Spark
  MLlib behind the same interfaces — a real rewrite of those bodies, not a flag flip.

## 8. Tech stack & keys
- Python 3.11 · pandas/numpy/scipy/scikit-learn/statsmodels · lightgbm/xgboost (optional; sklearn
  HistGradientBoosting fallback) · optuna · pydantic v2 · **langgraph** · langchain-core +
  langchain-google-genai (only for `--llm`) · tavily (optional web research) · pytest.
- `GOOGLE_API_KEY` (Gemini, for `--llm`) · `TAVILY_API_KEY` (optional) ·
  `GEMINI_PRICE_INPUT_PER_1K` / `_OUTPUT_PER_1K` (optional, override cost estimate).

## 9. Where to read more
- `decision.md` — numbered design decisions V1–V31 (the full change rationale).
- `ORCHESTRATION.md` — the graph, node→tool map, HITL/interrupt details.
- `REFACTOR_PLAN.md` — the original v1→v2 refactor plan.
- v1 legacy: `notebook.ipynb`, `prompts/`, `tools/` (reference only; not executed by v2).
