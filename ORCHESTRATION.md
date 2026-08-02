# Orchestration — flows & tool connections

Two drivers run the **same** node functions (`app/nodes.py`); pick per run via `run_v2.py --engine`.

- **linear** (`app/orchestrator.py`) — one-shot; HITL via `runtime.hitl.HITLGate` (auto / approval-file / callable).
- **graph** (`app/graph.py`) — LangGraph; HITL via `interrupt()` + checkpointer (true pause/resume).

## Node → tool map (shared by both drivers)

| Node (`app/nodes.py`) | Calls | Plane | State in → out |
|---|---|---|---|
| `node_alias` | `dataio.aliasing` | deterministic | df → aliased df + `alias_map` |
| `node_profile` | `pipeline.profiling.profile_dataset` | deterministic | df → `profile` |
| `node_research` | `advisors.research.research` | LLM (optional) | profile → `findings` (structured plan) |
| `node_domain_fe` | `pipeline.feature_engineer.engineer` | deterministic | df + families → derived features |
| `node_design` | `advisors.design.design_config` | deterministic assembly of plan | profile, findings → `config` |
| `node_prepare` | `pipeline.preparation.prepare_data` | deterministic | df, config → `prepared` |
| `node_search_round` | `pipeline.training.run_search` (Optuna + GroupKFold) | deterministic | prepared, config → `all_trials` |
| `node_assess` | `advisors.search.should_continue` | LLM (optional) | trials → `should_continue` |
| `node_strategist` | `advisors.search.revise_search_space` | LLM (optional) | all_trials → revised `config.search` |
| `node_finalize` | `pipeline.runner.finalize_run` | deterministic | prepared, best → `result`, `best_model` |
| `node_narrate` | `advisors.narrative.summarize` | LLM (optional) | result → `summary` |

- `prepare_data` enforces the leakage-safe order: `temporal_split → fit_clean → select_features →
  preprocessor.fit` (all fit on train only; domain FE already done pre-alias in `node_domain_fe`).
- **Research plan** (`node_research`) is structured: `models` / `windows` / `hpo` /
  `feature_families`, each with its own **reason + source**. `node_design` assembles it into a
  validated `RunConfig` (models→algorithms, windows→snapshots, `hpo`→`SearchSpace.hpo_overrides`,
  metric from tier/objective); HPO ranges are used by training only if valid, else defaults.
- **Cross-cutting (both engines):** every node call is wrapped by `runtime.run_logger.RunLogger`
  (live `▶ step … ✓` + `log1_cot.md` reason+source per decision); every LLM call is metered by
  `runtime.cost.CostTracker` (`cost_log.jsonl` / `cost_summary.json`, capturing input/cached/
  output/thinking tokens). In the graph these live in the registry (`logger`, `cost`, `df`,
  `cached_llm`, `prepared`, `best_model` are HEAVY_KEYS).
- **Input guard:** both engines call `_validate_columns` before the run, so a wrong target/date/
  group name fails fast with a clear message rather than a mid-pipeline `KeyError`.

## LangGraph graph (`app/graph.py`)

```
START → alias → profile → hitl_profiling → research → domain_fe → design → hitl_design
                                                                              │
                                          ┌──────── regenerate ───────────────┘  (conditional → design)
                                          │
                                    accept/override
                                          ▼
        prepare → hitl_features → search → assess ──┐
                                                    │  route_after_assess (feature b):
                          ┌── strategist ◄──────────┤  should_continue? ── yes → strategist → search
                          │  (loop edge)            │                      no  → finalize
                          └─────────────────────────┘
                                          ▼ (no)
                                    finalize → hitl_evaluation → narrate → END
```

- `alias` (first node): renames every column — incl. target/date/group — to neutral aliases
  (`col_0001`, `target`, `snapshot_date`, `entity_id`). The LLM only ever sees aliases; the
  real↔alias mapping is written to `column_alias_map.json` / `.csv`. Disable with `alias_columns=False`.
- `assess` (feature b): after each round the agent decides **continue vs stop** via
  `advisors.search.should_continue` — hard cap on rounds, offline = single round, LLM judges from
  the leaderboard with a deterministic improvement fallback. Logged to `search_log.json`.

Conditional edges (the graph's payoff over linear):
- `route_after_design`: **regenerate → design** (cycle) else **prepare**.
- `route_after_assess`: **strategist → search** (cycle) while the agent says keep going, else **finalize**.

### HITL = interrupt/resume
Each `hitl_*` node calls `interrupt({"stage", "decisions"})`; the graph pauses and returns. The
caller resumes with `Command(resume=action)` on the same `thread_id`. `run_graph()` auto-resolves
via a `resolver(payload)->action` (default: accept) so the one graph serves interactive, headless,
and test use.

```python
# interactive / service: drive interrupts yourself
graph = build_graph()                       # MemorySaver checkpointer
cfg = {"configurable": {"thread_id": "run-123"}}
out = graph.invoke(init_state, config=cfg)            # pauses at first gate
payload = out["__interrupt__"][0].value               # show to user
out = graph.invoke(Command(resume=action), config=cfg)  # continue
```

### State vs registry (why)
The checkpointer **serialises** state every step, so heavy/non-serialisable objects (DataFrame,
fitted models, LLM client) must not live in state — you would never checkpoint a 500k-row frame.
They live in a process-local `_REGISTRY` keyed by `thread_id`; graph state carries only small
serialisable data (`config`, reports, leaderboard, scalars). For durable cross-process resume,
back the registry with disk paths.

## Run it
```bash
python run_v2.py --data d.csv --target y --date snapshot_date --group customer_id            # linear
python run_v2.py --data d.csv --target y --date snapshot_date --group customer_id --engine graph
# add --llm to enable the advisor plane (Gemini); --hitl approval_file for headless gated runs
```
