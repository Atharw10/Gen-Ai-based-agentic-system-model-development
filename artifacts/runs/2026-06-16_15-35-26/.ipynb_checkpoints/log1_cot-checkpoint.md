# Run 2026-06-16_15-35-26


## Stage: `profiling`
_2026-06-16T15:36:26_

**Inputs from previous stages:**
- user inputs (data_path, target_col, date_col)

**NS keys read:** `df_train`, `target_col`, `date_col`, `business_objective`

**What this stage did:**
Profiled 600000 rows & 50 cols in 5.51s

**Decisions made:**
- `n_rows` = 600000
- `n_cols` = 50
- `target` = cc_conversion_next_1m
- `pos_ratio` = 0.0503
- `n_snapshots` = 6

**Assumptions:**
- Pure observation - no decisions made here
- Only df_train profiled (df_test may be None)
- Date column parsed to datetime

**Outputs flowing to next stages:**
- model_design (consumes profile dict)

**NS keys written:** `profile`

---

## Stage: `model_design`
_2026-06-16T15:41:00_

**Inputs from previous stages:**
- profiling (NS['profile'])
- research_tool

**NS keys read:** `profile`, `business_objective`, `llm`

**What this stage did:**
Model Design completed in 7.05s. Research mode: gemini. Algorithms selected: ['XGBoost', 'LightGBM', 'LogisticRegression']. Errors during design: 0

**Decisions made:**
- `algorithm_shortlist` = ['XGBoost', 'LightGBM', 'LogisticRegression']
- `primary_metric` = ROC_AUC
- `imbalance_tier` = severe
- `snapshots_for_training` = ['2025-01', '2025-02', '2025-03', '2025-04']
- `oot_snapshots` = ['2025-05', '2025-06']
- `imputation_numeric` = median
- `outlier_strategy` = winsorize_99

**Assumptions:**
- OOT snapshots are the latest months (no future leakage)
- Algorithm shortlist constrained to 1-3 models
- Tier derived from profile's pos_ratio
- Validation: snapshots_for_training and oot_snapshots are disjoint

**Outputs flowing to next stages:**
- all downstream stages (windows, algos, imputation)

**NS keys written:** `model_design`, `research_findings`

---

## Stage: `eda`
_2026-06-16T15:51:53_

**Inputs from previous stages:**
- profiling (NS['profile'])
- model_design (NS['model_design'])

**NS keys read:** `df_train`, `target_col`, `date_col`, `profile`, `model_design.windows.snapshots_for_training`

**What this stage did:**
Stage executed successfully on attempt 1 in 24.77s. Generated 3241 chars of Python code. Output produced 2007 chars of stdout.

**Decisions made:**
- `snapshots_used` = ['2025-01', '2025-02', '2025-03', '2025-04']
- `n_top_correlations` = 15
- `eda_sample_rows` = 400000

**Assumptions:**
- EDA restricted to training snapshots to avoid OOT leakage
- Only top-15 correlated features visualized for clarity
- Histograms limited to top-3 by abs corr with target

**Outputs flowing to next stages:**
- cleaning, FE - eda_report is informational

**NS keys written:** `eda_report`

---

## Stage: `cleaning`
_2026-06-16T15:54:12_

**Inputs from previous stages:**
- model_design (imputation & outlier strategies)

**NS keys read:** `df_train`, `df_test`, `target_col`, `model_design.data_strategy`

**What this stage did:**
Stage executed successfully on attempt 1 in 17.77s. Generated 4160 chars of Python code. Output produced 301 chars of stdout.

**Decisions made:**
- `imputation_strategy` = median/mode
- `outlier_strategy` = winsorize_99
- `cols_dropped_high_null` = []
- `duplicates_removed` = 0
- `rows_train_after` = 600000

**Assumptions:**
- Imputation values learned from train only, applied to test (no leakage)
- Outlier bounds learned from train, clipped on both
- Duplicates dropped from train ONLY - test rows preserved
- Cols with >70% nulls dropped from both

**Outputs flowing to next stages:**
- feature_engineering (cleaned df_train, df_test)

**NS keys written:** `df_train`, `df_test`, `cleaning_report`, `imputation_map`, `outlier_bounds`

---

## Stage: `feature_engineering`
_2026-06-16T15:55:49_

**Inputs from previous stages:**
- cleaning (df_train, df_test)
- model_design (windows, thresholds)

**NS keys read:** `df_train`, `df_test`, `target_col`, `date_col`, `llm`, `model_design.windows`, `model_design.data_strategy`

**What this stage did:**
Stage executed successfully on attempt 1 in 35.95s. Generated 7491 chars of Python code. Output produced 6739 chars of stdout.

**Decisions made:**
- `final_n_features` = 36
- `derived_created` = ['credit_velocity_3m_6m', 'debit_velocity_3m_6m', 'credit_debit_ratio_3m', 'mandate_share_3m', 'bounce_rate_3m']
- `dropped_psi` = ['debit_velocity_3m_6m', 'txn_count_3m', 'txn_count_6m', 'total_debit_amt_6m', 'credit_velocity_3m_6m', 'salary_credit_months_3m', 'total_debit_amt_3m', 'total_credit_amt_6m', 'bounce_count_3m', 'mandate_share_3m', 'total_credit_amt_3m']
- `dropped_corr` = 10
- `woe_applied` = False
- `scaling_applied` = True

**Assumptions:**
- Temporal split: OOT is FUTURE data - never random
- PSI threshold auto-drops unstable features (user can override via HITL)
- Correlation pruning keeps feature with higher target-correlation
- Scaling only applied if model_design requires it (e.g., for LogReg)
- Column mapping inferred by LLM with sample rows (overridable via HITL)

**Outputs flowing to next stages:**
- training (X_train, y_train)
- evaluation (X_oot, y_oot)

**NS keys written:** `X_train`, `X_oot`, `y_train`, `y_oot`, `preprocessor`, `feature_names`, `column_map`, `psi_report`, `iv_report`, `fe_report`

---

## Stage: `training`
_2026-06-16T17:28:59_

**Inputs from previous stages:**
- feature_engineering (X_train, y_train)
- model_design (algorithm shortlist, metric)

**NS keys read:** `X_train`, `y_train`, `model_design.algorithm_shortlist`, `model_design.expected_metric_targets`

**What this stage did:**
Stage executed successfully on attempt 1 in 3353.57s. Generated 4101 chars of Python code. Output produced 976 chars of stdout.

**Decisions made:**
- `models_trained` = ['XGBoost', 'LightGBM', 'GradientBoosting']
- `imbalance_tier` = moderate
- `cv_scores` = {'XGBoost': [0.7098750428093752], 'LightGBM': [0.7136379168916165], 'GradientBoosting': [0.7256290073604912]}

**Assumptions:**
- Only algorithms from model_design shortlist trained (not full zoo)
- Imbalance handled via class_weight='balanced' or scale_pos_weight
- StratifiedKFold preserves class balance across CV folds
- Models verified to be distinct objects (no shared state)
- Score diversity check catches potential target leakage

**Outputs flowing to next stages:**
- evaluation (fitted_models)

**NS keys written:** `fitted_models`, `cv_results`, `imbalance_info`

---

## Stage: `evaluation`
_2026-06-16T17:34:06_

**Inputs from previous stages:**
- training (fitted_models)
- feature_engineering (X_oot, y_oot)

**NS keys read:** `X_oot`, `y_oot`, `fitted_models`, `imbalance_info`, `business_objective`, `model_design.expected_metric_targets`

**What this stage did:**
Stage executed successfully on attempt 3 in 40.94s. Generated 8361 chars of Python code. Output produced 1939 chars of stdout.

**Decisions made:**
- `best_model` = GradientBoosting
- `primary_metric` = PR_AUC
- `optimal_threshold` = 0.1062274204493241
- `business_thresholds` = {'f1_optimal': 0.1062274204493241, 'precision_50': None, 'recall_80': 0.04271480819418522}

**Assumptions:**
- OOT is the only true performance signal - CV scores not used for ranking
- Primary metric chosen by mlkit.metrics.select_primary_metric
- Threshold tuning: F1-optimal AND business-aware (precision >=0.5, recall >=0.8)
- Lift table computed for top-decile actionability (banking standard)

**Outputs flowing to next stages:**
- output (best_model, threshold, leaderboard)

**NS keys written:** `leaderboard`, `best_model`, `best_model_name`, `optimal_threshold`, `business_thresholds`, `test_predictions`, `lift_table`, `metric_choice`

---

## Stage: `output`
_2026-06-16T17:36:13_

**Inputs from previous stages:**
- all upstream stages

**NS keys read:** `preprocessor`, `best_model`, `feature_names`, `column_map`, `model_design`, `optimal_threshold`, `imbalance_info`

**What this stage did:**
Stage executed successfully on attempt 1 in 3.32s. Generated 2050 chars of Python code. Output produced 1294 chars of stdout.

**Decisions made:**
- `artifact_path` = artifacts\runs\2026-06-16_15-35-26/models/best_model.joblib
- `summary_path` = artifacts\runs\2026-06-16_15-35-26/run_summary.json

**Assumptions:**
- Artifact bundles preprocessor + model + thresholds + column_map for deployment
- summary JSON is the audit-trail snapshot for the run

**Outputs flowing to next stages:**
- disk (best_model.joblib, run_summary.json)

**NS keys written:** `artifact`, `summary`

---
