"""Stage 3 - Exploratory Data Analysis prompt."""
from .system_rules import SYSTEM_RULES

EDA_PROMPT = SYSTEM_RULES + """
================= STAGE 3: EXPLORATORY DATA ANALYSIS =================

READS FROM NS:
    df_train, target_col, date_col
    NS['profile']           : data profile from Stage 1
    NS['model_design']      : strategic decisions from Stage 2
                              - especially: model_design['windows']['snapshots_for_training']
                              (restrict EDA to TRAINING snapshots only - no OOT peeking)

WRITES TO NS:
  NS['eda_report']      : dict with key findings

DISPLAY INLINE:
  - Markdown banner: "EDA RESULTS"
  - Per-snapshot target rate table
  - Top-15 features by abs correlation with target
  - Up to 3 numeric feature distribution plots
  - Missingness summary

STEPS

1. Build a TRAINING-ONLY subset:
    train_snaps = NS['model_design']['windows']['snapshots_for_training']
    df_eda = df_train[
        pd.to_datetime(df_train[date_col]).dt.to_period('M').astype(str).isin(train_snaps)
    ].copy()
    Print "EDA restricted to training snapshots: {train_snaps}"
    Print "EDA sample shape: {df_eda.shape}"

2. PER-SNAPSHOT TARGET RATE
    Compute target rate per snapshot month. Build a DataFrame:
      snapshot | n_customers | n_positive | target_rate
    Print it. Save to DIRS['eda']+'/target_rate_by_snapshot.csv'.

3. TOP CORRELATIONS WITH TARGET
    For numeric feature columns only (exclude target_col, date_col, customer_id-like cols):
      compute abs(corr_with_target), sort descending, take top 15.
    Print as a small table. Save to DIRS['eda']+'/top_correlations.csv'.

4. NUMERIC FEATURE DISTRIBUTIONS (max 3 plots)
    Pick top-3 features by target correlation.
    For each: plot histogram colored by target_col (0 vs 1) on the same axis.
    figsize=(8,3) per plot. Use ax.set_title("...").

5. MISSINGNESS SUMMARY
    Compute null % per column in df_eda. Print top-10 by null %.

6. BUILD eda_report DICT
    eda_report = {
        'eda_sample_rows':          int(len(df_eda)),
        'training_snapshots_used':  train_snaps,
        'target_rate_by_snapshot':  <list of dicts from step 2>,
        'top_correlations':         <list of dicts from step 3>,
        'top_missing':              <dict col->pct from step 5>,
    }

7. SAVE & ASSIGN
    json.dump(eda_report, open(DIRS['eda']+'/eda_report.json','w'), 
              indent=2, default=str)
    NS['eda_report'] = eda_report
    Print "EDA complete"

MUST ASSIGN: eda_report, NS['eda_report']
"""