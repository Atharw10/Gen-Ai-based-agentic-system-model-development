# Model Card — Build CC propensity model

- **Primary metric:** PR_AUC
- **Best algorithm:** HistGradientBoosting (trial 16)
- **Trials run:** 36
- **Data hash:** `857188fe077623eb…`
- **Seed:** 42

## Data
- Train rows: 500,000 (['2025-01', '2025-05']); pos_ratio=0.0501
- OOT rows:   100,000 (['2025-06', '2025-06']); pos_ratio=0.0511
- Imbalance tier: moderate

## Feature selection
- In: 52 → hygiene: 52 → univariate: 47 → redundancy: 40 → **selected: 40**
- Top selected features (real names): segment, total_credit_amt_1m, net_flow_1m, avg_txn_size_1m, mcc_travel_amt_1m, total_credit_amt_3m, txn_count_1m, total_debit_amt_1m, bounce_count_1m, bounce_amt_1m

## OOT performance (best model)
- precision: 0.0000
- recall: 0.0000
- f1: 0.0000
- roc_auc: 0.7210
- pr_auc: 0.1203
- ks: 0.3157
- gini: 0.4419

## Thresholds
- F1-optimal: 0.1053
- f1_optimal: 0.10529090759904489
- precision_50: None
- recall_80: 0.04346223761033223

## Reproducibility
Re-run `run_config.json` against data matching `data_hash.txt` with the same seed to reproduce this model byte-for-byte. No LLM is involved in reproduction.