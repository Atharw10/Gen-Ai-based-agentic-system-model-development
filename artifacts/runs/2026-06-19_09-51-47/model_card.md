# Model Card — Build cc propensity model

- **Primary metric:** PR_AUC
- **Best algorithm:** LightGBM (trial 5)
- **Trials run:** 36
- **Data hash:** `857188fe077623eb…`
- **Seed:** 42

## Data
- Train rows: 500,000 (['2025-01', '2025-05']); pos_ratio=0.0501
- OOT rows:   100,000 (['2025-06', '2025-06']); pos_ratio=0.0511
-alance tier: moderate
 Imb
## Feature selection
- In: 52 → hygiene: 52 → univariate: 47 → redundancy: 40 → **selected: 40**
- Top selected features (real names): segment, total_credit_amt_1m, net_flow_1m, avg_txn_size_1m, mcc_travel_amt_1m, total_credit_amt_3m, txn_count_1m, total_debit_amt_1m, bounce_count_1m, bounce_amt_1m

## OOT performance (best model)
- precision: 0.0911
- recall: 0.6807
- f1: 0.1606
- roc_auc: 0.7223
- pr_auc: 0.1206
- ks: 0.3184
- gini: 0.4446

## Thresholds
- F1-optimal: 0.6597
- f1_optimal: 0.6597297751983545
- precision_50: None
- recall_80: 0.45872612147050146

## Reproducibility
Re-run `run_config.json` against data matching `data_hash.txt` with the same seed to reproduce this model byte-for-byte. No LLM is involved in reproduction.