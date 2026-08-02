# Model Card — Build CC propensity model

- **Primary metric:** PR_AUC
- **Best algorithm:** XGBoost (trial 11)
- **Trials run:** 20
- **Data hash:** `5b4d0cd41241f005…`
- **Seed:** 42

## Data
- Train rows: 4,800 (['2025-01', '2025-04']); pos_ratio=0.0806
- OOT rows:   2,400 (['2025-05', '2025-06']); pos_ratio=0.09
- Imbalance tier: moderate

## Feature selection
- In: 41 → hygiene: 39 → univariate: 24 → redundancy: 24 → **selected: 24**

## OOT performance (best model)
- precision: 0.9070
- recall: 0.7222
- f1: 0.8041
- roc_auc: 0.9867
- pr_auc: 0.9173
- ks: 0.9226
- gini: 0.9733

## Thresholds
- F1-optimal: 0.0856
- f1_optimal: 0.08560071545392307
- precision_50: 0.001984062455200494
- recall_80: 0.2533600229237771

## Reproducibility
Re-run `run_config.json` against data matching `data_hash.txt` with the same seed to reproduce this model byte-for-byte. No LLM is involved in reproduction.