# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 0.12s
- `profile data` — 3.63s
- `research (models / windows / HPO / features)` — 11.16s
- `domain feature engineering (research families)` — 1.11s
- `model design` — 0.07s
- `prepare data (split/clean/select/encode)` — 21.63s
- `search round (HPO)` — 1282.91s
- `assess: continue or stop?` — 0.01s
- `finalize (refit + OOT eval + persist)` — 9.66s
- `write model summary` — 1.71s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are chosen for their efficiency and performance on tabular data, especially with moderate imbalance. HistGradientBoosting is a strong contender for its robustness and speed. LogisticRegression is a good baseline but may struggle with complex interactions. RandomForest and GradientBoosting are also viable but often outperformed by tree-based gradient boosting methods in terms of speed and accuracy for this type of problem. [https://www.altexsoft.com/blog/propensity-model, https://antonsruberts.github.io/propensity-model, https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/windows] observation=5m, OOT=1 snapshot(s)
- **Why:** Using 5 months for observation and 1 month for out-of-time (OOT) validation provides a reasonable balance for capturing trends and evaluating model performance on unseen, recent data, given the availability of 6 monthly snapshots. This allows for sufficient historical context while ensuring the OOT period is representative of current customer behavior. [https://www.altexsoft.com/blog/propensity-model]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. The specified spaces cover key parameters for LightGBM, XGBoost, and HistGradientBoosting. Feature engineering should precede hyperparameter tuning to ensure the model learns from meaningful features. [https://www.altexsoft.com/blog/propensity-model, https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [feature_engineering/families] Suggested families: ['Demographics', 'Transaction History', 'Existing Products', 'Credit Bureau Data', 'Customer Interactions']
- **Why:** A comprehensive set of feature families is essential for building a robust propensity model. These families cover various aspects of customer behavior and attributes that are likely to influence credit card application decisions. [https://www.altexsoft.com/blog/propensity-model, https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data, https://www.oajaiml.com/uploads/archivepdf/237151184.pdf]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are chosen for their efficiency and performance on tabular data, especially with moderate imbalance. HistGradientBoosting is a strong contender for its robustness and speed. LogisticRegression is a good baseline but may struggle with complex interactions. RandomForest and GradientBoosting are also viable but often outperformed by tree-based gradient boosting methods in terms of speed and accuracy for this type of problem. [https://www.altexsoft.com/blog/propensity-model, https://antonsruberts.github.io/propensity-model, https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** Using 5 months for observation and 1 month for out-of-time (OOT) validation provides a reasonable balance for capturing trends and evaluating model performance on unseen, recent data, given the availability of 6 monthly snapshots. This allows for sufficient historical context while ensuring the OOT period is representative of current customer behavior. [https://www.altexsoft.com/blog/propensity-model]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. The specified spaces cover key parameters for LightGBM, XGBoost, and HistGradientBoosting. Feature engineering should precede hyperparameter tuning to ensure the model learns from meaningful features. [https://www.altexsoft.com/blog/propensity-model, https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c]
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']
