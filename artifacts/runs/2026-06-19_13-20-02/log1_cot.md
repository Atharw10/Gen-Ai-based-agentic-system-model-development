# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 0.09s
- `profile data` — 3.38s
- `research (models / windows / HPO / features)` — 9.8s
- `domain feature engineering (research families)` — 0.94s
- `model design` — 0.06s
- `prepare data (split/clean/select/encode)` — 19.63s
- `search round (HPO)` — 1252.02s
- `assess: continue or stop?` — 3.02s
- `strategist: revise algorithm budget` — 4.72s
- `search round (HPO)` — 1263.98s
- `assess: continue or stop?` — 0.01s
- `finalize (refit + OOT eval + persist)` — 9.57s
- `write model summary` — 1.52s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Gradient boosting models like LightGBM, XGBoost, and HistGradientBoosting are highly effective for tabular data and can handle moderate class imbalance well. They offer good performance and interpretability, crucial for banking applications. Logistic Regression is a good baseline but may struggle with complex interactions. RandomForest is also a strong contender but can be slower than gradient boosting methods for large datasets.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [research/windows] observation=5m, OOT=1 snapshot(s)
- **Why:** Using 5 months for observation and 1 month for out-of-time (OOT) validation provides a sufficient lookback period to capture customer behavior trends while ensuring a recent and relevant OOT set for performance evaluation, aligning with the monthly snapshot data.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. The provided spaces cover key parameters for LightGBM, XGBoost, and HistGradientBoosting, focusing on learning rate, tree complexity, and regularization. Feature engineering should precede HPO, as stated in https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [feature_engineering/families] Suggested families: ['Demographics', 'Transaction History', 'Existing Products & Services', 'Credit Bureau Data', 'Engagement Metrics']
- **Why:** A comprehensive set of feature families is essential for building a robust propensity model. These families cover demographic, behavioral, financial, and engagement aspects of customers, drawing from established practices in propensity modeling (e.g., https://www.altexsoft.com/blog/propensity-model, https://antonsruberts.github.io/propensity-model, https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data). Feature engineering will be critical to extract meaningful signals from these raw data points, as highlighted in https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Gradient boosting models like LightGBM, XGBoost, and HistGradientBoosting are highly effective for tabular data and can handle moderate class imbalance well. They offer good performance and interpretability, crucial for banking applications. Logistic Regression is a good baseline but may struggle with complex interactions. RandomForest is also a strong contender but can be slower than gradient boosting methods for large datasets.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** Using 5 months for observation and 1 month for out-of-time (OOT) validation provides a sufficient lookback period to capture customer behavior trends while ensuring a recent and relevant OOT set for performance evaluation, aligning with the monthly snapshot data.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. The provided spaces cover key parameters for LightGBM, XGBoost, and HistGradientBoosting, focusing on learning rate, tree complexity, and regularization. Feature engineering should precede HPO, as stated in https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://antonsruberts.github.io/propensity-model', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data']
