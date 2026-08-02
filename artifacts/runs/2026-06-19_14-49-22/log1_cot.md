# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 0.26s
- `profile data` — 4.07s
- `research (models / windows / HPO / features)` — 8.51s
- `domain feature engineering (research families)` — 1.25s
- `model design` — 0.07s
- `prepare data (split/clean/select/encode)` — 21.64s
- `search round (HPO)` — 1631.97s
- `assess: continue or stop?` — 0.02s
- `finalize (refit + OOT eval + persist)` — 11.47s
- `write model summary` — 2.22s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle imbalanced datasets well. HistGradientBoosting is a strong alternative, often providing good performance with less tuning. These algorithms are well-suited for propensity modeling as described in various sources (e.g., Altexsoft, Comarch). LogisticRegression is a baseline but may struggle with complex interactions in imbalanced data. RandomForest and GradientBoosting are also viable but LightGBM and XGBoost often offer better performance and speed.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/windows] observation=5m, OOT=1 snapshot(s)
- **Why:** With 6 monthly snapshots, a common practice is to use the first 5 months for training and feature engineering, and the last month for out-of-time (OOT) validation. This allows for a realistic assessment of model performance on unseen, recent data, as suggested by the need for monthly snapshots.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. For imbalanced datasets, `scale_pos_weight` is essential for LightGBM and XGBoost. The provided ranges for `learning_rate` and `num_leaves`/`max_depth`/`max_leaf_nodes` are common starting points for these algorithms, as suggested by general ML best practices and the need for fine-tuning (Datatonic).
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [feature_engineering/families] Suggested families: ['Customer Demographics', 'Transaction History', 'Product Holdings', 'Interaction Data', 'Derived Features']
- **Why:** A comprehensive set of feature families is necessary for building a robust propensity model. These families cover key aspects of customer behavior and relationship with the bank, aligning with the principles of feature engineering for predictive modeling (Medium, Altexsoft, Faraday.ai).
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle imbalanced datasets well. HistGradientBoosting is a strong alternative, often providing good performance with less tuning. These algorithms are well-suited for propensity modeling as described in various sources (e.g., Altexsoft, Comarch). LogisticRegression is a baseline but may struggle with complex interactions in imbalanced data. RandomForest and GradientBoosting are also viable but LightGBM and XGBoost often offer better performance and speed.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** With 6 monthly snapshots, a common practice is to use the first 5 months for training and feature engineering, and the last month for out-of-time (OOT) validation. This allows for a realistic assessment of model performance on unseen, recent data, as suggested by the need for monthly snapshots.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is crucial for maximizing model performance. For imbalanced datasets, `scale_pos_weight` is essential for LightGBM and XGBoost. The provided ranges for `learning_rate` and `num_leaves`/`max_depth`/`max_leaf_nodes` are common starting points for these algorithms, as suggested by general ML best practices and the need for fine-tuning (Datatonic).
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://www.comarch.com/trade-and-services/loyalty-marketing/blog/ai-driven-propensity-modeling', 'https://faraday.ai/blog/choose-ml-tools-propensity-modeling', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']
