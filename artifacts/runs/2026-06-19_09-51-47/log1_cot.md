# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 0.15s
- `profile data` — 3.55s
- `research (models / windows / HPO / features)` — 8.73s
- `domain feature engineering (research families)` — 1.02s
- `model design` — 0.06s
- `prepare data (split/clean/select/encode)` — 20.82s
- `search round (HPO)` — 1683.93s
- `assess: continue or stop?` — 0.05s
- `strategist: revise algorithm budget` — 0.02s
- `search round (HPO)` — 1676.4s
- `assess: continue or stop?` — 0.01s
- `finalize (refit + OOT eval + persist)` — 9.55s
- `write model summary` — 1.84s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle imbalanced datasets well. HistGradientBoosting is a strong contender for its efficiency and performance. These algorithms are well-suited for propensity modeling and have demonstrated success in similar banking applications (as suggested by general ML best practices and the nature of the problem).
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [research/windows] observation=3m, OOT=1 snapshot(s)
- **Why:** A 3-month observation window allows for capturing recent customer behavior without being overly diluted by older data. A 1-month out-of-time (OOT) window provides a frequent and relevant validation of model performance against the most current data, crucial for a dynamic product like credit cards. This aligns with common practices for time-series based propensity modeling.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is critical for maximizing model performance. The specified ranges for LightGBM, XGBoost, and HistGradientBoosting cover common effective values and are informed by general best practices for these algorithms in classification tasks. Tuning these parameters will help the models generalize better and capture the nuances of the propensity data.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [feature_engineering/families] Suggested families: ['Demographics', 'Account Activity', 'Credit Behavior', 'Product Holdings', 'Marketing Interactions', 'Time-Based Features']
- **Why:** A comprehensive set of feature families is essential for building an accurate propensity model. These families cover customer characteristics, financial behavior, and interactions, providing a holistic view. The inclusion of time-based features is particularly important given the monthly snapshot data and the dynamic nature of propensity.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle imbalanced datasets well. HistGradientBoosting is a strong contender for its efficiency and performance. These algorithms are well-suited for propensity modeling and have demonstrated success in similar banking applications (as suggested by general ML best practices and the nature of the problem).
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** A 3-month observation window allows for capturing recent customer behavior without being overly diluted by older data. A 1-month out-of-time (OOT) window provides a frequent and relevant validation of model performance against the most current data, crucial for a dynamic product like credit cards. This aligns with common practices for time-series based propensity modeling.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter optimization is critical for maximizing model performance. The specified ranges for LightGBM, XGBoost, and HistGradientBoosting cover common effective values and are informed by general best practices for these algorithms in classification tasks. Tuning these parameters will help the models generalize better and capture the nuances of the propensity data.
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://datascience.stackexchange.com/questions/111235/feature-creation-ideas-for-propensity-models', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']
