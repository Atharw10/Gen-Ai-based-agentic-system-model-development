# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 0.11s
- `profile data` — 3.7s
- `research (models / windows / HPO / features)` — 31.8s
- `domain feature engineering (research families)` — 2.33s
- `model design` — 0.11s
- `prepare data (split/clean/select/encode)` — 28.27s
- `search round (HPO)` — 1352.36s
- `assess: continue or stop?` — 10.99s
- `strategist: revise algorithm budget` — 3.19s
- `search round (HPO)` — 1642.75s
- `assess: continue or stop?` — 0.01s
- `finalize (refit + OOT eval + persist)` — 20.85s
- `write model summary` — 3.5s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'LogisticRegression']
- **Why:** LightGBM and XGBoost are chosen for their efficiency and performance on tabular data, especially with imbalanced datasets. Logistic Regression is included as a baseline and for its interpretability, which is crucial in banking. (Cited: https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data, https://www.oajaiml.com/uploads/archivepdf/237151184.pdf)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/windows] observation=3m, OOT=1 snapshot(s)
- **Why:** A 3-month observation window is chosen to capture recent customer behavior, while a 1-month out-of-time (OOT) window is used for robust model evaluation and to simulate real-world deployment. This balances capturing sufficient historical data with reflecting current trends. (Cited: https://www.altexsoft.com/blog/propensity-model)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost']
- **Why:** Hyperparameter optimization is critical for maximizing model performance. The spaces for LightGBM and XGBoost are defined to explore key parameters that influence model complexity and learning speed. Feature engineering should precede HPO for better results. (Cited: https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [feature_engineering/families] Suggested families: ['Demographics', 'Transaction_History', 'Existing_Products', 'Credit_Behavior', 'Digital_Engagement']
- **Why:** A comprehensive set of feature families is crucial for building an accurate propensity model. These families cover various aspects of customer behavior and profile, providing a rich dataset for model training. (Cited: https://www.altexsoft.com/blog/propensity-model, https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data, https://www.oajaiml.com/uploads/archivepdf/237151184.pdf)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'LogisticRegression']
- **Why:** LightGBM and XGBoost are chosen for their efficiency and performance on tabular data, especially with imbalanced datasets. Logistic Regression is included as a baseline and for its interpretability, which is crucial in banking. (Cited: https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data, https://www.oajaiml.com/uploads/archivepdf/237151184.pdf)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** A 3-month observation window is chosen to capture recent customer behavior, while a 1-month out-of-time (OOT) window is used for robust model evaluation and to simulate real-world deployment. This balances capturing sufficient historical data with reflecting current trends. (Cited: https://www.altexsoft.com/blog/propensity-model)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost']
- **Why:** Hyperparameter optimization is critical for maximizing model performance. The spaces for LightGBM and XGBoost are defined to explore key parameters that influence model complexity and learning speed. Feature engineering should precede HPO for better results. (Cited: https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c)
- **Source:** Gemini + web: ['https://www.altexsoft.com/blog/propensity-model', 'https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://www.artefact.com/blog/scoring-customer-propensity-using-machine-learning-models-on-google-analytics-data', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c']
