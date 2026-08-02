# Run reasoning log — `log1_cot.md`

## Steps
- `alias columns (privacy shield)` — 2.4s
- `profile data` — 4.86s
- `research (models / windows / HPO / features)` — 12.89s
- `domain feature engineering (research families)` — 1.41s
- `model design` — 0.04s
- `prepare data (split/clean/select/encode)` — 25.47s
- `search round (HPO)` — 1750.09s
- `assess: continue or stop?` — 9.41s
- `strategist: revise algorithm budget` — 2.8s
- `search round (HPO)` — 1652.42s
- `assess: continue or stop?` — 0.0s
- `finalize (refit + OOT eval + persist)` — 9.21s

## Decisions & reasoning (with sources)
### [research/models] Recommended models: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle large datasets efficiently, often outperforming traditional methods. HistGradientBoosting is a strong contender for its speed and performance, especially with large datasets. These algorithms are well-suited for classification tasks with moderate imbalance and can capture complex non-linear relationships. LogisticRegression is a good baseline but may struggle with the complexity of the data. RandomForest and GradientBoosting are also viable but LightGBM and XGBoost generally offer better performance and scalability.
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [research/windows] observation=6m, OOT=1 snapshot(s)
- **Why:** Using 6 months of historical data for observation provides a sufficiently rich dataset to capture recent customer behavior and trends. A 1-month out-of-time (OOT) window is standard for evaluating model performance on the most recent, unseen data, ensuring the model's ability to generalize to current conditions. This aligns with the monthly snapshot approach.
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [research/hpo] HPO ranges proposed for: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter tuning is crucial for optimizing model performance, especially with imbalanced datasets. The proposed search spaces for LightGBM, XGBoost, and HistGradientBoosting cover key parameters that significantly influence model accuracy and generalization. Prioritizing feature engineering before HPO, as suggested by [https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c], ensures that the tuning process operates on a more informative feature set, leading to better results. This approach is also supported by general ML best practices [https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml].
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [feature_engineering/families] Suggested families: ['Demographics', 'Transaction History', 'Existing Products', 'Credit Bureau Data', 'Marketing Interactions', 'Behavioral Features']
- **Why:** A comprehensive set of feature families is essential for building a robust CC propensity model. By considering demographics, transaction history, existing products, credit bureau data, marketing interactions, and derived behavioral features, we can capture a wide range of customer attributes and behaviors that influence their likelihood of applying for a credit card. This multi-faceted approach to feature engineering is critical for addressing the moderate imbalance and achieving high predictive accuracy.
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/algorithms] Shortlist applied: ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** LightGBM and XGBoost are highly effective for tabular data and handle large datasets efficiently, often outperforming traditional methods. HistGradientBoosting is a strong contender for its speed and performance, especially with large datasets. These algorithms are well-suited for classification tasks with moderate imbalance and can capture complex non-linear relationships. LogisticRegression is a good baseline but may struggle with the complexity of the data. RandomForest and GradientBoosting are also viable but LightGBM and XGBoost generally offer better performance and scalability.
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/windows] train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05'], OOT=['2025-06']
- **Why:** Using 6 months of historical data for observation provides a sufficiently rich dataset to capture recent customer behavior and trends. A 1-month out-of-time (OOT) window is standard for evaluating model performance on the most recent, unseen data, ensuring the model's ability to generalize to current conditions. This aligns with the monthly snapshot approach.
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']

### [model_design/metric] Primary metric: PR_AUC
- **Why:** deterministic metric policy for objective + tier 'moderate'
- **Source:** deterministic policy

### [model_design/hpo] Using research HPO ranges for ['LightGBM', 'XGBoost', 'HistGradientBoosting']
- **Why:** Hyperparameter tuning is crucial for optimizing model performance, especially with imbalanced datasets. The proposed search spaces for LightGBM, XGBoost, and HistGradientBoosting cover key parameters that significantly influence model accuracy and generalization. Prioritizing feature engineering before HPO, as suggested by [https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c], ensures that the tuning process operates on a more informative feature set, leading to better results. This approach is also supported by general ML best practices [https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml].
- **Source:** Gemini + web: ['https://dzone.com/articles/hyperparameter-tuning-feature-engineering-ml', 'https://www.oajaiml.com/uploads/archivepdf/237151184.pdf', 'https://medium.com/data-science/why-you-should-do-feature-engineering-first-hyperparameter-tuning-second-as-a-data-scientist-334be5eb276c', 'https://datatonic.com/insights/propensity-modelling-tensorflow-cloud-ai', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11389638']
