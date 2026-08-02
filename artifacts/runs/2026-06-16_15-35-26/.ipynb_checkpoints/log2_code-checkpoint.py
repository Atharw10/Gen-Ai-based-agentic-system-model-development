# Generated code - run 2026-06-16_15-35-26


# ================= eda (attempt 1) =================
print("## EDA RESULTS")

# 1. Build a TRAINING-ONLY subset
train_snaps = NS['model_design']['windows']['snapshots_for_training']
df_eda = df_train[
    pd.to_datetime(df_train[date_col]).dt.to_period('M').astype(str).isin(train_snaps)
].copy()
print(f"EDA restricted to training snapshots: {train_snaps}")
print(f"EDA sample shape: {df_eda.shape}")

# 2. PER-SNAPSHOT TARGET RATE
target_rate_by_snapshot = df_eda.groupby(pd.to_datetime(df_eda[date_col]).dt.to_period('M').astype(str))[target_col].agg(['count', 'sum', 'mean'])
target_rate_by_snapshot.columns = ['n_customers', 'n_positive', 'target_rate']
target_rate_by_snapshot = target_rate_by_snapshot.reset_index()
target_rate_by_snapshot.rename(columns={'index': 'snapshot'}, inplace=True)
print("\nTarget Rate by Snapshot:")
print(target_rate_by_snapshot.to_markdown(index=False))
target_rate_by_snapshot.to_csv(os.path.join(DIRS['eda'], 'target_rate_by_snapshot.csv'), index=False)

# 3. TOP CORRELATIONS WITH TARGET
# Identify potential numeric feature columns
numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
# Exclude target and date columns, and potential ID columns
potential_id_cols = [col for col in df_eda.columns if 'id' in col.lower() and col != target_col]
feature_cols = [col for col in numeric_cols if col not in [target_col, date_col] + potential_id_cols]

if feature_cols:
    correlations = df_eda[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    top_correlations = correlations.drop(target_col).head(15).reset_index()
    top_correlations.columns = ['feature', 'abs_correlation']
    print("\nTop 15 Features by Absolute Correlation with Target:")
    print(top_correlations.to_markdown(index=False))
    top_correlations.to_csv(os.path.join(DIRS['eda'], 'top_correlations.csv'), index=False)
else:
    top_correlations = pd.DataFrame()
    print("\nNo numeric features found for correlation analysis.")

# 4. NUMERIC FEATURE DISTRIBUTIONS (max 3 plots)
features_for_plots = top_correlations['feature'].tolist()[:3]
if features_for_plots:
    print("\nNumeric Feature Distributions:")
    for feature in features_for_plots:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(data=df_eda, x=feature, hue=target_col, ax=ax, kde=True, common_norm=False)
        ax.set_title(f"Distribution of {feature} by Target")
        # plt.show() # Orchestrator handles display

# 5. MISSINGNESS SUMMARY
missingness = df_eda.isnull().sum()
missing_pct = (missingness / len(df_eda)) * 100
top_missing = missing_pct.sort_values(ascending=False).head(10)
print("\nTop 10 Columns by Missingness Percentage:")
print(top_missing.to_markdown())

# 6. BUILD eda_report DICT
eda_report = {
    'eda_sample_rows': int(len(df_eda)),
    'training_snapshots_used': train_snaps,
    'target_rate_by_snapshot': target_rate_by_snapshot.to_dict('records'),
    'top_correlations': top_correlations.to_dict('records') if not top_correlations.empty else [],
    'top_missing': top_missing.to_dict(),
}

# 7. SAVE & ASSIGN
report_path = os.path.join(DIRS['eda'], 'eda_report.json')
with open(report_path, 'w') as f:
    json.dump(eda_report, f, indent=2, default=str)
NS['eda_report'] = eda_report
print("\nEDA complete")


# ================= cleaning (attempt 1) =================
# 1. CAPTURE BEFORE STATE
rows_train_before = int(df_train.shape[0])
nulls_train_before = int(df_train.isna().sum().sum())
dupes_train_before = int(df_train.duplicated().sum())

if df_test is not None:
    rows_test_before = int(df_test.shape[0])
    nulls_test_before = int(df_test.isna().sum().sum())
else:
    rows_test_before = 0
    nulls_test_before = 0

# 2. READ STRATEGY
ds = NS['model_design']['data_strategy']
imp_num = ds.get('imputation_numeric', 'median')
imp_cat = ds.get('imputation_categorical', 'mode')
outlier_str = ds.get('outlier_strategy', 'winsorize_99')

# 3. DROP HIGH-MISSING COLUMNS
null_pct = df_train.isna().mean()
dropped_cols = null_pct[null_pct > 0.70].index.tolist()

if dropped_cols:
    df_train = df_train.drop(columns=dropped_cols)
    if df_test is not None:
        df_test = df_test.drop(columns=[c for c in dropped_cols if c in df_test.columns])
    print(f"Dropped {len(dropped_cols)} high-missing columns: {dropped_cols}")
else:
    dropped_cols = []

# 4. IMPUTE
df_train, imp_map = cleaning.smart_impute(
    df_train, num_strategy=imp_num, cat_strategy=imp_cat, target_col=target_col)

if df_test is not None:
    df_test = cleaning.apply_imputation(df_test, imp_map)

print(f"Imputation applied: {len(imp_map)} columns ({imp_num}/{imp_cat})")
NS['imputation_map'] = imp_map

# 5. OUTLIER CAPPING
outlier_bounds = {}
if outlier_str == 'winsorize_99':
    for col in df_train.select_dtypes(include=[np.number]).columns:
        if col == target_col: continue
        capped, lo, hi = cleaning.winsorize(df_train[col], lo_pct=0.01, hi_pct=0.99)
        if lo is not None and hi is not None:
            df_train[col] = capped
            outlier_bounds[col] = (lo, hi)
            if df_test is not None and col in df_test.columns:
                df_test[col] = df_test[col].clip(lo, hi)
elif outlier_str == 'iqr_safe':
    for col in df_train.select_dtypes(include=[np.number]).columns:
        if col == target_col: continue
        capped, lo, hi = cleaning.safe_iqr_cap(df_train[col])
        if lo is not None and hi is not None:
            df_train[col] = capped
            outlier_bounds[col] = (lo, hi)
            if df_test is not None and col in df_test.columns:
                df_test[col] = df_test[col].clip(lo, hi)

print(f"Outlier capping ({outlier_str}): {len(outlier_bounds)} columns capped")
NS['outlier_bounds'] = outlier_bounds

# 6. DROP DUPLICATES FROM df_train ONLY
train_dupes_removed = int(df_train.duplicated().sum())
df_train = df_train.drop_duplicates().reset_index(drop=True)
print(f"Dropped {train_dupes_removed} duplicates from df_train. New shape: {df_train.shape}")

# 7. BUILD cleaning_report dict
cleaning_report = {
    'dropped_columns':          dropped_cols,
    'rows_train_before':        rows_train_before,
    'rows_train_after':         int(df_train.shape[0]),
    'rows_test_before':         rows_test_before,
    'rows_test_after':          int(df_test.shape[0]) if df_test is not None else 0,
    'nulls_train_before':       nulls_train_before,
    'nulls_train_after':        int(df_train.isna().sum().sum()),
    'nulls_test_before':        nulls_test_before,
    'nulls_test_after':         int(df_test.isna().sum().sum()) if df_test is not None else 0,
    'train_duplicates_removed': train_dupes_removed,
    'imputation_strategy':      f"{imp_num}/{imp_cat}",
    'outlier_strategy':         outlier_str,
    'imputation_columns':       list(imp_map.keys()),
    'outlier_columns':          list(outlier_bounds.keys()),
    'df_test_provided':         df_test is not None,
}

# 8. DISPLAY a small before/after table
summary_df = pd.DataFrame({
    'metric': ['rows_train', 'nulls_train', 'duplicates_train'],
    'before': [rows_train_before, nulls_train_before, dupes_train_before],
    'after':  [df_train.shape[0], int(df_train.isna().sum().sum()), 0],
})
print("\n" + summary_df.to_string(index=False))

# 9. SAVE
json.dump(cleaning_report, open(os.path.join(DIRS['clean'], 'cleaning_report.json'), 'w'),
          indent=2, default=str)
NS['cleaning_report'] = cleaning_report
NS['df_train'] = df_train
NS['df_test'] = df_test
print("Cleaning complete")


# ================= feature_engineering (attempt 1) =================
print("====================== STAGE 5: FEATURE ENGINEERING ======================")

# 1. ENGINEER DERIVED FEATURES via mlkit
df_train, created_feats, column_map = features.engineer_propensity_features(
    df_train, column_map=None, llm=NS['llm'])
if df_test is not None:
    df_test, _, _ = features.engineer_propensity_features(
       df_test, column_map=column_map, llm=None)
NS['column_map'] = column_map
json.dump(column_map, open(DIRS['fe']+'/column_map.json','w'),
          indent=2, default=str)

print("\n--- Column Mapping ---")
print(pd.DataFrame(list(column_map.items()), columns=['Concept', 'Column Name']).to_string())

# 2. RECOMPUTE COLUMN LISTS
feature_cols_now = [c for c in df_train.columns
                    if c not in (target_col, date_col)]
numeric_cols     = df_train[feature_cols_now].select_dtypes(include=['number']).columns.tolist()
categorical_cols = [c for c in feature_cols_now if c not in numeric_cols]

# 3. DROP CONSTANTS + HIGH-CARDINALITY using mlkit
df_train, hc_dropped = features.drop_high_cardinality(
    df_train, max_unique=50, exclude=[target_col, date_col])
if df_test is not None:
    df_test = df_test.drop(columns=[c for c, _ in hc_dropped if c in df_test.columns])
feature_cols_now = [c for c in df_train.columns if c not in (target_col, date_col)]
numeric_cols     = df_train[feature_cols_now].select_dtypes(include=['number']).columns.tolist()
categorical_cols = [c for c in feature_cols_now if c not in numeric_cols]
print(f"\nDropped {len(hc_dropped)} high-cardinality cols")

# 4. PSI ANALYSIS across training snapshots
ds = NS['model_design']['data_strategy']
psi_threshold = ds.get('psi_threshold', 0.25)
psi_df = stats.compute_psi_table(df_train, date_col,
                                 feature_cols=numeric_cols + categorical_cols,
                                 target_col=target_col)
print("\n--- PSI Report ---")
print(psi_df.to_string())
psi_df.to_csv(DIRS['fe']+'/psi_report.csv', index=False)
reporting.psi_chart(psi_df)

unstable = psi_df[psi_df['PSI'] > psi_threshold]['feature'].tolist()
if unstable:
    print(f"⚠️ Auto-dropping {len(unstable)} unstable features: {unstable}")
    df_train = df_train.drop(columns=unstable)
    if df_test is not None:
       df_test  = df_test.drop(columns=[c for c in unstable if c in df_test.columns])
    numeric_cols     = [c for c in numeric_cols     if c not in unstable]
    categorical_cols = [c for c in categorical_cols if c not in unstable]
NS['psi_report'] = psi_df

# 5. IV RANKING (info only – do NOT auto-drop)
iv_df = stats.compute_iv_table(df_train, target_col,
                               feature_cols=numeric_cols + categorical_cols)
print("\n--- IV Report (Top 20) ---")
print(iv_df.head(20).to_string())
iv_df.to_csv(DIRS['fe']+'/iv_report.csv', index=False)
reporting.iv_chart(iv_df)

if (iv_df['IV'] > 0.5).any():
    leaky_features = iv_df[iv_df['IV'] > 0.5]['feature'].tolist()
    print(f"⚠️ WARNING: Potential data leakage detected in features with IV > 0.5: {leaky_features}")
NS['iv_report'] = iv_df

# 6. CORRELATION PRUNING (mlkit handles the logic)
corr_thresh = ds.get('correlation_threshold', 0.85)
kept_cols, pairs_dropped = features.drop_correlated_features(
    df_train, feature_cols=numeric_cols, target_col=target_col,
    threshold=corr_thresh)
dropped_by_corr = [d[3] for d in pairs_dropped]
if dropped_by_corr:
    df_train = df_train.drop(columns=dropped_by_corr)
    if df_test is not None:
       df_test  = df_test.drop(columns=[c for c in dropped_by_corr if c in df_test.columns])
    numeric_cols = kept_cols
print(f"\nCorrelation pruning: dropped {len(dropped_by_corr)} features")

# 7. TEMPORAL SPLIT using mlkit
oot_start = NS['model_design']['windows']['oot_snapshots'][0] + '-01'
df_tr, df_oot, split_info = splits.temporal_split(
    df_train, date_col=date_col, oot_start=oot_start)
print(f"\nTemporal split: train={df_tr.shape}, oot={df_oot.shape}")
print(f"  train range: {split_info['train_range']}")
print(f"  oot range:   {split_info['oot_range']}")

# 8. BUILD PREPROCESSOR – respects model_design['data_strategy']['scaling_required']
from sklearn.compose     import ColumnTransformer
from sklearn.pipeline    import Pipeline
from sklearn.impute      import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

scaling_required = ds.get('scaling_required', False)
num_steps = [('imp', SimpleImputer(strategy='median'))]
if scaling_required:
    num_steps.append(('sc', StandardScaler()))
numeric_pipe = Pipeline(num_steps)
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('oh',  OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                          max_categories=50, min_frequency=10))
])
final_num = [c for c in numeric_cols if c in df_tr.columns]
final_cat = [c for c in categorical_cols if c in df_tr.columns]
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, final_num),
    ('cat', cat_pipe,    final_cat),
], remainder='drop', verbose_feature_names_out=False)

# 9. TARGET ENCODE + FIT
y_train = df_tr[target_col].values
y_oot   = df_oot[target_col].values
if df_tr[target_col].dtype == object:
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train); y_oot = le.transform(y_oot)
    joblib.dump(le, DIRS['fe']+'/label_encoder.pkl')

preprocessor.fit(df_tr[final_num + final_cat])
X_train = np.asarray(preprocessor.transform(df_tr[final_num + final_cat]), dtype=np.float32)
X_oot   = np.asarray(preprocessor.transform(df_oot[final_num + final_cat]), dtype=np.float32)
feature_names = list(preprocessor.get_feature_names_out())

# 10. CONTRACT CHECKS (must pass)
assert isinstance(X_train, np.ndarray) and X_train.dtype != object
assert isinstance(X_oot,   np.ndarray) and X_oot.dtype != object
assert X_train.shape[1] == X_oot.shape[1] == len(feature_names)
assert len(np.unique(X_train, axis=0)) >= 100,         "X_train too degenerate – check cleaning"
print(f"\nFE contract OK: X_train={X_train.shape} X_oot={X_oot.shape}")

# 11. WoE TRANSFORMATION (if model_design said so)
woe_applied = False
if ds.get('woe_transformation', False):
    top_iv = iv_df.head(20)['feature'].tolist()
    top_iv = [c for c in top_iv if c in df_tr.columns]
    if top_iv:
        woe_df_tr, woe_maps = stats.woe_transform(df_tr, top_iv, target_col)
        joblib.dump(woe_maps, DIRS['fe']+'/woe_maps.pkl')
        print(f"WoE applied to {len(top_iv)} features")
        woe_applied = True
    else:
        print("WoE transformation requested but no suitable features found.")

# 12. SAVE & ASSIGN
fe_report = {
    'created_derived_features': created_feats,
    'column_map':               column_map,
    'final_n_features':         int(len(feature_names)),
    'dropped_high_cardinality': [c for c, _ in hc_dropped],
    'dropped_psi_unstable':     unstable,
    'dropped_correlation':      [[p[0], p[1], p[2]] for p in pairs_dropped],
    'temporal_split':           split_info,
    'scaling_applied':          scaling_required,
    'woe_applied':              woe_applied,
}
json.dump(fe_report, open(DIRS['fe']+'/fe_report.json','w'),
          indent=2, default=str)

NS['X_train']       = X_train
NS['X_oot']         = X_oot
NS['y_train']       = y_train
NS['y_oot']         = y_oot
NS['preprocessor']  = preprocessor
NS['feature_names'] = feature_names
NS['fe_report']     = fe_report
NS['column_map']    = column_map # Redundant but explicit for contract

print("\nFeature engineering complete")


# ================= training (attempt 1) =================
print("TRAINING")

# 0. INPUT CONTRACT CHECK
X_train = np.asarray(X_train, dtype=np.float32)
assert X_train.dtype != object
assert len(y_train) == X_train.shape[0]

# 1. DETECT IMBALANCE
imb = metrics.detect_imbalance_tier(y_train)
NS['imbalance_info'] = imb
print(f"Imbalance tier: {imb['tier']}, pos_ratio={imb.get('pos_ratio')}")

# 2. READ MODEL_DESIGN constraints
shortlist = NS['model_design']['algorithm_shortlist']
primary_metric = NS['model_design']['expected_metric_targets']['primary_metric']
metric_map = {
    'PR_AUC':    'average_precision', 'ROC_AUC': 'roc_auc',
    'F1':        'f1',                'Recall':  'recall',
    'Precision': 'precision',         'Accuracy':'accuracy',
    'R2':        'r2',                'KS':      'roc_auc',
}
scoring = metric_map.get(primary_metric, 'roc_auc')

# 3. BUILD ONLY THE SHORTLISTED MODELS
from copy import deepcopy
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

use_cw = imb['tier'] != 'balanced'
cw     = 'balanced' if use_cw else None
spw    = imb.get('scale_pos_weight', 1.0)

model_factory = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=2000, n_jobs=-1, class_weight=cw),
    'RandomForest':       lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                         random_state=42, class_weight=cw),
    'XGBoost':            lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.08,
                                                    max_depth=6, tree_method='hist',
                                                    eval_metric='aucpr',
                                                    scale_pos_weight=spw if use_cw else 1.0,
                                                    n_jobs=-1, random_state=42),
    'LightGBM':           lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                                     num_leaves=63, n_jobs=-1,
                                                     random_state=42, class_weight=cw),
    'GradientBoosting':   lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
}
models = {}
for name in shortlist:
    if name in model_factory:
        models[name] = model_factory[name]()
    else:
        print(f"⚠️ Unknown algorithm '{name}' - skipping")
assert len(models) >= 1, "No valid algorithms in shortlist"
print(f"Training {len(models)} models from Model Design shortlist: {list(models)}")

# 4. CV STRATEGY based on tier
n_splits = 3 if imb['tier'] in ('severe', 'extreme') else 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 5. TRAIN with deepcopy
fitted_models = {}
cv_results = []
for name, mdl in models.items():
    print(f" Training {name} ...")
    m = deepcopy(mdl)
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    m.fit(X_train, y_train)
    fitted_models[name] = m
    cv_results.append({
        'model': name,
        f'cv_{primary_metric}_mean': float(scores.mean()),
        f'cv_{primary_metric}_std':  float(scores.std()),
        'scoring': scoring,
    })
    print(f"        -> cv_{primary_metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. SANITY CHECKS
ids = set(id(m) for m in fitted_models.values())
assert len(ids) == len(fitted_models), "Duplicate fitted models!"
probs = [m.predict_proba(X_train[:300])[:,1] for m in fitted_models.values()
         if hasattr(m, 'predict_proba')]
if len(probs) >= 2:
    diff = float(np.abs(probs[0] - probs[1]).mean())
    print(f" Score diversity: |p0-p1|={diff:.6f} {'⚠️ possible leakage' if diff<1e-4 else '✅ distinct'}")

# 7. SAVE & ASSIGN
json.dump({'cv_results': cv_results, 'imbalance_info': imb},
          open(os.path.join(DIRS['models'], 'cv_results.json'),'w'),
          indent=2, default=str)
NS['fitted_models'] = fitted_models
NS['cv_results']    = cv_results
print(f"Training complete - {len(fitted_models)} models fitted")


# ================= training (attempt 2) =================
# 0. INPUT CONTRACT CHECK
X_train = np.asarray(X_train, dtype=np.float32)
assert X_train.dtype != object, "X_train contains object dtype"
assert len(y_train) == X_train.shape[0], "Length of y_train does not match X_train rows"

# 1. DETECT IMBALANCE
imb = metrics.detect_imbalance_tier(y_train)
NS['imbalance_info'] = imb
print(f"Imbalance tier: {imb['tier']}, pos_ratio={imb.get('pos_ratio')}")

# 2. READ MODEL_DESIGN constraints
shortlist = NS['model_design']['algorithm_shortlist']
primary_metric = NS['model_design']['expected_metric_targets']['primary_metric']
metric_map = {
    'PR_AUC':    'average_precision', 'ROC_AUC': 'roc_auc',
    'F1':        'f1',                'Recall':  'recall',
    'Precision': 'precision',         'Accuracy':'accuracy',
    'R2':        'r2',                'KS':      'roc_auc',
}
scoring = metric_map.get(primary_metric, 'roc_auc')
print(f"Primary metric: {primary_metric}, using scoring: {scoring}")

# 3. BUILD ONLY THE SHORTLISTED MODELS
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

use_cw = imb['tier'] != 'balanced'
cw     = 'balanced' if use_cw else None
spw    = imb.get('scale_pos_weight', 1.0)

model_factory = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=2000, n_jobs=-1, class_weight=cw),
    'RandomForest':       lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                         random_state=42, class_weight=cw),
    'XGBoost':            lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.08,
                                                    max_depth=6, tree_method='hist',
                                                    eval_metric='aucpr',
                                                    scale_pos_weight=spw if use_cw else 1.0,
                                                    n_jobs=-1, random_state=42),
    'LightGBM':           lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                                     num_leaves=63, n_jobs=-1,
                                                     random_state=42, class_weight=cw),
    'GradientBoosting':   lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
}
models = {}
for name in shortlist:
    if name in model_factory:
        models[name] = model_factory[name]()
    else:
        print(f"⚠️ Unknown algorithm '{name}' - skipping")

assert len(models) >= 1, "No valid algorithms in shortlist"
print(f"Training {len(models)} models from Model Design shortlist: {list(models)}")

# 4. CV STRATEGY based on tier
n_splits = 3 if imb['tier'] in ('severe', 'extreme') else 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 5. TRAIN with deepcopy
fitted_models = {}
cv_results = []
for name, mdl in models.items():
    print(f" Training {name} ...")
    m = deepcopy(mdl)
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    m.fit(X_train, y_train)
    fitted_models[name] = m
    cv_results.append({
        'model': name,
        f'cv_{primary_metric}_mean': float(scores.mean()),
        f'cv_{primary_metric}_std':  float(scores.std()),
        'scoring': scoring,
    })
    print(f"        -> cv_{primary_metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. SANITY CHECKS
ids = set(id(m) for m in fitted_models.values())
assert len(ids) == len(fitted_models), "Duplicate fitted models!"

probs = []
for m in fitted_models.values():
    if hasattr(m, 'predict_proba'):
        try:
            probs.append(m.predict_proba(X_train[:300])[:,1])
        except Exception as e:
            print(f"Warning: Could not get predict_proba for {type(m).__name__}: {e}")

if len(probs) >= 2:
    diff = float(np.abs(probs[0] - probs[1]).mean())
    print(f" Score diversity: |p0-p1|={diff:.6f} {'⚠️ possible leakage' if diff<1e-4 else '✅ distinct'}")
elif len(probs) == 1:
    print(" Score diversity: Only one model supports predict_proba.")
else:
    print(" Score diversity: No models support predict_proba.")


# 7. SAVE & ASSIGN
json.dump({'cv_results': cv_results, 'imbalance_info': imb},
          open(os.path.join(DIRS['models'], 'cv_results.json'),'w'),
          indent=2, default=str)

NS['fitted_models'] = fitted_models
NS['cv_results']    = cv_results
print(f"Training complete - {len(fitted_models)} models fitted")

