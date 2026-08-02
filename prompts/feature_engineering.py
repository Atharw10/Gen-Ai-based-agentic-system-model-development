"""Stage 5 - Feature Engineering prompt."""
from .system_rules import SYSTEM_RULES

FE_PROMPT = SYSTEM_RULES + """
====================== STAGE 5: FEATURE ENGINEERING ======================

READS FROM NS:
  df_train, df_test, target_col, date_col, llm
  NS['model_design']['windows']['oot_snapshots']     : for temporal split
  NS['model_design']['data_strategy']:
    - correlation_threshold   (default 0.85)
    - psi_threshold           (default 0.25, auto-drop above this)
    - woe_transformation       (bool)
    - scaling_required         (bool)

WRITES TO NS:
  NS['column_map']      : resolved {concept: column_name} dict
  NS['psi_report']      : DataFrame of PSI per feature
  NS['iv_report']       : DataFrame of IV per feature
  NS['fe_report']       : full report dict
  X_train, X_oot, y_train, y_oot   (also as NS keys)
  preprocessor, feature_names      (also as NS keys)

DISPLAY INLINE:
  - "FE RESULTS" banner
  - Column-mapping table
  - "Created N derived features: [...]"
  - PSI bar chart (color-coded by stability)
  - IV bar chart
  - Final shape confirmation

STEPS

1. ENGINEER DERIVED FEATURES via mlkit:
    df_train, created_feats, column_map = features.engineer_propensity_features(
        df_train, column_map=None, llm=NS['llm'])
    if df_test is not None:
        df_test, _, _ = features.engineer_propensity_features(
           df_test, column_map=column_map, llm=None)  
    NS['column_map'] = column_map
    json.dump(column_map, open(DIRS['fe']+'/column_map.json','w'),
              indent=2, default=str)

2. RECOMPUTE COLUMN LISTS:
    feature_cols_now = [c for c in df_train.columns 
                        if c not in (target_col, date_col)]
    numeric_cols     = df_train[feature_cols_now].select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in feature_cols_now if c not in numeric_cols]

3. DROP CONSTANTS + HIGH-CARDINALITY using mlkit:
    df_train, hc_dropped = features.drop_high_cardinality(
        df_train, max_unique=50, exclude=[target_col, date_col])
    if df_test is not None:
        df_test = df_test.drop(columns=[c for c, _ in hc_dropped if c in df_test.columns])
    feature_cols_now = [c for c in df_train.columns if c not in (target_col, date_col)]
    numeric_cols     = df_train[feature_cols_now].select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in feature_cols_now if c not in numeric_cols]
    Print f"Dropped {len(hc_dropped)} high-cardinality cols"

4. PSI ANALYSIS across training snapshots:
    ds = NS['model_design']['data_strategy']
    psi_threshold = ds.get('psi_threshold', 0.25)
    psi_df = stats.compute_psi_table(df_train, date_col,
                                     feature_cols=numeric_cols + categorical_cols,
                                     target_col=target_col)
    Print psi_df.to_string().Save to DIRS['fe']+'/psi_report.csv'.
    Generate bar chart: reporting.psi_chart(psi_df)
    AUTO-DROP features where PSI > psi_threshold:
    unstable = psi_df[psi_df['PSI'] > psi_threshold]['feature'].tolist()
    if unstable:
        print(f"⚠️ Auto-dropping {len(unstable)} unstable features: {unstable}")
        df_train = df_train.drop(columns=unstable)
        if df_test is not None:
           df_test  = df_test.drop(columns=[c for c in unstable if c in df_test.columns])
        numeric_cols     = [c for c in numeric_cols     if c not in unstable]
        categorical_cols = [c for c in categorical_cols if c not in unstable]
    NS['psi_report'] = psi_df

5. IV RANKING (info only – do NOT auto-drop):
    iv_df = stats.compute_iv_table(df_train, target_col,
                                   feature_cols=numeric_cols + categorical_cols)
    Print iv_df.head(20).to_string().Save to DIRS['fe']+'/iv_report.csv'.
    Generate bar chart: reporting.iv_chart(iv_df)
    Print warnings for any IV > 0.5 (potential leakage).
    NS['iv_report'] = iv_df

6. CORRELATION PRUNING (mlkit handles the logic):
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
    Print f"Correlation pruning: dropped {len(dropped_by_corr)} features"

7. TEMPORAL SPLIT using mlkit:
    oot_start = NS['model_design']['windows']['oot_snapshots'][0] + '-01'
    df_tr, df_oot, split_info = splits.temporal_split(
        df_train, date_col=date_col, oot_start=oot_start)
    Print f"Temporal split: train={df_tr.shape}, oot={df_oot.shape}"
    Print f"  train range: {split_info['train_range']}"
    Print f"  oot range:   {split_info['oot_range']}"

8. BUILD PREPROCESSOR – respects model_design['data_strategy']['scaling_required']:
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

9. TARGET ENCODE + FIT:
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

10. CONTRACT CHECKS (must pass):
    assert isinstance(X_train, np.ndarray) and X_train.dtype != object
    assert isinstance(X_oot,   np.ndarray) and X_oot.dtype != object
    assert X_train.shape[1] == X_oot.shape[1] == len(feature_names)
    assert len(np.unique(X_train, axis=0)) >= 100, \
        "X_train too degenerate – check cleaning"
    Print f"FE contract OK: X_train={X_train.shape} X_oot={X_oot.shape}"

11. WoE TRANSFORMATION (if model_design said so):
    if ds.get('woe_transformation', False):
        top_iv = iv_df.head(20)['feature'].tolist()
        top_iv = [c for c in top_iv if c in df_tr.columns]
        woe_df_tr, woe_maps = stats.woe_transform(df_tr, top_iv, target_col)
        joblib.dump(woe_maps, DIRS['fe']+'/woe_maps.pkl')
        print(f"WoE applied to {len(top_iv)} features")

12. SAVE & ASSIGN:
    joblib.dump(preprocessor, DIRS['fe']+'/preprocessor.pkl')
    fe_report = {
        'created_derived_features': created_feats,
        'column_map':               column_map,
        'final_n_features':         int(len(feature_names)),
        'dropped_high_cardinality': [c for c, _ in hc_dropped],
        'dropped_psi_unstable':     unstable,
        'dropped_correlation':      [[p[0], p[1], p[2]] for p in pairs_dropped],
        'temporal_split':           split_info,
        'scaling_applied':          scaling_required,
        'woe_applied':              ds.get('woe_transformation', False),
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
    Print "Feature engineering complete"

MUST ASSIGN: X_train, X_oot, y_train, y_oot, preprocessor, feature_names,
            fe_report, column_map, NS['column_map'], NS['psi_report'],
            NS['iv_report'], NS['fe_report'], NS['X_train'], NS['X_oot'],
            NS['y_train'], NS['y_oot'], NS['preprocessor'], NS['feature_names']
"""