"""Stage 6 - Training prompt (constrained to Model Design's shortlist)."""
from .system_rules import SYSTEM_RULES

TRAINING_PROMPT = SYSTEM_RULES + """
====================== STAGE 6: TRAINING ======================

READS FROM NS:
  X_train, y_train (or NS['X_train'], NS['y_train'])
  NS['model_design']['algorithm_shortlist']        : ONLY train these
  NS['model_design']['data_strategy']['imbalance_tier']
  NS['model_design']['expected_metric_targets']['primary_metric']

WRITES TO NS:
  NS['fitted_models']   : dict {name: fitted estimator}
  NS['cv_results']      : list of dicts
  NS['imbalance_info']  : dict from mlkit.metrics.detect_imbalance_tier

DISPLAY INLINE:
  - "TRAINING" banner
  - Imbalance tier + scoring used
  - One progress line per model with CV score
  - Score diversity check

STEPS

0. INPUT CONTRACT CHECK (DO NOT touch df_train or feature_cols - use X_train only):
    X_train = np.asarray(X_train, dtype=np.float32)
    assert X_train.dtype != object
    assert len(y_train) == X_train.shape[0]

1. DETECT IMBALANCE via mlkit:
    imb = metrics.detect_imbalance_tier(y_train)
    NS['imbalance_info'] = imb
    Print f"Imbalance tier: {imb['tier']}, pos_ratio={imb.get('pos_ratio')}"

2. READ MODEL_DESIGN constraints:
    shortlist = NS['model_design']['algorithm_shortlist']
    primary_metric = NS['model_design']['expected_metric_targets']['primary_metric']
    metric_map = {
        'PR_AUC':    'average_precision', 'ROC_AUC': 'roc_auc',
        'F1':        'f1',                'Recall':  'recall',
        'Precision': 'precision',         'Accuracy':'accuracy',
        'R2':        'r2',                'KS':      'roc_auc',
    }
    scoring = metric_map.get(primary_metric, 'roc_auc')

3. BUILD ONLY THE SHORTLISTED MODELS:
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
    Print f"Training {len(models)} models from Model Design shortlist: {list(models)}"

4. CV STRATEGY based on tier:
    n_splits = 3 if imb['tier'] in ('severe', 'extreme') else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

5. TRAIN with deepcopy (distinct objects):
    fitted_models = {}
    cv_results = []
    for name, mdl in models.items():
        Print f" Training {name} ..."
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
        Print f"        -> cv_{primary_metric}: {scores.mean():.4f} ± {scores.std():.4f}"

6. SANITY CHECKS:
    ids = set(id(m) for m in fitted_models.values())
    assert len(ids) == len(fitted_models), "Duplicate fitted models!"
    probs = [m.predict_proba(X_train[:300])[:,1] for m in fitted_models.values()
             if hasattr(m, 'predict_proba')]
    if len(probs) >= 2:
        diff = float(np.abs(probs[0] - probs[1]).mean())
        Print f" Score diversity: |p0-p1|={diff:.6f} {'⚠️ possible leakage' if diff<1e-4 else '✅ distinct'}"

7. SAVE & ASSIGN:
    json.dump({'cv_results': cv_results, 'imbalance_info': imb},
              open(DIRS['models']+'/cv_results.json','w'),
              indent=2, default=str)
    NS['fitted_models'] = fitted_models
    NS['cv_results']    = cv_results
    Print f"Training complete - {len(fitted_models)} models fitted"

MUST ASSIGN: fitted_models, cv_results, NS['fitted_models'],
            NS['cv_results'], NS['imbalance_info']
"""