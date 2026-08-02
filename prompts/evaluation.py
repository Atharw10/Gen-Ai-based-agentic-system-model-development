"""Stage 7 - Evaluation prompt (on OOT, adaptive metric)."""
from .system_rules import SYSTEM_RULES

EVALUATION_PROMPT = SYSTEM_RULES + """

# ========= STAGE 7: EVALUATION (on OOT) =========

# READS FROM NS:
#   X_oot, y_oot, fitted_models
#   business_objective
#   NS['imbalance_info']
#   NS['model_design']['expected_metric_targets']['primary_metric']

# WRITES TO NS:
#   NS['leaderboard']             : pd.DataFrame
#   NS['best_model'], NS['best_model_name']
#   NS['optimal_threshold']        : float (F1-optimal)
#   NS['business_thresholds']     : dict {f1, precision_50, recall_80}
#   NS['test_predictions']        : pd.DataFrame
#   NS['lift_table']              : pd.DataFrame
#   NS['metric_choice']           : dict from mlkit.metrics.select_primary_metric

# DISPLAY INLINE:
#   - "EVALUATION ON OOT" banner
#   - Metric selection rationale
#   - Leaderboard sorted by primary metric
#   - Threshold tuning summary table
#   - ROC / PR / Confusion matrix plots
#   - Decile lift table

# STEPS

1. RESOLVE PRIMARY METRIC via mlkit:
imb = NS['imbalance_info']
m_choice = metrics.select_primary_metric(
    business_objective=NS.get('business_objective', ''),
    imbalance_tier=imb.get('tier', 'balanced'),
    is_binary=imb.get('is_binary', True)
)
primary = m_choice['primary']
print(f"=== METRIC SELECTION ===")
print(f"Objective: {NS.get('business_objective')}")
print(f"Imbalance: {imb.get('tier')}")
print(f"Primary:   {primary}")
print(f"Secondary: {m_choice['secondary']}")
print(f"Rationale: {m_choice['rationale']}")

2. SAFE PROBABILITY HELPER:
def _pos_score(mdl, X):
    if hasattr(mdl, 'predict_proba'):
        p = mdl.predict_proba(X)
        if p.shape[1] == 1: return p[:, 0]
        cls = list(getattr(mdl, 'classes_', [0, 1]))
        return p[:, cls.index(1)] if 1 in cls else p[:, 1]
    return mdl.decision_function(X) if hasattr(mdl, 'decision_function') else mdl.predict(X)

3. BUILD LEADERBOARD on OOT:
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             RocCurveDisplay, PrecisionRecallDisplay)

rows = []
for name, mdl in NS['fitted_models'].items():
    y_pred = mdl.predict(X_oot)
    y_score = _pos_score(mdl, X_oot)
    row = {
        'model':          name,
        'Accuracy':       accuracy_score(y_oot, y_pred),
        'Precision':      precision_score(y_oot, y_pred, zero_division=0),
        'Recall':         recall_score(y_oot, y_pred, zero_division=0),
        'F1':             f1_score(y_oot, y_pred, zero_division=0),
    }
    if np.ndim(y_score) == 1 and len(np.unique(y_oot)) == 2:
        row['ROC_AUC'] = roc_auc_score(y_oot, y_score)
        row['PR_AUC']  = average_precision_score(y_oot, y_score)
        row['KS']      = stats.ks_statistic(y_oot, y_score)
        row['Gini']    = stats.gini_coefficient(y_oot, y_score)
    rows.append(row)
leaderboard = pd.DataFrame(rows)

4. RANK BY PRIMARY METRIC (with fallback):
for col in [primary, m_choice['secondary'], 'F1', 'ROC_AUC', 'Accuracy']:
    if col in leaderboard.columns:
        leaderboard = leaderboard.sort_values(col, ascending=False).reset_index(drop=True)
        sort_key = col
        break

print(f"\n=== LEADERBOARD (ranked by {sort_key}) ===")
print(leaderboard.round(4).to_string())
best_model_name = leaderboard.iloc[0]['model']
best_model = NS['fitted_models'][best_model_name]
print(f"\n🎯 Best model: {best_model_name}")

5. THRESHOLD TUNING (binary + tier != 'balanced'):
optimal_threshold = 0.5
business_thresholds = {'f1_optimal': 0.5, 'precision_50': None, 'recall_80': None}
if imb.get('is_binary') and imb.get('tier') != 'balanced':
    y_score = _pos_score(best_model, X_oot)
    if np.ndim(y_score) == 1:
        prec, rec, thr = precision_recall_curve(y_oot, y_score)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.argmax(f1s[:-1]))
        optimal_threshold = float(thr[best_idx])
        p50_idx = np.where(prec[:-1] >= 0.5)[0]
        p50_thr = float(thr[p50_idx[0]]) if len(p50_idx) else None
        r80_idx = np.where(rec[:-1] >= 0.8)[0]
        r80_thr = float(thr[r80_idx[-1]]) if len(r80_idx) else None
        business_thresholds = {
            'f1_optimal': optimal_threshold,
            'precision_50': p50_thr, 
            'recall_80': r80_thr
        }
        print("\nThreshold tuning:")
        print(f"  Default (0.5)      F1={f1_score(y_oot, (y_score>=0.5).astype(int), zero_division=0):.4f}")
        print(f"  Precision >= 0.5 thr = {p50_thr}")
        print(f"  Recall    >= 0.8 thr = {r80_thr}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(thr, prec[:-1], label='Precision')
        ax.plot(thr, rec[:-1], label='Recall')
        ax.plot(thr, f1s[:-1], label='F1', linestyle='--')
        ax.axvline(optimal_threshold, color='red', linestyle=':',
                   label=f'Optimal={optimal_threshold:.3f}')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
        ax.set_title(f"Threshold Tuning ({imb.get('tier')} imbalance)")
        ax.legend(); ax.grid(True)
        plt.tight_layout()

6. CORE PLOTS - only when binary & both classes present:
if imb.get('is_binary') and (y_oot == 1).any() and (y_oot == 0).any():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    try: RocCurveDisplay.from_estimator(best_model, X_oot, y_oot, ax=axes[0])
    except Exception as e: axes[0].text(0.1, 0.5, f"ROC: {e}")
    axes[0].set_title('ROC')
    try: PrecisionRecallDisplay.from_estimator(best_model, X_oot, y_oot, ax=axes[1])
    except Exception as e: axes[1].text(0.1, 0.5, f"PR: {e}")
    axes[1].set_title("PR (more honest for imbalance)" if imb.get('tier') != 'balanced' else 'PR')
    y_score = _pos_score(best_model, X_oot)
    if np.ndim(y_score) == 1 and imb.get('tier') != 'balanced':
        y_pred_tuned = (y_score >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_oot, y_pred_tuned)
        ax_title = f'Confusion @ thr={optimal_threshold:.3f}'
    else:
        cm = confusion_matrix(y_oot, best_model.predict(X_oot))
        ax_title = 'Confusion @ thr=0.5'
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title(ax_title)
    axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')
    plt.tight_layout()

7. LIFT TABLE (banking standard):
if imb.get('is_binary'):
    y_score = _pos_score(best_model, X_oot)
    if np.ndim(y_score) == 1:
        lift = reporting.lift_table(y_oot, y_score, n_bins=10)
        print("\n=== DECILE LIFT TABLE ===")
        print(lift.to_string())
        lift.to_csv(DIRS['eval']+'/lift_table.csv')
        NS['lift_table'] = lift

8. SAVE PREDICTIONS & ARTIFACTS:
if imb.get('is_binary'):
    y_score = _pos_score(best_model, X_oot)
    test_pred = pd.DataFrame({'prediction_default': best_model.predict(X_oot)})
    if np.ndim(y_score) == 1:
        test_pred['score'] = y_score
        test_pred['prediction_tuned'] = (y_score >= optimal_threshold).astype(int)
        test_pred['threshold_used'] = optimal_threshold
else:
    test_pred = pd.DataFrame({'prediction': best_model.predict(X_oot)})
test_pred.to_csv(DIRS['eval']+'/test_predictions.csv', index=False)
leaderboard.to_csv(DIRS['eval']+'/leaderboard.csv', index=False)

eval_meta = {
    'tier':                imb.get('tier'),
    'primary_metric':      primary,
    'secondary_metric':    m_choice['secondary'],
    'rationale':           m_choice['rationale'],
    'sort_key':            sort_key,
    'best_model':          best_model_name,
    'optimal_threshold':   optimal_threshold,
    'business_thresholds': business_thresholds,
}
json.dump(eval_meta, open(DIRS['eval']+'/eval_meta.json', 'w'),
          indent=2, default=str)

9. ASSIGN TO NS:
NS['leaderboard']         = leaderboard
NS['best_model']          = best_model
NS['best_model_name']     = best_model_name
NS['optimal_threshold']    = optimal_threshold
NS['business_thresholds'] = business_thresholds
NS['test_predictions']    = test_pred
NS['metric_choice']       = m_choice
print("\nEvaluation complete")

# MUST ASSIGN: leaderboard, best_model, best_model_name, test_pred,
#              optimal_threshold, NS['leaderboard'], NS['best_model'],
#              NS['best_model_name'], NS['optimal_threshold'],
#              NS['business_thresholds'], NS['test_predictions'],
#              NS['metric_choice']
"""

