"""Stage 8 - Output (artifact persistence) prompt."""
from .system_rules import SYSTEM_RULES

OUTPUT_PROMPT = SYSTEM_RULES + """

# ========= STAGE 8: OUTPUT =========

# READS FROM NS:
#   Everything produced upstream (preprocessor, best_model, model_design,
#   column_map, feature_names, optimal_threshold, leaderboard, etc.)

# WRITES TO NS:
#   NS['artifact']   : final model bundle
#   NS['summary']    : human-readable summary dict

# DISPLAY INLINE:
#   - "FINAL SUMMARY" banner
#   - Summary JSON
#   - List of saved artifacts with paths

# STEPS

# 1. BUILD ARTIFACT bundle:
artifact = {
    'preprocessor':       NS['preprocessor'],
    'model':              NS['best_model'],
    'model_name':         NS['best_model_name'],
    'feature_names':      NS['feature_names'],
    'column_map':         NS['column_map'],
    'model_design':       NS['model_design'],
    'optimal_threshold':  NS['optimal_threshold'],
    'business_thresholds':NS['business_thresholds'],
    'imbalance_info':     NS['imbalance_info'],
    'target_col':         target_col,
    'date_col':           date_col,
}
joblib.dump(artifact, DIRS['models']+'/best_model.joblib')

# 2. BUILD SUMMARY dict:
summary = {
    'objective':            NS.get('business_objective'),
    'best_model':           NS['best_model_name'],
    'algorithm_shortlist':  NS['model_design']['algorithm_shortlist'],
    'primary_metric':       NS['metric_choice']['primary'],
    'primary_metric_value': float(NS['leaderboard'].iloc[0].get(NS['metric_choice']['primary'], np.nan)),
    'optimal_threshold':    NS['optimal_threshold'],
    'imbalance_tier':       NS['imbalance_info'].get('tier'),
    'n_features':           len(NS['feature_names']),
    'rows_trained':         int(len(NS['y_train'])),
    'rows_oot':             int(len(NS['y_oot'])),
    'training_snapshots':   NS['model_design']['windows']['snapshots_for_training'],
    'oot_snapshots':        NS['model_design']['windows']['oot_snapshots'],
}
json.dump(summary, open(DIRS['root']+'/run_summary.json','w'),
          indent=2, default=str)

# 3. PRINT FINAL SUMMARY (text only - no plots):
print('\n' + '='*60)
print(' FINAL ARTIFACT SUMMARY')
print('='*60)
for k, v in summary.items():
    print(f" {k:24s}: {v}")
print('='*60)
print('\nSaved artifacts:')
print(f" {DIRS['models']}/best_model.joblib")
print(f" {DIRS['root']}/run_summary.json")
print(f" {DIRS['eval']}/leaderboard.csv")
print(f" {DIRS['eval']}/test_predictions.csv")
print(f" {DIRS['eval']}/lift_table.csv")
print(f" {DIRS['fe']}/preprocessor.pkl")
print(f" {DIRS['fe']}/column_map.json")
print(f" {DIRS['fe']}/iv_report.csv")
print(f" {DIRS['fe']}/psi_report.csv")

# 4. ASSIGN TO NS:
NS['artifact'] = artifact
NS['summary'] = summary

# MUST ASSIGN: artifact, summary, NS['artifact'], NS['summary']

"""