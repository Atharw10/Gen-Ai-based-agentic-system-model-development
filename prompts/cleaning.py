"""Stage 4 - Data Cleaning prompt."""
from .system_rules import SYSTEM_RULES

CLEANING_PROMPT = SYSTEM_RULES + """
===== STAGE 4: DATA CLEANING =====

IMPORTANT: df_test may be None (when user provides only a single CSV).
           All operations on df_test MUST be guarded by `if df_test is not None`.

READS FROM NS:
  df_train, df_test (may be None), target_col
  NS['model_design']['data_strategy']:
    - imputation_numeric     ('median' | 'mean' | 'zero')
    - imputation_categorical ('mode' | 'constant')
    - outlier_strategy       ('winsorize_99' | 'iqr_safe' | 'none')

WRITES TO NS:
  df_train, df_test (cleaned in-place - overwrite the keys)
  NS['cleaning_report']
  NS['imputation_map']
  NS['outlier_bounds']

STEPS

1. CAPTURE BEFORE STATE (handle df_test = None):
   rows_train_before = int(df_train.shape[0])
   nulls_train_before = int(df_train.isna().sum().sum())
   dupes_train_before = int(df_train.duplicated().sum())
   if df_test is not None:
       rows_test_before  = int(df_test.shape[0])
       nulls_test_before = int(df_test.isna().sum().sum())
   else:
       rows_test_before  = 0
       nulls_test_before = 0

2. READ STRATEGY from NS['model_design']['data_strategy']:
   ds = NS['model_design']['data_strategy']
   imp_num     = ds.get('imputation_numeric', 'median')
   imp_cat     = ds.get('imputation_categorical', 'mode')
   outlier_str = ds.get('outlier_strategy', 'winsorize_99')

3. DROP HIGH-MISSING COLUMNS (>70% missing in df_train):
   null_pct = df_train.isna().mean()
   dropped_cols = null_pct[null_pct > 0.70].index.tolist()
   if dropped_cols:
       df_train = df_train.drop(columns=dropped_cols)
       if df_test is not None:
           df_test = df_test.drop(columns=[c for c in dropped_cols if c in df_test.columns])
   print(f"Dropped {len(dropped_cols)} high-missing columns: {dropped_cols}")

4. IMPUTE using mlkit.cleaning.smart_impute:
   df_train, imp_map = cleaning.smart_impute(
       df_train, num_strategy=imp_num, cat_strategy=imp_cat, target_col=target_col)
   if df_test is not None:
       df_test = cleaning.apply_imputation(df_test, imp_map)
   print(f"Imputation applied: {len(imp_map)} columns ({imp_num}/{imp_cat})")
   NS['imputation_map'] = imp_map

5. OUTLIER CAPPING per outlier_str:
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

6. DROP DUPLICATES FROM df_train ONLY (not df_test - test rows are sacred):
   df_train = df_train.drop_duplicates().reset_index(drop=True)
   train_dupes_removed = rows_train_before - df_train.shape[0]
   print(f"Dropped {train_dupes_removed} duplicates from df_train. New shape: {df_train.shape}")

7. BUILD cleaning_report dict:
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

8. DISPLAY a small before/after table:
   summary_df = pd.DataFrame({
       'metric': ['rows_train', 'nulls_train', 'duplicates_train'],
       'before': [rows_train_before, nulls_train_before, dupes_train_before],
       'after':  [df_train.shape[0], int(df_train.isna().sum().sum()), 0],
   })
   print("\n" + summary_df.to_string(index=False))

9. SAVE:
   json.dump(cleaning_report, open(DIRS['clean']+'/cleaning_report.json','w'),
             indent=2, default=str)
   NS['cleaning_report'] = cleaning_report
   NS['df_train']        = df_train
   NS['df_test']         = df_test
   print("Cleaning complete")

MUST ASSIGN: df_train, df_test, cleaning_report,
             NS['df_train'], NS['df_test'], NS['cleaning_report'],
             NS['imputation_map'], NS['outlier_bounds']
"""