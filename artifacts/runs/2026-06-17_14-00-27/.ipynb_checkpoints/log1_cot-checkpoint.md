# Run 2026-06-17_14-00-27


## Stage: `profiling`
_2026-06-17T14:01:44_

**Inputs from previous stages:**
- user inputs (data_path, target_col, date_col)

**NS keys read:** `df_train`, `target_col`, `date_col`, `business_objective`

**What this stage did:**
Profiled 600000 rows & 50 cols in 5.08s

**Decisions made:**
- `n_rows` = 600000
- `n_cols` = 50
- `target` = cc_conversion_next_1m
- `pos_ratio` = 0.0503
- `n_snapshots` = 6

**Assumptions:**
- Pure observation - no decisions made here
- Only df_train profiled (df_test may be None)
- Date column parsed to datetime

**Outputs flowing to next stages:**
- model_design (consumes profile dict)

**NS keys written:** `profile`

---
