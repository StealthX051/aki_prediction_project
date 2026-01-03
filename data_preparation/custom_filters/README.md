# Custom cohort filters

Functions in this folder are imported by `data_preparation/inputs.py` and
applied inside `step_01_cohort_construction.py`. Each helper should take a
`pd.DataFrame` and return a new `pd.DataFrame`, leaving index alignment intact.
Current filters include:

- `add_aki_label`: Derives `aki_label` from postoperative creatinine trends.
- `filter_preop_cr`: Drops cases with missing or extreme baseline creatinine.
- `filter_postop_cr`: Removes cases lacking postoperative creatinine needed for
  adjudication.
- `ensure_sample_independence`: Enforces sampling independence before the
  train/test split.

Add new filters as separate modules in this directory and import them in
`inputs.py` to include them in the cohort pipeline.
