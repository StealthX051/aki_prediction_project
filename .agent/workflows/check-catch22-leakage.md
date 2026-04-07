---
description: Check for potential data leakage in the Catch22 pipeline
---

# Check for potential data leakage

Inspect the touched files plus the usual leakage hotspots:

- `data_preparation/step_03_preop_prep.py`
- `data_preparation/step_04_intraop_prep.py`
- `data_preparation/step_05_data_merge.py`
- `model_creation/step_06_run_hpo.py`
- `model_creation/step_07_train_evaluate.py`
- `model_creation/postprocessing.py`
- `results_recreation/metrics_summary.py`

Verify all of the following:

- The grouped split is created once in Step 03 and propagated via the
  outcome-specific split columns (`split_group_any_aki`,
  `split_group_icu_admission`, and related fields).
- Learned preprocessing statistics are fit on training data only.
- Hyperparameter optimization uses training/CV data only.
- Calibration and threshold selection are fit on training or out-of-fold
  predictions only, then applied unchanged to held-out data.
- Reporting and results-recreation consume saved artifacts only; they do not
  silently refit models, calibrators, or thresholds.

Common red flags:

- recomputing or overwriting split metadata after Step 03
- computing encodings, percentiles, or imputations on the full dataframe
- selecting thresholds or features with held-out test metrics
- fitting calibration on held-out test predictions
- merges that drop, regenerate, or misalign outcome-specific split columns

When you find a risk, cite the specific file and line, explain the leakage
path, and propose the smallest viable fix.
