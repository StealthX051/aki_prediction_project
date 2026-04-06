---
description: Check for potential data leakage in VitalDB Catch22 Project
---

# Check for potential data leakage

Inspect the touched files plus the usual leakage hotspots:

- `data_preparation/step_03_preop_prep.py`
- `data_preparation/step_04_intraop_prep.py`
- `data_preparation/step_05_data_merge.py`
- `model_creation/step_06_run_hpo.py`
- `model_creation/step_07_train_evaluate.py`
- `model_creation/postprocessing.py`

Verify all of the following:

- The train/test split is created once in step 03 and propagated via
  `split_group`.
- Outlier thresholds, encodings, imputations, and any other learned
  preprocessing statistics are fit on training data only.
- Full-case and windowed branches both preserve the original split assignment.
- Hyperparameter optimization uses training/CV data only.
- Calibration and threshold selection are fit from training or out-of-fold
  predictions only, then applied unchanged to the held-out test set.
- Reporting and results-recreation scripts only consume saved artifacts; they do
  not silently refit calibrators, thresholds, or models.

Common red flags:

- recomputing the split after preprocessing
- computing percentiles or encodings on the full dataframe
- selecting features or thresholds with test metrics
- fitting calibration on held-out test predictions
- merges that drop or regenerate `split_group`

When you find a risk, cite the specific file/line, explain the leakage path,
and propose the smallest viable fix.
