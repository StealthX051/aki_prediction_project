---
description: Run the Catch22 pipeline
---

# Run AKI pipeline end-to-end

Use this workflow for the validated Catch22 path. Prefer the smoke path unless
the task truly requires a larger run.

## Environment

```bash
conda activate aki_prediction_project
```

## Default validation path

```bash
python -m run_catch22 smoke
```

This is the preferred end-to-end verification path after code changes.

## Full experiment launcher

```bash
python -m run_catch22 experiments
python -m run_catch22 experiments --prep force
python -m run_catch22 experiments --prep skip
python -m run_catch22 experiments --only-xgboost
python -m run_catch22 experiments --only-ebm --prep skip
```

Use full sweeps only when necessary.

## Stepwise modules

```bash
python -m data_preparation.step_01_cohort_construction
python -m data_preparation.step_02_catch_22
python -m data_preparation.step_03_preop_prep
python -m data_preparation.step_04_intraop_prep
python -m data_preparation.step_05_data_merge
python -m model_creation.step_07_train_evaluate \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type xgboost
python -m results_recreation.metrics_summary \
  --results-dir results/catch22/experiments \
  --delta-mode reference \
  --reference-feature-set preop_only
python -m reporting.make_report
```

Use `model_creation.step_06_run_hpo` only for targeted ad hoc holdout tuning;
it is not required for the default launcher path.
