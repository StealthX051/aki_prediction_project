---
description: Runs catch22 VitalDB Pipeline
---

# Run AKI pipeline end-to-end

Use this workflow for the primary Catch22-based pipeline. Prefer the smoke test
unless the task truly requires a larger run.

When experimenting, prefer environment-variable overrides such as
`SMOKE_ROOT`, `DATA_DIR`, `PROCESSED_DIR`, `RESULTS_DIR`, and `PAPER_DIR`
instead of writing into canonical project outputs.

## Environment

```bash
conda env create -f environment.yml
conda activate aki_prediction_project
```

## Fastest validation path

```bash
./run_smoke_test.sh
```

This runs:
- steps 01 through 05 of `data_preparation/`
- a windowed `any_aki` + `all_waveforms` + `xgboost` smoke HPO/train/eval
- `results_recreation.metrics_summary`
- `reporting.make_report`

Smoke outputs are isolated under `smoke_test_outputs/`.

## Stepwise production pipeline

### 1. Cohort construction

```bash
python -m data_preparation.step_01_cohort_construction
```

Key outputs:
- `data/processed/aki_pleth_ecg_co2_awp.csv`
- `results/catch22/paper/metadata/cohort_flow_counts.json`

### 2. Catch22 feature extraction

```bash
python -m data_preparation.step_02_catch_22
```

Key outputs:
- `data/processed/aki_pleth_ecg_co2_awp_inf.csv`
- `data/processed/aki_pleth_ecg_co2_awp_inf_windowed.csv` when windowed export is enabled

### 3. Preoperative prep and split

```bash
python -m data_preparation.step_03_preop_prep
python -m data_preparation.step_03_preop_prep --impute-missing
```

Key output:
- `data/processed/aki_preop_processed.csv`

### 4. Intraoperative wide-table prep

```bash
python -m data_preparation.step_04_intraop_prep
```

Key outputs:
- `data/processed/aki_intraop_wide.csv`
- `data/processed/aki_intraop_wide_windowed.csv`

### 5. Merge modeling tables

```bash
python -m data_preparation.step_05_data_merge
python -m data_preparation.step_05_data_merge --impute-missing
```

Key outputs:
- `data/processed/aki_features_master_wide.csv`
- `data/processed/aki_features_master_wide_windowed.csv`

### 6. Hyperparameter search

```bash
python -m model_creation.step_06_run_hpo \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type xgboost
```

Parameter files are written under:
- `results/catch22/experiments/params/{model_type}/{outcome}/{branch}/{feature_set}.json`

### 7. Final training and evaluation

```bash
python -m model_creation.step_07_train_evaluate \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type xgboost
```

EBM example:

```bash
python -m model_creation.step_07_train_evaluate \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type ebm \
  --export_ebm_explanations
```

Key outputs live under:
- `results/catch22/experiments/models/...`
- `results/catch22/paper/...`

### 8. Standardized evaluation and reporting

```bash
python -m results_recreation.metrics_summary \
  --results-dir results/catch22/experiments \
  --delta-mode reference \
  --reference-feature-set preop_only

python -m reporting.make_report
```

## Full grid launcher

```bash
./run_experiments.sh
./run_experiments.sh --only-xgboost
./run_experiments.sh --only-ebm
./run_experiments.sh --prep skip
```

Use full sweeps only when needed; they are much more expensive than the smoke
path.
