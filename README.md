# AKI Prediction Project

This repository generates reproducible results for postoperative acute kidney
injury prediction from VitalDB intraoperative waveforms plus preoperative
clinical data.

- Primary paper path: Catch24/Catch22 feature extraction, XGBoost/EBM model
  training, and paper-facing reporting.
- Experimental path: Aeon time-series models under `model_creation_aeon/`.
  Aeon is not part of the validated default workflow.

Raw VitalDB data are not shipped in this repository. You need local access to
the source CSVs and waveform downloads to run the pipeline.

## Canonical docs

- `README.md`: operational entrypoint and run commands
- `METHODS.md`: scientific definitions and preprocessing/evaluation rules
- `RESULTS.md`: artifact layout and regeneration of paper outputs
- `AGENTS.md`: repository-specific working rules for coding agents
- `experimental/aeon/README.md`: experimental Aeon commands only

## Environment

Use the Conda environment in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate aki_prediction_project
python -m pip install -r requirements-test.txt
```

Update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

## Quick start

The Python launcher is the canonical entrypoint.

```bash
# Fastest end-to-end validation on isolated outputs
python -m run_catch22 smoke

# Full Catch22 experiment grid
python -m run_catch22 experiments

# Descriptive tables and cohort-flow figures from saved artifacts
python -m run_catch22 descriptive
```

Recommended first check after code changes:

```bash
python -m run_catch22 smoke
```

## Launcher behavior

`python -m run_catch22` exposes three subcommands:

- `smoke`: isolated real-data validation run using a small cohort, holdout
  evaluation, and report generation
- `experiments`: full Catch22 grid over the active outcomes, branches, feature
  sets, and model families
- `descriptive`: preoperative descriptives, cohort flow, and missingness tables
  from saved artifacts

Useful patterns:

```bash
# Rebuild processed artifacts before the full grid
python -m run_catch22 experiments --prep force

# Reuse validated processed artifacts
python -m run_catch22 experiments --prep skip

# Stage model families separately
python -m run_catch22 experiments --only-xgboost --prep auto
python -m run_catch22 experiments --only-ebm --prep skip
```

`--prep auto` and `--prep skip` validate the Step 01/03/05 metadata sidecars
before reuse and fail fast on stale processed artifacts.

## Stepwise pipeline

Run the modules directly when you want an explicit staged workflow.

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

Optional ad hoc holdout-only tuning remains available:

```bash
python -m model_creation.step_06_run_hpo \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type ebm
```

## Artifact roots and environment overrides

Generated Catch22 artifacts default to:

```text
/media/volume/catch22/data/aki_prediction_project
```

Relevant environment variables:

- `AKI_ARTIFACT_ROOT`: canonical generated-artifact root
- `AKI_STORAGE_POLICY`: heavy-output enforcement policy (`enforce`, `warn`,
  `off`)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`: explicit data roots
- `RESULTS_DIR`, `PAPER_DIR`: explicit experiment and paper roots
- `SMOKE_ROOT`: isolated smoke output root
- `LOG_FILE`: explicit log path

Use overrides when experimenting so exploratory runs do not pollute canonical
artifacts.

## Paper-facing outputs

The publication surface lives under `results/catch22/paper/` and is rebuilt
from saved artifacts rather than retraining models at report time.

Key outputs:

- `tables/metrics_summary.{csv,md,docx,pdf}`
- `tables/results_*_{main,delta}.csv` and companion formatted tables
- `figures/` for ROC, PR, calibration, cohort flow, and manuscript figures
- `reports/report.{md,docx,pdf}`
- `metadata/cohort_flow_counts.json`
- `manifest.json`

See `RESULTS.md` for the full layout and regeneration commands.

## Repository map

- `data_preparation/`: cohort construction, Catch22 extraction, preop prep,
  intraop pivoting, and merge
- `model_creation/`: HPO, training, calibration, thresholding, and prediction
  export
- `results_recreation/`: standardized metrics aggregation and bootstrap CIs
- `reporting/`: paper tables, plots, cohort flow, descriptives, missingness
- `model_creation_aeon/`: experimental Aeon models
- `experimental/aeon/README.md`: Aeon execution notes
- `archive/`: historical notebooks and scripts, not the default path

## Publication-compliance notes

- The validated default workflow is the Catch22/XGBoost/EBM path only.
- Experimental code paths are labeled and separated from the paper path.
- Reporting regenerates tables and figures from saved predictions and metadata.
- Scientific definitions and evaluation rules are documented in `METHODS.md`.
- Legacy notebooks and archived scripts are retained for reference, not as the
  default reproducible workflow.
