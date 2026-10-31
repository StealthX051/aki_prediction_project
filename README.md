# AKI Prediction Project

Postoperative AKI and ICU-admission prediction from VitalDB intraoperative
Catch24 waveform features and preoperative clinical data. The manuscript path
uses logistic regression, XGBoost, and EBM. Aeon remains experimental.

**Pre-run checkpoint:** the implementation is prepared for definitive execution;
full-cohort reconstruction and manuscript model runs have not started. Offline
checks verify code and numerical contracts. Earlier real-data diagnostics
predate the latest cleaning, anti-aliasing, and comparison changes; they do not
validate the definitive results. See the [evidence register](.agent/workflows/manuscript-reorientation.md).

- [METHODS.md](METHODS.md): scientific definitions and analysis protocol
- [RESULTS.md](RESULTS.md): saved artifacts and paper-output regeneration
- [AGENTS.md](AGENTS.md): repository working rules
- [Preprocessing stages](data_preparation/README.md)
- [Experimental Aeon workflow](experimental/aeon/README.md)

## Environment and offline checks

Linux x86_64 / Python 3.12 CPU is the supported environment. On this VM:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate aki_prediction_project
bash scripts/cpu/verify-offline.sh
```

For a new environment:

```bash
conda env create -f environment.yml
conda activate aki_prediction_project
python -m pip install -r requirements-test.txt
```

Runtime pins are in `requirements-cpu.txt`; exact replay and dependency-change
procedures are in [locks/README.md](locks/README.md). Preserve the existing
validated environment when testing dependency changes. Use `xgboost-cpu`, never
both XGBoost distributions. The GPU manifest is historical; Aeon, CUDA,
AutoGluon, notebooks, and optional browser image exports are outside this profile.

`python -m pytest -q` runs the regression suite; append a test filename for a
focused check. Pytest isolates study paths before project imports and blocks
live VitalDB requests. Native-library checks also verify Catch24 reference
values, CPU models, SHAP additivity, serialization, and report exports.

## Configure a fresh study

Raw data are not shipped. This VM's source CSVs are at the path below. Override
it on other machines. The launcher also needs VitalDB waveform access.

```bash
unset DATA_DIR PROCESSED_DIR RESULTS_DIR PAPER_DIR LOG_FILE SMOKE_ROOT INPUT_FILE LAB_DATA_FILE
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1

# Read-only inspection: no preprocessing, training, or output directories
python -m runtime_resources --volume "$AKI_ARTIFACT_ROOT"
python -m run_catch22 experiments --preset core --dry-run
```

Use a fresh study root on the attached volume. Historical extraction outputs
are incompatible with the current anti-aliasing protocol. The launcher checks
processed schemas, source hashes, cohort rosters, and extraction settings before
reuse. `--prep auto` prepares absent inputs or validates existing ones;
`--prep skip` only validates. `--prep force` requires an empty processed directory
and preserves existing artifacts. Resume is enabled for matching complete folds.

The core order is `preop_only`, `all_waveforms`, `preop_and_all_waveforms`.
Main core: 36 configurations across two outcomes, two branches, and three
families. `main_shared` uses the same AKI/ICU-eligible operations within a branch;
`icu_waveform_sensitivity` uses the broader ICU-eligible population. Repeat
operations remain grouped by patient. The full main grid has 180 configurations.

## Execution after run approval

The following commands start research work; they are not readiness checks.
Use the same root and unchanged code/environment throughout a staged study.

```bash
# First family prepares both waveform branches once
python -m run_catch22 experiments --preset core --model-types logistic_regression
python -m run_catch22 experiments --preset core --model-types xgboost --prep skip
python -m run_catch22 experiments --preset core --model-types ebm --prep skip

# Broader ICU sensitivity, then the remaining supplementary main grid
python -m run_catch22 experiments --preset core --cohort-scope icu_waveform_sensitivity --prep skip
python -m run_catch22 experiments --preset full --prep skip
```

`--outcomes`, `--branches`, and `--feature-sets` restrict configurations;
`--model-types` selects families. Definitive validation uses five outer, three
HPO, and five training-only calibration folds, with one repeat. HPO defaults
are 20 LR trials and 50 each for XGBoost/EBM. Reduced settings require
`--diagnostic`; holdout/smoke outputs are also diagnostic. Full-data refits are
optional and disabled by default. Strict manifests and complete fold bundles
prevent resuming incompatible or incomplete analyses.

Model workers/threads respect CPU affinity and quota. On the audited 64-CPU VM,
the default is five outer workers with 12 threads each. LR uses one BLAS thread;
`--blas-threads` can increase it within `--threads-per-model`. Other families
keep BLAS at one alongside native model threads. Extraction has its own budget:
start with `AKI_EXTRACTION_WORKERS=16`, then adjust only from measured throughput
and memory. Avoid simultaneous study writers or extra configuration-level
parallelism. `scripts/cpu/benchmark-resources.py` offers a bounded synthetic LR
benchmark; it is not a production runtime estimate.

## Diagnostics and saved-result reporting

Real-data smoke needs a fresh/empty disposable `SMOKE_ROOT`, local source CSVs,
and live VitalDB access; it can exceed the requested case count to preserve
outcome/class patient support. It does not overwrite existing smoke outputs.

```bash
# Research diagnostic; run only when requested
SMOKE_ROOT="$AKI_ARTIFACT_ROOT/diagnostics/smoke-v1" AKI_EXTRACTION_WORKERS=16 \
python -m run_catch22 smoke --raw-source-dir "$RAW_DIR" --case-limit 40 \
  --model-types logistic_regression,xgboost,ebm

# Saved results only; no estimator, calibrator, or threshold fitting
python -m run_catch22 experiments --report-only
python -m run_catch22 descriptive --branch non_windowed
python -m run_catch22 descriptive --branch windowed
python -m reporting.fold_explanations
```

Reporting defaults to 5,000 unstratified patient-cluster bootstrap draws, seed 0.
`--bootstrap`, `--bootstrap-seed`, and `--bootstrap-stratified` change the recorded
reporting policy without fitting models. Family-stage reports remain partial
until the full requested core is available. See [RESULTS.md](RESULTS.md) for
saved-prediction comparisons, DCA, explanations, and publication outputs.

## Paths and advanced entrypoints

Generated artifacts default to `/media/volume/catch22/data/aki_prediction_project`.
`AKI_ARTIFACT_ROOT` selects a study; `RAW_DIR` selects source CSVs. Advanced
`PROCESSED_DIR`, `RESULTS_DIR`, `PAPER_DIR`, and `LOG_FILE` overrides remain
available. Results/paper roots append `<cohort_scope>/<validation_scheme>` unless
already scoped. Heavy outputs must remain on the attached volume under the
normal `AKI_STORAGE_POLICY=enforce`. Convenience symlinks change only with
`--refresh-links`.

Use [Steps 01–05](data_preparation/README.md) for explicit staged preparation and
`python -m model_creation.step_07_train_evaluate --help` for standalone training.
`step_06_run_hpo` is an ad hoc holdout-tuning entrypoint, not a prerequisite for
nested validation. `archive/` retains historical code; it is not the manuscript
workflow. Optional legacy artifact readers require explicit legacy mode.
