# AKI Prediction Project

Postoperative AKI and ICU-admission prediction from VitalDB intraoperative
Catch24 waveform features and preoperative clinical data. The manuscript path
uses logistic regression, XGBoost, and EBM. Aeon remains experimental.

The main core study has completed (36 configurations at frozen revision
`fcff7c7`). The optimized scheduler and broader ICU/mortality extension use a
fresh study root and a full-budget numerical equivalence gate. Existing main
artifacts remain read-only. See the [evidence register](.agent/workflows/manuscript-reorientation.md).

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
it on other machines. Fresh extraction needs VitalDB access; this extension
reuses sealed waveform features.

```bash
unset DATA_DIR PROCESSED_DIR RESULTS_DIR PAPER_DIR LOG_FILE SMOKE_ROOT INPUT_FILE LAB_DATA_FILE
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1
export PROCESSED_DIR=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1/data/processed

# Read-only inspection: no preprocessing, training, or output directories
python -m runtime_resources --volume "$AKI_ARTIFACT_ROOT"
python -m run_catch22 experiments --preset core --prep skip \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity \
  --reuse-from /media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1 --dry-run
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
operations remain grouped by patient. `mortality_waveform_sensitivity` uses valid
in-hospital mortality labels in the broader waveform population and remains
exploratory. The two new core scopes have 18 configurations each. The full main
grid has 180 configurations.

## Execution after run approval

Execution is held at readiness. After explicit approval, run the gated supervisor
from the sealed study checkout. It validates the old 36 configurations, runs the
single broader ICU XGBoost canary with 12 and four threads, and admits the two
18-configuration batches only after the complete numerical comparison passes.
It never selects the 144 main supplementary configurations.

```bash
cd /media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1/code
python scripts/cpu/run-optimized-core.py --print-commands  # read-only preview

# Starts research work; survives closing the terminal/SSH connection
RUN_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1
mkdir -p "$RUN_ROOT/control"
nohup python -u scripts/cpu/run-optimized-core.py \
  > "$RUN_ROOT/control/supervisor.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/control/supervisor.pid"
```

Status is `<new-study>/control/supervisor_status.json`; a failed gate stops
execution. Root overrides are listed by `--help`. Reuse the sealed processed
features; this extension requires no preprocessing or waveform extraction.
The supervisor uses the active Conda interpreter. The same command resumes
verified completed folds after an interruption; a VM shutdown still requires
relaunching it. Routine stage transitions require no input, and a scientific
gate or integrity failure stops execution with its reason in the status/log.

`--outcomes`, `--branches`, and `--feature-sets` restrict configurations;
`--model-types` selects families. Definitive validation uses five outer, three
HPO, and five training-only calibration folds, with one repeat. HPO defaults
are 20 LR trials and 50 each for XGBoost/EBM. Reduced settings require
`--diagnostic`; holdout/smoke outputs are also diagnostic. Full-data refits are
optional and disabled by default. Strict manifests and complete fold bundles
prevent resuming incompatible or incomplete analyses.

The default `--scheduler optimized` mixes independent configuration processes
within affinity/quota-aware CPU and memory budgets. On this 64-CPU VM it admits
up to 60 reserved CPUs: LR uses BLAS 4, XGBoost uses four threads per fit, and
EBM retains 12 jobs with its shared bag pool. HPO remains serial inside each
fold. `--max-workers` bounds outer folds; `--cpu-budget` and
`--max-concurrent-configs` bound admission. Explicit thread overrides remain
available; numerical settings stay fixed throughout each configuration.
`--scheduler legacy` runs configurations sequentially. Per-job logs, sampled
CPU/RSS/I/O and controller summaries live under `<study>/control/`;
`--no-profile-resources` disables sampling. Summed RSS can include shared pages.

`--reuse-from` verifies protected completed configurations before skipping them;
it never rewrites old fingerprints. Complete local runs are also verified before
skipping. Future main supplements can use `--preset full --reuse-from ...` to
reuse main core references. Deliberate repeats require a fresh output root,
`--rerun-completed --no-resume`, and no imported inventory. Extraction retains
its independent `AKI_EXTRACTION_WORKERS` budget; this extension reuses features.

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
python -m run_catch22 experiments --preset core --report-only \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity
python -m run_catch22 descriptive --cohort-scope mortality_waveform_sensitivity --branch non_windowed
python -m run_catch22 descriptive --cohort-scope mortality_waveform_sensitivity --branch windowed
COHORT_SCOPE=mortality_waveform_sensitivity python -m reporting.fold_explanations
```

Reporting defaults to 5,000 unstratified patient-cluster bootstrap draws, seed 0.
`--bootstrap`, `--bootstrap-seed`, and `--bootstrap-stratified` change the recorded
reporting policy without fitting models. Family-stage reports remain partial
until the full requested core is available. See [RESULTS.md](RESULTS.md) for
saved-prediction comparisons, DCA, explanations, and publication outputs.
The supervised run pins process reporting. Its 1,800-second bootstrap deadline
applies per configuration task; bounded process retries fail visibly. Explicit
thread/single-worker execution cannot enforce this deadline. Setting
`--bootstrap-timeout 0` on the metrics entrypoint permits backend fallbacks.

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
