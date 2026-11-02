# AKI Prediction Project

Postoperative AKI and ICU-admission prediction from VitalDB intraoperative
Catch24 waveform features and preoperative clinical data. The manuscript path
uses logistic regression, XGBoost, and EBM. Aeon remains experimental.

The main core study completed (36 configurations at frozen revision `fcff7c7`),
followed by the numerical equivalence gate, broader ICU/mortality cores, main
supplements and broader ICU supplements. Frozen artifacts remain read-only.
See the [evidence register](.agent/workflows/manuscript-reorientation.md).

- [METHODS.md](METHODS.md): scientific definitions and analysis protocol
- [RESULTS.md](RESULTS.md): saved artifacts and paper-output regeneration
- [AGENTS.md](AGENTS.md): repository working rules
- [Preprocessing stages](data_preparation/README.md)
- [Original waveform processing](METHODS.md#waveform-processing): loading,
  operative cropping, quality handling, filters, anti-alias resampling and branch rates
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

The ICU/mortality core run was started in tmux session `vitaldb-sensitivity`.
Live phase/status is in `<new-study>/control/supervisor_status.json`; attach with
`tmux attach -t vitaldb-sensitivity`. The launch command below starts or resumes
the gated supervisor after an explicit run request; avoid a second controller
while this session is active. It validates the old 36 configurations, runs the
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

The default `--scheduler optimized --admission-policy adaptive` mixes independent
configuration processes. On this 64-CPU VM, `--cpu-budget 60` targets near-full
CPU use rather than imposing a ceiling on summed reservations. After a complete
60-second window below 93% of that budget, another configuration can start;
admissions beyond nominal reservations are at least 60 seconds apart. Up to 12
configurations, including multiple EBM configurations, may run when memory permits.
`--max-concurrent-configs` can lower this cap. Peaks may use all CPUs; numerical
settings remain fixed: LR BLAS 4, XGBoost four threads, EBM 12 jobs, and serial HPO
within each fold. `--max-workers` bounds outer-fold concurrency.
`--admission-policy reserved` restores strict summed reservations and the single
EBM limit; `--scheduler legacy` runs configurations sequentially. Missing CPU
telemetry disables extra adaptive admissions. Existing memory safeguards apply.

Per-job CPU/RSS/I/O logs and controller summaries use separate execution epochs
under `<study>/control/scheduler/`; `control/active_scheduler.json` identifies the
current summary. `--no-profile-resources` disables detailed job logs, while CPU
pressure sampling continues for admission. Summed RSS can include shared pages.
To finish active configurations and stop admission, run the supervisor with
`--drain-current`. Relaunch the normal command to resume verified unfinished work.
The drain helper also supports the original controller through a verified
process-only handoff; it never suspends training workers.

`--reuse-from` verifies protected completed configurations before skipping them;
it never rewrites old fingerprints. Complete local runs are also verified before
skipping. Future main supplements can use `--preset full --reuse-from ...` to
reuse main core references. Deliberate repeats require a fresh output root,
`--rerun-completed --no-resume`, and no imported inventory. Extraction retains
its independent `AKI_EXTRACTION_WORKERS` budget; this extension reuses features.

## Completed main supplementary run

The 144 remaining main configurations were authorized and started in tmux
session `vitaldb-supplementary`. They reuse the verified 36 main core
configurations, sealed processed features and the validated adaptive scheduler.
All 144 new configurations and reporting completed successfully November 1.
The completed ICU/mortality study remains separate. Outputs are under
`/media/volume/catch22/data/aki_prediction_project/studies/manuscript-supplementary-v1`.

```bash
tmux attach -t vitaldb-supplementary
# Run status and full log:
cat /media/volume/catch22/data/aki_prediction_project/studies/manuscript-supplementary-v1/control/launch-status.json
tail -F /media/volume/catch22/data/aki_prediction_project/studies/manuscript-supplementary-v1/control/launch.log
```

After an interruption, resume with the same Conda interpreter and
`<supplementary-study>/control/run.py`; avoid a second controller while active.
The script pins the full main grid, reuse source, 5/3/5 validation and 5,000
bootstrap draws. Training, metrics and report generation run without routine
intervention; failures remain visible in the status and log.

## Optional broader ICU and mortality supplements

The broader ICU supplement completed successfully November 1, after launch in tmux
`vitaldb-icu-supplementary`: 72 new configurations plus 18 verified core references.
Study root: `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-icu-supplementary-v1`.
Check `control/launch-status.json` and `control/launch.log`; whole-machine CPU
history is `control/machine-usage.jsonl`. It retains the existing scientific
settings and runs reporting automatically. Resume only its frozen
`control/run.py` after confirming no active controller. Mortality supplements
remain optional and unstarted.

The full feature grid is available for both broader cohorts. This read-only
preview verifies the completed core studies and plans separate supplementary
outputs in a fresh root; it does not authorize or start another grid. It is a
planning example, not a request to repeat the completed ICU supplement.

```bash
unset DATA_DIR RESULTS_DIR PAPER_DIR LOG_FILE N_TRIALS THREADS_PER_MODEL
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export PROCESSED_DIR=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1/data/processed
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-broader-supplementary-v1
python -m run_catch22 experiments --preset full --prep skip \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity \
  --reuse-from /media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1 \
  --outer-folds 5 --inner-folds 3 --calibration-folds 5 --repeats 1 \
  --max-workers 5 --bootstrap 5000 --bootstrap-seed 0 --no-bootstrap-stratified \
  --dry-run
```

Expected: 180 logical configurations, 36 verified core references and 144 pending
(72 per cohort), covering both branches and all three model families. Use
`--cohort-scope` to select either cohort alone. Completed source models and their
original fingerprints stay read-only; the new study records verified references.
Mortality retains its existing source labels and exploratory status. Freeze the
reviewed execution checkout before any subsequently authorized launch.

## Window-duration decision

Retain 10-second windows with a nominal five-second stride for this manuscript;
the duration search is closed. The completed feature-only pilot selected no
duration under its full rule. Forty seconds improved numerical coverage for one
respiratory descriptor, without establishing better AKI prediction. No further
duration ablation or gate tuning is planned. See the
[methods rationale](METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis),
[pilot artifacts and pending saved-model audit](RESULTS.md#window-duration-pilot-and-feature-audit),
and [completed pilot record](.agent/workflows/window-duration-pilot.md).

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
