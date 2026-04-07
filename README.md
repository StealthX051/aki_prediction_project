# AKI Prediction Project

This project implements a machine learning pipeline to predict Postoperative Acute Kidney Injury (AKI) using intraoperative physiological waveforms (ECG, PPG, EtCO2) from the VitalDB dataset.

## 📋 Project Overview
The pipeline consists of three main stages:
1.  **Cohort Selection**: Filtering patients based on clinical criteria and waveform availability.
2.  **Feature Extraction**: Extracting time-series features (Catch22) from high-frequency waveforms.
3.  **Modeling**: Training XGBoost and Explainable Boosting Machine (EBM) classifiers to predict AKI (primary) and ICU admission (secondary), with mortality retained as a supported broader-cohort outcome outside the default run grid.

## 🗂️ Results layout (quick reference)
- **Canonical generated-artifact root**: `/media/volume/catch22/data/aki_prediction_project` by default. Set via `AKI_ARTIFACT_ROOT`.
- **Storage policy**: heavy generated outputs fail fast off the attached media volume by default (`AKI_STORAGE_POLICY=enforce`; override with `warn` or `off`).
- **Experiments root**: derived from `AKI_ARTIFACT_ROOT` as `results/catch22/experiments` unless `RESULTS_DIR` is set explicitly.
- **Paper surface**: derived from `AKI_ARTIFACT_ROOT` as `results/catch22/paper` unless `PAPER_DIR` is set explicitly. Symlinks from `experiments/{tables,figures,metadata}` point here for convenience.
- **XAI surfacing**: `results/catch22/xai/ebm/<o>/<b>/<fs>/ebm_xai` and `results/catch22/xai/shap/xgboost/<o>/<b>/<fs>/shap_summary_*.png` (symlinks to artifacts under experiments).
- **Archive**: `results/catch22/archive/legacy` holds legacy grids (`results/xgboost_*`).
- **Aeon**: parallel structure under `results/aeon/experiments` and `results/aeon/paper` for the experimental branch only; it is not part of the validated paper pipeline.
- Key publication files (defaults):
  - Metrics: `results/catch22/paper/tables/metrics_summary.{csv,md,docx,pdf}`
  - Figures: `results/catch22/paper/figures/` (Catch22 paper figures emitted as `.svg` + publication `.png`)
  - Reports: `results/catch22/paper/reports/report.{md,docx,pdf}`
  - Descriptives/Missingness: `results/catch22/paper/tables/preop_descriptives.*`, `missingness_table.*`
  - Manifest: `results/catch22/paper/manifest.json`

## 🛠️ Prerequisites & Setup

### 1. Environment Setup
Use the Conda environment as the canonical runtime for this repository. Agents
and contributors should prefer this environment over ad-hoc `venv` setups so
commands, paths, and package versions stay aligned.

```bash
# First-time setup
conda env create -f environment.yml

# Sync an existing environment after dependency changes
conda env update -f environment.yml --prune

# Activate before running project commands
conda activate aki_prediction_project
```

If you need the pinned test extras inside the same Conda environment, install
them after activation:

```bash
python -m pip install -r requirements-test.txt
```

Static Plotly exports rely on the bundled `kaleido` dependency. The environment
also pins `matplotlib` to a Plotly-compatible version to keep visualization
outputs reproducible across CLI runs.

Shell runners (`run_experiments.sh`, `run_smoke_test.sh`) honor `PYTHON_BIN` as
a whitespace-separated command prefix. If you do not activate Conda in the
shell first, invoke them with
`PYTHON_BIN='conda run -n aki_prediction_project python' ...` so the smoke and
experiment paths run inside the pinned project environment.

Generated Catch22 artifacts default to the attached media volume. The runners
derive `PROCESSED_DIR`, `RESULTS_DIR`, `PAPER_DIR`, `SMOKE_ROOT`, and log-file
locations from `AKI_ARTIFACT_ROOT` unless you override them explicitly.

### 2. VitalDB Access
Ensure you have access to the VitalDB API. The `vitaldb` Python package is used to download waveforms on-demand.

## 🚀 Execution Guide (Refactored Pipeline)

**Recommended**: This new pipeline uses modular scripts for better reproducibility, data leakage prevention, and support for both full and windowed datasets. The Catch22 + XGBoost/EBM workflow is the primary, production-ready path; Aeon remains **experimental** and is not kept in strict feature-parity with the modular Catch22 cohort logic.

### Advantages over Legacy Pipeline
*   **Reproducibility**: Modular Python scripts replace monolithic notebooks, making execution deterministic and easier to automate.
*   **Leakage Prevention**: The patient-level train/test split is performed **once** at the beginning of the preprocessing stage (`step_03`) and propagated to the final datasets. This guarantees that no patient from the test set is used to calculate training statistics (e.g., outlier thresholds).
*   **Dual Mode Support**: Automatically handles both **Full** (entire case) and **Windowed** (segmented) feature sets.

### Step 1: Configuration
**File**: `data_preparation/inputs.py`
Central configuration file. Defines input/output paths, mandatory waveforms/columns, and **AEON export settings** (e.g., padding policy, save formats). Verify `INPUT_FILE`, `MANDATORY_WAVEFORMS`, and `TARGET_SR`.

> Results layout: by default Catch22 generated outputs are rooted at
> `/media/volume/catch22/data/aki_prediction_project`. `RESULTS_DIR` and
> `PAPER_DIR` can still be overridden explicitly, but the runners now fail fast
> if heavy generated outputs resolve off `/media/volume/catch22` unless
> `AKI_STORAGE_POLICY` is relaxed.

### Step 2: Cohort Construction
**File**: `data_preparation/step_01_cohort_construction.py`
Filters the raw dataset to create a valid cohort.
*   **Logic**: Builds a shared outer cohort using outcome-agnostic criteria only: required `opend`, all `MANDATORY_WAVEFORMS`, and ASA I-IV. Multiple operations per patient are retained, but later train/validation/test splits are grouped by patient ID.
*   **Output**: A cohort CSV containing valid `caseid`s, shared derived outcomes, and per-outcome eligibility metadata.
    *   `eligible_any_aki` / `eligible_severe_aki`: Require baseline creatinine present, baseline creatinine `<= 4.0`, and at least one postoperative creatinine in the adjudication window.
    *   `eligible_icu_admission` / `eligible_mortality`: Require only the shared outer cohort plus a non-missing source label.
    *   `aki_label`: Primary outcome (KDIGO AKI) for AKI-eligible rows only; ineligible rows are left missing rather than forced to `0`.
    *   `y_icu_admit`: ICU admission (>0 days) — secondary outcome used in the current experiment grid.
    *   `y_inhosp_mortality`: In-hospital mortality — supported by the modular cohort path but not part of the default experiment launcher.
    *   `y_prolonged_los_postop`: Prolonged postoperative LOS (>= 75th percentile; archival only).
    *   `y_severe_aki`: Severe AKI (KDIGO Stage 2 or 3; archival only).
    *   Versioned sidecar metadata: each Step 01/03/05 CSV now writes a `.metadata.json` sidecar so downstream steps can fail fast on stale processed artifacts instead of mixing schemas silently.
```bash
python -m data_preparation.step_01_cohort_construction
```

To regenerate a cohort flow diagram from saved counts/metadata without rerunning
the full pipeline, supply the JSON counts file to the reporting utility. The renderer
now requires Graphviz (`dot`); keep the Conda environment in sync with
`conda env update -f environment.yml --prune` before running it. The diagram
skips no-op steps, uses friendly labels, and renders a CONSORT-style layout:
waveform checks are grouped into a single “High-fidelity Waveform Availability” box
with a required-channel footnote, exclusion reasons are drawn to the right in dashed
boxes, and the AKI vs. No AKI split uses two direct arrows from the bottom center of
the Final Cohort box. The default counts JSON remains AKI-specific even though the
saved cohort now contains the broader shared outer cohort. Per-step removals are shown
in the exclusion boxes, and AKI False/True counts are displayed when present in the
counts JSON produced by step 01. The baseline creatinine eligibility stage reflects the
AKI cohort rule `preop_cr <= 4.0 mg/dL`, not a second missingness check.

```bash
python -m reporting.cohort_flow --counts-file results/catch22/paper/metadata/cohort_flow_counts.json
```
Outputs are written to `results/catch22/paper/figures/` by default as
`cohort_flow.dot`, `cohort_flow.svg`, and `cohort_flow.png`.

### Step 3: Feature Extraction
**File**: `data_preparation/step_02_catch_22.py`
Extracts 22 time-series features (Catch22) from each waveform channel.
*   **Technical Details**:
    *   Resamples waveforms to `TARGET_SR` (default 10Hz) for both full-case and windowed branches.
    *   **Full Mode**: Extracts features from the entire case duration.
    *   **Windowed Mode**: Segments waveforms into windows (defined by `WIN_SEC`, `SLIDE_SEC`) and extracts features per window.
    *   Handles multiprocessing for efficiency.
```bash
python -m data_preparation.step_02_catch_22
```

### Step 4: Preoperative Data Prep & Split
**File**: `data_preparation/step_03_preop_prep.py`
Processes clinical data and defines the **stratified train/test split**.

#### 1. Variable Selection & Derivation
We extract a comprehensive set of preoperative variables from `clinical_data.csv` and `lab_data.csv`:

*   **Demographics**: Age, Sex, Height, Weight, BMI.
*   **Surgical Context**: Emergency Operation (`emop`), Department, Approach, ASA Class, Operation Type (`optype`), Anesthesia Type (`ane_type`).
*   **Comorbidities**: Hypertension (`preop_htn`), Diabetes (`preop_dm`), ECG abnormalities (`preop_ecg`), Pulmonary Function Test (`preop_pft`).
*   **Labs (Clinical Table)**: Hb, Platelets, PT (INR), aPTT, Na, K, Glucose, Albumin, AST, ALT, BUN, Creatinine, Bicarbonate.
*   **ABG (Clinical Table)**: pH, Base Excess (`preop_be`), PaO2, PaCO2, SaO2.
*   **Labs (Lab Table)**: Last preoperative value (within 30 days) for:
    *   `preop_wbc`: White Blood Cell count.
    *   `preop_crp`: C-Reactive Protein.
    *   `preop_lac`: Lactate.
*   **Derived Features**:
    *   `inpatient_preop`: Binary flag for inpatient admission prior to surgery.
    *   `preop_egfr_ckdepi_2021`: eGFR calculated using the CKD-EPI 2021 creatinine-only, race-free equation (per the National Kidney Foundation). The dataset-supplied `preop_gfr` column is intentionally excluded in favor of this derived value.
    *   Clinical flag helpers (e.g., `bun_high`, `hypoalbuminemia`) are computed for intermediate use but removed before the processed preop table is saved.

#### 2. Processing Steps
*   **Splitting**: Performs approximately 80/20 patient-grouped stratified splits for each trainable outcome and saves them as `split_group_any_aki`, `split_group_severe_aki`, `split_group_icu_admission`, and `split_group_mortality`. Outcomes that are too sparse to support a valid grouped holdout split are marked in the Step 03 metadata sidecar instead of blocking unrelated outcomes. A backward-compatible `split_group` alias still points at the AKI split.
*   **Deterministic only**: Step 03 now stops after raw feature derivation plus outcome-specific split assignment. No train-fitted encoding, rare-category collapsing, outlier clipping, or imputation rules are baked into `aki_preop_processed.csv`.
*   **Leakage control**: All learned preprocessing now happens inside model-time folds only, so inner CV, calibration, and holdout training all fit preprocessing statistics on the current training partition only.
*   **Missingness**: The saved processed tables keep `NaN` values by default. Step 03 still accepts `--impute-missing` for CLI compatibility, but that flag is now a no-op and does not modify the saved outputs.

```bash
python data_preparation/step_03_preop_prep.py
```

### Step 5: Intraoperative Data Prep
**File**: `data_preparation/step_04_intraop_prep.py`
Processes waveform features (pivoting, imputation) for both full and windowed modes.
*   **Technical Details**:
    *   **Pivoting**: Converts long-format Catch22 results into a wide format (one row per case/window).
    *   **Flattening**: Renames columns to `{waveform}_{feature}` (e.g., `SNUADC_PLETH_DN_HistogramMode_5`).
    *   **Missing Data**: Keeps `NaN` values for wholly missing waveform segments instead of zero-filling. Partial gaps remain
        linearly interpolated by upstream Catch22 extraction, and empty 10s windows are silently dropped as before.
```bash
python data_preparation/step_04_intraop_prep.py
```

### Step 6: Data Merge
**File**: `data_preparation/step_05_data_merge.py`
Combines preop and intraop data into master datasets.
*   **Technical Details**:
    *   Merges intraop features with processed preop data on `caseid` only, then carries all outcome, eligibility, and split metadata forward unchanged.
    *   **Integrity Check**: Validates one-to-one `caseid` coverage between the intraop and preop tables rather than dropping rows based on a single outcome split.
    *   **Missing Data**: Leaves NaNs in place by default. `--impute-missing` (or `IMPUTE_MISSING=True`) is a backward-compatible Step 05-only option that fills merge-introduced NaNs with `-99`; the primary nested-CV pipeline should leave NaNs intact.
    *   Outputs final wide CSVs ready for training.
```bash
python data_preparation/step_05_data_merge.py
# Optional legacy sentinel path during merge only
python data_preparation/step_05_data_merge.py --impute-missing
```

### Step 7: Model Training & Evaluation
**Directory**: `model_creation/`

`model_creation/step_07_train_evaluate.py` is now the primary entrypoint. By default it runs a leakage-hardened patient-grouped nested CV workflow (`--validation-scheme nested_cv`) rather than a single holdout split. Inputs retain NaNs by default; pass `--legacy_imputation` only if you intentionally want fold-local modeling-time imputation instead of preserving missing values for the estimators. This flag is separate from Step 03/05's backward-compatible `--impute-missing` switches.

*   **Primary validation mode**: `nested_cv` with patient-grouped `5 outer x 5 inner` CV, `repeats=1`, `max_workers=4`, `threads_per_model=8`, and Optuna `n_trials=100` per outer fit. `repeats > 1` is currently rejected because repeat-aware reporting is not implemented yet.
*   **Legacy mode**: `holdout` remains available via `--validation-scheme holdout`, but it now requires the persisted outcome-specific split created in Step 03. Downstream code no longer regenerates, guesses, or reuses stale generic `split_group` assignments for non-AKI outcomes.
*   **Fold-local preprocessing**: rare-category merging, one-hot encoding, numeric clipping, and optional imputation are all fit inside each training partition only. HPO, calibration, and threshold tuning never inspect held-out fold outcomes.
*   **Nested postprocessing**: after inner HPO, each outer fold generates grouped OOF predictions on outer-train, fits logistic recalibration on those predictions only, chooses the Youden-J threshold on calibrated outer-train OOF predictions only, then freezes both objects before scoring outer-test.
*   **Strict resume safety**: nested checkpoints are reused only when the stored validation fingerprint matches the current run configuration exactly; stale fold checkpoints are ignored and recomputed rather than mixed into the new run.
*   **Optional final refit**: `--save-final-refit` now works in both `nested_cv` and `holdout` mode and saves an additive full-data artifact bundle without changing the reported validation metrics.
*   **Standalone HPO helper**: `model_creation/step_06_run_hpo.py` remains available for ad hoc tuning on the persisted outcome-specific holdout training cohort, but the default experiment runner no longer requires a separate Step 06 pass. Invalid configs, stale processed artifacts, or missing holdout metadata now terminate Step 06 with a nonzero exit code instead of logging-and-continuing.
*   **Shared utilities**: `model_creation/validation.py`, `model_creation/preprocessing.py`, `model_creation/postprocessing.py`, and `model_creation/prediction_io.py` implement grouped splitting, fold-local preprocessing, calibration/thresholding, checkpointing, and prediction validation.

```bash
# Default primary analysis: nested 5x5 patient-grouped CV
python -m model_creation.step_07_train_evaluate \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type xgboost

# Legacy patient-grouped holdout
python -m model_creation.step_07_train_evaluate \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type xgboost \
  --validation-scheme holdout

# Optional standalone HPO on the holdout training cohort
python -m model_creation.step_06_run_hpo \
  --outcome any_aki \
  --branch windowed \
  --feature_set all_waveforms \
  --model_type ebm
```

**EBM explainability exports (global, local, per-term)**  
Step 7 now emits calibrated, publication-ready EBM interpretability artifacts by default when `--export_ebm_explanations` is set (the flag is auto-applied inside `run_experiments.sh` for all EBM runs; smoke tests still skip it). Behavior:
* Global: `global_explanation.json` plus styled term-importance plots (`global_importances.png`, `global_importances_plotly.{html,png}`) using display-dictionary labels, improved fonts, muted palettes, and light grids.
* Interactions: `interaction_importances.{json,png,html,png}` (empty when interactions=0, the default for stability).
* Local: `local_explanations.json` (raw + calibrated probabilities, logits, predicted labels, threshold, calibration params, case IDs), `local_attributions.csv` (reconciled table), and a diverging-color bar plot centered at zero (`local_contributions.png`).
* Per-feature partial plots: For every term, `terms/<feature-slug>/partial_dependence.{html,png}` plus `explanation.json`. Plots show contribution curves with a density overlay (secondary axis), styled legends, fonts, and grids. An `index.html` in `ebm_xai/` links to all plots.
* Robust export loop: per-term exports run in a thread pool with progress logs, per-term timeouts (90s), up to 2 retries, and non-blocking shutdown to avoid hangs. KeyboardInterrupt cancels outstanding tasks promptly.

**EBM HPO (small-N optimized)**  
During HPO we now prioritize speed and stability for ~2.5k rows: `inner_bags=0`, `outer_bags=1`, `max_bins∈{32,64,128,256}`, `max_leaves∈{2,3}`, `smoothing_rounds∈{0,25,50,75,100,150,200,350,500}`, `learning_rate∈{0.0025,0.005,0.01,0.015,0.02,0.03,0.04}`, `validation_size∈{0.1,0.15,0.2}`, `early_stopping_rounds∈{100,200}`, `early_stopping_tolerance∈{0,1e-5}`, `min_samples_leaf∈{2,3,4,5,10}`, `min_hessian∈{0,1e-6,1e-4,1e-2}`, `greedy_ratio∈{0,5,10}`, `cyclic_progress∈{0,1}`. Frozen bins are computed per fold-training subset and per candidate `max_bins`, not once on the full outer-train table. Final training in Step 7 keeps the heavier bagging defaults (`inner_bags=20`, `outer_bags=14`) while honoring the selected `max_bins`.

*Parallelism*: the default launcher keeps Optuna trial parallelism at `1` inside each nested task, parallelizes outer folds instead, and budgets the host at `max_workers x threads_per_model = 4 x 8 = 32` threads by default.

#### Full experiment grid (Catch22 + XGBoost/EBM)
Use the provided shell script to sweep the default grid of outcomes, branches, feature sets, and both model families. Logs are written to `experiment_log.txt`.

```bash
# Runs Catch22-based XGBoost + EBM models (primary pipeline)
./run_catch22_experiments.sh
```

**What happens during training & evaluation?**
*   **Nested default**: `predictions/test.csv` contains pooled outer-fold out-of-fold predictions, exactly one row per operation in the current supported workflow (`repeats=1`). In holdout mode it remains the single held-out test cohort.
*   **OOF Calibration**: Each outer fold fits logistic recalibration on grouped outer-train OOF predictions only; outer-test outcomes are never used to fit the calibrator.
*   **Threshold Selection**: Each outer fold chooses its own Youden-J threshold from calibrated outer-train OOF predictions only. Nested outputs therefore allow row-varying thresholds; reporting uses stored `y_pred_label` rather than assuming one global threshold per file.
*   **Artifacted Outputs**: For each configuration, the script saves `predictions/train_oof.csv`, `predictions/test.csv`, `artifacts/calibration.json`, `artifacts/threshold.json`, `artifacts/metadata.json`, and `artifacts/validation.json`. Nested runs additionally save per-fold checkpoints under `artifacts/folds/`; resume only reuses checkpoints whose stored validation fingerprint exactly matches the current run.
*   **Model-Type Specific Artifacts**: XGBoost runs export `model.json` plus SHAP figures, while EBM runs export `model.pkl` and can also emit calibrated explainability artifacts (`--export_ebm_explanations`).
    *   **EBM Explainability outputs** live under `results/catch22/experiments/models/ebm/{outcome}/{branch}/{feature_set}/artifacts/ebm_xai/` (also symlinked at `results/catch22/xai/ebm/`) and include:
        * `global_explanation.json`: raw `interpret` global explanation payload for all learned terms.
        * `global_importances.png` and `global_importances_plotly.(html|png)`: bar plots of term importances (Matplotlib + Plotly with Kaleido fallback).
        * `interaction_importances.(json|png|html|png)`: ranked interaction terms derived from the model’s native `term_features_` and `term_importances_`.
        * `local_explanations.json`: local `interpret` explanations augmented with case IDs, raw logits, raw/calibrated probabilities, predicted labels, and calibration parameters.
        * `local_attributions.csv`: reconciliation table aligning each test case ID with raw logits, calibrated probabilities, predicted label, and per-term contributions where available.
        * `README.md`: describes how calibrated probabilities were produced from raw logits (`p = sigmoid(intercept + slope * logit)`) and the threshold used for decisions.
    *   The CLI flag is ignored during smoke tests to keep runs lightweight.
*   **Fixed Application at Evaluation Time**: The stored calibrator and threshold are applied **without further fitting** when evaluating each outer test fold (or the held-out cohort in holdout mode); these same fixed row-level outputs are later reused by the reporting scripts.

#### Available Options
*   **Outcomes**: `any_aki`, `icu_admission`, and `mortality` are supported by the modular cohort code. The standard run scripts currently exercise `any_aki` and `icu_admission`; other retained labels remain archival unless launched explicitly.
*   **Branches**: `non_windowed` (Full Case), `windowed` (Segmented).
*   **Feature Sets**: `preop_only`, `all_waveforms`, `preop_and_all_waveforms`, `pleth_only`, `ecg_only`, etc.
*   **Model Types**: `xgboost` (default) or `ebm`.
*   **Default Grid (`run_experiments.sh`)**: Primary runs cover preop-only, single-waveform models (`pleth_only`, `ecg_only`, `co2_only`, `awp_only`), all waveforms, and fused preop + all waveforms. Ablations pair preop with each single waveform (`preop_and_<waveform>`) and with all waveforms minus one (`preop_and_all_minus_<waveform>`). Two-channel waveform-only combinations (e.g., **AWP+CO2** or **ECG+PLETH**) are intentionally excluded from default sweeps and should be launched manually if needed.
*   **Model Families**: `run_experiments.sh` now supports staging and reuse. By default it runs both `xgboost` and `ebm`, uses `nested_cv`, resumes completed outer folds when the validation fingerprint matches, and keeps configs sequential while parallelizing outer folds within each config. PREP modes remain `--prep auto` (default), `--prep force`, and `--prep skip`. `--prep auto` and `--prep skip` both validate the Step 01/03/05 `.metadata.json` sidecars before reuse and fail fast on stale processed artifacts; `--prep force` rebuilds them. Validation knobs (`--validation-scheme`, `--outer-folds`, `--inner-folds`, `--repeats`, `--max-workers`, `--threads-per-model`, `--n-trials`, `--resume`, `--save-final-refit`) are available directly on the shell runner.
*   **Launcher failure contract**: the main launcher finishes the requested grid, prints a compact summary of failed configurations, skips `metrics_summary`/`make_report` on partial runs, and exits nonzero if any validation job failed. This prevents stale artifacts from earlier runs being mistaken for current outputs.
*   **Reporting/Plotting Defaults**: `reporting.make_report` now runs with quantile calibration bins (`CALIBRATION_BIN_STRATEGY=quantile`, `CALIBRATION_N_BINS=10`), per-bin counts on the calibration curve, probability histograms beneath the calibration plot, auto x-axis zoom with a full-range inset, and parallel plotting (`PLOT_N_JOBS=-2`). PR curves render as step functions with a prevalence baseline; class-count annotations are off by default (`PR_SHOW_CLASS_BALANCE=false`). These defaults are exported by the run scripts; override via env if needed (e.g., `CALIBRATION_SHOW_BIN_COUNTS`, `CALIBRATION_MAX_COUNT_ANNOTATE`, `CALIBRATION_SHOW_PROB_HIST`, `CALIBRATION_SHOW_XLIM_INSET`, `PLOT_PREFER_CALIBRATED`, `PR_SHOW_CLASS_BALANCE`).

##### Running staged experiments

* Full fresh run (prep + both families, nested CV default):  
  `./run_experiments.sh --prep force`
* Holdout-only rerun for backward comparison:  
  `./run_experiments.sh --validation-scheme holdout --prep skip`
* XGBoost first, reuse data, skip EBM:  
  `./run_experiments.sh --only-xgboost --prep auto`
* Later EBM-only reuse of the same processed data:  
  `./run_experiments.sh --only-ebm --prep skip`

All CLI options also flow through `run_catch22_experiments.sh`, so you can pass the same flags there.

### Real-data smoke test (shell)
Use `run_smoke_test.sh` to exercise the full pipeline on a handful of real cases before launching the full experiment grid. The script writes all intermediate data and results to an isolated directory (`smoke_test_outputs/` by default) so production artifacts remain untouched.

`run_smoke_test.sh` uses the same `PYTHON_BIN` contract as the main launcher: pass a whitespace-separated command prefix such as `PYTHON_BIN='conda run -n aki_prediction_project python'` when the Conda environment is not already activated.

```bash
# Default: 10 cases, 2 Optuna trials, windowed all-waveform model
./run_smoke_test.sh

# Override knobs
CASE_LIMIT=5 HPO_TRIALS=1 SMOKE_ROOT=/tmp/aki_smoke RAW_SOURCE_DIR=/data/raw ./run_smoke_test.sh
```

Key behaviors:
* Activates environment variable overrides added to `data_preparation.inputs` and `model_creation.utils` so cohort, features, merged data, and results are written under `SMOKE_ROOT`.
* Trims the generated cohort to `CASE_LIMIT` rows with `data_preparation.smoke_trim_cohort`, preserving both classes for the active smoke outcomes only (`SMOKE_OUTCOMES`, default `any_aki,icu_admission`). The trim step does not try to guarantee unrelated sparse outcomes such as mortality.
* Runs Steps 1-5, then executes Step 07 in `holdout` mode for each smoke outcome (`--validation-scheme holdout --n-trials HPO_TRIALS --smoke_test`). Smoke intentionally uses the persisted Step 03 holdout splits rather than nested CV because the sampled cohort is too small for stable grouped inner/outer folds.
* Logs progress to `$SMOKE_ROOT/smoke_test.log` and leaves artifacts under `$SMOKE_ROOT/data/processed` and `$SMOKE_ROOT/results` for inspection.
* Reporting: smoke scripts now run `metrics_summary` with stratified/paired bootstrap (Δ vs `preop_only`, all cores) and `reporting/make_report` to emit the main + Δ tables/figures bundle.

### Step 8: Post-hoc Analysis & Visualization
**Primary Scripts**: `results_recreation/metrics_summary.py` → `reporting/make_report.py`

Generates publication-ready tables and figures for both the **Primary Pipeline** (Catch22/XGBoost) and the **Experimental Pipeline** (Aeon/Multirocket). Metrics are computed once from stored predictions/artifacts; no refitting happens at report time.

Display labels for outcomes, branches, feature sets, waveforms, and Catch22
statistics are centralized in `metadata/display_dictionary.json`. See
`reporting/DISPLAY_DICTIONARY.md` for the schema and helper utilities that keep
tables/figures synchronized.

*   **Artifact Validation & Aggregation**: `results_recreation/metrics_summary.py` crawls `RESULTS_DIR/models/**/predictions/test.csv`, confirms that the paired validation artifacts from Step 7 are present, and computes metrics directly from stored probabilities plus stored `y_pred_label`. In nested mode this means pooled outer-fold OOF predictions rather than a single held-out cohort. A consolidated metrics table is written to `results/catch22/paper/tables/metrics_summary.csv`; manuscript-ready companion exports are emitted as `metrics_summary.{md,docx,pdf}` (with the CSV still reachable via the `results/catch22/experiments/tables` symlink; optional bootstrap samples can be saved alongside).
*   **Calibration & Thresholds**: No report-time refitting occurs. Holdout runs use one frozen calibrator/threshold pair; nested runs preserve the per-fold calibrator/threshold decisions already applied in Step 7 and report from the stored row-level outputs.
*   **Uncertainty Estimation**: When prediction files include `subjectid`, bootstrap confidence intervals are computed with patient-clustered resampling so all operations from a sampled patient stay together within each replicate.
*   **Bootstrapping**: Non-parametric, outcome-stratified bootstrap of the saved evaluation predictions (default 1000 reps). Bootstrap samples are **paired across models within each Outcome × Branch × Pipeline group**, enabling Δ CIs. Default reference for Δ is the `preop_only` feature set.
    * Parallelization defaults to the process backend with `PARALLEL_BACKEND=processes` in the run shells; retries and timeouts are baked in (`--bootstrap-timeout 1800`, `--bootstrap-max-retries 2`) so a stuck pool falls back to threads, then sequential, instead of hanging.
*   **Unified Reporting**: `reporting/make_report.py` consumes `metrics_summary.csv` to generate Markdown, Word, PDF, and HTML tables plus ROC/PR/Calibration figures across both pipelines. Reports display distinct rows for each model type (`xgboost` vs. `ebm`) within every Outcome × Branch × Feature Set combination, and add a separate delta table (Δ vs reference) with heatmap shading only when the Δ CI excludes 0.
*   **Outputs**:
    *   **Reports**:
        *   `results/catch22/paper/reports/report.md`: Scan-friendly manuscript index of all consolidated result tables.
        *   `results/catch22/paper/reports/report.docx`: Formatted Word document containing all results tables with selective bolding and background gradients.
        *   `results/catch22/paper/reports/report.pdf`: Aggregated PDF report of all tables.
        *   `results/catch22/paper/tables/results_*.{html,md,docx,pdf}` plus `results_*_{main,delta}.csv`: Individual table bundles for each outcome/branch/model.
    *   **Figures**: High-quality ROC, PR, and Calibration curves saved in `results/catch22/paper/figures/` as `.svg` and publication `.png`.
    *   **Data**: `results/catch22/paper/tables/metrics_summary.csv` containing all calculated metrics and confidence intervals, plus optional bootstrap Parquet files for downstream analysis.
```bash
python results_recreation/metrics_summary.py \
  --delta-mode reference \
  --reference-feature-set preop_only \
  --parallel-backend processes \
  --bootstrap-timeout 1800 \
  --bootstrap-max-retries 2 \
  --n-jobs -1       # Precompute consolidated metrics + Δs using all cores

python reporting/make_report.py  # Build figures and reports from the precomputed CSV
```

### Additional reporting utilities
These helpers live under `reporting/` and reuse the shared display dictionary
(`metadata/display_dictionary.json`) so figure and table labels remain
consistent across manuscripts and dashboards.

- **Cohort flow diagram** — Recreate the consort-style flow from saved counts
  emitted by `step_01_cohort_construction.py`. The renderer requires Graphviz
  (`dot`); keep the project environment synced with
  `conda env update -f environment.yml --prune`. It skips no-op/increasing
  steps, uses friendly labels (waveforms, filters), groups all waveform checks into
  one stage with a footnote, and shows rightward dashed exclusion boxes labeled with
  removal reasons and counts. Two direct arrows connect the Final Cohort box to the
  AKI vs. No AKI split boxes when `label_split` is present in the JSON (default:
  `results/catch22/paper/metadata/cohort_flow_counts.json`). The `filter_preop_cr`
  stage is labeled as baseline creatinine eligibility because it excludes
  `preop_cr > 4.0 mg/dL`. Outputs are written as DOT/SVG/PNG under
  `results/catch22/paper/figures/`.

  ```bash
  python -m reporting.cohort_flow \
    --counts-file results/catch22/paper/metadata/cohort_flow_counts.json \
    --display-dictionary metadata/display_dictionary.json
  ```

- **Preoperative descriptive table** — Summarize baseline characteristics using
  the raw cohort CSV or `aki_preop_processed.csv`, both of which now retain the
  raw categorical columns needed for counts. The default `--report-cohort paper_default`
  view matches the AKI paper cohort rather than the broader shared outer cohort.
  Each continuous variable
  undergoes a Shapiro–Wilk test; report mean ± SD if normal, otherwise median (IQR).
  Binary categoricals render as False/True; all labels/units come from the display
  dictionary. Outputs: CSV/Markdown/DOCX/PDF plus preserved HTML/LaTeX at
  `results/catch22/paper/tables/preop_descriptives.*`.

  ```bash
  python -m reporting.preop_descriptives \
    --dataset data/processed/aki_pleth_ecg_co2_awp.csv \
    --processed-dataset data/processed/aki_preop_processed.csv \
    --report-cohort paper_default \
    --display-dictionary metadata/display_dictionary.json
  ```

- **Model-feature missingness table** — Computes per-column missingness for the
  merged modeling dataset (default: `data/processed/aki_features_master_wide.csv`)
  while skipping identifiers and outcomes. Columns are human-readable (friendly
  headers, one-hot labels resolved via the display dictionary) and written to
  CSV/Markdown/DOCX/PDF plus preserved HTML in `results/catch22/paper/tables/`.

  ```bash
  python -m reporting.missingness_table \
    --dataset data/processed/aki_features_master_wide.csv \
    --display-dictionary metadata/display_dictionary.json
  ```

---

## 🕰️ Legacy Execution Guide (Notebook Based)

Use this approach if you need to replicate older results. Note that this method has potential data leakage risks due to inconsistent splitting between notebooks.

### Step 1-3: Same as above (Cohort & Feature Extraction)

### Step 3: Data Wrangling, Preoperative data extraction and preprocessing
**File**: `notebooks/03_tabular_data_prep.ipynb`

Open this notebook to prepare data for the next step.
*   **Technical Details**:
    *   Performs data transformation (pivots from long to wide format) for the intraoperative data.
    *   Extracts and cleans preoperative clinical data.
    *   **Note**: This legacy method may not enforce the strict train/test split boundaries used in the new pipeline. 

### Step 4: Model Training & HPO
**File**: `notebooks/04_hpo_xgboost.ipynb`

Open this notebook to train the model.
*   **Technical Details**:
    *   **Monolithic Workflow**: Performs data wrangling, preprocessing, HPO, and training in a single notebook execution.
    *   **HPO**: Uses Optuna for hyperparameter tuning.
    *   **Visualization**: Includes code for plotting ROC curves and SHAP values inline.

**To run:**
```bash
jupyter notebook notebooks/04_hpo_xgboost.ipynb
```

## 📂 Directory Structure

*   `data_preparation/`: Python scripts for the data pipeline.
    *   `inputs.py`: Global configuration and AEON export settings.
    *   `step_01_cohort_construction.py`: Cohort filtering.
    *   `step_02_catch_22.py`: Feature extraction.
    *   `step_03_preop_prep.py`: Preop processing & splitting (New).
    *   `step_04_intraop_prep.py`: Intraop processing (New).
    *   `step_05_data_merge.py`: Data merging (New).
    *   `aeon_io.py`: **(New)** Helper for exporting data to AEON formats (Nested, 3D NumPy) with padding support.
    *   `waveform_processing.py`: Helper functions for signal processing.
*   `model_creation/`: **(New)** Modular modeling pipeline.
    *   `utils.py`: Shared logic for data loading and preprocessing.
    *   `step_06_run_hpo.py`: Hyperparameter optimization script (Optuna, AUPRC objective).
    *   `step_07_train_evaluate.py`: Final training, OOF calibration, Youden-J threshold selection, and artifacted prediction export.
    *   `postprocessing.py`: Logistic recalibration, patient-grouped stratified OOF prediction helpers, threshold search, and JSON persistence utilities.
    *   `prediction_io.py`: Prediction file validation and standardized CSV writing for train/test splits.
*   `model_creation_aeon/`: **(New)** Experimental Aeon pipeline.
    *   `classifiers.py`: Custom `FusedClassifier`, `RocketFused`, and `FreshPrinceFused` classes implementing early fusion.
    *   `step_06_aeon_train.py`: CLI-driven script for training Aeon models.
*   `notebooks/`: Jupyter notebooks.
    *   `03_tabular_data_prep.ipynb`: Legacy data transformation + preoperative data extraction notebook.
    *   `04_hpo_xgboost.ipynb`: Legacy modeling notebook.
*   `data/`: Data storage (raw and processed).
*   `results/`: Model outputs (params, metrics, plots).

## 🧪 Experimental Pipeline: Aeon (Time Series Classification)

> **Note**: This pipeline operates separately from the main Catch22/XGBoost pipeline and is treated as **experimental**. For production or publication runs, prefer the Catch22 + XGBoost/EBM scripts above.

This branch implements an end-to-end Deep Learning/State-of-the-Art Time Series Classification pipeline using the **Aeon** library. It bypasses manual feature engineering (Catch22) in favor of direct waveform processing and fusion.

### Overview
*   **Resampling**: All waveforms are resampled to **1 Hz** (1 data point per second) to preserve temporal structure while managing dimensionality.
*   **Padding**: Variable-length sequences are right-padded with zeros to a fixed maximum length of **16 hours** (57,600 timepoints).
*   **Fusion**: Early fusion of these 1 Hz waveforms (transformed via Rocket) with 115 static preoperative features. FreshPRINCE is currently disabled due to computational constraints.
*   **Goal**: Compare manual feature engineering (Catch22+XGB) against SOTA Time Series Classifiers (Rocket, FreshPRINCE).
*   **Fusion Strategy**: **Early Fusion**. Preoperative tabular features are concatenated with waveform embeddings before the final classifier head.

### Pipeline Steps
### Pipeline Components

| Component | Script | Description |
| :--- | :--- | :--- |
| **Export** | `data_preparation/step_02_aeon_export.py` | Loads waveforms, rescales to 1 Hz, and writes per-case sequences padded/truncated to **57,600** samples (16 hours) in `.npz` format. <br> **Args**: `--limit` (debug) |
| **Preop Prep** | `data_preparation/step_04_aeon_prep.py` | Prepares tabular data: preserves `NaN` by default; add `--impute-missing` to apply median imputation with missingness indicators. This branch remains experimental and is not guaranteed to mirror the modular Catch22 cohort/split semantics exactly. |
| **Training** | `model_creation_aeon/step_06_aeon_train.py` | Trains separate or fused models. <br> **Models**: `multirocket` (default `n_kernels=10000`), `minirocket`, `freshprince`. <br> **HPO**: Optuna optimization (100 trials) for linear head (`C`) maximizing AUPRC. Class weight is fixed to 'balanced'. <br> **Fusion**: Concatenates preop features with Rocket embeddings. <br> **Outputs**: Saves `predictions.csv` for unified analysis. |
| **Reference** | `model_creation_aeon/classifiers.py` | Contains `RocketFused` and `FreshPrinceFused` class definitions. |
| **Analysis** | `results_recreation/metrics_summary.py` | Aggregates saved `predictions/test.csv` from the Aeon runs (same schema as Catch22), then feeds `reporting/make_report.py` for tables/figures. |

### Technical Specifications
*   **Library**: `aeon` (Sktime fork), `tsfresh`.
*   **Input Shape**: `(N_samples, N_channels=4, Length=57,600)` after 1 Hz resampling with right-padding/truncation to 16 hours.
*   **Fusion Type**: Early Fusion (Preop features concatenated to transform embeddings).
*   **Evaluation**:
    *   **Outcomes**: `any_aki` (primary) and `icu_admission` (secondary), matching the maintained Catch22 pipeline scope. Other labels in the cohort are not part of current Aeon runs.
    *   **Metrics**: 1000-fold bootstrapped CIs for AUROC, AUPRC, etc.
    *   **Ablations**: Single Channel, Leave-One-Out, and Fusion impact analysis.

### 5. Troubleshooting & Learnings
*   **Aeon v1.1.0 Compatibility**:
    *   The pipeline explicitly supports `aeon >= 1.1.0`. `MultiRocket` and `MiniRocket` are imported from `aeon.transformations.collection.convolution_based`.
    *   The correct parameter for kernel count is `n_kernels` (not `num_kernels`).
    *   Estimators must be passed as instantiated Objects
    *   **Head**: A **Logistic Regression** classifier is trained on the fused, standardized feature set.
    *   **Optimization (HPO)**: We optimize the linear head using **Optuna** with 5-fold stratified cross-validation on the training set.
        *   **Objective**: Maximize **AUPRC**.
        *   **Search Space**:
            *   Regularization Strength (`C`): Log-uniform distribution [1e-3, 10.0].
            *   Class Weight (`class_weight`): Fixed to 'balanced'.
        *   **Scaling**: A `StandardScaler` is fitted within each CV fold to prevent data leakage.
        *   **Trials**: 100 trials.
*   **Class Balance in Testing**:
    *   When running with `--limit` (e.g., smoke testing), the export script enforces a balanced selection of positive and negative cases. This is critical because `LogisticRegression` will crash if the training fold contains only a single class.
*   **Performance**:
    *   Crucial: Ensure `n_jobs=-1` is passed to the Aeon transformer. For HPO, we utilize **parallel Optuna trials** (`n_jobs=-1`) with **sequential** scikit-learn fits (`n_jobs=1`).
    *   **Threading**: We explicitly set `OMP_NUM_THREADS=1` (and related BLAS variables) to prevent thread contention. This ensures that the 32 parallel trials (processes) each use a single core efficiently, rather than each trial attempting to spawn multiple threads.
*   **Convergence**:
    *   **Max Iterations**: The default `lbfgs` solver in `LogisticRegression` may fail to converge on the high-dimensional MiniRocket features (10k+). We increased `max_iter` to **5000** to resolve `ConvergenceWarning`s.

### Execution
Run the full experimental suite:
```bash
bash run_full_pipeline_aeon.sh
```
Or individual experiments:
```bash
bash run_experiments_aeon.sh
```
The Aeon experiment script currently evaluates the `any_aki` and `icu_admission` outcomes. Treat it as an exploratory branch rather than a validated paper path; the modular outcome-specific split contract described above is enforced in the Catch22 pipeline, not retrofitted across every Aeon helper.
