# Methods

## Study Design and Data Source
This retrospective observational study used high-resolution intraoperative data from **VitalDB** (Vital Signs DataBase), an open-access repository of 6,388 non-cardiac surgical cases from Seoul National University Hospital.

## Cohort Selection
The saved Step 01 cohort is now a **shared outer cohort** used across outcomes. Patients are included in that outer cohort when all of the following are satisfied:
1.  **Data availability**: Operation end time (`opend`) was present.
2.  **Waveform availability (strict)**: All four high-fidelity intraoperative channels were required - `SNUADC/PLETH`, `SNUADC/ECG_II` (with `SNUADC/ECG_V5` substitution when necessary), `Primus/CO2`, and `Primus/AWP`. Intersection logic across VitalDB case lists enforced a complete four-channel set, and missing channels removed the case. Each segment was drawn from `opstart`-`opend`; segments with >5% NaNs were discarded and shorter gaps were linearly interpolated.
3.  **Clinical criteria**:
    * ASA physical status V-VI excluded.
    * Multiple surgeries per patient are retained, but every hold-out split and cross-validation fold is grouped on patient identifier (`subjectid`) so no patient spans train/validation/test partitions.

Outcome-specific eligibility is then layered on top of the shared outer cohort:
* **AKI / severe AKI cohort**: Requires preoperative creatinine present, preoperative creatinine <= 4.0 mg/dL, and at least one postoperative creatinine measurement within 7 days of surgery.
* **ICU admission / mortality cohort**: Uses the shared outer cohort only and requires a non-missing source label; no creatinine availability requirement is imposed.

**Outcome Definition**
* **Primary**: Postoperative AKI per **KDIGO** creatinine criteria - any rise >= 0.3 mg/dL within 48 hours or >= 1.5x baseline within 72 hours.
* **Secondary (active grid)**: Postoperative ICU admission (`y_icu_admit`, length of stay > 0 days).
* **Additional supported label**: In-hospital mortality (`y_inhosp_mortality`) is retained in the shared cohort and can be trained on under the broader non-AKI cohort definition.
* **Archival labels**: Severe AKI (`y_severe_aki`) remains available under the AKI-eligible cohort, and prolonged postoperative length of stay (`y_prolonged_los_postop`) remains archival only because its current threshold is cohort-derived.

## Data Preprocessing

### Clinical Data
Preoperative variables were drawn from VitalDB clinical and laboratory tables. Last-observation values within 30 days pre-surgery were used for labs (`preop_wbc`, `preop_crp`, `preop_lac`) via `lab_data.csv`, and merged to the cohort.

#### Variable Definitions
* **Demographics**: Age, sex, height, weight, BMI.
* **Surgical context**: Emergency operation (`emop`), department, approach, ASA class, operation type (`optype`), anesthesia type (`ane_type`).
* **Comorbidities/tests**: Hypertension, diabetes, ECG findings, pulmonary function tests.
* **Laboratory values**: Hemoglobin, platelets, INR (`preop_pt`), aPTT, sodium, potassium, glucose, albumin, AST, ALT, BUN, creatinine, bicarbonate, ABG components (pH, base excess, PaO2, PaCO2, SaO2), plus CRP, lactate, and WBC from `lab_data.csv`.

#### Derived Features
* **Preoperative admission**: `inpatient_preop = adm < 0`.
* **eGFR (CKD-EPI 2021)**: Creatinine converted to mg/dL with unit checks (>20 assumed umol/L/88.4); nonpositive values set missing. Formula mirrors the National Kidney Foundation specification. The raw `preop_gfr` column is excluded.
* **Physiology flags (QA only)**: High BUN (>27), hypoalbuminemia (<3.5), sex-specific anemia (Hb <13.0 male, <12.0 female), hyponatremia (<135), metabolic acidosis (HCO3 <22 or BE < -2), hypercapnia (PaCO2 >45), hypoxemia (PaO2 <80 or SaO2 <95). These helpers are dropped before modeling.

#### Split and Fold-local Preprocessing
* **Hold-out metadata**: Step 03 creates one patient-grouped, approximately 80/20 stratified split per trainable outcome (`split_group_any_aki`, `split_group_severe_aki`, `split_group_icu_admission`, `split_group_mortality`). Each split is created only inside that outcome's eligible cohort. Outcomes that cannot support a valid grouped holdout split in a given artifact are marked `unsupported_in_artifact` in the Step 03 metadata sidecar rather than blocking other outcomes. A backward-compatible `split_group` alias is retained for the AKI split only after the active outcome-specific split has been validated. All downstream merges preserve these columns, and no `subjectid` may appear in both train and test within any outcome-specific split.
* **No learned preprocessing in Step 03**: The saved `data/processed/aki_preop_processed.csv` retains raw categorical columns, raw numeric columns, derived features, outcome eligibility columns, and split metadata, but it does not apply train-fitted encoding, rare-category collapsing, clipping, or imputation.
* **Processed artifact compatibility**: Step 01, Step 03, and Step 05 each emit a `.metadata.json` sidecar with a schema version and column inventory. Downstream loaders verify these sidecars and fail fast with rebuild instructions when stale processed artifacts are reused.
* **Categoricals**: In modeling, departments with <30 training rows within the current training partition collapse to `other`, then one-hot encoding is fit on that same partition only; validation/test rows use `handle_unknown="ignore"`.
* **Numeric preprocessing**: Continuous preoperative variables are clipped using quantiles learned on the current training partition only. Optional imputation, when requested, also fits on the current training partition only.
* **Imputation**: Default behavior preserves `NaN`. Step 03's `--impute-missing` flag is now a backward-compatible no-op, and Step 05's `--impute-missing` flag remains a backward-compatible sentinel-fill path for merged tables only. Fold-local modeling-time imputation in the Catch22 branch is controlled by `--legacy_imputation` on Step 06/07. Aeon fusion models still apply their own downstream median-imputation path after replacing any sentinel values with `NaN`.

### Waveform Signal Processing (Catch22 Branch)
Filtering and resampling follow channel-specific specifications in `data_preparation/waveform_processing.py`, grounded in prior signal-processing literature:
* **PPG (SNUADC/PLETH)**: 500 Hz native -> band-pass 0.1-10 Hz (4th-order Butterworth) -> resample to 100 Hz (Lapitan 2024 Sci Rep; Park 2022 Front Physiol).
* **ECG (SNUADC/ECG_II or V5)**: 500 Hz native -> band-pass 0.5-40 Hz (4th-order Butterworth) -> resample to 100 Hz (Kligfield 2007 Circulation; Pan and Tompkins 1985 IEEE TBME).
* **Capnography (Primus/CO2)**: 62.5 Hz native -> low-pass 8 Hz (4th-order Butterworth) -> retained at 62.5 Hz (Gutierrez 2018 PLoS One; Leturiondo 2017 CinC).
* **Airway pressure (Primus/AWP)**: 62.5 Hz native -> low-pass 12 Hz (4th-order Butterworth) -> retained at 62.5 Hz (de Haro 2024 Crit Care; Thome 1998 J Appl Physiol).

Segments are cut at native resolution between `opstart` and `opend`, interpolated if <=5% NaNs, and rejected otherwise. Full-case Catch24 features are computed after downsampling the filtered signal to 10 Hz for efficiency. Windowed features use the filtered/resampled signal at its channel-specific target sampling rate.

## Feature Engineering
We use **Catch24** (22 canonical Catch22 features plus mean and standard deviation) per channel.
1. **Full-case (non-windowed)**: Catch24 on the entire surgery after 10 Hz downsampling.
   * Per case, this yields 24 features per channel (96 across four channels) before merging with preop features.
2. **Windowed**: 10-second windows with 5-second overlap; per-window Catch24 features are aggregated by mean, standard deviation, minimum, and maximum for each feature/channel, yielding fixed-length vectors.
   * Each channel produces 24 features x 4 statistics = 96 aggregated features; with four channels this yields 384 intraoperative features prior to fusion with preop data.
3. **Wide assembly**: Long-form waveform features are pivoted to wide format (`data_preparation/step_04_intraop_prep.py`) and merged with preop data on `caseid` only (`step_05_data_merge.py`). The merged tables carry outcome labels, eligibility flags, and split metadata forward unchanged. Non-windowed and windowed branches are saved as `aki_features_master_wide.csv` and `aki_features_master_wide_windowed.csv`.

## Model Development

**Branches and feature sets**: Both non-windowed and windowed branches are trained. Feature grids include `preop_only`, single-waveform sets (`pleth_only`, `ecg_only`, `co2_only`, `awp_only`), `all_waveforms`, and `preop_and_all_waveforms`, plus ablations `preop_and_<waveform>` and `preop_and_all_minus_<waveform>`. Two-channel waveform-only models are intentionally excluded outside the all-waveform condition.

**Validation design**:
* The primary internal validation procedure is patient-grouped nested cross-validation with 5 outer folds and 5 inner folds (`repeats=1` by default). Repeated nested CV is intentionally disabled in the current implementation (`repeats > 1` raises an error) until repeat-aware aggregation and uncertainty estimation are added. The legacy single holdout remains available as a secondary mode but is no longer the default analysis.
* Every split, calibration fold, threshold-selection fold, and bootstrap resample is grouped on `subjectid` to prevent patient-level leakage when multiple operations belong to the same patient.
* Outer test folds are used exactly once: after model selection, calibration fitting, and threshold selection have already been completed on the corresponding outer-training cohort.
* Interrupted nested runs may resume from fold checkpoints only when the stored validation fingerprint exactly matches the active run configuration; otherwise stale checkpoints are ignored and recomputed.
* The primary Catch22 shell launcher (`run_experiments.sh`) treats any failed validation configuration as a failed run: it completes the requested grid, skips pooled reporting on partial runs, and exits nonzero so stale artifacts are not mistaken for current evidence.

**XGBoost**:
* Objective `binary:logistic`, `tree_method=hist`, `eval_metric=aucpr`.
* Optuna HPO runs inside the current training cohort with patient-grouped stratified CV (up to 5 folds; reduced when the minority class or unique patient count is small). `n_estimators` is tuned directly; the scored validation fold is not reused for early stopping. `scale_pos_weight` is recomputed inside each fold-training subset rather than once globally.
* Final models refit on the full outer-training data (or the full dataset for the optional final refit). SHAP summary plots are produced for holdout evaluations and optional final refits.

**Explainable Boosting Machines**:
* Optuna search space: `max_bins in {32,64,128,256}`, `max_leaves in {2,3}`, `smoothing_rounds in {0,25,50,75,100,150,200,350,500}`, `learning_rate in {0.0025-0.04}`, `validation_size in {0.1,0.15,0.2}`, `early_stopping_rounds in {100,200}`, `early_stopping_tolerance in {0,1e-5}`, `min_samples_leaf in {2,3,4,5,10}`, `min_hessian in {0,1e-6,1e-4,1e-2}`, `greedy_ratio in {0,5,10}`, `cyclic_progress in {0,1}` with `interactions=0`, `missing="gain"`, `inner_bags=0`, and `outer_bags=1`.
* Frozen bins are computed inside each fold-training subset only, never once on the full outer-training data, so inner validation rows do not influence EBM discretization. The fold cache keys these bins by the candidate `max_bins` value so HPO, grouped calibration, outer-fold refits, and optional full-data refits all use a coherent discretization contract.
* HPO itself stays on the lighter bagging settings above for runtime control, but post-HPO grouped calibration fits, outer-fold final models, and optional full-data refits switch to `inner_bags=20` and `outer_bags=14` while honoring the selected `max_bins`. Optional EBM XAI exports include global/interaction plots, local attributions, per-term partial dependence, and an HTML index with display-dictionary labels; term exports run in a bounded thread pool with timeouts and retries.

## Post-hoc Analysis and Statistical Evaluation
* **Calibration and thresholding**: Within each outer fold, patient-grouped out-of-fold predictions on the outer-training cohort feed a logistic recalibration model (`calibration.json`). Calibrated outer-train OOF probabilities are thresholded by Youden's J statistic (`threshold.json`). The locked calibrator and threshold are then applied unchanged to the corresponding outer-test fold; no outer-test outcomes are used for recalibration or threshold selection. In nested mode, pooled `test.csv` predictions must contain exactly one row per operation.
* **Evaluation target**: Performance is reported at the operation level using pooled outer-fold predictions (or the legacy holdout test cohort when holdout mode is invoked). Primary metric is AUPRC; secondary metrics include AUROC, F1, sensitivity, specificity, accuracy, precision, and Brier score computed from calibrated probabilities and stored binary labels.
* **Bootstrapping**: `results_recreation/metrics_summary.py` validates artifacts, then runs 1,000 stratified, patient-clustered non-parametric bootstrap resamples of the saved evaluation predictions (configurable). Patients are resampled with replacement and all of each sampled patient's operations are retained in each replicate. Percentile (2.5, 97.5) CIs are attached per Outcome x Branch x Feature Set x Model, with paired delta-CIs against a reference (default `preop_only`) when predictions share identical case and patient ordering. Joblib process pools are used with guarded fallbacks to threads/sequential execution and retryable timeouts.
* **Interpretability**: XGBoost SHAP summaries are saved for test data. EBM relies on native additive explanations when exports are enabled; SHAP is not run for EBMs.

## Reporting Artifacts
All figures and tables regenerate from saved artifacts using `metadata/display_dictionary.json` for consistent labels/units.
* **Cohort flow**: `reporting/cohort_flow.py` converts `results/metadata/cohort_flow_counts.json` into Graphviz-backed DOT/SVG/PNG consort diagrams, consolidating waveform checks into a single availability box, labeling the baseline creatinine eligibility step as `preop_cr <= 4.0 mg/dL`, and rendering the terminal AKI split as two direct arrows from the final cohort box when present.
* **Baseline table**: `reporting/preop_descriptives.py` uses raw cohort data or `aki_preop_processed.csv`, both of which retain raw categorical variables after Step 03; Shapiro-Wilk testing guides mean+/-SD vs. median[IQR] display. The default `--report-cohort paper_default` view matches the AKI paper cohort rather than the broader shared outer cohort. Outputs: HTML/LaTeX/DOCX at `results/tables/preop_descriptives.*`.
* **Missingness table**: `reporting/missingness_table.py` summarizes feature-level missingness from `data/processed/aki_features_master_wide.csv` to CSV/HTML for supplements and QA.

## Experimental Pipeline: Time-Series Classification (Aeon)
This parallel pipeline benchmarks modern time-series classifiers against the Catch22/XGBoost baseline. It remains experimental and is not the validated paper path for the modular outcome-specific cohort workflow.

### Data Preprocessing (Aeon)
* Waveforms are filtered as above, resampled to channel-specific targets, then downsampled to **1 Hz** per case; cases with >5% NaNs are dropped, and remaining gaps are interpolated.
* Series are right-padded with zeros to **57,600 samples (16 hours)** with a strict four-channel requirement; longer cases are truncated. Labels and channels are saved to `outputs/aeon/X_nonwindowed.npz` and `y_nonwindowed.csv`.
* Preoperative features for Aeon are aligned to the retained cases, augmented with `anesthesia_duration_minutes`, and median-imputed with missingness indicators before linear heads are trained.

### Modeling Strategy (Early Fusion)
* **MiniRocket/MultiRocket**: 10,000 convolutional kernels (Aeon implementations) produce fixed embeddings. Embeddings are concatenated with imputed preoperative vectors when `--include_preop` is set (early fusion), scaled with `StandardScaler`, and fed to a balanced logistic regression head. Optuna (100 trials, 5-fold stratified CV) tunes `C` over [1e-3, 10] to maximize AUPRC.
* **FreshPRINCE**: TSFresh-based feature extraction with a 200-tree Rotation Forest head for fused inputs (no HPO).
* **Calibration and inference**: Out-of-fold predictions undergo the same logistic recalibration and Youden-J thresholding as the primary pipeline. Test predictions are fixed-calibrated; no test-time recalibration.

### Evaluation
Outcomes mirror the main pipeline (`any_aki`, `icu_admission`). Ablations cover single-channel, leave-one-out, and fusion vs. waveform-only comparisons. Test-set metrics and 1,000-rep stratified bootstraps (paired when case sets match) are produced via `results_recreation/metrics_summary.py`, enabling delta-CIs against the preoperative-only reference feature set. No BCa adjustments are applied.
