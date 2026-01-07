# Methods

## Study Design and Data Source
This retrospective observational study used high-resolution intraoperative data from **VitalDB** (Vital Signs DataBase), an open-access repository of 6,388 non-cardiac surgical cases from Seoul National University Hospital.

## Cohort Selection
Patients were included when all of the following were satisfied:
1.  **Data availability**: Preoperative creatinine (`preop_cr`) and operation end time (`opend`) were present.
2.  **Waveform availability (strict)**: All four high-fidelity intraoperative channels were required - `SNUADC/PLETH`, `SNUADC/ECG_II` (with `SNUADC/ECG_V5` substitution when necessary), `Primus/CO2`, and `Primus/AWP`. Intersection logic across VitalDB case lists enforced a complete four-channel set, and missing channels removed the case. Each segment was drawn from `opstart`-`opend`; segments with >5% NaNs were discarded and shorter gaps were linearly interpolated.
3.  **Clinical criteria**:
    * At least one postoperative creatinine measurement within 7 days of surgery (filtering performed before labeling).
    * Preoperative creatinine <= 4.0 mg/dL to exclude baseline end-stage kidney disease.
    * ASA physical status V-VI excluded.
    * Sample independence enforced by randomly selecting one surgery per patient (`subjectid`, seed=42).

**Outcome Definition**
* **Primary**: Postoperative AKI per **KDIGO** creatinine criteria - any rise >= 0.3 mg/dL within 48 hours or >= 1.5x baseline within 72 hours.
* **Secondary (active grid)**: Postoperative ICU admission (`y_icu_admit`, length of stay > 0 days).
* **Archival labels**: Severe AKI (`y_severe_aki`), in-hospital mortality (`y_inhosp_mortality`), and prolonged postoperative length of stay (`y_prolonged_los_postop`) remain available but are not trained in the current experiment sweep.

## Data Preprocessing

### Clinical Data
Preoperative variables were drawn from VitalDB clinical and laboratory tables. Last-observation values within 30 days pre-surgery were used for labs (`preop_wbc`, `preop_crp`, `preop_lac`) via `lab_data.csv`, and merged to the cohort.

#### Variable Definitions
* **Demographics**: Age, sex, height, weight, BMI.
* **Surgical context**: Emergency operation (`emop`), department (rare levels <30 merged to "other" using training counts), approach, ASA class, operation type (`optype`), anesthesia type (`ane_type`).
* **Comorbidities/tests**: Hypertension, diabetes, ECG findings, pulmonary function tests.
* **Laboratory values**: Hemoglobin, platelets, INR (`preop_pt`), aPTT, sodium, potassium, glucose, albumin, AST, ALT, BUN, creatinine, bicarbonate, ABG components (pH, base excess, PaO2, PaCO2, SaO2), plus CRP, lactate, and WBC from `lab_data.csv`.

#### Derived Features
* **Preoperative admission**: `inpatient_preop = adm < 0`.
* **eGFR (CKD-EPI 2021)**: Creatinine converted to mg/dL with unit checks (>20 assumed umol/L/88.4); nonpositive values set missing. Formula mirrors the National Kidney Foundation specification. The raw `preop_gfr` column is excluded.
* **Physiology flags (QA only)**: High BUN (>27), hypoalbuminemia (<3.5), sex-specific anemia (Hb <13.0 male, <12.0 female), hyponatremia (<135), metabolic acidosis (HCO3 <22 or BE < -2), hypercapnia (PaCO2 >45), hypoxemia (PaO2 <80 or SaO2 <95). These helpers are dropped before modeling.

#### Split, Encoding, and Outliers
* **Hold-out split**: A single 80/20 stratified split on `aki_label` is created before any encoding or winsorization and stored as `split_group`; all downstream merges preserve this split.
* **Categoricals**: Departments with <30 training cases collapse to "other". One-hot encoding is fit using training levels; test columns are aligned to the training design matrix.
* **Outlier handling**: Training-set percentiles drive winsorization. Values below the 1st percentile are replaced with draws from the [0.5th, 5th] percentile range; values above the 99.5th percentile are replaced using [95th, 99.5th] percentiles. Percentiles are computed on numeric training data only.
* **Imputation**: Default behavior preserves `NaN`. An opt-in flag (`--impute-missing` or `IMPUTE_MISSING=True`) restores the legacy sentinel (-99) for compatibility with older workflows. Aeon fusion models invoke median imputation with missingness indicators (`SimpleImputer(add_indicator=True)`) after replacing any sentinel values with `NaN`.
* **Outputs**: Preop processing saves `data/processed/aki_preop_processed.csv` with `split_group` retained for all downstream branches.

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
3. **Wide assembly**: Long-form waveform features are pivoted to wide format (`data_preparation/step_04_intraop_prep.py`) and merged with preop data on `caseid`, `aki_label`, and `split_group` (`step_05_data_merge.py`). Rows missing `split_group` after the merge are dropped to preserve the original split. Non-windowed and windowed branches are saved as `aki_features_master_wide.csv` and `aki_features_master_wide_windowed.csv`.

## Model Development

**Branches and feature sets**: Both non-windowed and windowed branches are trained. Feature grids include `preop_only`, single-waveform sets (`pleth_only`, `ecg_only`, `co2_only`, `awp_only`), `all_waveforms`, and `preop_and_all_waveforms`, plus ablations `preop_and_<waveform>` and `preop_and_all_minus_<waveform>`. Two-channel waveform-only models are intentionally excluded outside the all-waveform condition.

**XGBoost**:
* Objective `binary:logistic`, `tree_method=hist`, `eval_metric=aucpr`.
* Optuna HPO on the training set with stratified CV (up to 5 folds; reduced when the minority class is small) and early stopping (50 rounds). `scale_pos_weight` is derived from the training split. Default HPO budget is 100 trials unless smoke-test mode is invoked.
* Final models refit on full training data. SHAP summary plots are produced for test sets.

**Explainable Boosting Machines**:
* Optuna search space: `max_bins in {32,64,128,256}`, `max_leaves in {2,3}`, `smoothing_rounds in {0,25,50,75,100,150,200,350,500}`, `learning_rate in {0.0025-0.04}`, `validation_size in {0.1,0.15,0.2}`, `early_stopping_rounds in {100,200}`, `early_stopping_tolerance in {0,1e-5}`, `min_samples_leaf in {2,3,4,5,10}`, `min_hessian in {0,1e-6,1e-4,1e-2}`, `greedy_ratio in {0,5,10}`, `cyclic_progress in {0,1}` with `interactions=0`, `missing="gain"`, `inner_bags=0`, `outer_bags=1`, `n_jobs=-2`. Datasets with <50 rows force single-threaded, low-bag settings to avoid hangs.
* Final training uses `inner_bags=20`, `outer_bags=14`, `max_bins=1024`, `n_jobs=-2` (or reduced for very small data). Optional EBM XAI exports include global/interaction plots, local attributions, per-term partial dependence, and an HTML index with display-dictionary labels; term exports run in a bounded thread pool with timeouts and retries.

## Post-hoc Analysis and Statistical Evaluation
* **Calibration and thresholding**: Stratified out-of-fold predictions on the training split feed a logistic recalibration model (`calibration.json`). Calibrated OOF probabilities are thresholded by Youden's J statistic (`threshold.json`). The same calibrator and threshold are applied to held-out test predictions; no test-time refitting occurs.
* **Test metrics**: Primary metric AUPRC; secondary metrics AUROC, F1, sensitivity, specificity, accuracy, precision, and Brier score computed on calibrated test probabilities.
* **Bootstrapping**: `results_recreation/metrics_summary.py` validates artifacts, then runs 1,000 stratified, non-parametric bootstrap resamples of test predictions (configurable). Stored thresholds remain fixed. Percentile (2.5, 97.5) CIs are attached per Outcome x Branch x Feature Set x Model, with paired delta-CIs against a reference (default `preop_only`) when predictions share identical case sets. Joblib process pools are used with guarded fallbacks to threads/sequential execution and retryable timeouts.
* **Interpretability**: XGBoost SHAP summaries are saved for test data. EBM relies on native additive explanations when exports are enabled; SHAP is not run for EBMs.

## Reporting Artifacts
All figures and tables regenerate from saved artifacts using `metadata/display_dictionary.json` for consistent labels/units.
* **Cohort flow**: `reporting/cohort_flow.py` converts `results/metadata/cohort_flow_counts.json` into SVG/PNG consort diagrams, consolidating waveform checks into a single availability box and rendering the terminal AKI split when present.
* **Baseline table**: `reporting/preop_descriptives.py` uses raw cohort data (pre-encoding) and winsorized continuous variables; Shapiro-Wilk testing guides mean+/-SD vs. median[IQR] display. Outputs: HTML/LaTeX/DOCX at `results/tables/preop_descriptives.*`.
* **Missingness table**: `reporting/missingness_table.py` summarizes feature-level missingness from `data/processed/aki_features_master_wide.csv` to CSV/HTML for supplements and QA.

## Experimental Pipeline: Time-Series Classification (Aeon)
This parallel pipeline benchmarks modern time-series classifiers against the Catch22/XGBoost baseline.

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
