# Methods

This file captures the active scientific contract for the repository. It is
intended to be more manuscript-oriented than `README.md`: the operational run
commands live in the README, while this file preserves the scientific
definitions, preprocessing contract, modeling design, and evaluation rules for
the validated Catch22 pipeline.

The default validated analysis is the Catch24/Catch22 plus XGBoost/EBM path.
Aeon remains experimental and is documented separately.

## Study design and data source

This is a retrospective observational study using VitalDB, an open-access
intraoperative dataset from Seoul National University Hospital containing 6,388
non-cardiac surgical cases.

The repository combines:

- preoperative clinical and laboratory data
- intraoperative physiological waveforms
- patient-grouped model development and evaluation

The current paper-facing path is the modular Python pipeline under
`data_preparation/`, `model_creation/`, `results_recreation/`, and `reporting/`.

## Cohort construction

### Shared outer cohort

Step 01 defines a shared outer cohort that is reused across supported outcomes.
A case enters that cohort when all of the following are satisfied:

1. `opend` is present
2. ASA physical status is I-IV
3. all four required waveforms are available:
   - `SNUADC/PLETH`
   - `SNUADC/ECG_II`, with `SNUADC/ECG_V5` substitution when needed
   - `Primus/CO2`
   - `Primus/AWP`

Multiple operations per patient are retained. Downstream splitting, calibration,
and resampling are grouped on `subjectid` so that one patient never spans both
training and evaluation partitions within a single run.

### Outcome-specific eligibility

Outcome-specific eligibility is layered onto the shared outer cohort:

- `any_aki` and `severe_aki` require:
  - preoperative creatinine present
  - preoperative creatinine `<= 4.0 mg/dL`
  - at least one postoperative creatinine in the adjudication window
- `icu_admission` and `mortality` require:
  - the shared outer cohort
  - a non-missing source label

This design allows the repository to preserve a common waveform-qualified cohort
while still enforcing outcome-specific clinical requirements where needed.

## Outcome definitions

### Primary outcome

`any_aki` is defined using KDIGO creatinine criteria:

- absolute increase `>= 0.3 mg/dL` within 48 hours, or
- relative increase to `>= 1.5x` baseline within 7 days

### Secondary and additional labels

- `icu_admission`: postoperative ICU admission
- `y_inhosp_mortality`: in-hospital mortality
- `severe_aki`: KDIGO stage 2 or 3 AKI

The active experiment grid focuses on `any_aki` and `icu_admission`. Mortality
is supported by the modular pipeline but is not part of the default launcher
grid. Severe AKI remains available primarily for archival or explicit secondary
analysis.

## Clinical data preprocessing

### Source variables

Preoperative variables are derived from VitalDB clinical and laboratory tables.
The production preoperative table includes:

- demographics and anthropometrics
- surgical context and anesthesia metadata
- comorbidities and test flags
- routine preoperative labs
- arterial blood gas values
- last-observed CRP, lactate, and WBC within 30 days before surgery

### Derived variables

Derived preoperative variables include:

- `inpatient_preop`
- `preop_egfr_ckdepi_2021`

The repository intentionally prefers a derived CKD-EPI 2021 eGFR over the raw
dataset `preop_gfr` field.

### Persisted split metadata

Step 03 writes one patient-grouped, approximately 80/20 holdout split per
trainable outcome:

- `split_group_any_aki`
- `split_group_severe_aki`
- `split_group_icu_admission`
- `split_group_mortality`

A backward-compatible `split_group` alias is retained only for the AKI split.
Outcomes that are too sparse to support a valid grouped holdout split are marked
as unsupported in the Step 03 metadata sidecar instead of blocking unrelated
outcomes.

### Leakage protection in preprocessing

The persisted `aki_preop_processed.csv` does not contain training-fitted
encoding, clipping, or imputation. Learned preprocessing is fit only inside the
active training partition during model development.

Step 01, Step 03, and Step 05 each emit `.metadata.json` sidecars. Reuse of
processed artifacts is gated on successful sidecar validation so stale schemas
fail fast instead of mixing incompatible outputs.

## Waveform processing

Waveform handling is implemented in `data_preparation/waveform_processing.py`
and `data_preparation/step_02_catch_22.py`.

### Signal extraction and quality handling

- segments are cut between `opstart` and `opend`
- short gaps are interpolated
- segments with excessive missingness are rejected upstream

### Channel-specific filtering and resampling

- PPG:
  - source: `SNUADC/PLETH`
  - filtered and resampled from 500 Hz to 100 Hz
- ECG:
  - source: `SNUADC/ECG_II` or fallback `SNUADC/ECG_V5`
  - filtered and resampled from 500 Hz to 100 Hz
- CO2:
  - source: `Primus/CO2`
  - low-pass filtered and retained at 62.5 Hz
- Airway pressure:
  - source: `Primus/AWP`
  - low-pass filtered and retained at 62.5 Hz

For full-case feature extraction, filtered signals are downsampled to 10 Hz for
computational efficiency. Windowed extraction uses the filtered signal at the
channel-specific target rate before window aggregation.

## Feature engineering

The production branch uses Catch24 features: the 22 canonical Catch22
statistics plus mean and standard deviation.

### Non-windowed branch

The `non_windowed` branch computes Catch24 on the full case after downsampling
to 10 Hz.

### Windowed branch

The `windowed` branch uses:

- 10-second windows
- 5-second overlap
- per-feature aggregation by mean, standard deviation, minimum, and maximum

### Wide-table assembly

Waveform features are pivoted to wide format in Step 04 and merged with
preoperative variables in Step 05. The resulting modeling tables are:

- `aki_features_master_wide.csv`
- `aki_features_master_wide_windowed.csv`

Outcome labels, eligibility flags, and split metadata are carried forward
through the merge rather than regenerated downstream.

## Feature sets and branches

The default Catch22 grid spans both `non_windowed` and `windowed` branches and
the following feature-set families:

- `preop_only`
- single-waveform sets:
  - `pleth_only`
  - `ecg_only`
  - `co2_only`
  - `awp_only`
- `all_waveforms`
- `preop_and_all_waveforms`
- preop-plus-single-waveform ablations
- leave-one-waveform-out fused ablations

Two-channel waveform-only combinations are intentionally excluded from the
default grid.

## Model development

### Supported model families

The validated primary path supports:

- XGBoost
- Explainable Boosting Machines

### Validation design

The primary internal validation procedure is patient-grouped nested
cross-validation with:

- 5 outer folds
- 5 inner folds
- `repeats=1`

Legacy grouped holdout evaluation remains available as a secondary mode but is
not the default analysis.

### Fold-local preprocessing

The following steps are fit on training data only within the active partition:

- categorical handling and rare-category collapse
- numeric clipping
- optional imputation
- hyperparameter optimization
- calibration
- threshold selection

No held-out outer-test or final holdout-test outcomes are used to fit
preprocessing statistics, model parameters, calibrators, or decision
thresholds.

### Calibration and thresholding

Within each outer fold, the workflow:

1. fits the model on the outer-training cohort
2. generates grouped out-of-fold predictions on that same outer-training cohort
3. fits logistic recalibration on those outer-training OOF predictions only
4. selects the decision threshold from calibrated outer-training OOF
   predictions only
5. applies the locked calibrator and threshold to the outer-test fold

Holdout mode follows the same principle using the persisted Step 03 split.

### Resume behavior

Interrupted nested runs may resume from saved fold checkpoints only when the
stored validation fingerprint matches the current run configuration exactly.
Stale checkpoints are ignored and recomputed.

### Optional final refit

An additive full-data refit can be saved after validation without changing the
reported validation metrics.

## Model-specific notes

### XGBoost

XGBoost uses a binary logistic objective with histogram-based tree building.
Hyperparameter optimization occurs inside patient-grouped validation folds, and
class imbalance handling is recomputed inside each fold-training subset rather
than once globally.

### Explainable Boosting Machines

EBM training uses a small-sample-oriented search space and retains optional
explainability exports. When enabled, those exports include:

- global term importances
- local explanations
- per-term partial dependence views
- machine-readable JSON/CSV artifacts

The EBM path remains part of the primary validated grid, but the explanation
artifacts are additive and do not alter reported predictive performance.

## Evaluation and statistical analysis

### Saved prediction contract

`results_recreation.metrics_summary` works from saved prediction and validation
artifacts rather than retraining models at report time. In nested CV, the
primary evaluation target is the pooled outer-fold prediction set. In holdout
mode, the primary evaluation target is the held-out test cohort.

### Reported metrics

Primary discrimination metric:

- AUPRC

Additional metrics:

- AUROC
- F1
- sensitivity
- specificity
- accuracy
- precision
- Brier score

### Bootstrap uncertainty

Uncertainty is estimated with patient-clustered, non-parametric bootstrap
resampling of the saved predictions. When possible, delta confidence intervals
are computed against a reference feature set (`preop_only` by default) using
paired resamples.

## Reporting artifacts

Paper-facing outputs are rebuilt from saved artifacts under `reporting/`.
Important components include:

- `reporting.make_report` for consolidated tables and plots
- `reporting.cohort_flow` for cohort flow figures
- `reporting.preop_descriptives` for baseline descriptive tables
- `reporting.missingness_table` for feature missingness summaries

Display labels and units are centralized in
`metadata/display_dictionary.json` so manuscript-facing outputs remain
consistent across tables and figures.

## Experimental scope

Aeon time-series models are exploratory and are intentionally separated from
the default Catch22 paper path. Their commands and caveats live in
`experimental/aeon/README.md`.
