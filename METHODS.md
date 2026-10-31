# Methods

This file captures the active scientific contract for the repository. It is
intended to be more manuscript-oriented than `README.md`: the operational run
commands live in the README, while this file preserves the scientific
definitions, preprocessing contract, modeling design, and evaluation rules for
the Catch22 manuscript pipeline.

The manuscript implementation uses Catch24/Catch22 with logistic regression,
XGBoost, and EBM. The main AKI/ICU core analysis has completed; broader ICU and
exploratory mortality analyses are specified below but have not been run at
full budget. Diagnostic outputs verify implementation and are not manuscript
performance estimates. All full analyses use five patient-grouped outer folds,
three HPO folds, and five training-only calibration folds. Run inventories and
execution status live in `RESULTS.md` and the evidence register; Aeon remains
separate and experimental.

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
2. source anesthesia type is `General`
3. ASA physical status V-VI is excluded; missing ASA is retained
4. all four required waveforms are available:
   - `SNUADC/PLETH`
   - `SNUADC/ECG_II`, with `SNUADC/ECG_V5` substitution when needed
   - `Primus/CO2`
   - `Primus/AWP`

Multiple operations per patient are retained. All operations of a patient stay
together within each outer, inner, and calibration split and each EBM internal
training/validation bag. Each operation receives one outer-test prediction per
configuration. A patient contributes training data to other outer-fold fits,
as expected in cross-validation; patient separation applies within each split.

### Outcome-specific eligibility

Outcome-specific eligibility is layered onto the shared outer cohort:

- `any_aki` and `severe_aki` require:
  - finite, positive preoperative creatinine `<= 4.0 mg/dL`
  - at least one finite, positive postoperative creatinine strictly after
    `opend` and at or before `opend + 72 hours`
- `icu_admission` and `mortality` require:
  - the shared outer cohort
  - outcome-specific eligibility and a valid binary label

This design allows the repository to preserve a common waveform-qualified cohort
while still enforcing outcome-specific clinical requirements where needed.

### Manuscript comparison scopes

`main_shared` requires valid AKI eligibility/labels and valid ICU labels and is
used for both manuscript outcomes. `icu_waveform_sensitivity` requires valid ICU
labels without requiring baseline or postoperative creatinine for eligibility.
`mortality_waveform_sensitivity` uses the broader waveform-qualified population
with valid mortality eligibility and binary in-hospital mortality labels. It
requires neither creatinine ascertainment nor ICU admission/label eligibility.
Mortality analyses are exploratory, with the same full validation and patient
clustering as the primary analyses. Report positive operations and positive
patients separately because repeat operations remain present. No event count
alone establishes outcome stability; report uncertainty and fold-level support.

The scope selector runs before feature selection. Every family/feature set
within a scope, outcome, and processing branch uses the same operation roster.
Those configurations also reuse the same outcome-specific fold assignments.
Partial waveform-feature NaNs are retained; successful extraction of every
channel is not an additional exclusion. Extraction failures can change branch
membership, so branch-specific manifests and their intersection are recorded.
An operation with no successful channel-extraction record is absent from that
branch, including its clinical-only comparator. Cross-branch differences must
be interpreted with these denominators.

## Outcome definitions

### Primary outcome

`any_aki` is creatinine-defined postoperative AKI within 72 hours using KDIGO
thresholds:

- absolute increase `>= 0.3 mg/dL` within 48 hours, or
- relative increase to `>= 1.5x` baseline within 72 hours

Both windows begin at operation end (`opend`), excluding measurements exactly
at operation end and including measurements at the upper boundary. Baseline
and postoperative creatinine are interpreted in mg/dL. Missing, nonnumeric,
nonpositive, or infinite results do not establish ascertainment. Ineligible
operations retain missing AKI stage, any-AKI, and severe-AKI labels.

Baseline creatinine is the source clinical-table `preop_cr` value. The pipeline
does not independently select that baseline from a laboratory lookback window.
The 72-hour horizon is an intentional study-specific choice, with its decision
provenance recorded in the evidence register. It is not complete seven-day KDIGO
ascertainment. A valid 72-hour measurement does not ensure a measurement within
48 hours or complete surveillance throughout follow-up; absence of an observed
rise is not proof that no AKI occurred. Measurement frequency and timing must
be reported as ascertainment limitations.

### Secondary and additional labels

- `icu_admission`: positive source ICU length of stay (`los_icu`, with
  `icu_days` as an alias); zero denotes no ICU admission, and nonfinite or
  negative durations leave the outcome unascertained
- `mortality` (`y_inhosp_mortality`): the source `death_inhosp` indicator,
  representing in-hospital mortality rather than a fixed 30-day endpoint
- `severe_aki`: KDIGO stage 2 or 3 AKI

Among operations meeting the any-AKI definition, stage 3 is assigned for
creatinine `>= 3x` baseline or `>= 4.0 mg/dL` within 72 hours; otherwise stage 2
requires `>= 2x` baseline within 72 hours, and remaining AKI is stage 1. The
absolute stage-3 threshold requires the acute rise established by the any-AKI
definition: stable baseline/postoperative creatinine of 4.0 mg/dL is not AKI.
Urine-output and renal-replacement-therapy criteria are not used.

The main analysis evaluates `any_aki` and `icu_admission` in `main_shared`.
The broader sensitivity analyses evaluate ICU admission in
`icu_waveform_sensitivity` and exploratory in-hospital mortality in
`mortality_waveform_sensitivity`. Mortality is not added to the main shared
cohort's default outcome grid. Severe AKI is implemented but is outside these
core manuscript comparisons.

## Clinical data preprocessing

### Source variables

Preoperative variables are derived from VitalDB clinical and laboratory tables.
The production preoperative table includes:

- demographics and anthropometrics
- surgical context and anesthesia metadata
- comorbidities and test flags
- routine preoperative labs
- arterial blood gas values
- last finite numeric CRP, lactate, and WBC in `[-30 days, 0)` relative to
  case-recording start; this laboratory-selection rule does not redefine the
  source baseline creatinine

### Derived variables

Derived preoperative variables include:

- `inpatient_preop`
- `preop_egfr_ckdepi_2021`

The repository intentionally prefers a derived CKD-EPI 2021 eGFR over the raw
dataset `preop_gfr` field.
Source creatinine is explicitly in mg/dL; a high magnitude is not treated as
evidence of another unit. Nonpositive/nonfinite creatinine values remain missing
in predictors and eGFR derivation. eGFR requires a finite age and a recorded
female/male sex; unknown inputs leave it missing. Unknown/nonfinite admission
time leaves `inpatient_preop` missing. This indicator records admission before
recording start (`adm < 0`); it is not a separately adjudicated administrative
inpatient classification. ICU admission requires finite,
nonnegative source ICU duration; invalid durations leave its label missing.

### Leakage protection and secondary holdout analysis

Clinical preparation preserves source values and outcome-specific eligibility;
it does not fit encodings, clipping thresholds, or imputations. The modeling
loader admits only registered preoperative and waveform predictors. Patient and
operation identifiers, outcomes, eligibility flags, postoperative surveillance
fields, and split metadata are excluded from predictors.

Patient-grouped, approximately 80/20 holdout assignments are also retained per
outcome for explicitly requested secondary analyses. They do not replace the
nested cross-validation used for the manuscript. An unsupported sparse holdout
outcome does not block other outcomes. Source, cohort, extraction, and processed
file identities are checked before reuse; the exact schema/checkpoint contracts
are described in [RESULTS.md](RESULTS.md).

## Waveform processing

Waveform handling is implemented in `data_preparation/waveform_processing.py`
and `data_preparation/step_02_catch_22.py`.

### Signal extraction and quality handling

Waveforms are restricted to `[opstart, opend)` before interpolation, filtering,
resampling, or feature extraction. Features summarize the complete operation;
the prediction time is operation end. This workflow does not evaluate
predictions made earlier during surgery.

- segments are cut between `opstart` and `opend`
- whole-operation segments with more than 5% unusable samples (NaN or infinity)
  are rejected; exactly 5% is allowed
- remaining gaps are linearly interpolated before filtering and windowing;
  leading/trailing gaps use the nearest finite endpoint; there is no separate
  maximum contiguous-gap duration
- empty or nonfinite filtered/resampled signals fail extraction explicitly

### Channel-specific filtering and resampling

Channels use Butterworth filters with design order `N=4`, implemented as
second-order sections with forward-backward filtering. The design order is
reported explicitly because bandpass transformation and forward-backward
application affect the resulting filter order/response.

- PPG:
  - source: `SNUADC/PLETH`
  - bandpass 0.1-10 Hz; resampled from 500 Hz to 100 Hz
- ECG:
  - source: `SNUADC/ECG_II` or fallback `SNUADC/ECG_V5`
  - bandpass 0.5-40 Hz; resampled from 500 Hz to 100 Hz
- CO2:
  - source: `Primus/CO2`
  - low-pass 8 Hz; retained at 62.5 Hz
- Airway pressure:
  - source: `Primus/AWP`
  - low-pass 12 Hz; retained at 62.5 Hz

For full-case feature extraction, filtered signals are downsampled to 10 Hz for
computational efficiency. Windowed extraction uses the filtered signal at the
channel-specific target rate before window aggregation.
All rate changes use SciPy polyphase resampling with a centered low-pass FIR
filter (Kaiser window, beta 5.0), including 500→100 Hz and 100/62.5→10 Hz.
Exact rational rate ratios preserve fractional sampling rates. Linear endpoint
extension uses only the operative segment; filter delay is compensated. The
output grid starts at the first input sample and advances by `1 / target_sr`,
with `ceil(n * target_sr / source_sr)` samples covering the half-open input
duration. The final grid point can require endpoint extension beyond the last
observed sample; no neighboring recording data are read. Same-rate resampling calls add no resampling FIR;
channel-specific Butterworth filtering still applies. Full-series features describe this 10 Hz, band-limited
representation; preservation of higher-frequency ECG morphology is not claimed.
The existing quality rules do not implement a separate spike/saturation detector
or a blanket amplitude-based exclusion.

## Feature engineering

The production branch uses Catch24 features: the 22 canonical Catch22
statistics plus mean and standard deviation, obtained with
`pycatch22.catch22_all(..., catch24=True)`. There is no additional project-level
waveform z-scoring before this call. The mean and standard deviation describe
the filtered/resampled signal, rather than unprocessed physiological levels.

### Non-windowed branch

The `non_windowed` branch computes Catch24 on the full case after downsampling
to 10 Hz. The filtered channel must first have population standard deviation
at least `1e-6` at its channel-specific target rate.

### Windowed branch

The `windowed` branch uses:

- 10-second windows
- nominal 5-second stride (approximately 50% overlap)
- per-feature aggregation by mean, standard deviation, minimum, and maximum

The stride uses integer sample counts: 500 samples at 100 Hz and 312 samples
(4.992 seconds) at 62.5 Hz. Only complete windows are used; a terminal partial
window is discarded. Windows
with population standard deviation below `1e-6` are excluded. A channel with no
valid windows fails extraction. PPG/ECG windows retain 100 Hz, and CO2/airway
pressure windows retain 62.5 Hz. Whole-case versus windowed comparisons
therefore change temporal resolution as well as windowing and aggregation;
they are not an isolated test of windowing at a common sampling rate.

Aggregation uses float64 and sample standard deviation (`ddof=1`). A single
valid window has missing standard deviation. Undefined descriptors in any
included window propagate to that descriptor's aggregates; no available-window
averaging or feature imputation is performed during extraction.

### Wide-table assembly

Waveform features are pivoted to wide format in Step 04 and merged with
preoperative variables in Step 05. The resulting modeling tables are:

- `aki_features_master_wide.csv`
- `aki_features_master_wide_windowed.csv`

Outcome labels, eligibility flags, and split metadata are carried forward
through the merge rather than regenerated downstream.
Case/channel duplicates are rejected. Extraction settings, code and package
versions, source/roster identities, file hashes, and failure counts are recorded
in sidecars; unidentified historical features cannot be reused automatically.
Nonfinite descriptors become NaN; partial descriptor NaNs are preserved.
Extraction protocol 3 records numeric-validity rules, anti-alias resampling
settings and SciPy version, and rejects prior extraction fingerprints for both
branches. Atomic batched snapshots support interrupted
extraction without treating a partially appended row as a completed pair.

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
The core comparisons use preoperative clinical variables alone, all four
waveforms alone, and their combination, in that fixed presentation order.
Whole-case comparisons are the main presentation; windowed and additional
feature-set comparisons are supplementary.

## Model development

### Supported model families

The primary path supports:

- L2-regularized logistic regression
- XGBoost
- Explainable Boosting Machines

### Validation design

The primary internal validation procedure is patient-grouped nested
cross-validation with:

- 5 outer folds
- 3 inner HPO folds
- 5 training-only calibration cross-fitting folds
- `repeats=1`

Legacy grouped holdout evaluation remains available as a secondary mode but is
not the default analysis.
Reduced folds require explicit diagnostic mode and are labeled accordingly.
Definitive mode requires exactly five outer, three HPO, and five calibration
folds and rejects missing patient identities or silent fold reduction.
Grouped stratification evaluates up to 64 deterministic seed candidates and
chooses the feasible assignment with the smallest normalized squared deviation
of fold class counts from their target. Exact balance ends the search early;
assignments are cached by ordered labels, patient identities, fold count, seed,
selector code, and NumPy/scikit-learn versions. Persistent cache entries must
pass patient-separation, coverage, class-support, and balance-score checks.
No model-performance metric is used to choose a split.

Within each outer-training partition, a seeded Optuna tree-structured Parzen
estimator (TPE) search runs serially within that study. Its objective is the
unweighted mean of the three inner-fold AP scores computed from raw model
probabilities, before recalibration. Each outer fold selects its own
hyperparameters. The default budgets are 20 LR trials and 50 XGBoost/EBM trials,
with base random seed 42 and outer-fold seed offsets. The prespecified search
domains below can be used for a supplementary methods table; final selected
parameters are retained per outer fold.

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

For logistic regression, numeric median imputation, numeric standardization,
and training-derived missingness indicators are fit inside each training fold.
All-empty numeric columns use zero. One-hot columns and missingness indicators
are not standardized. XGBoost and EBM retain native NaN handling.
Categorical variables are one-hot encoded using training-fitted categories;
unseen levels are ignored by that encoder. Rare-category pooling applies
specifically to departments observed fewer than 30 times in the active training
partition. Missing departments remain separate from pooled rare departments.

All three families coerce registered numeric predictors to numeric and treat
nonnumeric/nonfinite values as missing before learning statistics. Numeric
schema takes precedence over CSV object dtype. Blank categorical values and
numeric infinities in categorical fields are missing; recorded labels such as
`Unknown` are not treated as empty. Missing department values
are not pooled with rare observed departments.

Continuous preoperative predictors are winsorized at the training partition's
0.5th and 99.5th percentiles, learned separately in every training fit and
applied unchanged to validation/test data. Waveform descriptors are not
winsorized. These cutoffs limit influence rather than establish physiological
validity. Raw source extremes remain in descriptive tables; no blanket
clinical outlier deletion or data-dependent feature-missingness exclusion is
applied. Missingness tables count unusable numeric values before fold-local
imputation/clipping, with the same operation denominator as Table 1.

### Calibration and thresholding

Within each outer fold, the workflow:

1. selects hyperparameters using three grouped folds within outer training
2. generates grouped out-of-fold predictions on that outer-training cohort
   using five folds and the selected hyperparameters
3. fits logistic recalibration on those outer-training OOF predictions only
4. selects the decision threshold from calibrated outer-training OOF
   predictions only
5. refits preprocessing and the selected base model on the full outer-training
   cohort
6. applies the fitted model, locked calibrator, and threshold to the outer-test
   fold

The recalibration estimator is sklearn logistic regression with its L2 penalty
and C=1 default; its intercept and slope are training-derived transformation
parameters, not held-out calibration-performance estimates. The same
recalibration method is used for logistic regression, XGBoost, and EBM:
`p_calibrated = sigmoid(intercept + slope * logit(p_raw))`.
These are separately fitted recalibrators for each outer fold and model
configuration. The base-model OOF predictions exclude each patient's rows from
the corresponding estimator fit, but the hyperparameters were selected using
the outer-training labels. The recalibrator and threshold are also fitted on
these training labels. Calibrated training OOF metrics are development
diagnostics; only outer-test predictions evaluate the complete selected
workflow.

Holdout mode follows the same principle using the persisted Step 03 split.
The classification threshold maximizes Youden's J (sensitivity + specificity
minus one) over the observed calibrated training-OOF probabilities. A case is
classified positive when its calibrated probability is at least the threshold;
equal-J ties retain the lowest observed threshold. This threshold is used for
classification metrics, not to choose the decision-curve threshold range.

### Reproducibility and optional full-data fitting

Each outer fold saves its fitted preprocessing, model, recalibrator, threshold,
and held-out predictions with source, environment, operation, and fold
identities. Scheduling may change execution order but not partitions, seeds,
HPO budgets, or learned preprocessing boundaries. Checkpoint and reuse rules
are specified in [RESULTS.md](RESULTS.md).

An optional full-data refit may be saved after validation. It is disabled for
the core comparisons and cannot replace outer-test predictions as the source
of reported performance.

## Model-specific notes

### Logistic regression

The model uses L2 regularization with `lbfgs`, no class/sample reweighting, and a
seeded Optuna search over `C` from 1e-4 to 1e4 (20 trials by default). Convergence
warnings trigger a bounded deterministic retry and unresolved failures are
reported: the iteration limit increases from 5,000 to 10,000 if needed.
Calibration and threshold selection follow the shared protocol.

### XGBoost

XGBoost uses a binary logistic objective with histogram-based tree building.
Hyperparameter optimization occurs inside patient-grouped validation folds, and
class imbalance handling is recomputed inside each fold-training subset rather
than once globally.
The imbalance ratio is applied once through `scale_pos_weight`; positive sample
weights do not apply the same ratio a second time.
The positive weight is the number of negative divided by positive operations
in the fitting partition. The number of boosting rounds is tuned; outer-test
data are not used as an early-stopping set.

### Explainable Boosting Machines

EBM is additive (`interactions=0`). Numeric cut points are derived from finite
values in the active training partition using quantiles at the selected bin
resolution, with duplicate cuts removed. Positive-operation sample weights use
the fitting partition's negative-to-positive ratio; negative weights are one.

Explicit deterministic patient-grouped internal training/validation bags are
passed to EBM for early stopping in HPO, cross-fitting, and final fitting. These
internal partitions do not cross patient boundaries. HPO uses one outer bag and
no inner bags. Calibration fits and final outer-training refits use 14 outer
bags and 20 inner bags. Candidate validation fractions of patients are 0.10,
0.15, and 0.20; training and validation sides must both contain both classes.
Explicit diagnostic mode uses a bounded EBM compute profile: a 200-round cap,
smoothing at most 50 rounds, one HPO outer bag, two refit outer bags, and no
inner bags. Manuscript mode retains 14 refit outer and 20 inner bags and the
original search. Compute profiles and thread environments are fingerprinted;
bounded diagnostic artifacts cannot be relabeled as definitive results.

The EBM path remains part of the primary grid. Explanation exports do not alter
reported predictive performance.

### Hyperparameter search domains

Continuous domains are linear-scale unless marked logarithmic; listed values
are categorical choices. TPE samples adaptively within these domains. Parameters not searched retain the recorded
implementation/package defaults. All selected values remain local to the
corresponding outer-training model.

| Model | Parameter | Search domain |
| --- | --- | --- |
| LR | Inverse L2 penalty `C` | 1e-4 to 1e4, logarithmic |
| XGBoost | Boosting rounds | 100 to 1,000, step 100 |
| XGBoost | Learning rate | 1e-4 to 0.1, logarithmic |
| XGBoost | Maximum depth | Integers 3 to 10 |
| XGBoost | Row / column subsampling | Each 0.6 to 1.0 |
| XGBoost | Split penalty `gamma` | 0 to 5 |
| XGBoost | L1 / L2 penalties | Each 1e-8 to 1.0, logarithmic |
| EBM | Maximum bins / leaves | Bins: 32, 64, 128, 256; leaves: 2, 3 |
| EBM | Learning rate | 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 |
| EBM | Smoothing rounds | 0, 25, 50, 75, 100, 150, 200, 350, 500 |
| EBM | Validation fraction | 0.10, 0.15, 0.20 |
| EBM | Early-stopping rounds / tolerance | Rounds: 100, 200; tolerance: 0, 1e-5 |
| EBM | Minimum samples per leaf | 2, 3, 4, 5, 10 |
| EBM | Minimum Hessian | 0, 1e-6, 1e-4, 1e-2 |
| EBM | Greedy ratio / cyclic progress | Greedy: 0, 5, 10; cyclic: 0, 1 |

## Evaluation and statistical analysis

### Saved prediction contract

`results_recreation.metrics_summary` works from saved prediction and validation
artifacts rather than retraining models at report time. In nested CV, the
primary evaluation target is the pooled outer-fold prediction set. In holdout
mode, the primary evaluation target is the held-out test cohort.

### Reported metrics

Primary discrimination metric:

- average precision (AP), computed by `average_precision_score`; the saved
  artifact key remains `auprc`. This is not trapezoidal PR-curve integration.

Additional metrics:

- AUROC
- F1
- sensitivity
- specificity
- accuracy
- precision
- Brier score

AP, AUROC, and Brier score use pooled calibrated outer-test predictions;
classification metrics use the saved classes from each fold's locked
threshold. This evaluation differs from the raw-probability inner-fold AP used
to select hyperparameters. AP is interpreted alongside the event prevalence of
the same evaluated cohort. Calibration is assessed with Brier score and
reliability curves comparing mean predicted risk with observed event frequency
in up to 10 quantile bins; tied edges are collapsed. Held-out calibration
intercept/slope and their confidence intervals are not estimated.

### Bootstrap uncertainty

Uncertainty is estimated with patient-clustered, non-parametric bootstrap
resampling of the saved predictions. When possible, delta confidence intervals
are computed against a reference feature set (`preop_only` by default) using
paired resamples.
Intervals are the 2.5th and 97.5th percentiles; the current defaults are 5,000
replicates, seed 0, and unstratified patient-cluster sampling. Requested and
metric-specific valid replicate counts are recorded. Single-class resamples
remain usable for defined metrics such as Brier score; AUROC requires both
classes, AP requires a positive outcome, sensitivity requires a positive
outcome, and specificity requires a negative outcome. Undefined metrics are
excluded individually. Intervals with fewer than two valid values are missing
and warn rather than reporting spurious precision. Zero-denominator precision
and F1 retain the declared zero convention.
Resampling preserves all operations of a sampled patient; explicit optional
stratification uses the patient's maximum observed operation outcome. Paired
comparisons align cases and verify patient, label, and fold identities within
the same cohort/outcome/branch. Existing reference summaries retain each
family's `preop_only` reference; a different family's baseline is never
substituted. Separate long-form comparisons explicitly identify both sides of
cross-family contrasts. All metric and decision-curve differences use the same
draws, verified by hashes of their actual sampled row indices. Statistics are
operation-weighted: every operation from a selected patient is retained, and
the resampled operation count supplies the denominator.
These intervals condition on saved fitted models and predictions; they do not
include the full variability of repeating cohort construction, HPO, and model
fitting. Comparisons across the supplementary grid are exploratory unless a
multiplicity policy is specified before reviewing definitive results.
These percentile intervals are conditional uncertainty summaries, not a claim
of exact coverage for the complete nested model-development procedure.

### Prespecified comparisons and DeLong eligibility

Within each family, the core contrasts are combined minus preop, waveform-only
minus preop, and combined minus waveform-only. Cross-family contrasts compare
XGBoost minus LR, EBM minus LR, and EBM minus XGBoost at each core input set.
Each requested noncore ablation is compared with its same-family preop
reference. This yields 18 unique core contrasts per cohort/outcome/branch.
The six combined-versus-preop AP contrasts across three families and two
outcomes in the nonwindowed main cohort are primary. Other core contrasts are
secondary, and windowed, sensitivity-cohort, and ablation contrasts are
exploratory. Nominal confidence intervals are not simultaneous inference.

Ordinary paired DeLong results are gated to verified held-out predictions with
one evaluated operation per patient, at least two observations per class, and
matched training/test provenance. All pooled-CV or repeat-patient analyses
instead export `unsupported_design` with a missing p-value, while retaining
paired bootstrap AUROC differences. Patient clustering alone would not remove
shared-training dependence in pooled CV. Invalid inputs fail; zero variance
produces a `degenerate` status rather than a p-value of 1 or an arbitrary
covariance ridge. Eligible tests use tie-aware placement differences and stable
survival/log-survival probabilities. Unique eligible DeLong contrasts receive
Holm correction within cohort, validation scheme, outcome, and branch. No
formal AP p-values or significance-based model selection are introduced.

### Decision-curve analysis

Exploratory decision curves use saved calibrated predictions with model, treat-all, and treat-none
net benefit. Default probability ranges are 0.01-0.30 for AKI and 0.05-0.60 for
ICU, in 0.01 steps. Exploratory mortality uses 0.001-0.050 in 0.001 steps
(0.1-5% risk). The threshold grids are fixed analysis settings, not selected by
examining held-out outcomes. Plots alone do not establish clinical utility.
At probability threshold `p_t`, net benefit is
`TP/N - (FP/N) * p_t/(1-p_t)`, where `N` counts evaluated operations. Treat-none
net benefit is zero; treat-all uses the same cohort prevalence and threshold
odds. Paired resamples use the resampled operation denominator.
Decision curves export absolute net-benefit intervals and paired net-benefit
differences from the same patient resamples. Companion difference plots show
pointwise 95% bands; absolute main plots retain only the core inputs, with
ablation values retained numerically. Threshold-wise intervals are not
simultaneous evidence of dominance across the range. Saved probabilities are
used directly: score-to-probability
conversion models must not be fitted on evaluation outcomes. A decision/action
and plausible threshold preferences are required before interpreting these
exploratory curves as clinical utility.

## Descriptive reporting and model interpretation

Descriptives, cohort flow, and missingness summaries use the same fixed
scope/branch rosters as model evaluation. Counts of operations and unique
patients are reported separately. Categorical summaries retain source categories
and display missing/unknown values separately.

Model interpretation uses saved outer-fold bundles without refitting or
selecting new features. XGBoost TreeSHAP values are computed for held-out
operations on the raw, uncalibrated log-odds scale. EBM exports native global
term importance and response curves from each fitted training model, together
with local contributions for held-out operations and separate missing/unknown
bin scores. Logistic-regression coefficients describe penalized log-odds per
training standard deviation for numeric features; categorical and missingness
coefficients retain their encoded scale. Encodings and coefficients remain
fold-specific. Individual one-hot coefficients are not reference-category odds
ratios; category contrasts require coefficient differences within the same
fold. All explanations describe associations, remain separate from logistic
recalibration, and are not causal effects.

Display labels and units are centralized in
`metadata/display_dictionary.json` so manuscript-facing outputs remain
consistent across tables and figures.
Publication descriptives use median (IQR), missingness, operation/patient counts,
and display-only NSR/abnormal ECG grouping. Main ROC/PR figures and legend scores
use pooled outer predictions, matching the tables; fold variability is an
explicit optional view. Main calibration axes span [0,1].

## Software and analysis provenance

The frozen main analysis used Python 3.12, scikit-learn 1.6.1, XGBoost CPU 2.1.4,
interpret-core 0.7.8, pycatch22 0.4.5, and Optuna 4.6.0. Exact package versions,
code revisions, source hashes, settings, and selected-case/fold identities are
retained with the analysis artifacts. Older completed configurations retain
their original provenance. Computational reuse requires verified compatibility;
original fingerprints are not rewritten. Cold split preparation parallelizes
independent cache keys while preserving candidate order and tie-breaking
within each key. This optimization does not change the statistical design.

The manuscript should supply its verified dataset citation, collection period,
and applicable ethics/data-use statement from source and study records; those
administrative details are not inferred from the modeling code. Aeon models
remain outside this manuscript pathway.
