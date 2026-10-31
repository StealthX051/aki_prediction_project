# VitalDB manuscript: pre-run checkpoint

## Status and authorization

Implementation, cleanup, and offline verification are complete at the pre-run
checkpoint. The user authorized committing and pushing this work, and explicitly
prohibited starting full runs. No definitive reconstruction/model grid has been
launched. Drive/email/manuscript external writes remain separately authorized.
Earlier real-data diagnostics precede the latest protocol and are not validation
of current full-cohort results. This file is the single decision/evidence
register; definitions belong in METHODS, commands in README, and outputs in RESULTS.

## Scientific decisions

| Decision | Implemented contract |
| --- | --- |
| AKI horizon | Absolute rise ≥0.3 mg/dL by 48 hours or relative rise ≥1.5× by 72 hours; acute-rise gate for stage 3 |
| AKI ascertainment | Finite positive baseline ≤4 mg/dL and ≥1 finite positive result in `(opend, opend+72h]`; otherwise stages/labels remain missing |
| Population | General anesthesia; exclude ASA V/VI, retain missing ASA and repeat operations |
| Scopes | `main_shared` for both AKI/ICU; broader `icu_waveform_sensitivity` for ICU only |
| Matching | Freeze branch rosters/folds before feature sets; same operations across families/input sets within a branch; partial-feature NaNs retained |
| Predictors | Preoperative clinical variables plus operative waveforms only; postoperative audit metadata excluded |
| Cleaning | Explicit source-mg/dL creatinine; unknown admission/sex/age inputs remain missing; invalid ICU durations are ineligible; latest finite preop extra labs |
| Outliers/NaNs | Training-only 0.5th/99.5th preop caps; no blanket waveform clipping; LR median/indicators/scaling, trees native NaNs; nonfinite numeric values become missing |
| Waveforms | Preserve full-series 10 Hz and 10-second windows/5-second overlap; centered polyphase anti-alias FIR resampling; sampling-rate sensitivity deferred by user |
| Models | L2 LR, XGBoost, EBM; single XGBoost weighting; deterministic patient-grouped EBM internal bags |
| Validation | Five grouped outer / three HPO / five calibration folds, one repeat; 20 LR trials and 50 trials each for XGBoost/EBM; no silent fold reduction |
| Calibration/threshold | Same regularized logistic intercept/slope recalibration and training-only Youden selection for all families; only pooled outer-test estimates are publication performance |
| Comparisons | Shared 5,000 unstratified patient-cluster draws, seed 0; paired AP/AUROC/Brier and DCA; ordinary DeLong only for eligible independent holdouts |
| Presentation | Whole-case core primary; windowed/sensitivity/ablations supplementary; retain null and negative contrasts; saved-fold explanations without refitting |

Missing waveform samples (NaN/infinity) above 5% reject a segment; remaining gaps
use interpolation/endpoint filling. There is no contiguous-gap cap or complete
spike/saturation detector. A 72-hour creatinine result does not guarantee 48-hour coverage.
These prespecified limitations remain explicit rather than being silently changed.
The 10 Hz representation does not preserve higher-frequency ECG morphology.
Additional model families, longer windows, Aeon expansion, and post-hoc
monotonicity are outside the definitive pathway.

## Source authority

Sources below were reviewed during the handoff/audits; no new external changes
are implied by retaining these links.

- [Baseline master b75138f](https://github.com/StealthX051/aki_prediction_project/commit/b75138f19f10cd09f5329371165ec8b85a7ba999) and local CPU branch `c208fc3`: scientific/environment baselines; retain the four environment commits.
- [Mentor status/comments](https://docs.google.com/document/d/1ihwbwJUilHMOc2YfoX3h5h0vaaynMHn_wEcJLzMY26I/edit): Wingert's August 4/10, 2026 scope/core/windowing direction, corroborated by August 4/10/11 Outlook messages.
- [Three-day decision](https://docs.google.com/document/d/1GEIKOE9O_PgoNMJkZ-xOwg7lp5ZFSdrTEZYndJpHTvI/edit): Kothari, December 10, 2025; implementation replies December 23/31. Unresolved comment state does not reverse the decision.
- [Current methods comments](https://docs.google.com/document/d/1vrIdN5BiJ7edGaGULUAjelBikxedo24Bt6H7sHLQ8tI/edit): January 26, 2026 general-anesthesia restriction; older holdout/deduplication descriptions are historical.
- [Descriptives comments](https://docs.google.com/document/d/17OnmP44M7LtdLKRiuDYZQmL0fvcgJaJdpXbEwJEXz1U/edit), [performance comments](https://docs.google.com/document/d/1w5TzywiVV5QydszXJltSvmCYSsoB-ga0/edit), [VitalDB codebook](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md), and local parameter/track dictionaries: clinical sections and units; PT %, CRP mg/dL, aPTT seconds, ECG findings, AWP hPa. Do not import INSPIRE's missingness exclusion.
- [KDIGO](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf): establish AKI before the absolute-creatinine stage 3 gate; our 72-hour horizon is intentionally shorter than standard seven-day ascertainment.
- [SciPy 1.13.1 polyphase resampling](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.signal.resample_poly.html): Kaiser beta 5 FIR, operation-local linear boundary extension, compensated delay, exact rational rate ratios and physical output grid.
- [Pinned binary dcurves reference](https://github.com/MSKCC-Epi-Bio/dcurves/tree/168c8b654808691d70a1ea6b92d2a5013915efc2), [R benchmark](https://www.danieldsjoberg.com/dcurves/reference/dca.html), [Sun/Xu reference](https://github.com/yandexdataschool/roc_comparison), and [pROC](https://xrobin.github.io/pROC/): numerical audit references, not endorsement of ordinary DeLong for pooled CV/repeat patients.

## Source/historical evidence and its limits

Source files are read-only under
`/media/volume/catch22/projects/aki_prediction_project/data/raw`.
Earlier aggregate audits found 6,388 operations/6,090 patients, including 6,043
operations with general anesthesia. These are source counts, not final branch
rosters. Corrected baseline/72-hour coverage versus seven-day coverage was:

| Population | Seven-day coverage | 72-hour coverage | Late-only |
| --- | ---: | ---: | ---: |
| Source | 3,834 | 3,801 | 33 |
| Source general anesthesia | 3,805 | 3,779 | 26 |
| Historical cohort | 3,460 | 3,429 | 31 |
| Historical general anesthesia | 3,434 | 3,409 | 25 |

Among 72-hour-covered source operations, 49 lacked 48-hour measurements (47 general
anesthesia); 72-hour measurement counts had Q1/median/Q3 = 1/2/4. A pre-extraction
feasibility audit found main 3,581 operations/3,442 patients (191 AKI, 838 ICU), and
broader ICU 5,779 operations/5,557 patients (1,047 ICU). These are preliminary and
predate current cleaning/extraction. No invalid/stable 4→4 creatinine case was
observed; tests protect supported edge cases rather than claiming prevalence.
Source aPTT reached 400 seconds and ECG was 98.7% NSR; unusual values were not
silently deleted. Two source creatinine values exceeded 20 mg/dL, reinforcing
explicit units rather than magnitude-based conversion.

Historical 120-configuration XGBoost/EBM results are under
`/media/volume/catch22/projects/aki_prediction_project/results/catch22`.
A saved summary's 3,453-operation denominator conflicted with its 691-operation
test file; the audit used independently aligned test predictions instead.
Whole-case historical contrasts (691 patients, 34 AKI/154 ICU events) were:

| Outcome | Family | Clinical AP | Combined AP | ΔAP (95% interval) |
| --- | --- | ---: | ---: | --- |
| AKI | XGBoost | 0.383 | 0.333 | −0.050 (−0.153, 0.045) |
| AKI | EBM | 0.424 | 0.421 | −0.003 (−0.080, 0.069) |
| ICU | XGBoost | 0.666 | 0.702 | 0.036 (−0.012, 0.084) |
| ICU | EBM | 0.659 | 0.695 | 0.036 (−0.006, 0.080) |

These do not establish a primary whole-case gain. The nominal windowed EBM ICU
contrast 0.059 (0.009, 0.109) was exploratory among 120 unadjusted configurations;
never select it as a positive main finding. Historical LR is unavailable. New
small enriched diagnostics have different prevalence and cannot be compared
numerically with those historical results.

## Implementation and verification inventory

- A–D implemented: scientific eligibility/staging/scopes; three-family models;
  core/full orchestration; fitted fold bundles; strict paired comparisons;
  publication tables, curves, DCA, and saved-fold explanations.
- E/resource checks: earlier isolated real-data diagnostics plus current offline
  tests. Exact current scientific reproduction remains pending execution.
- Final cleanup removes unused discarded flags, redundant copies/imports/parser
  calls, and import-time report writes; streams hashes and defers SHAP imports.
  Scientific defaults, artifact basenames, and legacy explicit entrypoints remain.
- Compatibility: clinical/preop schema 4; merged schema 5; extraction protocol 3;
  modeling protocol 4/preprocessing v4; fold bundle integrity 2; uncertainty 3.
  Source/settings/model/environment identities reject incompatible reuse.

Earlier diagnostic root:
`/media/volume/catch22/data/aki_prediction_project/validation-20261030.7e5XNoU7`.
Its `final_checks/validation_inventory.json` records nine reduced 2×2 nested
configurations, 18 outer bundles and 360 pooled rows across 40 operations;
18 holdout configurations/144 test rows; successful resume with 144 saved files
unchanged; and saved-fold explanations with fitting disabled. Other evidence:
`full_source_5x5_preflight.json`, `final_checks/nested_console.log`,
`final_checks/holdout_console.log`, `final_checks/resume_integrity.json`, and
`final_checks/saved_bundle_explanations/verification.json`. These runs predate
current 5/3/5 validation and anti-alias/cleaning/statistical amendments. Interrupted
initial diagnostics remain preserved and are not certified as successful.

Final pre-run verification:

| Check | Evidence |
| --- | --- |
| Complete suite/native dependencies | 483 tests, 48.38 s, zero skips; `/tmp/aki-cpu-offline.bDBfCBZV/` |
| Read-only launch/CLI checks | Main core 36, full main 180, family stages 12 each, sensitivity 18; `/tmp/aki-prerun-readiness.u69v_5p1/`; source CSV size/mtime unchanged and future study root absent |
| Current VM inventory | 64 available CPUs, 239.9 GiB available RAM, 852.7 GiB free on the research volume; same readiness evidence |
| All 27 test modules audited; seven obsolete scripts removed | Scientific cases retained, native fitted fixtures shared; details in test names and assertions |
| Four deliberate faults rejected | Unfiltered decimation, patient-ignorant calibration folds, DCA without threshold odds, row bootstrap; `/tmp/aki-suite-faults.a16g3b0l/` |
| Pytest isolation/live-data guard | Inherited study paths untouched; `/tmp/aki-pytest-isolation-proof.h73l5es1/` |
| Anti-alias numerical/performance probe | `/tmp/aki-antialias-probe.dq1m8nmw/resampling-evidence.json`: 75–83 dB rejection for tested tones, <0.02% 1 Hz amplitude error; synthetic one-hour resampling 0.043 s at 500→100→10 Hz/0.005 s at 62.5→10 Hz |

Numerical audits matched binary DCA references to 1e−12; DeLong matched 999
nondegenerate reference fixtures (AUC error ≤1.45e−15; p-error ≤1.18e−14), while
invalid/degenerate designs now fail explicitly. Synthetic/native checks establish
implementation behavior, not complete clinical validity or exact inferential
coverage. Resampling timings exclude loading, channel filters, and Catch24.
The final cleanup checks passed with no research reconstruction or experiments.
The commit is explicitly a pre-run checkpoint; its revision is available through
`git log -1`. Commit/push do not authorize executing the pending study.

## Pending execution

1. On explicit run approval, select a fresh root, source paths, and the frozen
   commit/environment. Prepare both branches once; validate provenance, actual
   fold support, scope counts, branch intersections, and extraction failures.
2. Stage main core LR, XGBoost, then EBM, reviewing complete family reports and
   measured runtime/memory. Initial full-data refits remain disabled.
3. Run broader ICU sensitivity and remaining supplements; regenerate manuscript
   artifacts from saved predictions/bundles. Preserve null/negative comparisons.

Main core 36 configurations; sensitivity 18; full main 180 (144 additional after
core). At 5/3/5 with 20 LR/50 tree trials, main core requires about 22,680 estimator
fits before retries/internal bagging. This is not a wall-time forecast.
Use affinity/quota-aware outer workers, bounded native/BLAS threads, validated
fold caches and resumable bundles. Extraction has an independent worker budget;
start at 16 and measure before increasing it. Heavy artifacts belong on the 1 TB
volume. Resource inventory and dry runs are read-only; no new experiment,
reconstruction, or external document update is authorized by this checkpoint.
