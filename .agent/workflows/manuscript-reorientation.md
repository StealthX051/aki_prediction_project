# VitalDB manuscript: execution and optimization register

## Status and authorization

The user subsequently authorized the main core run. All 36 configurations and
their bootstrap/report stages completed successfully on October 31, 2026 at
frozen modeling revision `fcff7c7c9794e0ea8405814c68176760a2eb9afc`. The pre-run
code was committed and pushed; local, remote, and frozen revisions were verified
identical before optimization benchmarking. Broader ICU and supplementary grids
remain unstarted. Current authorization covers bounded optimization diagnostics
and a reviewable implementation plan, preserving completed research artifacts.
Drive/email/manuscript external writes remain separate. This is the single
decision/evidence register; definitions belong in METHODS, commands in README,
and outputs in RESULTS. Earlier diagnostics below retain their stated limits.

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
- E/resource checks: earlier isolated diagnostics and offline tests, followed by
  the completed main core run recorded below. Scientific interpretation and
  publication review remain distinct from successful execution.
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
The frozen modeling commit is the pre-run checkpoint `fcff7c7`; subsequent user
authorization and execution are recorded below.

## Completed main core and pending extensions

Study root: `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1`.
`control/status.json` records completed/exit 0; the main nested manifest records
validated and 12 successful configurations per family. Metrics contain all 36
configurations. Five grouped outer/three HPO/five calibration folds, 20 LR/50
tree trials, and 5,000 patient-cluster resamples were retained; no full-data refits.
Both branches have 3,571 main operations/3,432 patients; the broader ICU selector
has 5,765 operations/5,543 patients. These are refreshed branch-qualified counts.

Next scientific work is review of core results, then broader ICU sensitivity and
selected supplements. Preserve null/negative comparisons. No additional grid is
part of the optimization diagnostics. Saved predictions/bundles support further
reports and explanations without refitting.

Main core 36 configurations; sensitivity 18; full main 180 (144 additional after
core). At 5/3/5 with 20 LR/50 tree trials, main core requires about 22,680 estimator
fits before retries/internal bagging. This is not a wall-time forecast.
Use affinity/quota-aware worker budgets, validated fold caches and resumable
bundles. The main run increased extraction from 16 to 64 workers while preserving
saved rows; its operational record is `control/resource-adjustment.json`.
Heavy artifacts and diagnostic scripts stay on the 1 TB volume.

## Bounded runtime investigation, October 31, 2026

No production Python, environment, clinical definition, or waveform setting was
changed for this investigation. Diagnostic scripts call the frozen production
functions. Completed model fitting/validation took LR 59.9 minutes, XGBoost
255.2 minutes, EBM 470.1 minutes; all three report stages totaled 17.0 minutes.
One windowed AKI EBM configuration took 250.2 minutes; four folds finished within
34.8 minutes, leaving a 215.4-minute single-fold tail. Tables below measure
diagnostic work, not the remaining grid's runtime.

Diagnostic roots under `/media/volume/catch22/data/aki_prediction_project/diagnostics/`:

| Evidence | Current result |
| --- | --- |
| `runtime-plan-20261031-mixed-v2` | Three repetitions of 22 fixed jobs: median total wall 66.0 s current staged threads, 59.2 s staged processes, 42.6 s shared process queue; startup, data loading, prediction, saving, and shutdown included |
| `runtime-plan-20261031-large` | One larger 32-job confirmation: 148.3 s current versus 64.3 s shared queue |
| `runtime-plan-20261031-split-cache` | Four cold split keys: one worker 102.0 s, four workers 33.9 s; cache envelopes and partitions byte-identical |
| `runtime-plan-20261031-whole-input` | Original outer-0 training operations only (2,850); two of its five grouped calibration partitions reused as diagnostic evaluation partitions; original outer-test patients excluded |
| `runtime-plan-20261031-whole-v2` | Six complete fold jobs per policy: 189.4 s current staged execution, 96.8 s overlapping configurations, 91.6 s individual-fold process queue; one pass each, common warmed split cache |

Kernel jobs retain identical inputs, seeds, parameters and EBM patient bags.
LR keeps BLAS 4; tree families keep BLAS 1. Kernel candidate tree fits use four jobs,
one-bag EBM HPO uses one. Fixed EBM kernels cap HPO at 100 rounds and bagged
refits at 25 rounds (14 outer/20 inner bags). Every saved training probability
across all rows/jobs/repetitions matched exactly. The shared queue reserves at
most 60 CPUs. Peak summed process RSS was below 10 GiB, with over 233 GiB system
memory available; summed RSS includes shared pages. CPU figures are sampled
process-tree accounting, not exact whole-tree resource integrals.

`analysis.json`, `runtime_summary.csv`, and `numerical-agreement.json` contain
the checks. Scripts are saved with their diagnostic roots. Earlier single-fit
probes are in `cpu-profile-20261031-v1` and `cpu-profile-20261031-lr-repeats`.
Changing LR BLAS 4 to 1 changed some probabilities; retain BLAS 4 when evaluating
scheduling alone. Initial prototypes exposed nested-pool shutdown waits and an
overly strict float32 CSV assertion; failed timings are preserved and excluded.

The whole-function replay uses three HPO trials, three inner folds and five
calibration folds. EBM uses the existing diagnostic protocol (200-round cap,
two outer/no inner bags), not production's 14 outer/20 inner bags. All 18 fold
jobs passed exact inner/calibration training-index containment, grouped EBM bag
separation, nine preprocessing fits, sequential trial tracing and bundle reload.
Trial order/parameters/scores, chosen parameters, calibration, thresholds,
feature order and every train/held-out prediction matched across all policies.
`scientific-agreement.json` records these checks. Native float32 probabilities
round-trip exactly after restoring float32; their float64 CSV representations
differ from the native values by up to 2.81e-8, not a model discrepancy.
`runtime-plan-20261031-final-audit/checkpoint-audit.json` records rejection of
incomplete metadata, changed protocol and a corrupted model; saved loading did
not fit. These checks do not replace an actual interrupted-controller test.

## Approved optimized extension (October 31, 2026)

The approved implementation replaces the prior proposal: use a bounded queue of
configuration subprocesses, cold split-cache preparation with up to eight
single-threaded workers, LR BLAS 4, XGBoost four threads, and unchanged EBM 12-job
production settings. CPU admission defaults to 60 on this 64-CPU VM. One EBM
configuration is admitted at a time. Numerical fitting policies remain fixed.

- [x] Add independent `mortality_waveform_sensitivity` selector and read-only
  legacy-sidecar overlays; preserve main/ICU manifest shapes.
- [x] Add queue, lightweight resource sampling, protected inventory validation,
  strict completed-run skipping and saved trial histories.
- [x] Add exploratory mortality DCA, 0.001–0.050/step 0.001.
- [x] Offline environment/native checks passed; final complete suite: 539 tests
  passed. Source changes during preflight now fail explicitly.
- [x] Sealed real-data mixed diagnostic passed: six configurations, 12 bundles,
  actual interruption/resume, unchanged saved checkpoint, no live stopped workers,
  and zero fits pending on relaunch. Full suite: 539 passed; subsequent narrow
  compatibility-guard tests: 49 passed.
- [ ] Validate one new broader ICU XGBoost combined/non-windowed configuration
  at full 5/3/5 and 50 trials: legacy 12 threads versus optimized four threads.
  This replaces the earlier per-family replay proposal; no old main fits repeat.
- [ ] Only after the gate passes, finish 18 broader ICU and 18 broader mortality
  core configurations (the optimized canary counts toward ICU's 18).
- [ ] Verify 180 new fold bundles, complete OOF coverage, reports, zero-fit
  repeat planning, and unchanged protected-study integrity inventory.

Protected root: `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1`.
New root: `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1`.
Reuse the protected processed features; no extraction/reconstruction. The
supervisor `scripts/cpu/run-optimized-core.py` controls the gate, batch and status.
The 144 remaining main supplementary configurations stay unstarted. Mortality
is exploratory but uses full validation; infeasible folds/bags fail explicitly.

Baseline aggregate estimates are 5,765 operations/5,543 patients, with 52
mortality-positive operations across 47 patients. Freeze exact counts per branch
before execution. Preserve legacy `fcff7c7` provenance and never patch old
fingerprints. Current run status/evidence belong under the new root's control
inventory rather than proliferating support documents.

The first disposable mixed diagnostic was interrupted after checkpoints appeared
because final integration edits overlapped its fold-preparation phase. Its
outputs under `diagnostics/mixed-integration` are excluded from acceptance;
repeat from the sealed checkout under `diagnostics/sealed-integration-v2`. Scheduler
process groups were reaped after SIGTERM. No protected main artifacts changed.

The first sealed interruption check exposed a consumer-side verifier error:
valid loaded folds carry a `resumed` manifest status. The verifier now accepts
that status only with intact bundles, exact hashes and full prediction coverage.
A regression also rejects a self-consistent bundle whose input differs from the
original frozen roster. Neither correction changes estimator fitting. The
initial sealed diagnostic remains diagnostic evidence, not production results.

### Readiness stop requested

The user subsequently requested stopping before any full runs. Full-budget
canary and ICU/mortality production execution are held; no new full runs have
started. Finish bounded readiness diagnostics only. The supervised launch command
remains available for later explicit execution.

The interrupted XGBoost check also exposed CSV dtype promotion when mixing
resumed float64-loaded scores with native float32 test scores. Checkpoint
metadata now records floating dtypes and loading uses round-trip parsing; this
preserves exact train/test values and pooled/fold unions without changing fit
calculations or weakening equivalence tolerances. Earlier disposable failed
checks remain excluded from acceptance. Final evidence is under
`diagnostics/sealed-readiness-final` and `control/integration-verification.json`.

Readiness evidence: the final modeling diagnostic used frozen revision
`f3b5dcf` and passed for LR, XGBoost and EBM across both new scopes. Subsequent
consumer-only compatibility checks pin reuse to that reviewed modeling digest;
they change neither fitting nor partition selection. The full main preview
resolves 36 protected configurations reused and 144 pending supplementary
configurations; none of those supplementary jobs were launched. The new core
preview resolves 18 ICU plus 18 exploratory mortality configurations. The first
future full-budget action remains the supervised XGBoost equivalence canary.
