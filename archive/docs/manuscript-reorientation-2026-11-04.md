# Historical manuscript execution and review log

Archived on 4 November 2026. Run states and pending-work statements below describe
their recorded checkpoints. Use the [current handoff](../../.agent/workflows/manuscript-reorientation.md)
for present status and [METHODS.md](../../METHODS.md) for scientific definitions.
The event record is preserved; relative links have been updated for this location.

# VitalDB manuscript: execution and optimization register

## Status and authorization

The user subsequently authorized the main core run. All 36 configurations and
their bootstrap/report stages completed successfully on October 31, 2026 at
frozen modeling revision `fcff7c7c9794e0ea8405814c68176760a2eb9afc`. The pre-run
code was committed and pushed; local, remote, and frozen revisions were verified
identical before optimization benchmarking. The optimized scheduler and broader
ICU/mortality extension completed, including the full-budget equivalence gate,
36 new configurations and 180 verified outer-fold bundles. The user authorized
the 144 remaining main supplementary configurations on October 31, 2026; the
batch ran in tmux with the completed 36 main configurations reused.
The main supplementary batch completed successfully on November 1 at 02:02 PST
(144 new configurations plus 36 reused cores). The broader ICU supplementary
batch also completed on November 1; its separate record is below. The duration
search is closed for this manuscript: retain 10/5 seconds and document the
completed outcome-blinded pilot without claiming optimality.
Main-core refits remain excluded. Mortality linkage verification continues
separately without changing labels or running additional mortality models.
Drive/email/manuscript external writes remain separate. This is the single
decision/evidence register; definitions belong in METHODS, commands in README,
and outputs in RESULTS. Earlier diagnostics below retain their stated limits.

## Scientific decisions

| Decision | Implemented contract |
| --- | --- |
| AKI horizon | Absolute rise ≥0.3 mg/dL by 48 hours or relative rise ≥1.5× by 72 hours; acute-rise gate for stage 3 |
| AKI ascertainment | Finite positive baseline ≤4 mg/dL and ≥1 finite positive result in `(opend, opend+72h]`; otherwise stages/labels remain missing |
| Population | General anesthesia; exclude ASA V/VI, retain missing ASA and repeat operations |
| Scopes | `main_shared` for AKI/ICU; broader `icu_waveform_sensitivity` for ICU; independent `mortality_waveform_sensitivity` for exploratory in-hospital mortality |
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
The [production processing contract](../../METHODS.md#waveform-processing)
records native-grid loading, ECG loading-only substitution, integer-truncated
operative cropping, interpolation, operation-wide forward-backward filtering,
anti-alias resampling and the separate branch rates. Windowing occurs after
operation-wide filtering. Finite library outputs remain unchanged in production;
the pilot's qualification masks are diagnostic only. Historical helpers under
`custom_filters/` do not define the active signal or cohort pathway.
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
configuration was initially admitted at a time. The adaptive follow-up below
replaces that admission limit; numerical fitting policies remain fixed.

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
- [x] Validate one new broader ICU XGBoost combined/non-windowed configuration
  at full 5/3/5 and 50 trials: legacy 12 threads versus optimized four threads.
  This replaces the earlier per-family replay proposal; no old main fits repeat.
- [x] Only after the gate passes, finish 18 broader ICU and 18 broader mortality
  core configurations (the optimized canary counts toward ICU's 18).
- [x] Verify 180 new fold bundles, complete OOF coverage, reports, zero-fit
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
outputs under `diagnostics/mixed-integration` are excluded from acceptance.
The later sealed attempts and final successful check are distinguished below. Scheduler
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

Final readiness checks also passed at production partition counts: both outcomes
and both branches support the exact 5/3/5 plan, with all production EBM bags
checked. The 22 unique grouped partition keys are warmed under the new study's
control directory. Both scoped diagnostic reports regenerated from saved
predictions with 100 diagnostic bootstrap draws (no fitting), including mortality
DCA; these are diagnostic artifacts, not definitive manuscript results. New
canonical model namespaces remain empty. User-requested launch hold remains in
force for the full-budget canary and both production batches.

## Unattended-readiness audit

Independent operations, scientific-boundary and bloat reviews found no new
leakage or fitting-method changes. The reviewed modeling digest remains
`926b0f7a5eb8710b3d4de7a88ecab8ddad3a43de52e18a3c574e599ec3d30ce0`.
Fixed clean-terminal supervisor imports, foreground hangup cleanup while
preserving nohup operation, interpreter/runtime provenance, and real cancellable
process bootstrap deadlines. Removed the ineffective thread timeout wrapper and
repaired delayed-task sequential fallback; resampling draws/calculations are
unchanged. Mortality's exploratory role and cohort denominators now appear in
all human report/table exports. Removed dead code and stale/duplicate docs;
retained focused behavioral/scientific tests without new dependencies or service
infrastructure. Full suite: 550 passed, including bounded subprocess probes.
No full-budget canary or production run was launched; the user's hold persists.

## Authorized tmux launch

The user explicitly requested starting the ICU sensitivity and mortality runs
using tmux. Launched session `vitaldb-sensitivity` with `run` and `monitor`
windows, frozen source `94c087f`, supported Conda interpreter and a 60-CPU
budget. The supervisor started successfully and entered `legacy_canary`.
It automatically proceeds through the optimized canary, exact equivalence gate,
18 ICU plus 18 exploratory mortality core configurations, and saved-result
reporting only when each preceding gate succeeds. No main-core refits or 144
supplementary configurations are selected. Live status/logs and exact launch
authorization are under the new study's `control/` directory. Earlier readiness
holds above are historical and superseded by this explicit run request.

## Adaptive admission and no-work-loss handoff

The user authorized replacing static reservations with measured CPU admission,
targeting the existing 60-CPU budget on the 64-CPU VM. Production profiling
showed about 32 busy cores on average and EBM about 6.6 cores per configuration;
the original single-EBM limit would leave a substantial serial tail. Adaptive
admission permits multiple EBM configurations, at most 12 configurations, and
one extra admission per 60 seconds after a full 60-second pressure window below
93% of the CPU budget. Memory reservations/floors remain strict. Peaks at 100%
CPU are acceptable; throughput and memory, not utilization alone, establish value.
The `reserved` admission policy retains the original behavior for rollback.

The selected handoff finishes active configurations. The tested drain helper
suspends only the old queue, waits for its independent training workers, verifies
their artifacts, and retires the queue before restart. Known multiprocessing
resource trackers are coordination helpers, not training jobs. The first live
attempt rejected that tracker and safely resumed the queue; the corrected
attempt stopped admissions with all three training jobs continuing normally.
During this legacy drain, resource sampling has an explicit gap. New controllers
support native drain requests without suspending their sampler.

Scientific files, Git HEAD and numerical settings remain frozen. A committed,
allowlisted orchestration overlay records exact before/after source hashes in a
new execution epoch; original snapshots, fingerprints and resource logs remain
intact. The passed full-budget XGBoost canary is verified from saved integrity
hashes rather than repeated. Live drain/migration evidence and diagnostic results
remain under the existing study's control and diagnostic directories.

Verification evidence: the offline environment/native checks and 566-test suite
passed, followed by 70 focused final integration checks and 17 scheduler checks.
Six audited outer jobs produced identical raw/calibrated predictions and trial
scores under reserved versus adaptive scheduling (176.75 versus 102.71 seconds
in this bounded diagnostic). Twelve simultaneous production-bagged EBM kernels
(14 outer/20 inner bags, diagnostic 25-round cap) matched the reference exactly,
reached 100% machine CPU, and retained at least 219 GiB available memory. All
25 diagnostic process groups stopped. These timings use shortened diagnostic
admission windows and concurrent production activity; they do not forecast
production runtime. The diagnostic manifest distinguishes subsequent summary-
telemetry edits from the executed controller source. Durable evidence is
`diagnostics/adaptive-admission-20261101-v1/acceptance.json` on the study volume.

## Mortality counting and admission-linkage audit (October 31, 2026 PDT)

Decision: retain operation-level prediction at operation end and the source
`death_inhosp` flag for each eligible case. Report positive operations and unique
positive patients separately. Do not select the last operation only for patients
who died, change earlier positives to negatives, or propagate one patient's
positive flag across admissions. Operation-weighted metrics and patient-cluster
uncertainty remain the current analysis. First-eligible-operation-per-admission
would be a separate, outcome-independent sensitivity design after valid linkage.

The completed extension contains 18 ICU plus 18 exploratory mortality
configurations/180 fold bundles. Original main artifacts were unchanged and
zero-fit relaunch passed. This updates computational status; mortality admission
membership and death timing remain unadjudicated.

Read-only checks used the frozen clinical CSV, both processed branches and all
18 mortality pooled OOF prediction sets. `caseid` is unique, patient IDs and
labels match the source, and patients are separated in all 90 outer splits.
The current official API agrees with the frozen source in `caseid`, `subjectid`,
`casestart`, `adm`, `dis`, and `death_inhosp`. Source file SHA-256:
`7d6edb471e5eee3fde75e417084240c97bdbf6eff41cbd61e5dace44f1585ecf`.

| Population | Operations | Patients | Positive operations | Positive patients |
| --- | ---: | ---: | ---: | ---: |
| Raw VitalDB snapshot | 6,388 | 6,090 | 57 | 48 |
| Main shared cohort | 3,571 | 3,432 | 31 | 29 |
| Broader mortality, either branch | 5,765 | 5,543 | 52 | 47 |

In the broader cohort, 42 patients have one positive operation and five have two.
Six patients have mixed positive/negative source flags. Three have mixed flags
within a matching reported stay length: subjects 10 (54 days), 563 (16 days), and
4561 (3 days). This does not prove a common admission or an erroneous label.
All 6,388 source `casestart` values are zero and every `dis-adm` is a whole number
of days. Neither random case IDs nor equal stay lengths provide a validated
admission key or cross-admission operation order.

Admission-linkage route: provider slides identify `operations.case_id` as the
VitalDB link; INSPIRE provides `subject_id`, `hadm_id`, and `op_id`. No approved
operations/schema table was found in the known local project/research roots.
Its files are credentialed/DUA-controlled. A small authorized mapping/operations
extract or provider clarification of the flagged records is sufficient as the
next source check; downloading the full INSPIRE database is not required.
Validate mapping cardinality/coverage, patient consistency, relative-time units,
admission chronology and death endpoint before adjudicating any labels. The
separate dataset's numeric patient IDs or 30-day mortality example are not
substitutes for those checks.

Evidence is under
`/media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1/control/audits/mortality-linkage-20261101/`:
`audit.py`, `audit.json`, and `mixed-label-records.csv`. These are audit artifacts,
not new predictors or rewritten study results. No labels, cohorts or completed
models were changed, and no experiments were launched.

Sources: [VitalDB codebook](https://vitaldb.net/dataset/),
[official clinical API](https://api.vitaldb.net/cases),
[SNUH linkage presentation, pp. 33 and 46–47](https://www.ohdsi.org/wp-content/uploads/2025/06/OHDSI-APAC-Scientific-Forum-20250605.pdf),
[INSPIRE 1.4.2 identifiers and access rules](https://physionet.org/content/inspire/1.4.2/).

## Main supplementary launch (October 31, 2026, 8:37 PM PDT)

The user explicitly authorized starting the 144 supplementary configurations
while INSPIRE downloads. A fresh study at
`/media/volume/catch22/data/aki_prediction_project/studies/manuscript-supplementary-v1`
is isolated from both completed studies. Its independent code copy matches the
validated extension snapshot plus recorded adaptive scheduler overlay; 155 source
files were hash-checked. The scientific protocol and environment remain unchanged.

The read-only preview verified 36 protected main configurations/180 bundles for
reuse and exactly 144 pending main configurations across AKI/ICU, both branches,
and LR/XGBoost/EBM. No mortality or broader ICU configurations were selected.
The controller uses adaptive admission, a 60-CPU target, up to 12 concurrent
configurations, five outer workers, serial HPO and fixed numerical thread
settings. Validation remains 5/3/5, one repeat, 20 LR or 50 tree-model trials,
and 5,000 patient-cluster bootstrap draws. No preparation/extraction is requested.

Session: `vitaldb-supplementary` (run and monitor windows). Run-specific
`control/run.py`, `launch-authorization.json`, `preflight.json`,
`launch-status.json` and `launch.log` record the exact request, copied source
hashes, commands and execution status. The controller supplies split-plan
validation, checkpoint recovery, per-configuration resource logs and saved-
artifact reporting. Launching does not establish completion; inspect the status
and run manifest before interpreting results. INSPIRE remains separate raw data
and is not an input to this supplementary analysis.

## Conversation handoff: supplementary monitoring and INSPIRE admission audit

Current request: continue useful coding and scientific review while the 144 main
supplementary configurations run. Watch the existing batch and INSPIRE download;
verify hospital admission/outcome linkage; discuss what INSPIRE could improve in
outcome derivation; assess whether supplementary feature comparisons are worth
adding to the broader ICU and mortality populations. Do not start further grids
or change current labels/cohorts without discussing the scientific change first.
The user wants short, plain explanations and minimal code/docs/tests.

### First checks in the new conversation

- Work in `/home/exouser/projects/aki_prediction_project`. Read `AGENTS.md`,
  `README.md`, `METHODS.md` and this handoff section. Conda Python:
  `/home/exouser/miniforge3/envs/aki_prediction_project/bin/python`.
- Supplementary root:
  `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-supplementary-v1`.
  Tmux session: `vitaldb-supplementary`, with run and monitor windows.
  `control/launch-status.json` records batch exit; `control/active_scheduler.json`
  points to the live `scheduler_summary.json`. Inspect configuration states,
  failures, CPU/memory logs and `control/launch.log`. The last check found nine
  running configurations and no recorded failures; this is a snapshot, not a
  completion claim. Recheck current files when resuming. Give meaningful updates
  while working; do not promise monitoring after a turn ends.
- The study run manifest is
  `results/catch22/experiments/main_shared/nested_cv/run_manifest.json`.
  It must ultimately cover 180 logical main configurations: 36 read-only core
  references plus 144 newly fitted supplementary configurations (720 new outer
  bundles), complete OOF predictions and saved-artifact reports.
- Resume only when the existing controller is no longer active, using the same
  interpreter with `<supplementary-root>/control/run.py`. The controller locks,
  validates checkpoints, prepares exact folds, profiles resources and reports
  failures. Avoid launching duplicate controllers or editing frozen run code.
- INSPIRE 1.4.2 downloads in tmux session `inspire-download` to
  `/media/volume/catch22/data/inspire/1.4.2`. The user entered their password
  directly; do not request, read or store it. The last check found a running
  wget process and `operations.csv.gz`, `schema.csv`, `SHA256SUMS.txt`, compressed
  labs/vitals/medications/diagnoses and other files. Their presence does not prove
  the whole download is complete: check gzip integrity and SHA256SUMS, then
  verify download exit/completeness. Do not read download terminal scrollback
  unnecessarily. No INSPIRE data have been joined or fed into models.

### Scientific and execution boundaries

The supplementary launch started October 31, 2026 at 8:37 PM PDT. It reuses the
sealed processed CSVs in `manuscript-core-v1/data/processed`; no reconstruction
or extraction is planned. The dry run verified all 36 protected configurations
and exactly 144 pending. The split-plan barrier passed before training began.
Its independent frozen code copy matches the completed extension's validated
source plus adaptive scheduling overlay; source file hashes and the exact
command are in `control/launch-authorization.json` and `control/preflight.json`.
The old code revision is `94c087f`; orchestration overlay was reviewed/committed
at `4be027f`. The modeling digest is
`926b0f7a5eb8710b3d4de7a88ecab8ddad3a43de52e18a3c574e599ec3d30ce0`.

Settings remain 5 outer / 3 HPO / 5 calibration grouped folds, one repeat,
20 LR trials and 50 XGBoost/EBM trials, 5,000 unstratified patient-cluster
bootstrap draws, seed 0. HPO is serial within each fold; training-only
preprocessing/calibration/threshold selection and grouped EBM bags remain fixed.
Adaptive scheduling targets 60 CPUs on this 64-CPU/246-GiB VM, up to 12 concurrent
configurations, five outer workers, LR BLAS 4, XGBoost 4 native threads and EBM
12 jobs. Keep numerical settings fixed during the run. All outputs remain on
the mounted volume. Workspace edits do not affect the isolated live code copy.

Completed studies under the same `studies` parent:
`manuscript-core-v1` contains 36 main AKI/ICU configurations at `fcff7c7`;
`manuscript-optimized-v1` contains 18 broader ICU plus 18 exploratory mortality
configurations and 180 verified bundles. The full-budget XGBoost 12-versus-four
thread equivalence gate passed. Original main integrity and zero-fit relaunch
checks passed. Do not overwrite results or rerun any completed grids.

### Admission and outcome work

Retain operation-level prediction at operation end, repeat operations, patient-
grouped validation and patient-cluster uncertainty. Multiple positive operations
can correspond to a single death. Do not keep only decedents' last operations,
turn earlier positive operations negative, or propagate patient mortality flags
across admissions. Report positive operations and positive patients separately.

The broader mortality population has 5,765 operations/5,543 patients, including
52 positive operations in 47 patients (42 with one positive operation, five with
two), identical across branches. Main shared mortality has 31 positive operations
in 29 patients among 3,571 operations/3,432 patients. Fifty-two positive operation
rows are not 52 independent deaths, and the exploratory models still have few
positive patients. Assess stability and uncertainty before proposing more fits.

VitalDB's current CSV has random case IDs, patient IDs, case-relative timestamps
and no admission ID or shared chronological origin. Reported stay lengths are
integer days in all 6,388 source cases. Matching patient/stay length is not proof
of a shared admission; case ID order is not chronology. Six selected patients
have mixed mortality flags, three within matching reported stay lengths; these
are unresolved, not established source errors. Labels match the official API,
preprocessed rows and all 18 mortality OOF outputs. All 90 outer train/test
patient separations were checked. Durable audit code/results:
`manuscript-optimized-v1/control/audits/mortality-linkage-20261101/`.

INSPIRE creators explicitly describe linkage through the operations table's
`case_id` to VitalDB cases, with hospital admission identifiers. Start from the
actual downloaded schema and operations table, checking case mapping coverage,
cardinality, patient identity and timestamp units. Do not join separate datasets
on coincident numeric patient IDs. Use admission and death timing to investigate
repeated-positive/mixed-label cases and verify whether the same admission is
involved. Public INSPIRE coverage is a subset; quantify unmatched cases and any
selection differences before using linked outcomes. Its public mortality example
uses 30-day mortality; our endpoint remains in-hospital mortality. Any proposed
change needs an explicit definition, ascertainment/coverage audit and impact
counts before new models. Keep admission/death/postoperative audit fields out
of predictors. Do not silently change active study data or fingerprints.

Consider and verify whether the actual tables support better admission-specific
in-hospital mortality, timing-based mortality horizons, ICU transfer timing or
creatinine surveillance. These are questions, not established benefits. Compare
source definitions, observation windows, missingness and linkage coverage first.
Discuss whether additional broader ICU/mortality waveform ablations/windowed
comparisons answer a worthwhile question given existing core results, rare death
counts and compute cost. At this handoff no extension supplementary grid had
been authorized; the later completed ICU supplement is recorded below. Mortality
supplements remain unstarted. Preserve completed results and label any
revised-outcome study as a separate analysis.

Primary sources for verification:
[VitalDB codebook](https://vitaldb.net/dataset/),
[INSPIRE 1.4.2](https://physionet.org/content/inspire/1.4.2/),
[creator presentation, pp. 33 and 46–47](https://www.ohdsi.org/wp-content/uploads/2025/06/OHDSI-APAC-Scientific-Forum-20250605.pdf).
Keep credentialed INSPIRE data and generated model artifacts out of Git.

### Linkage audit follow-up (October 31, 2026, PDT)

Workspace utility: `scripts/audit_inspire_linkage.py`; five focused synthetic
tests pass. Aggregate evidence is in
`outputs/audits/inspire-linkage-20261101.json` (ignored by Git). Explicit inputs,
release checksums and before/after input hashes are checked; the utility does
not import pipeline configuration, derive replacement labels or export records.

- INSPIRE links 2,595/6,388 source operations, 1,437/3,571 main operations and
  2,336/5,765 broader ICU/mortality operations. Linked patient crosswalks are
  one-to-one; within linked admissions, admission/discharge times are consistent.
- Broader mortality: four source-negative operations have a recorded death
  after operation end within the admission; ten of twenty linked source-positive
  operations have no recorded death time. Treat these as unresolved discordances.
  Only one of six mixed-flag patients is fully linked, with separate admissions;
  none of five repeat-positive patients has all positive operations linked.
- Linked/unmatched broader mortality prevalence is 20/2,336 versus 32/3,429;
  ICU prevalence is 407/2,336 versus 636/3,429. Both groups have median age 59
  among numeric ages; seven unmatched ages are censored/nonnumeric. These checks
  do not establish representative linkage. Patients can occur in both groups.
- INSPIRE times are patient-relative minutes, unlike VitalDB case-relative
  seconds. Median absolute operation-duration disagreement is five minutes.
  ICU timing is documented for transfers within 24 hours, and some actual times
  fall outside that horizon or the linked admission. Do not substitute it for
  the current ICU LOS endpoint. Death times also require admission checks.
- Among 2,594 source-linked operations with valid stay chronology, 1,780 have
  postoperative creatinine within 48 hours, 1,800 within 72 hours and 1,818 within
  seven days, ending follow-up at discharge. Values are binned/clipped; counts
  describe observed surveillance, not complete ascertainment or new AKI labels.
- Saved mortality core results have 9-10 positive patients per outer test fold;
  all six combined-minus-preop AP and AUROC intervals cross zero. Prioritize
  ascertainment over additional mortality fits. A limited broader ICU sensor
  comparison could answer a specific exploratory question after discussion;
  neither a new outcome study nor any extra grid is authorized.

The live frozen source copy passed all 155 launch-authorized file hashes.
At 8:51 PM PDT the supplementary scheduler had two configurations complete,
eight running and no recorded failures. Eleven of twelve INSPIRE manifest files
passed checksums; completed gzip files passed integrity checks. Ward vitals was
still growing and wget remained active. Recheck these snapshots before acting;
do not assume a completed download or promise monitoring after the turn ends.

### Completed INSPIRE preparation (October 31, 2026, PDT)

The user authorized extraction/preparation and a linkage recheck. All twelve
release checksums pass. `scripts/prepare_inspire.py` extracts six byte-preserved
CSV tables with gzip integrity, row counts and hashes into
`/media/volume/catch22/data/inspire/prepared-1.4.2-v1/tables`. Compressed originals
remain in `inspire/1.4.2`. The preparation root contains a README, readiness
record, `linkage/audit.json` and a private 6,388-row case/admission crosswalk;
these remain outside Git and the study's predictors/labels. Audit reruns use
original gzip inputs; extracted tables use `preparation_manifest.json` hashes.

Coverage remains 2,595 source cases and 2,336 broader study operations. Sex
agrees for all 2,595 links, ASA for 2,519/2,537 comparable records, and age
differences fit INSPIRE's five-year bins. INSPIRE is a larger 2011-2020 hospital
extract with exclusions and a public 50% patient sample, not a complete copy of
the original VitalDB release. These facts do not fully explain the exact shortfall.
Its 21,099 nonmissing case IDs include 7,261 negatives and span -32,763 to 32,767;
18,504 IDs are outside the original VitalDB IDs. Their meaning remains unresolved;
do not infer an expanded public release or apply arithmetic remapping. The
[official-repository question](https://github.com/vitaldb/inspire/issues/3)
has no maintainer clarification at this check. No labels or grids were changed.

### Optional broader supplementary code prepared

The user requested runnable code for possible broader ICU and mortality
supplements, without authorizing fits. The existing full grid now supports
verified read-only import of the optimized study's 18 completed cores per scope
into a fresh output root. Original fingerprints, labels and folds are preserved;
reporting validates each scope's source references. New fits must use the same
outer folds as imported cores. README contains the dry-run command for either
scope or both. Actual preflight resolved 180 configurations: 36 reused and 144
pending (72 per scope); the proposed output root remained absent. Strict saved
prediction reporting validation passed all 36 imported cores without fitting or
bootstrap. No additional grid was launched.

### Completed duration pilot: retain 10/5 seconds

The duration search is closed for this manuscript. Retain 10-second windows with
nominal five-second stride; do not add durations, retune gates, drop reducers or
launch another duration grid. Forty seconds remains an exploratory alternative,
not a selected or optimal replacement. Whole-case core contrasts remain primary;
the windowed branch remains supplementary.

The main AKI models preceded this exploratory pilot. No AKI outcomes, model
scores or predictive fits informed its duration comparisons. The screen used
120 distinct patients, one operation each, from the main-shared AKI/windowed
outer-fold-0 training partition, and seven lengths through 640 seconds. It
cannot retrospectively serve as the original window-selection procedure or
select a replacement independent of all five existing outer-test partitions.

The pilot completed November 1 at 01:16 PDT: 480 channel tasks, 475 frozen
10-second baseline matches, five channels rejected by the existing >5%
nonfinite-sample rule, and 120 paired synthetic traces. Every duration passed
its availability gate. Forty seconds first achieved 95% numerical coverage for
all 24 descriptors in every channel; the improvement concerned the respiratory
embedding-distance descriptor. No duration passed the complete grid-origin rule;
all 2,000 patient-bootstrap draws selected no candidate. Raw finite-only and
matched-count diagnostics also selected none. These engineering thresholds
have not been validated as clinical or prognostic acceptance criteria.

This supports retaining a fixed representation without claiming optimality.
Grid displacement, coverage and window counts change with duration; extrema
can capture real events, and synthetic checks are not physiological validation.
Implementation-defined finite fallbacks are not waveform errors by definition.
Raw canonical values and the original failed selection rule remain preserved.
An exact 10/5-second VitalDB segmentation precedent supports engineering
plausibility, not Catch24-to-AKI validity. The complete methods, evidence and
manuscript wording are in [the completed pilot record](../../.agent/workflows/window-duration-pilot.md).

Artifact root:
`/media/volume/catch22/data/aki_prediction_project/pilots/window-duration-v2`.
Preserve its frozen `REPORT.md`, manifest, code and outputs. The partial v1 is
also preserved; v2 reused 12 verified filtered signals. The adaptive scheduling
handoff preserved 14 complete channel outputs and unchanged numerical functions;
provenance is under `control/resource-adjustments/20261101T052748Z/`. Its 24
focused tests passed and all 155 main supplementary source hashes matched.
Resource files contain final snapshots, not complete pilot utilization histories.

Pending follow-up, not yet performed: audit the embedding-distance descriptor
in saved windowed AKI models across outer folds/families, including CO2/AWP
mean/SD/min/max variants and associated missingness indicators where present.
Use verified saved models and family-appropriate explanations without refitting
or comparing importance magnitudes across families. Low importance reduces
concern but does not prove harmlessness; high importance motivates checking the
values' provenance, not automatic feature removal or a new model grid.

### Completed broader ICU supplements

The user explicitly authorized broader ICU supplements on November 1.
Study: `/media/volume/catch22/data/aki_prediction_project/studies/manuscript-icu-supplementary-v1`;
tmux: `vitaldb-icu-supplementary` (run/monitor), started 05:20 PST and
completed successfully at 09:15 PST on November 1.
Scope is only `icu_waveform_sensitivity/icu_admission`: 72 new configurations
and 18 verified core references from `manuscript-optimized-v1`, both branches
and all three model families. The separate frozen snapshot retains the reviewed
modeling digest, original labels/rosters/folds, 5/3/5 validation, 20/50 HPO budgets,
fixed estimator threads and 5,000 unstratified patient-cluster resamples.
Adaptive budget 60 CPUs, maximum 12 simultaneous configurations. All 72 new
configurations and reporting completed; the manifest validates all 90 logical
configurations including reused cores. Preflight and 70 targeted runner/inventory
tests passed. Status and provenance:
`control/launch-status.json`, `active_scheduler.json`, `preflight.json`,
`launch-authorization.json`, `launch.log`. Whole-machine CPU/memory history
was recorded every ten seconds in `control/machine-usage.jsonl`.
Additional mortality supplements remain unstarted.

Historical main fitting telemetry (6h06m, excluding subsequent reporting) shows
whole-machine CPU use averaging 87.3% of 64 CPUs; 91.0% excluding its first 15
and last 30 minutes, and 53.7% in the last 30 minutes. This includes concurrent
pilot work, not only model processes. Pilot-overlap estimates average 92.3%
during adaptive extraction and 90.4% during synthetic work; overlap alignment
uses approximate UTC anchoring of the elapsed-time log. Reproducible calculations
and source hashes are saved under the new ICU study's `control/` as
`window-duration-cpu-history.py` and `window-duration-cpu-history-20261101.json`.
Do not treat final overwritten pilot telemetry as a full utilization history.


## Final repository review before merge

On November 1, 2026 (Pacific time), the final offline verification passed native
library checks and all **629 tests** (71.40 seconds; no failures or skips), with
no broken package requirements. Evidence: `/tmp/aki-cpu-offline.PXdPVBxO/`.
Documentation links/anchors and whitespace checks passed. Independent reviews
covered protected supplementary reuse, INSPIRE utilities and duration-pilot
numerics/resume behavior.

The review fixed missing audit statistics to serialize as strict JSON null and
added checkpoint operation/channel identity checks plus exact assembled-array
validation before pilot summaries. A read-only audit of the completed pilot
verified all 480 channel checkpoints and every raw/audited/matched/phase slice;
all 535 checked frozen source/control/pilot files and five frozen pilot inputs
retained their hashes. No study results were recomputed or altered. These source
hardening changes apply to future execution; historical snapshots remain frozen.
The saved-model descriptor interpretation audit remains pending.


## Test effectiveness and dead-code audit

The November 1, 2026 (Pacific time) audit reviewed all 40 test modules (378
original test functions). Stronger checks now exercise real ECG loading/fallback,
Catch24 fallback masking, controller rejection before summary generation, INSPIRE
timing boundaries and exported figure content. A duplicate scheduler test was
removed; its warm-up assertion remains with the existing scheduler coverage.

The audit found and fixed prediction validation accepting invalid raw
probabilities, thresholds or fractional stored labels. Regression cases cover
those failures and valid boundary/optional-column behavior. Read-only validation
accepted all 288 saved pooled prediction files across the four completed studies.
Four unreferenced historical cohort helpers and unused launcher imports were
removed: production Python shrank by 129 net lines. Scientific compatibility
helpers covered by the reviewed modeling fingerprint remain unchanged.

Eight representative injected faults were all detected in a disposable code
copy, covering missingness boundaries, sample SD, anti-aliasing, the AKI horizon,
patient-cluster bootstrap/grouping, decision-curve odds and checkpoint integrity.
This is a sampled effectiveness check, not exhaustive mutation coverage. Native
checks and all 647 offline tests passed in 74.69 seconds, with no failures or
skips. Evidence: `/tmp/aki-test-effectiveness/` and
`/tmp/aki-cpu-offline.Ri3TDhIz/`. All 535 checked frozen files and the reviewed
modeling digest remain unchanged. No research fits or grids were launched.

## Publication asset exports and mentor review

The November 1, 2026 (Pacific) publication review verified current Journal of
Medical Systems Research limits: 4,000 main-text words; abstract instructions
conflict (300 versus 150–250), so draft to 250. Four main figures/two main tables
and six supplementary figures are editorial choices, not journal caps.
The separately versioned `publication-assets` mentor bundle contains native
174 mm SVG/PDF/EPS figures, PNG600/TIFF600 and JPG300, separate captions, and
styled HTML/Markdown/editable Word performance tables. Those dimensions are a
project convention. All 180 main, 90 broader-ICU and 36 core configurations retain
eight metrics and saved intervals; 18 mortality configurations remain separately
labeled exploratory meeting backup. Independent verification checked 4,752
formatted estimate/interval cells and frozen source equality.

Shared reporting now exports JPG previews and complete `_detailed` table
companions, propagates descriptive ranks across formats, corrects Brier/F1 delta
style mappings and favorable Brier direction, and labels continued PDF panels.
Final targeted reporting/export/comparison/DCA verification passed 92 tests;
evidence is `/tmp/publication-reporting-tests.log`. No models, bootstrap draws,
labels, thresholds, canonical study results or connected documents were changed.

Mentor artwork review restored Figure 1 to the earlier retained-cohort spine with
explicit exclusion branches (historical renderer 089deeb), using the current
counts and a native 174 mm monochrome layout. The model-workflow column was removed;
methods retain its content. The final cohort remains 3,571 operations/3,432 patients,
with nonexclusive AKI/ICU outcome annotations. The existing fallback renderer gained
an optional publication profile; default output behavior is preserved. Seventeen
focused cohort/export checks passed, including text containment at native size
and exclusion arithmetic. Caption, guide, narrative display plan and review packet
were refreshed; frozen results and the earlier mentor packet remain unchanged.

The full packet design review compared the historical PR/calibration renderers
and HTML/Word tables with all current displays. Figures now share outcome/family
ordering, input keys, legible legends and explicit AP percentage-point contrasts;
EBM panels retain native functions, fold variation and observed-value support.
Main tables use clinical/outcome/family groups; detailed tables preserve all eight
metrics in semantic panels, with a portable 15-input component key. Captioned
review proofs preserve native vector artwork, while submission masters remain
captionless. The narrative now emphasizes within-outcome incremental ICU gains,
rather than comparing raw AP across prevalences. Abbreviated methods/readiness
were restored with the frozen endpoint and comparison hierarchy. Main displays
are six pages; supplement methods/figures are eight. Final affected checks passed
107 tests. Independent audits retained 4,752 full-table and 72 main-table
estimate/interval entries and all figure coverage; no study results were changed.

## Portable workspace publication polish — November 2, 2026 (Pacific)

The current writing workspace separates publication copies from author drafts,
audit notes and exhaustive electronic sources. Four main figures/two tables and
eight consecutively numbered supplementary figures (S1–S8) are retained. Captions
now keep figure-specific decoding information; shared procedures appear once in
supplementary methods. Native Word Table 2 retains all 19 best displayed metric
cells, including ties, after assembly. Table 1 remains portrait and Table 2 landscape.

Curated supplementary tables retain all 163 clinical source records in 82 aligned
characteristics, 39 representative saved-grid decision-curve comparisons and all
five folds for seven displayed EBM terms (35 records). Complete 180/90-configuration
eight-metric tables, paired contrasts, exhaustive missingness/EBM/DCA records,
imported data, predictions and portable rebuild code remain available. Independent
verification matched all 72 main-table values/intervals and all 7,318 original
scientific file hashes. PDF proofs preserve native vector artwork; native Word
formatting was inspected directly because no Word layout renderer is installed.

Deliverables are sibling `VitalDB_Publication_Packet_2026-11-02_Polished.zip` and
`VitalDB_Manuscript_Workspace_2026-11-02_Polished.zip` under the November 2
visualization export directory. The earlier ZIP is preserved. No model fits,
bootstrap resampling, canonical result changes or cloud uploads were performed.

## Narrative-led display revision — November 3, 2026 (UTC)

The user authorized a figure-led narrative review and flexible information density.
The current review book interleaves Figure 1, Table 1, Table 2, then Figures 2–5:
population/context, absolute performance, paired AP/AUROC differences, calibration,
global native EBM importance, and selected response functions. The main set is
five figures/two tables; ten sequential supporting figures include PR/ROC,
channel/window/cohort sensitivities, both DCA views, additional EBM functions and
four held-out local examples. Standalone missing-bin dotplots are excluded;
missing-value support remains tabulated. AKI remains primary and ICU secondary.

Table 1 adds only operative duration, preoperative eGFR below 60 and stage 2–3 AKI.
It remains one portrait page with all original entries. New Table S7 supplies all
procedure categories with outcome frequencies, full AKI stages, repeated patients
and creatinine ascertainment. The editable story map and working manuscript draft
follow the revised display order. The main PDF has seven pages; methods/supporting
figures have twelve. All full eight-metric performance tables remain available.

Four native local EBM examples were selected at outcome-blind risk percentiles
and explained with their exact held-out models/calibrators. Exported signed terms
and groups reconstruct predictions within 1.2e-16; ROC curves match all 18 frozen
AUROCs. Independent review verified all 72 Table 2 entries, 19 bold metric cells,
15 synchronized captions, display geometry and 7,318 original scientific hashes.
No fits, new bootstrap estimates, outcome changes or cloud writes occurred.
The November 3 narrative-revision ZIPs preserve the previous polished exports.

## Shared publication appearance — November 3, 2026 (Pacific)

Publication appearance is now defined once per working environment: the live
repository uses `reporting/publication_style.json`; the portable workspace carries
an identical self-contained copy in `05_CODE`. Current repository plotting paths
and all 21 portable current/reference figure sets use the shared input/fold/branch
encodings, fonts, neutral roles, line weights and export quality. Word writers
explicitly resolve Arial/dark headings and remove theme overrides; metric bolding
and larger Excel editing fonts are retained. The figure sequence and data remain
unchanged. Historical snapshots/imported documents are preserved as provenance.

`python 05_CODE/rebuild_publication_workspace.py --package` refreshes all displays,
books, inventories and ZIPs from bundled data. Style fingerprints, dimensions,
raster quality, native Word styles and the 19 Table 2 bold metric cells are
validated. An isolated changed-profile replay verified actual color propagation
in five SVGs and font propagation through two concrete Word writers. Independent
review retained all 84 figure CSV/caption hashes and all 7,318 original scientific
hashes. Focused reporting/export/cohort/table/style tests passed. The shared-style
ZIP revision preserves the earlier narrative and polished exports.

## Publication wording — November 3, 2026 (Pacific)

An editorial pass covered all 21 current/reference figure sets and 25 editable
table copies, including the complete workspace. Titles, labels and captions now
use concise publication language; notes retain units, denominators, reference
definitions and uncertainty keys. Figure titles/captions are canonical source
inputs; workbook headings and notes derive from the corresponding Markdown tables.
An inventory of rendered reader text is included in the portable workspace.

Independent review verified all 15 active captions across source, Word and PDF,
unique supplementary titles, 19 bold Table 2 metric cells, unchanged 57 figure
CSVs and all 7,318 original scientific files. Historical imports retain their
wording. The editorial ZIP revision preserves earlier exports; no model fits,
new bootstrap estimates or cloud writes occurred.

## Reader navigation and supporting definitions — November 4, 2026 (UTC)

The user approved navigation, calibration support and a waveform descriptor key,
while reserving manuscript writing. The offline rebuild now adds PDF section
bookmarks and links, matching native Word links, and a linked supplementary-table
index. Actual destinations were validated in 24 PDFs and 30 Word files; the main
and supplementary books retain seven and twelve pages.

Figure 3 now shows held-out predicted-risk distributions beneath the existing
calibration curves. New one-page Tables S8 and S9 give calibration diagnostics
and definitions of all six waveform descriptors displayed in the EBM figures.
Definitions use the pinned pycatch22 0.4.5 source, including its decreasing-run
count convention. All 120 calibration points, 480 histogram counts and twelve
diagnostic records were reproduced from bundled held-out predictions.

All 57 existing figure CSVs and both manuscript starting files remain byte
identical. Display rebuilds preserve author edits by default; the separate
`--refresh-author-material` flag explicitly regenerates the starting material.
The reader-revision ZIPs preserve earlier exports. No model or calibration fits,
new bootstrap estimates, manuscript writing or cloud writes occurred.

## Repository cleanup before shelving — November 4, 2026 (UTC)

Five unreferenced historical scripts/notebooks moved into `archive/`; current
Catch22, experimental Aeon and scientific audit entry points retain their paths.
Unused generated split lists, training logs and a temporary clinical fixture
were removed from the tracked working tree after preservation under ignored
`outputs/legacy-reference/`. No active research data or study artifacts changed.
README now identifies completed studies, the active code layout and resume paths.

The current portable manuscript builders are preserved as 34 source/configuration
templates under `reporting/manuscript_workspace/`, with the two canonical shared
style files supplied at installation. The read-only workspace check verified all
36 mapped files against the delivered writing workspace. Explicit `--apply`
updates code only and preserves data and author files.

The supported offline verification passed native library checks and all 666 tests.
Evidence is retained under ignored
`outputs/cpu-validation/pre-shelve-2026-11-04/`. No training controller was running;
remaining tmux panes were completed shells or log tails. The research volume was
mounted at `/media/volume/catch22` on `/dev/sdb`; it holds the study artifacts and
must be retained independently of Git and the writing ZIPs. Git staging, commit
and push remain for the user.

## Test relevance review — November 4, 2026 (UTC)

Reviewed the suite for duplicate coverage, vacuous assertions and unnecessary
parameter expansion. Consolidated reading-table, PDF/style, launcher-failure and
JSON checks into stronger existing tests; retained scientific boundaries,
independent numerical references, patient leakage, checkpoint integrity and
safe-resume checks. Auxiliary research-tool tests remain because they protect
the numerical evidence used in Methods.

The suite decreased from 44 to 43 files, 401 to 391 test functions, and 666 to
647 collected cases. All 647 passed in 87.83 seconds; the 30 warnings are existing
dependency deprecations. No tests were hidden or excluded and no production code
changed. Evidence is under ignored `outputs/cpu-validation/test-review-2026-11-04/`;
README now gives focused test commands for ordinary edits.
