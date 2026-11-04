# Completed Catch24 duration pilot and manuscript decision

## Current decision

Retain **10-second windows with a nominal five-second stride** for this
manuscript. The duration search is closed: no additional lengths, gate tuning,
aggregation redesign or duration-prediction grid is planned. Forty seconds is
an exploratory alternative, not a selected or optimal replacement. Whole-case
comparisons remain primary; the windowed branch remains supplementary.

The main AKI prediction models preceded this pilot. Their existing outputs use
whole-case and 10-second windowed features; no AKI prediction comparison across
window lengths was performed. The pilot can characterize and support retaining
a fixed engineering choice retrospectively. It cannot be presented as the
prospective selection procedure for the original models.

A related VitalDB ECG/capnography study used exactly 10-second windows with a
five-second overlap [1]. That is implementation precedent, not validation of our
Catch24 estimates. Related physiological prediction work also considered longer
windows [2]; neither study establishes an optimum for operation-level AKI
prediction. Retaining the original representation is a pragmatic decision.

## Completed analysis and artifacts

Root: `/media/volume/catch22/data/aki_prediction_project/pilots/window-duration-v2`.
The controller completed on November 1, 2026 (Pacific time). Its frozen
`REPORT.md`, `code/`, `control/manifest.json`, checkpoints and summaries remain
unchanged. This workspace document records the subsequent manuscript decision;
it does not replace or rewrite the frozen protocol or results.

- Sample: 120 distinct patients, one operation each, from
  `main_shared/any_aki/windowed` outer-fold-0 **training** only. Sampling used
  `default_rng(0)` on sorted patient IDs, then selected one operation per patient.
  No replacements were made after channel rejection. Outcome values, model
  scores and held-out patients did not enter this screen.
- Lengths: 10/20/40/80/160/320/640 seconds. For each length L and rate fs,
  window samples were `int(L*fs)` and stride s was `int(L*fs/2)`. Origins were
  `0`, `floor(s/4)`, `floor(s/2)`, `floor(3*s/4)`. Only complete windows within
  the operative interval were retained.
- ECG/PPG retained 100 Hz; CO2/AWP retained 62.5 Hz. Filters, interpolation,
  `pycatch22==0.4.5`, flat-window exclusion (`SD < 1e-6`), float64 summaries,
  sample SD (`ddof=1`) and propagation of undefined values were unchanged.
- All 480 channel tasks finished. Five channels were rejected by the existing
  >5% nonfinite-sample rule; the other 475 matched the saved 10-second/origin-zero
  raw descriptors. This verifies replay, not physiological accuracy.
- Raw canonical values were preserved. A separate qualification view masked
  implementation-defined finite fallbacks verified against pinned library
  branches. Ordinary zeros were retained. Observed flags concerned only
  `CO_Embed2_Dist_tau_d_expfit_meandiff` in CO2/AWP; no audited fluctuation
  fallthrough was observed. A flagged descriptor is not a rejected waveform.

The initial partial v1 is preserved. V2 retained its identical sample and reused
12 hash-verified filtered signals. Its later scheduling adjustment retained 14
complete channel outputs and unchanged numerical functions; archived evidence
is in `control/resource-adjustments/20261101T052748Z/`. Scheduling changes did
not change features, inputs, labels or sample selection.

## Recorded engineering definitions

These are the rules used for the completed pilot, not validated clinical
standards or instructions to keep searching until a length passes.

**Numerical availability.** A descriptor/channel/operation was available only
when its mean, SD, minimum and maximum were finite at all four origins after
the fallback audit. Undefined included-window values propagated to the operation
summaries; they were not averaged away. One affected window could therefore
make that descriptor unavailable. This measures computability under a strict
rule, not signal quality, estimation accuracy or predictive usefulness.

The availability gate required, in every channel, at least **22/24 descriptors**
each available in at least **114/120 patients (95%)**. This is marginal coverage
per descriptor, not joint completeness of all descriptors in the same patients.

**Grid-origin sensitivity.** For each descriptor, channel, reducer and length,
let G be the operation summary and calculate:

`S = between-patient IQR of each patient's median G across origins`

`E = patient's (maximum G - minimum G across origins) / S`

The output passed when the empirical **90th percentile of E was <=0.25**, with
at least 114 complete patients. A zero/numerically negligible S was unevaluable
(`S <= 100*eps_float64*max(abs(patient origin-medians))`), not repaired with an
epsilon. A duration needed **20/24 descriptors** passing in **every** one of the
16 channel/reducer combinations. Unevaluable outputs did not count as passes.

**Selection and bootstrap.** The shortest L qualified only if both L and 2L
passed availability and origin sensitivity, with doubling improving each
channel/reducer count by no more than one descriptor (and availability counts
by no more than one per channel). The rule was repeated in **2,000 paired
patient-bootstrap draws**, seed 0, preserving channels/lengths/origins/reducers
together. The same candidate needed **>=80%** selection frequency. These cutoffs
were engineering conventions; the bootstrap quantifies the rule's resampling
behavior, not its validity for choosing prognostic features.

Stability screening has precedent [3], but the cited wearable series were
sampled once per minute, unlike these raw waveforms. Neither that work nor the
original Catch22 paper [4] validates these particular tolerances.

## Findings and limits

Every tested length passed the availability gate: ECG/PPG had 24/24 adequate
coverage at every length, CO2/AWP had 23/24 at 10/20 seconds and 24/24 from
40 seconds onward. For the embedding-distance descriptor, availability improved
from **47/120 AWP and 85/120 CO2 operations at 10 seconds** to **118/120 and
119/120 at 40 seconds**. These counts describe one descriptor satisfying the
all-origin/all-summary rule, not the number of usable waveforms.

No length passed the complete grid-origin gate; no technical candidate was
selected in any of the 2,000 bootstrap draws. Raw finite-only and matched-count
secondary analyses also selected no candidate. Means improved in some respiratory
outputs, but longer durations did not uniformly stabilize all channels or
reducers. Minima, maxima and SD were particularly sensitive. Thus 40 seconds
was the shortest tested length restoring coverage for every descriptor, but it
was **not selected by the original full rule**. A later preference for it would
be a post hoc judgment, not a prospectively validated threshold decision.

Interpretation must account for:

- Maximum origin displacement grows from about 3.75 seconds at L=10 to
  240 seconds at L=640. Window counts and covered edges also change. Matched
  candidate counts were equalized across origins within L, not across lengths
  or necessarily after flat-window exclusions.
- Minima/maxima can reflect real brief events, artifacts and changing coverage.
  Zero IQR makes this statistic undefined; it does not by itself show unreliable
  estimation. Overlapping grids are not independent test-retest observations.
- The 120 paired synthetic challenges used 1,200-second oscillatory/AR(1)
  signals with a central 60-second change, two sampling rates and two noise
  levels. They were algorithm checks, not physiological validation. At lengths
  >=80 seconds, no window fits wholly inside the episode. Missing contrasts
  remain missing; no clinical attenuation or latency gate was established.
- One training-partition screen cannot select a global duration independently
  of all existing outer-test patients: its patients enter other folds' test
  partitions. Existing outcome results had also already been inspected.
- Neither availability nor placement sensitivity establishes AKI, ICU or
  mortality predictive superiority or equivalence.

## Pending saved-model interpretation audit

The remaining proposed check concerns the embedding-distance descriptor in
saved **windowed AKI** models, not another duration experiment. Inspect all
mean/SD/min/max variants for CO2/AWP and associated missingness indicators where
present, across saved outer folds and model families. Verify bundle/data hashes;
use family-appropriate coefficients, SHAP or EBM term contributions without
refitting. Do not compare importance scales across families or rely on one
aggregate ranking. Feature removal or retraining is a separate scientific change.

Low importance would lower concern, not prove harmlessness; high importance
would motivate checking value provenance and interpretation, not automatically
invalidate a model. An implementation-defined fallback under a degenerate or
near-degenerate embedding-distance distribution is not automatically a waveform
error. The requested cross-family review remains incomplete.

## Manuscript wording

The windowed analysis used 10-second windows with a five-second stride. A
subsequent outcome-blinded duration analysis found improved numerical coverage
for an embedding-distance descriptor in respiratory channels at 40 seconds,
but no duration satisfied the complete placement-stability criterion. We
retained the original representation and report window-duration optimality and
comparative AKI performance across durations as unresolved. The pilot thresholds
were exploratory engineering conventions rather than validated clinical criteria.

## Selected evidence

1. [Idrobo-Ávila et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100950/):
   exact 10/5-second segmentation in VitalDB ECG/capnography signal-quality
   assessment, not a duration comparison or Catch24-to-AKI validation.
2. [Finseth et al., 2023](https://doi.org/10.1109/ACCESS.2023.3254134):
   compared 10–40-second epochs for personalized physiological stress
   classification using mixed feature libraries and selected features; it does
   not establish a transferable Catch24 duration for operation-level outcomes.
3. [Zhou et al., 2022](https://www.jmir.org/2022/7/e34669/):
   Catch22 stability assessment on minute-sampled heart-rate series; its lengths
   and tolerances do not transfer directly to raw waveforms.
4. [Lubba et al., 2019](https://link.springer.com/article/10.1007/s10618-019-00647-x):
   rationale for a compact feature library, without a universal clinical window
   duration or validation of this pilot's selection cutoffs.
