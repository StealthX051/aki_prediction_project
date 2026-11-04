# Manuscript status and handoff

Updated 4 November 2026. The [historical execution and review log](../../archive/docs/manuscript-reorientation-2026-11-04.md)
retains launch authorizations, benchmark results, source correspondence and prior
verification records. Current commands are in [README.md](../../README.md),
scientific definitions in [METHODS.md](../../METHODS.md), and artifact layouts in
[RESULTS.md](../../RESULTS.md).

## Completed analyses

Study roots are under
`/media/volume/catch22/data/aki_prediction_project/studies/`:

| Study | Completed scope |
| --- | --- |
| `manuscript-core-v1` | 36 main AKI/ICU core configurations; sealed processed features |
| `manuscript-optimized-v1` | Equivalence gate, 18 broader-ICU core and 18 exploratory mortality configurations |
| `manuscript-supplementary-v1` | 180 main configurations, including the 36 reused cores |
| `manuscript-icu-supplementary-v1` | 90 broader-ICU configurations, including 18 reused cores |

There are 288 logical configurations and 1,440 outer-fold bundles. Main-core fits
remain frozen at `fcff7c7c9794e0ea8405814c68176760a2eb9afc`; later studies retain
their own frozen code, fingerprints and verified reuse references. Workspace
reporting changes do not replace these research artifacts.

The main cohort has 3,571 operations in 3,432 patients: 189 AKI events and 834 ICU
admissions. The broader cohorts have 5,765 operations in 5,543 patients. Exploratory
mortality includes 52 positive operations in 47 patients; these are not 52
independent deaths. Exact eligibility, outcome and validation rules remain in
METHODS.

## Current paper decisions

- Predict at operation end. AKI is primary; ICU admission is secondary.
- Use whole-case comparisons in the main paper. Keep windowed, channel-ablation
  and broader-cohort results supplementary; mortality remains exploratory.
- Prioritize average precision and show AUROC alongside it. Retain negative and
  uncertain contrasts. The clearest incremental gains are ICU XGBoost and EBM;
  AKI findings are inconsistent.
- Use native EBM term importance, selected response functions and held-out case
  examples. Preserve surgical case-mix terms and training support. Local channel
  contributions are within the combined model, not differences between models.
- Retain 10-second windows with a nominal five-second stride. The duration search
  is closed; its technical pilot did not select a duration under the full rule.
- Additional mortality supplements, severe-AKI models, new model families and
  sampling-rate experiments are outside the completed paper plan.

Original mentor decisions and their sources remain in the historical log.
New fitting, outcome relabeling or study-design changes need a user request;
completed grids should not be relaunched for routine reporting work.

## Writing workspace

Current exports:

- `VitalDB_Publication_Packet_2026-11-04_Reader_Revision.zip`
- `VitalDB_Manuscript_Workspace_2026-11-04_Reader_Revision.zip`

The export directory on this VM is
`/home/exouser/.codex/visualizations/2026/11/02/01a19a1b-3219-7783-9fe8-85b89edb61b8/`.
The unpacked complete workspace is its `manuscript-workspace/` subdirectory.

The main reading book has five figures and two tables across seven pages.
Supplementary methods and ten figures occupy twelve pages. Supplementary tables
S1–S9 include paired-comparison companions and the input-set key. Tables S8 and
S9 give calibration diagnostics and waveform descriptor definitions. Long tables
have section links and bookmarks; all tables have editable Word copies.

The user is writing the manuscript. Display rebuilds preserve existing author
starting files by default. The separate `--refresh-author-material` option
regenerates them only when explicitly requested.

The [workspace source guide](../../reporting/manuscript_workspace/README.md)
describes the 34 versioned templates and two shared style files. The read-only
installer check verified all 36 against the delivered workspace. The shared
appearance profile is `reporting/publication_style.json`; no research data or
figure ZIPs belong in Git.

The workspace rebuild renders saved values without fitting models, calibrators
or new bootstrap samples. Selecting new EBM cases requires the trusted fitted
models; the included examples rebuild from CSV. Earlier ZIPs and imported source
documents remain historical records.

## Verification

| Check | Result and evidence |
| --- | --- |
| Current offline suite | 647 passed in 87.83 s; `outputs/cpu-validation/test-review-2026-11-04/` |
| Native libraries and environment | Passed before test consolidation; `outputs/cpu-validation/pre-shelve-2026-11-04/` |
| Manuscript displays | 21 figure sets, 27 native table documents, 19 bold Table 2 metric cells |
| Reader navigation | Actual destinations verified in 24 PDFs and 30 Word files |
| Calibration support | 120 saved points, 480 histogram counts and 12 diagnostic rows reproduced from held-out predictions |
| Preserved exports | 57 existing figure CSVs and both manuscript starting files unchanged |

Manuscript QA is under `00_START_HERE/qa/` in the complete workspace, including
`reader_upgrade_independent_review.json`, `publication_navigation_validation.json`
and `reader_support_revision.json`. Its `SHA256SUMS.csv` verified 8,086 files with
zero mismatches.

## Open items

- Complete manuscript writing, references and author declarations.
- Finish the targeted cross-family review of
  `CO_Embed2_Dist_tau_d_expfit_meandiff` variants in saved windowed AKI models.
  Selected EBM explanations have been examined; the requested LR/XGBoost/EBM
  review is not documented as complete. Preserve current features and models.
- Resolve admission/death-time discordances from the INSPIRE linkage audit before
  considering any mortality-label revision. Existing labels remain unchanged;
  linkage/timing audit fields are excluded from predictors.

## Shelving and resuming

At the 4 November check, research controllers had exited; remaining tmux panes
were shells or log tails. The INSPIRE download had exited successfully.

Retain the research volume mounted at `/media/volume/catch22` and the downloaded
writing ZIPs. Git does not contain fitted models or original waveforms. Exact
binary environment replay also needs the wheelhouse recorded in
[locks/README.md](../../locks/README.md).

On resuming, read README and this handoff, activate the pinned Conda environment,
and inspect saved manifests before starting any research command.
