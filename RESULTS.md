# Results artifacts

This document describes saved artifacts and paper-output regeneration.
Scientific definitions and inference limits are in [METHODS.md](METHODS.md);
execution and root selection are in [README.md](README.md).

The 36-configuration main core study is complete at frozen revision `fcff7c7`.
The numerical equivalence gate and broader ICU/exploratory mortality core studies
completed, followed by 144 new main supplementary configurations and 72 new
broader ICU supplementary configurations. Completed cores are reused as verified
references. Additional mortality supplements remain unstarted. Historical and
diagnostic outputs remain separate, explicitly labeled evidence.

## Layout and integrity

Under `$AKI_ARTIFACT_ROOT`, the main roots are:

```text
data/processed/
results/catch22/experiments/<cohort_scope>/<validation_scheme>/
results/catch22/paper/<cohort_scope>/<validation_scheme>/
logs/
```

Scopes are `main_shared`, `icu_waveform_sensitivity`, and
`mortality_waveform_sensitivity`; validation schemes are
`nested_cv` and diagnostic `holdout`. Explicit results/paper overrides are base
roots unless already scoped. Historical roots and model basenames are retained.

Each experiments root contains `run_manifest.json`: requested configurations,
study/code/environment identity, diagnostic status, frozen branch rosters,
source hashes, resource budgets, and execution history. Required configurations
must exist before strict reporting succeeds. Family stages extend the manifest;
partial coverage cannot be presented as a complete three-family comparison.

A model configuration lives at:

```text
models/<family>/<outcome>/<branch>/<feature_set>/
  predictions/train_oof.csv
  predictions/test.csv
  artifacts/calibration.json
  artifacts/threshold.json
  artifacts/metadata.json
  artifacts/validation.json
  artifacts/folds/repeat_00/outer_00/
```

Nested `test.csv` pools outer-test predictions. Training OOF predictions are
model-development diagnostics, not publication estimates. Each outer-fold
bundle stores its fitted model, preprocessor, calibrator, threshold, parameters,
train/test predictions, and integrity metadata. XGBoost uses `model.json`;
LR/EBM use `model.pkl`; preprocessing uses `preprocessor.pkl`. Resume requires
matching fingerprints and every promised artifact, not just a prediction file.

Clinical/preoperative schemas are version 4, merged features version 5, and
waveform extraction protocol version 3. Old waveform/merged artifacts require
fresh extraction; changing a statistics-only protocol requires regenerating
statistics, not fitting models again. Historical artifacts are never relabeled
or upgraded in place.

Step 02's per-branch extraction sidecars record cohort IDs/operative bounds via
a roster hash, extraction source hashes, channel/filter specifications, rates,
substitutions, library versions, anti-alias resampling and missing-value policies.
They identify the original signal-processing contract in
[METHODS.md](METHODS.md#waveform-processing). Production finite descriptors are
preserved; the duration pilot's fallback-masked aggregates are separate diagnostic
artifacts and must not replace the frozen modeling inputs.

## Regenerate from saved artifacts

Use the [scoped saved-result commands](README.md#diagnostics-and-saved-result-reporting)
after selecting the existing study and processed-data paths. These consume saved
data and models without fitting. The default reporting policy is 5,000
unstratified patient-cluster draws, seed 0. The report-only launcher records
changed bootstrap settings before regeneration; direct metrics commands must
agree with the manifest policy. Standalone explanations require the appropriate
`COHORT_SCOPE`.

Strict reports align unique cases and verify patient, truth, and fold identity.
They require the same family's clinical reference and reject missing requested
runs, altered predictions, stale intervals, mixed study identities, or a pooled
prediction file inconsistent with its verified fold union. Explicit `--legacy`
reading supports historical artifacts but does not certify them as definitive.

## Publication bundle

The scoped paper tree contains tables, figures, reports, metadata, and a
`manifest.json` linking exported files. The principal outputs are:

| Artifact | Purpose |
| --- | --- |
| `tables/manuscript_<outcome>_main.{docx,pdf,html,md}` | Whole-case core comparison across all three families |
| Companion CSVs and `tables/metrics_summary.*` | Full metrics, denominators, and inspectable values |
| `tables/results_*_{main,delta}.*` | Detailed and supplementary results |
| `figures/incremental_ap_<outcome>_main.{svg,png,csv}` | Paired combined-versus-clinical AP differences |
| ROC/PR/calibration figures | Pooled outer-test curves and scores; calibration axes cover 0–1 |
| `reports/report.{md,docx,pdf}` | Consolidated working/supplementary report |

Main model tables prioritize AP, AUROC, Brier score, and paired AP differences.
Supplementary tables retain branch identity and all requested classification
metrics. Shading and selective bolding are descriptive, not significance marks.
Fold mean/SD curves are optional supplemental output, explicitly not confidence
intervals. Diagnostic status remains visible throughout exports.

## Comparisons and decision curves

Scoped results and paper tables include:

| File | Contents |
| --- | --- |
| `paired_comparisons.csv` | Directed core/ablation differences with paired intervals |
| `decision_curve_summary.csv` | Model, treat-all, and treat-none net benefit with pointwise intervals |
| `decision_curve_comparisons.csv` | Paired net-benefit differences and intervals |
| `delong_comparisons.csv` | Design eligibility, numerical status, raw/Holm-adjusted p-values |
| `comparison_inventory.json` | Coverage, hashes, actual bootstrap-draw identities, and calculation policy |

`comparison_primary` bundles emphasize the six primary whole-case AP contrasts;
`comparison_families` covers secondary matched-input family comparisons. There
are 18 unique core contrasts per cohort/outcome/branch. Requested ablations
retain their same-family `preop_only` dependency. Intervals use shared patient
resamples and operation-weighted denominators, including all repeat operations.
Valid replicate counts and uncertainty protocol 3 accompany the requested draw
count. Intervals condition on saved models; they do not repeat model development.

Ordinary DeLong is unavailable for pooled CV or repeat-patient designs; these
rows retain `unsupported_design` and missing p-values. Verified independent
holdouts use guarded calculations and Holm correction. Degenerate results are
explicit. DCA thresholds are exploratory; `dca_delta_*` figures have pointwise
bands, while `dca_families_*` compares families. Neither interval overlap nor
these broad threshold grids establish simultaneous or clinical superiority.

## Descriptives and explanations

Table 1 uses source-value median/IQR, explicit missingness, clinical sections,
and operation/patient denominators. Cohort flow, missingness, and descriptives
share the frozen scope/branch selectors. Whole-case outputs use the base names
`preop_descriptives`, `missingness_table`, and `cohort_flow`; windowed outputs
append `_windowed` to avoid overwriting another population. Flow counts live in
`metadata/cohort_flow_counts*.json`.

Fold explanations are under `figures/fold_explanations/`, mirroring model/fold
identity. They verify saved bundles and modeling-data hashes before export:

- LR: coefficients, intercept, training-SD units, and fold-specific encodings.
  Individual one-hot coefficients are not reference-category odds ratios.
- XGBoost: held-out SHAP contributions, importance, beeswarms, and base values
  on the raw uncalibrated log-odds scale.
- EBM: native term importance, held-out contributions, response plots, and
  separate missing/unknown-bin scores.

Explanations describe fitted associations rather than causal mechanisms;
logistic recalibration remains separate. Legacy single-model SHAP/EBM HTML
exports are opt-in. Smoke outputs live only under their disposable `SMOKE_ROOT`.
Aeon and archived outputs are outside the manuscript pathway.

## Window-duration pilot and feature audit

The manuscript retains the original 10/5-second windowed representation.
The subsequent outcome-blinded duration pilot is a supplementary technical
sensitivity analysis, not an AKI predictive ablation or prospective justification
of the original choice. Its independent artifact root is
`/media/volume/catch22/data/aki_prediction_project/pilots/window-duration-v2`:

- `REPORT.md`: completed feature audit and 120 paired synthetic challenges.
- `control/manifest.json` and `code/`: frozen sample, inputs, methods and software.
- `channels/`, `grid_diagnostics.json`, `pilot.npz`: channel checkpoints, numerical
  diagnostics and raw/audited aggregates. Five of 480 channels were rejected by
  the existing missing-sample rule; 475 matched the frozen 10-second baseline.
- `summary/{primary,finite_only,matched,first,middle,final}/`: recorded summaries,
  descriptor metrics and availability. No primary candidate was selected in any
  of 2,000 patient-bootstrap draws. All lengths passed the availability gate;
  40 seconds first achieved 95% coverage for all 24 descriptors in every channel.

Preserve this frozen report and its original rule. The current interpretation
and decision are in [METHODS.md](METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis)
and the [completed pilot record](.agent/workflows/window-duration-pilot.md).
No further duration experiments or gate revisions are planned for this manuscript.

Pending follow-up: inspect `CO_Embed2_Dist_tau_d_expfit_meandiff` in the saved
windowed AKI models, including all mean/SD/min/max variants for CO2/AWP and
associated missingness indicators where present. Use verified saved outer-fold
bundles and family-appropriate explanations across folds; do not refit, compare
importance scales across families, or remove features on this basis. Low
importance does not establish harmlessness; high importance motivates checking
the values' provenance and interpretation. Implementation-defined fallback zeros
are not automatically waveform errors. This audit has not yet been performed.

## Optimized extension and protected references

The broader ICU and exploratory mortality core batches contain 18 configurations
each. Their results/paper namespaces are separate from the completed main core.
The new study's `control/reused_inventory.json` records absolute original paths,
original fingerprints and integrity hashes for the protected 36 main runs.
External baseline references are validated against their own frozen provenance;
new checkpoints retain strict current-source matching. Missing or corrupt
required references fail rather than trigger refitting.

`control/execution_plan.json`, `control/active_scheduler.json`, per-job
logs/resource samples, and `control/supervisor_status.json` describe execution.
The active-scheduler file points to the current execution epoch's summary;
earlier resource logs and the original execution snapshot are preserved.
Reviewed orchestration overlays record their source commit and exact source
hashes separately, without changing scientific fingerprints.
Full per-fold HPO histories are saved under each configuration's
`artifacts/trials/`. The equivalence gate exports a saved-artifact agreement
report before the remaining extension jobs are admitted. Mortality DCA records
its exact 0.1–5% grid and exploratory interpretation. Reports and explanations
consume saved predictions/bundles only.

Both extension batches completed: 18 ICU and 18 exploratory mortality
configurations, with 180 verified fold bundles and saved-result reports. The
protected main study remained unchanged, and the relaunch check selected zero
fits. Computational validation completeness does not establish admission-level
outcome adjudication or clinical validation.

The mortality source-label audit confirmed 52 positive operation records in 47
patients among 5,765 operations/5,543 patients in each branch (42 patients with
one positive operation and five with two). Six patients have mixed source
mortality flags; three also have matching reported stay lengths. These are
unresolved linkage questions, not confirmed same-admission label errors.
The source flags match the official API and all 18 saved OOF configurations.
Operation-level results and labels are retained; no first/last-operation
deduplication or model rerun was performed. See the mortality/admission section
of `METHODS.md` for interpretation and the evidence register for audit artifacts.

Fold metadata records floating prediction dtypes. Checkpoint reload uses those
dtypes and round-trip CSV parsing so partial resume preserves native XGBoost
float32 test predictions and float64 training predictions exactly. Legacy
metadata remains readable; protected source artifacts are never rewritten.
