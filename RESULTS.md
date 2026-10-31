# Results artifacts

This document describes saved artifacts and paper-output regeneration.
Scientific definitions and inference limits are in [METHODS.md](METHODS.md);
execution and root selection are in [README.md](README.md).

The 36-configuration main core study is complete at frozen revision `fcff7c7`.
Broader ICU and exploratory mortality runs are held at readiness; their
full-budget equivalence canary has not started. Historical and diagnostic
outputs remain separate, explicitly labeled evidence.

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

## Optimized extension and protected references

The broader ICU and exploratory mortality core batches contain 18 configurations
each. Their results/paper namespaces are separate from the completed main core.
The new study's `control/reused_inventory.json` records absolute original paths,
original fingerprints and integrity hashes for the protected 36 main runs.
External baseline references are validated against their own frozen provenance;
new checkpoints retain strict current-source matching. Missing or corrupt
required references fail rather than trigger refitting.

`control/execution_plan.json`, `control/scheduler/scheduler_summary.json`, per-job
logs/resource samples, and `control/supervisor_status.json` describe execution.
Full per-fold HPO histories are saved under each configuration's
`artifacts/trials/`. The equivalence gate exports a saved-artifact agreement
report before the remaining extension jobs are admitted. Mortality DCA records
its exact 0.1–5% grid and exploratory interpretation. Reports and explanations
consume saved predictions/bundles only.

Fold metadata records floating prediction dtypes. Checkpoint reload uses those
dtypes and round-trip CSV parsing so partial resume preserves native XGBoost
float32 test predictions and float64 training predictions exactly. Legacy
metadata remains readable; protected source artifacts are never rewritten.
