# Results artifacts

This file is more artifact-focused than `README.md` and more operational than
`METHODS.md`. It documents where the repository writes paper-facing outputs,
what those outputs mean, and how to regenerate them from saved artifacts.

## Artifact roots

Generated Catch22 artifacts default to:

```text
/media/volume/catch22/data/aki_prediction_project
```

Relevant environment controls:

- `AKI_ARTIFACT_ROOT`: canonical generated-artifact root
- `AKI_STORAGE_POLICY`: heavy-output policy (`enforce`, `warn`, `off`)
- `RESULTS_DIR`: explicit experiments root
- `PAPER_DIR`: explicit paper-output root
- `SMOKE_ROOT`: isolated smoke-test root

The default policy is to fail fast if heavy generated outputs are routed off the
attached media volume unless the policy is relaxed explicitly.

## Catch22 results layout

The validated default artifact tree is:

- `results/catch22/experiments/`: raw model outputs and intermediate artifacts
- `results/catch22/paper/`: paper-facing bundle
- `results/catch22/xai/`: convenience entry points to interpretability outputs
- `results/catch22/archive/legacy/`: retained historical outputs

Within a single modeling configuration, artifacts live under:

```text
results/catch22/experiments/models/<model_type>/<outcome>/<branch>/<feature_set>/
```

Typical contents:

- `predictions/train_oof.csv`
- `predictions/test.csv`
- `artifacts/calibration.json`
- `artifacts/threshold.json`
- `artifacts/metadata.json`
- `artifacts/validation.json`
- model bundle and any family-specific artifacts

For nested CV runs, `predictions/test.csv` contains the pooled outer-fold
evaluation predictions. For holdout runs, it contains the held-out test cohort.

## Primary regeneration commands

### Fastest isolated validation

```bash
python -m run_catch22 smoke
```

This writes a small end-to-end run under `SMOKE_ROOT` without touching the
canonical experiment tree.

### Full Catch22 experiment grid

```bash
python -m run_catch22 experiments
```

Useful variants:

```bash
python -m run_catch22 experiments --prep force
python -m run_catch22 experiments --prep skip
python -m run_catch22 experiments --only-xgboost
python -m run_catch22 experiments --only-ebm --prep skip
```

`--prep auto` and `--prep skip` validate the Step 01/03/05 processed-artifact
sidecars before reuse.

### Descriptive reporting bundle

```bash
python -m run_catch22 descriptive
```

This regenerates:

- preoperative descriptives
- cohort flow figures
- missingness tables

from saved artifacts rather than rerunning model training.

## Metrics and report regeneration

If predictions already exist and you only need consolidated paper outputs:

```bash
python -m results_recreation.metrics_summary \
  --results-dir results/catch22/experiments \
  --delta-mode reference \
  --reference-feature-set preop_only

python -m reporting.make_report
```

The reporting path is artifact-based:

- `metrics_summary` validates stored predictions and paired validation metadata
- tables and figures are rebuilt from saved predictions, calibrations, and
  labels
- no model, calibrator, or threshold is refit at report time

## Paper-facing bundle

The publication-facing surface lives under `results/catch22/paper/`.

Primary files:

- `tables/metrics_summary.{csv,md,docx,pdf}`
- `tables/results_*_{main,delta}.csv` and companion formatted exports
- `figures/` for ROC, PR, calibration, cohort flow, SHAP, and manuscript
  figures
- `reports/report.{md,docx,pdf}`
- `metadata/cohort_flow_counts.json`
- `manifest.json`

The paper tree is designed for downstream manuscript writing and supplemental
assembly without requiring model retraining.

## Metrics summary contract

`results_recreation.metrics_summary` consolidates saved predictions across
configurations and computes:

- discrimination metrics
- thresholded operating-point metrics
- confidence intervals
- paired delta summaries against a reference feature set

Bootstrap uncertainty is computed from saved predictions using patient-grouped
resampling when `subjectid` is available.

## Descriptive artifact bundle

The descriptive branch produces the non-model outputs most likely to be reused
in the manuscript and supplement:

### Cohort flow

- `figures/cohort_flow.svg`
- `figures/cohort_flow.png`
- counts source: `metadata/cohort_flow_counts.json`

### Preoperative descriptives

- `tables/preop_descriptives.csv`
- `tables/preop_descriptives.md`
- `tables/preop_descriptives.docx`
- `tables/preop_descriptives.pdf`
- plus HTML/LaTeX companion exports

### Missingness

- `tables/missingness_table.csv`
- `tables/missingness_table.md`
- `tables/missingness_table.docx`
- `tables/missingness_table.pdf`
- plus HTML companion export

## Interpretability outputs

The primary tree also supports model-family-specific interpretability artifacts.

- XGBoost: SHAP figures rebuilt from saved model bundles and predictions
- EBM: optional native explanation exports, with convenience entry points under
  `results/catch22/xai/`

These artifacts are additive to the predictive evaluation outputs and should be
treated as supplementary or exploratory unless explicitly incorporated into the
paper workflow.

## Smoke outputs

Smoke runs write a miniature but structurally valid artifact tree under
`SMOKE_ROOT`. They are intended for pipeline verification rather than final
evidence generation.

Smoke runs still exercise:

- Steps 01-05
- Step 07 holdout validation
- metrics aggregation
- report generation

## Experimental and legacy outputs

- Aeon outputs are experimental and documented in
  `experimental/aeon/README.md` and `results/aeon/README.md`.
- Archive outputs are retained for reference only and are not part of the
  validated default paper path.
