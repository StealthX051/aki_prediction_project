# Results artifacts

This repository saves publication-ready outputs under `results/` so writers and
analysts can reuse them without rerunning the full pipeline. All reporting
scripts respect `RESULTS_DIR` environment overrides and share the display
dictionary at `metadata/display_dictionary.json` to keep labels synchronized.
Artifacts from the Catch22 + XGBoost/EBM pipeline are considered the primary
source of truth; Aeon artifacts are optional and experimental.

## Core model evaluation outputs
The consolidated metrics table and plots are generated from previously trained
models without retraining:

1. Build the unified metrics summary from saved predictions and calibration
   metadata:
   ```bash
   python results_recreation/metrics_summary.py
   ```

2. Render styled tables, ROC/PR/calibration plots, and Word/PDF reports:
   ```bash
   python results_recreation/results_analysis.py
   ```

Result locations:
- `results/tables/metrics_summary.csv`: Consolidated AUROC/AUPRC and
  thresholded metrics with confidence intervals.
- `results/report.docx` and `results/report.pdf`: Styled reports with all
  tables.
- `results/figures/`: ROC, PR, and calibration curves for each configuration.

## Cohort flow diagram
`reporting/cohort_flow.py` transforms the saved cohort counts from
`step_01_cohort_construction.py` into a vertical flow diagram. Provide the
counts JSON (default: `results/metadata/cohort_flow_counts.json`) and an optional
custom display dictionary path. Outputs: `results/figures/cohort_flow.svg` and
`cohort_flow.png`.

```bash
python -m reporting.cohort_flow \
  --counts-file results/metadata/cohort_flow_counts.json \
  --display-dictionary metadata/display_dictionary.json
```

## Preoperative descriptive statistics
`reporting/preop_descriptives.py` summarizes baseline demographics and clinical
variables using the raw cohort CSV (before one-hot encoding). Continuous
features are tested for normality (Shapiro–Wilk, subsampled to 5,000) to decide
between mean ± SD or median (IQR) reporting; categorical features are presented
as counts with percentages. Outputs:
- `results/tables/preop_descriptives.html`
- `results/tables/preop_descriptives.tex`
- `results/tables/preop_descriptives.docx`

```bash
python -m reporting.preop_descriptives \
  --dataset data/processed/aki_pleth_ecg_co2_awp.csv \
  --display-dictionary metadata/display_dictionary.json
```

## Missingness summary for model features
`reporting/missingness_table.py` computes per-feature missing counts and
percentages for the merged modeling dataset
(`data/processed/aki_features_master_wide.csv` by default), excluding
identifiers and outcomes. Outputs are written to:
- `results/tables/missingness_table.csv`
- `results/tables/missingness_table.html`

```bash
python -m reporting.missingness_table \
  --dataset data/processed/aki_features_master_wide.csv \
  --display-dictionary metadata/display_dictionary.json
```

## Smoke test artifacts
For quick validation, run the lightweight smoke test without touching production
artifacts:
```bash
./run_smoke_test.sh
```
This generates a miniature set of `data/processed/` and `results/` outputs under
the directory specified by `SMOKE_ROOT` (default: `smoke_test_outputs/`).
