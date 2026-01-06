# Results artifacts

This repository saves publication-ready outputs under `results/` so writers and
analysts can reuse them without rerunning the full pipeline. All reporting
scripts respect `RESULTS_DIR` environment overrides and share the display
dictionary at `metadata/display_dictionary.json` to keep labels synchronized.
Artifacts from the Catch22 + XGBoost/EBM pipeline are considered the primary
source of truth; Aeon artifacts are optional and experimental.

## Running experiments (full or staged)
- Use `run_experiments.sh` for the Catch22/XGBoost/EBM grid. It supports:
  - `--prep auto` (default): reuse `data/processed/aki_features_master_wide*.csv` if present; rebuild only if missing.
  - `--prep force`: re-run Steps 01–05 (cohort, Catch22, preop, intraop, merge) and regenerate both non-windowed and windowed features.
  - `--prep skip`: assume processed features already exist; skip all prep.
- Model-family control:
  - XGBoost only: `./run_experiments.sh --only-xgboost --prep auto`
  - EBM only (reuse XGBoost-prepared data): `./run_experiments.sh --only-ebm --prep skip`
  - Both families: default or `--model-types xgboost,ebm`
- Environment roots (`DATA_DIR`, `PROCESSED_DIR`, `RAW_DIR`, `RESULTS_DIR`) are exported so both families share the same files; no wasted preprocessing.
- Metrics/report generation is skipped if no prediction files are present, so partial runs (only one family) are handled gracefully. All CLI flags also pass through `run_catch22_experiments.sh`.
- `results_recreation/metrics_summary.py` validates each `predictions/test.csv`; invalid files are skipped with a warning so a single bad artifact does not halt consolidation. If no valid predictions remain, it fails fast with a clear message.
- All run scripts (`run_experiments*.sh`, smoke tests) now invoke `metrics_summary` with stratified, paired bootstrapping, Δ vs `preop_only`, and full-core parallelism, then call `reporting/make_report` to emit the report bundle (main + Δ tables).
- EBM explainability is auto-enabled in `run_experiments.sh` (the script injects `--export_ebm_explanations` for EBM runs). Per-term exports run in a bounded thread pool with per-term timeouts and retries; logging is unbuffered (`PYTHONUNBUFFERED=1`) to surface progress and avoid silent hangs. If you need to regenerate XAI for existing EBM models only, reuse processed data and models:  
  ```bash
  PYTHON_BIN=/home/exouser/.conda/envs/aki_prediction_project/bin/python
  OUTCOMES=(any_aki icu_admission); BRANCHES=(non_windowed windowed)
  FEATURE_SETS=(preop_only pleth_only ecg_only co2_only awp_only all_waveforms preop_and_all_waveforms preop_and_pleth preop_and_ecg preop_and_co2 preop_and_awp preop_and_all_minus_pleth preop_and_all_minus_ecg preop_and_all_minus_co2 preop_and_all_minus_awp)
  for o in "${OUTCOMES[@]}"; do for b in "${BRANCHES[@]}"; do for fs in "${FEATURE_SETS[@]}"; do
    $PYTHON_BIN -m model_creation.step_07_train_evaluate --outcome "$o" --branch "$b" --feature_set "$fs" --model_type ebm --export_ebm_explanations
  done; done; done
  ```
  Outputs land in `results/models/ebm/<outcome>/<branch>/<feature_set>/artifacts/ebm_xai/` with `index.html` linking global, local, and per-term plots.

## Core model evaluation outputs
The consolidated metrics table and plots are generated from previously trained
models without retraining:

1. Build the unified metrics summary from saved predictions and calibration
   metadata (stratified, paired bootstrap with Δ vs preop reference using all
   cores):
   ```bash
   python results_recreation/metrics_summary.py \
     --delta-mode reference \
     --reference-feature-set preop_only \
     --parallel-backend processes \
     --n-jobs -1
   ```

2. Render styled tables, ROC/PR/calibration plots, and Word/PDF reports (with
   separate delta tables and heatmap shading where the Δ CI excludes 0):
   ```bash
   python reporting/make_report.py
   ```

Result locations:
- `results/tables/metrics_summary.csv`: Consolidated AUROC/AUPRC and
  thresholded metrics with confidence intervals, plus Δ columns.
- `results/report.docx` and `results/report.pdf`: Styled reports with main
  tables and separate Δ tables (Δ vs reference).
- `results/figures/`: ROC, PR, and calibration curves for each configuration.

## Descriptive figures bundle
Run all descriptive artifacts (preop demographics table, cohort flow diagram,
missingness table) with one command. The script reuses your training run's
paths (`DATA_DIR`, `PROCESSED_DIR`, `RESULTS_DIR`) if they are already set.

```bash
./run_descriptive_figures.sh
```

## Cohort flow diagram
`reporting/cohort_flow.py` transforms the saved cohort counts from
`step_01_cohort_construction.py` into a vertical flow diagram. Provide the
counts JSON (default: `results/metadata/cohort_flow_counts.json`) and an optional
custom display dictionary path. The renderer skips no-op/increasing steps, shows
per-step removals, applies friendly labels, and draws the AKI False/True split
when `label_split` is present in the JSON. Outputs: `results/figures/cohort_flow.svg`
and `cohort_flow.png`.

```bash
python -m reporting.cohort_flow \
  --counts-file results/metadata/cohort_flow_counts.json \
  --display-dictionary metadata/display_dictionary.json
```

The cohort excludes ASA V–VI cases prior to waveform availability checks and
other custom filters; the counts JSON reflects that step.

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
  --processed-dataset data/processed/aki_preop_processed.csv \
  --display-dictionary metadata/display_dictionary.json
```
Continuous summaries pull from the winsorized preop dataset (`aki_preop_processed.csv`) when present; binary categoricals
render 0/1 as False/True, and one-hot categoricals use human-readable labels from the display dictionary. Regenerate
`aki_preop_processed.csv` via
`python -m data_preparation.step_03_preop_prep` if you rebuild the cohort.

## Missingness summary for model features
`reporting/missingness_table.py` computes per-feature missing counts and
percentages for the merged modeling dataset
(`data/processed/aki_features_master_wide.csv` by default), excluding
identifiers and outcomes. Headers are human-friendly and one-hot columns resolve
to display labels. Outputs are written to:
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
