# Data preparation pipeline

This directory contains the Catch22-based preprocessing pipeline used by the
production runs (AKI primary, ICU admission secondary). The steps must be run in
order because each stage depends on artifacts written by the previous one. Key
settings (paths, sample rate, windowing, required waveforms) live in
`data_preparation/inputs.py`. Step 01, Step 03, and Step 05 also write
`.metadata.json` sidecars; downstream loaders validate them and fail fast on
stale processed artifacts.

## Steps
1. **Cohort construction** — filter cases, derive outcomes, and emit
   `results/catch22/paper/metadata/cohort_flow_counts.json` for reporting. ASA V–VI cases are
   excluded before waveform checks and custom filters:
   ```bash
   python -m data_preparation.step_01_cohort_construction
   ```
2. **Catch22 feature extraction** — resample waveforms to 10 Hz and compute
   Catch22 features for full and windowed branches:
   ```bash
   python -m data_preparation.step_02_catch_22
   ```
3. **Preoperative prep + split** — clean clinical variables and create one
   80/20 patient-grouped holdout split per supported outcome. Sparse outcomes
   are marked unsupported in the metadata sidecar instead of blocking the whole
   artifact, and downstream holdout runs must use the persisted
   outcome-specific split from this step:
   ```bash
   python -m data_preparation.step_03_preop_prep [--impute-missing]
   ```
4. **Intraoperative prep** — pivot/flatten waveform features and leave NaNs in
   place by default:
   ```bash
   python -m data_preparation.step_04_intraop_prep
   ```
5. **Data merge** — combine preop and intraop datasets into master modeling
   tables (one per branch/feature set), keeping split assignments intact:
   ```bash
   python -m data_preparation.step_05_data_merge [--impute-missing]
   ```

For smoke testing, prefer the root-level `run_smoke_test.sh`, which wires these
steps together, trims the cohort with `data_preparation.smoke_trim_cohort`,
uses holdout validation on the small sampled cohort, and writes outputs under
`smoke_test_outputs/` by default.
