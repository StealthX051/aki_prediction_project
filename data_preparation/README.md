# Data preparation pipeline

This directory contains the production Catch22 preprocessing stages. Run the
steps in order; each stage depends on artifacts from the previous one.

Key settings live in `data_preparation/inputs.py`. Step 01, Step 03, and Step
05 also write `.metadata.json` sidecars that are validated before artifact
reuse.

## Stages

1. Cohort construction

```bash
python -m data_preparation.step_01_cohort_construction
```

Outputs the shared cohort CSV plus cohort-flow metadata.

2. Catch22 feature extraction

```bash
python -m data_preparation.step_02_catch_22
```

Builds non-windowed and windowed waveform feature tables.

3. Preoperative prep and split

```bash
python -m data_preparation.step_03_preop_prep
```

Creates the processed preoperative table plus one outcome-specific grouped
holdout split per supported outcome.

4. Intraoperative wide-table prep

```bash
python -m data_preparation.step_04_intraop_prep
```

Pivots waveform features to the wide modeling format.

5. Data merge

```bash
python -m data_preparation.step_05_data_merge
```

Creates the final non-windowed and windowed master modeling tables.

## Preferred validation path

For an end-to-end check, prefer the launcher:

```bash
python -m run_catch22 smoke
```

That command wires Steps 01-05 together, trims the cohort for a fast run, and
writes outputs under an isolated smoke root.
