# Experimental Aeon branch

Aeon is exploratory. It is not part of the validated Catch22 paper pipeline,
and there is no maintained top-level launcher for it.

Use the Conda environment first:

```bash
conda activate aki_prediction_project
```

## Direct commands

```bash
# Export fixed-length waveform tensors
python -m data_preparation.step_02_aeon_export --limit 50

# Prepare aligned preoperative features for the retained Aeon cases
python -m data_preparation.step_04_aeon_prep

# Train one Aeon configuration
python -m model_creation_aeon.step_06_aeon_train \
  --model minirocket \
  --outcome any_aki \
  --channels all \
  --include_preop \
  --results_dir results/aeon/experiments

# Aggregate metrics from saved predictions
python -m results_recreation.metrics_summary \
  --results-dir results/aeon/experiments \
  --delta-mode reference \
  --reference-feature-set preop_only

# Rebuild reports from the aggregated metrics
RESULTS_DIR=results/aeon/experiments PAPER_DIR=results/aeon/paper \
python -m reporting.make_report
```

## Caveats

- Aeon is experimental and not kept in strict feature-parity with the primary
  Catch22 cohort and split contract.
- The removed root shell runners are not supported.
- Use `AEON_OUT_DIR`, `RESULTS_DIR`, and `PAPER_DIR` overrides when isolating
  exploratory outputs.
