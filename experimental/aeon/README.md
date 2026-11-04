# Experimental Aeon branch

Aeon is outside the validated Catch24 manuscript pipeline. Its retained module
interfaces are listed below; there is no top-level Aeon launcher.

Training requires an Aeon-capable candidate environment. `environment.yml` and
`requirements-cpu.txt` do not include Aeon or its optional feature-extraction
dependencies. Keep these separate from the validated CPU environment. Configure
existing cohort/preoperative inputs before using the preparation commands.

## Direct commands

Select isolated output paths under the research volume:

```bash
export AEON_OUT_DIR=/media/volume/catch22/data/aki_prediction_project/experimental/aeon/data
export AKI_AEON_RESULTS=/media/volume/catch22/data/aki_prediction_project/experimental/aeon/experiments

# Export fixed-length waveform tensors
python -m data_preparation.step_02_aeon_export --limit 50

# Prepare preoperative features for the retained Aeon cases
python -m data_preparation.step_04_aeon_prep

# Train one experimental configuration
python -m model_creation_aeon.step_06_aeon_train \
  --model minirocket \
  --outcome any_aki \
  --channels all \
  --include_preop \
  --results_dir "$AKI_AEON_RESULTS"

# Summarize saved unscoped predictions
python -m results_recreation.metrics_summary --legacy \
  --results-dir "$AKI_AEON_RESULTS" --delta-mode none
```

Aeon outputs use the legacy artifact schema and do not satisfy the strict Catch24
manifest contract. The trainer does not produce a `preop_only` reference, so the
example requests absolute metrics. Current paper-report generation also expects
scoped result paths; an Aeon report requires an adapted layout and explicit
legacy mode.

These interfaces were checked against the source argument parsers. Aeon training
and report generation have not been validated in the supported CPU environment.
Its cohort and split handling remain experimental.
