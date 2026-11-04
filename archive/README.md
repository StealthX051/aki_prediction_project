# Archive

This directory contains historical notebooks, scripts, and environment
snapshots retained for reference only.

Use the root docs and the production modules under `data_preparation/`,
`model_creation/`, `results_recreation/`, and `reporting/` for current work.

## Contents

- `archive/notebooks/`: exploratory preparation, training and results notebooks, including superseded top-level notebooks
- `archive/scripts/`: legacy utilities, the hard-coded duration inspection script, superseded XGBoost HPO and older training code
- `archive/environment_linux.yml`: archived environment snapshot
- [Historical manuscript log](docs/manuscript-reorientation-2026-11-04.md): dated launch, audit and manuscript decisions

Generated historical split CSVs and training logs are outside the code tree.
Local reference copies are under the ignored `outputs/legacy-reference/`;
previous committed copies remain available in Git history. Current research
inputs and fitted artifacts belong on the research volume.

## Guidance

- Do not treat archive code as the default reproducible workflow.
- Do not update archive contents unless the task is explicitly about legacy
  reproduction or inspection.
- Expect methods and assumptions here to differ from the current production
  pipeline.
