# AGENTS.md

## Project purpose

This repository generates reproducible results for postoperative acute kidney
injury prediction from VitalDB intraoperative waveforms plus preoperative
clinical data.

- Primary path: Catch24/Catch22 feature extraction, XGBoost/EBM training, and
  paper-facing reporting
- Experimental path: Aeon models under `model_creation_aeon/`

Treat the Catch22 path as the default unless the task explicitly calls for
Aeon.

## Read these first

Open the smallest relevant document first.

- `README.md`: current run commands and operational entrypoint
- `METHODS.md`: authoritative scientific definitions
- `RESULTS.md`: artifact layout and paper outputs
- `data_preparation/README.md`: ordered preprocessing stages
- `experimental/aeon/README.md`: Aeon-only commands
- `.agent/rules/` and `.agent/workflows/`: narrow internal guidance

## Working style

- Prefer minimal, local edits over broad refactors.
- Reuse existing modules and helpers before adding new abstractions.
- Delete stale code and stale docs when a replacement is clearly in place.
- Preserve current CLI contracts, artifact names, and result layout unless the
  task explicitly requires a change.
- Prefer environment-variable overrides when experimenting so exploratory work
  does not pollute canonical outputs.
- Do not edit generated artifacts or large result trees unless the task is
  specifically about those artifacts.

## Environment and execution

- Use the Conda environment in `environment.yml`.
- Activate `aki_prediction_project` before running project commands.
- Sync dependency changes with `conda env update -f environment.yml --prune`.
- Install pinned test extras with
  `python -m pip install -r requirements-test.txt` when needed.
- Do not introduce a parallel `venv` workflow unless explicitly asked.

## Repository map

- `data_preparation/`: Steps 01-05 for the production preprocessing pipeline
- `model_creation/`: HPO, training, evaluation, calibration, thresholding
- `results_recreation/`: metrics aggregation and bootstrap evaluation
- `reporting/`: figures, tables, cohort flow, descriptives, missingness
- `run_catch22.py`: primary launcher
- `experimental/aeon/README.md`: experimental Aeon execution notes
- `archive/`: historical notebooks and scripts, not the default workflow

## Scientific guardrails

- Treat `METHODS.md` as the scientific source of truth.
- Preserve the shared outer cohort plus outcome-specific eligibility contract.
- Preserve the outcome-specific split columns created in Step 03:
  `split_group_any_aki`, `split_group_severe_aki`,
  `split_group_icu_admission`, and `split_group_mortality`.
- Fit learned preprocessing on training data only.
- Never use held-out test data for feature engineering, HPO, calibration, or
  threshold selection.
- Keep experimental variants clearly labeled and avoid overwriting primary
  artifacts unless explicitly asked.

## Verification

- Prefer the smallest check that exercises the touched behavior.
- For pipeline changes, prefer `python -m run_catch22 smoke`.
- For reporting changes, rerun `python -m results_recreation.metrics_summary`
  and/or `python -m reporting.make_report` when relevant.
- For utility changes, prefer targeted `pytest` tests.
- Do not run the full experiment grid unless the task truly requires it.

## Documentation

- Keep root docs brief and operational.
- Remove stale, duplicate, or bloated text instead of layering more prose on
  top.
- Verify commands, paths, and filenames against the current repository before
  finalizing doc edits.

## Task-specific docs

- `.agent/workflows/run-catch22-pipeline.md`
- `.agent/workflows/check-catch22-leakage.md`
- `.agent/workflows/summarize-catch22-change.md`
- `.agent/rules/catch22-coding-guide.md`
- `.agent/rules/catch22-scientific-integrity.md`

## Subagents

- Use subagents when parallel exploration or independent verification will
  materially help.
- Prefer `explorer` for read-heavy repo mapping.
- Keep subagent prompts narrow and outcome-oriented.
