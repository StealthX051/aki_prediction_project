# AGENTS.md

## Project purpose

This repository generates reproducible results for a clinical informatics
machine learning project on postoperative acute kidney injury (AKI) prediction
from VitalDB intraoperative waveforms plus preoperative clinical data.

- Primary production path: Catch24/Catch22 feature extraction with
  XGBoost/EBM training and paper-facing reporting.
- Experimental path: Aeon-based time-series models under
  `model_creation_aeon/`, `run_experiments_aeon.sh`, and `results/aeon/`.
  Do not treat Aeon as the default analysis unless the task explicitly calls
  for it.

## Source documents

Read the smallest relevant document first.

- `README.md`: current execution guide, results layout, and run commands.
- `METHODS.md`: authoritative scientific definitions and preprocessing rules.
- `RESULTS.md`: paper-facing outputs and current result summaries.
- `data_preparation/README.md`: ordered preprocessing steps.
- `reporting/DISPLAY_DICTIONARY.md`: labels and units for report outputs.
- `.agent/rules/` and `.agent/workflows/`: deeper task-specific guidance.
  Keep this file concise and open those docs only when the task needs them.

## Working style

- Follow KISS: choose the simplest implementation that satisfies the
  scientific and operational requirement.
- Prefer minimal, targeted changes over broad refactors.
- Reuse and extend existing functions/modules before introducing new
  abstractions.
- Prefer a small parameter, helper, or conditional in an existing module over
  adding new layers, frameworks, or configuration surfaces.
- When a change makes prior code paths, helpers, or flags obsolete, prefer
  deleting them instead of leaving dead or parallel code behind.
- Preserve current CLI contracts, filenames, column names, and result layout
  unless the task explicitly requires a change.
- Keep code professional and reliable: deterministic seeds where supported,
  clear stage-level logging, and comments only where the logic is non-obvious.
- Optimize for low churn and auditability: keep control flow explicit, avoid
  speculative generalization, and make it easy to trace a result back to a
  small set of functions and files.
- Prefer straightforward validation, clear errors, and local fault handling
  over complex orchestration when making code more robust.
- Do not keep compatibility code, fallback branches, or duplicated
  implementations unless the repository actually needs them for a current
  interface or study path.
- Prefer environment-variable overrides (`SMOKE_ROOT`, `DATA_DIR`,
  `PROCESSED_DIR`, `RESULTS_DIR`, `PAPER_DIR`) when experimenting so
  exploratory work does not pollute canonical outputs.
- Do not edit generated artifacts, logs, or large result trees just to keep the
  repository tidy.

## Environment

- Use the Conda environment in `environment.yml` as the default runtime for
  this repository.
- Before running pipeline commands, smoke tests, or repo scripts, activate it:
  `conda activate aki_prediction_project`.
- If dependencies change, sync the environment with
  `conda env update -f environment.yml --prune`.
- If a task needs the pinned test extras, install them into the activated Conda
  environment with `python -m pip install -r requirements-test.txt`.
- Do not introduce a second, parallel `venv` workflow in docs or automation
  unless the repository is explicitly redesigned around that.

## Repository map

- `data_preparation/`: production preprocessing pipeline (`step_01` through
  `step_05`); `inputs.py` is the source of truth for paths and core settings.
- `model_creation/`: Catch22 model HPO, training, evaluation, calibration, and
  artifact export.
- `results_recreation/`: standardized evaluation, bootstrap metrics, and
  artifact crawling.
- `reporting/`: cohort flow, descriptive tables, missingness, and report
  generation.
- `model_creation_aeon/` plus Aeon preparation/export steps: experimental
  time-series branch.
- `run_smoke_test.sh`, `run_experiments.sh`, and the thin wrapper
  `run_catch22_experiments.sh`: Catch22 workflow entry points.
- `run_aeon_smoke_test.sh` and `run_experiments_aeon.sh`: Aeon workflow entry
  points.

## Scientific guardrails

- This is a scientific repository, not just an application codebase. When
  implementing new methods or changing analytical behavior, optimize for
  scientific rigor, reproducibility, and defensible interpretation.
- Scientific rigor should still be implemented simply: if two approaches are
  equally correct, prefer the one that is easier to audit, review, and rerun.
- Treat `METHODS.md` as authoritative for cohort definitions, AKI labeling,
  waveform requirements, preprocessing, and evaluation.
- Keep `split_group` semantics intact. The train/test split is created once in
  `data_preparation/step_03_preop_prep.py` and reused downstream.
- Fit preprocessing statistics on training data only. Never use held-out test
  data for feature engineering, imputation rules, HPO, calibration fitting, or
  threshold selection.
- If you add a new method, ablation, or exploratory analysis, keep it clearly
  labeled as experimental unless the task explicitly updates the primary study
  design.
- Keep experimental variants clearly labeled and avoid overwriting
  primary-analysis artifacts unless explicitly asked.

## Verification

- Prefer the smallest check that exercises the changed behavior.
- When simplifying or deleting code, run a targeted check that confirms the
  new path works and that previously supported behavior in the touched area has
  not regressed.
- For utilities, postprocessing, or reporting changes, run targeted `pytest`
  tests from `tests/`.
- For production pipeline changes, prefer `./run_smoke_test.sh` before
  considering larger runs.
- For reporting-only changes, rerun `python -m results_recreation.metrics_summary`
  and/or `python -m reporting.make_report` when relevant.
- For Aeon changes, use `./run_aeon_smoke_test.sh`.
- Do not run the full experiment grid (`./run_experiments.sh`,
  `./run_experiments_aeon.sh`) unless the task truly requires it.

## Documentation

- When updating docs, prefer concise, high-signal edits to the smallest
  relevant file.
- Do not layer new text on top of stale text. Remove or replace outdated,
  duplicate, bloated, or misleading content.
- Verify commands, paths, filenames, and environment instructions against the
  actual repository before finalizing doc changes.
- Keep root docs brief and operational; put narrow workflow detail in the
  linked `.agent/` docs instead of auto-loading everything into context.

## Task-specific docs

- Pipeline execution: `.agent/workflows/run-catch22-pipeline.md`
- Leakage review: `.agent/workflows/check-catch22-leakage.md`
- Change summaries: `.agent/workflows/summarize-catch22-change.md`
- Coding conventions: `.agent/rules/catch22-coding-guide.md`
- Scientific integrity: `.agent/rules/catch22-scientific-integrity.md`

## Subagents

- Use subagents when the task benefits from parallel exploration, bounded
  implementation work, or independent verification.
- Zero, one, or many subagents are all acceptable; choose the smallest number
  that materially helps.
- Prefer `explorer` for read-heavy repo mapping and evidence gathering.
- Keep subagent prompts narrow and outcome-oriented to limit context pollution
  and duplicated work.
