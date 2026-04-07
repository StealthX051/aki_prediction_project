---
trigger: always_on
---

# AKI Prediction Project coding guide

## 1. Entry points

- Treat `run_catch22.py` as the canonical launcher.
- Treat `data_preparation/step_01` through `step_05` as the canonical staged
  preprocessing path.
- Keep `inputs.py` as the source of truth for preprocessing settings and paths.

## 2. Change strategy

- Prefer minimal, auditable edits.
- Reuse existing modules and helpers before adding abstractions.
- Delete stale code and stale docs when a replacement is clearly in place.
- Avoid touching generated outputs or result trees unless the task is about
  those artifacts.

## 3. Runtime assumptions

- Assume the Conda environment in `environment.yml`.
- Prefer the existing launcher and module commands over ad hoc entrypoints.
- Reflect dependency changes in `environment.yml`.

## 4. CLI and artifact contracts

- Preserve CLI behavior unless the task explicitly requires a change.
- Preserve artifact names, key columns, and result layout unless the task
  explicitly requires a change.
- Preserve Step 03 outcome-specific split metadata and its downstream reuse.

## 5. Verification

- Prefer the smallest check that proves the surviving path works.
- For pipeline changes, prefer `python -m run_catch22 smoke`.
- For reporting changes, prefer targeted pytest plus
  `results_recreation.metrics_summary` or `reporting.make_report` as needed.

## 6. Documentation

- Keep `README.md` operational.
- Keep `METHODS.md` and `RESULTS.md` suitable for manuscript and supplement
  distillation.
- Remove duplicate or stale text instead of layering more prose on top.

## 7. Subagents

- Use subagents when parallel exploration or independent verification will
  materially help.
- Keep subagent prompts narrow and outcome-oriented.
