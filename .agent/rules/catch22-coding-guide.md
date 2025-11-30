---
trigger: always_on
---

# AKI Prediction Project – Coding Guide

These rules apply when working in the `aki_prediction_project` repository.

## 1. Project structure and entry points

- Treat the `data_preparation/` directory as the canonical data pipeline.
- Keep the step scripts (`step_01_...` through `step_05_...`) modular and composable; avoid turning them into monolithic scripts.
- Keep `inputs.py` as the single source of truth for input/output paths and configuration; prefer extending it rather than hardcoding new paths.
- When adding new functionality, prefer new, clearly named modules over large modifications to existing ones.

## 2. Environment and execution

- Assume the Conda environment defined in `environment.yml` is the default runtime.
- When writing “how to run” instructions or automation, use that environment name and file rather than ad-hoc installations.
- Avoid adding dependencies that are not reflected in `environment.yml`; if necessary, update that file and mention the change in the README.

## 3. Coding style and logging

- Follow the global Python style rules (PEP 8, type hints, docstrings).
- For data pipeline steps and model training, use the standard Python `logging` module with informative messages at key steps:
  - dataset loading, train/test split application, feature extraction, model training, evaluation.
- Keep logging concise; focus on shapes, counts, and key configuration, not every loop iteration.

## 4. Inputs, outputs, and CLI contracts

- Treat the current CLI and function signatures of `data_preparation` scripts and training scripts as an API.
- When modifying scripts, preserve:
  - input file locations and expected formats,
  - output file locations and key column names,
  - `split_group` semantics for train/test indicators.
- If an interface must change, document:
  - what changed,
  - how to migrate,
  - which downstream steps are affected.

## 5. Tests, smoke checks, and reproducibility

- Whenever feasible, add or update smoke checks that:
  - verify non-empty outputs,
  - check key columns (`caseid`, outcome labels, `split_group`) exist and have reasonable distributions.
- Use fixed seeds where the underlying libraries support them (e.g., model training, data splitting) to maintain reproducibility of results.
- Do not delete or disable existing tests or checks without a clear stated reason.
