---
trigger: always_on
---

# AKI Prediction Project – Scientific Integrity Rules

These rules ensure code changes remain aligned with the study’s scientific methods.

## 1. Treat METHODS.md as the source of truth

- Consider the definitions and preprocessing described in `METHODS.md` to be authoritative for the study.
- Do not change key clinical definitions (e.g., AKI definition, inclusion/exclusion criteria, waveform list) unless explicitly asked to update the study design.
- When adding new analyses or variants, clearly distinguish them from the primary analysis.

## 2. Cohort selection and outcome definition

- Preserve the cohort inclusion and exclusion logic:
  - required preoperative creatinine and operation end time,
  - required waveform availability,
  - exclusions for severe baseline renal dysfunction and inadequate signal quality.
- Maintain the KDIGO-based AKI outcome definition and binary labeling convention.
- If you introduce alternative outcomes (e.g., staging, time-to-event), implement them as separate variables and document them.

## 3. Data leakage prevention

- Respect the stratified train/test split:
  - perform the split once and reuse it downstream,
  - do not recalculate the split inside later steps.
- When computing statistics (outlier thresholds, imputations, encodings), use **only the training subset** and apply the resulting rules to both train and test.
- Never use information from the test set to:
  - tune hyperparameters,
  - select features,
  - derive thresholds, encodings, or normalization parameters.

## 4. Feature engineering and waveform processing

- Keep the Catch24 feature definition consistent (Catch22 features plus mean and standard deviation per channel).
- Maintain both **full-case** and **windowed** feature pipelines; do not break one while modifying the other.
- When adding new features or channels:
  - follow the same resampling and filtering conventions,
  - document them clearly in code and comments so they can be reflected in METHODS later if needed.

## 5. Modeling and evaluation

- Preserve the use of gradient boosted tree models (e.g., XGBoost) for the primary analysis unless explicitly requested otherwise.
- When changing modeling choices (model type, hyperparameters, evaluation metrics):
  - clearly label experimental scripts as such (e.g., `EXPERIMENTAL_*.py`),
  - avoid overwriting artifacts from the main analysis without explicit instruction.
- Always report performance separately for the held-out test set; do not present cross-validation results alone as final performance.

## 6. Documentation of changes

- For any change that affects the scientific interpretation (cohort, preprocessing, features, modeling, metrics):
  - add or update comments in the relevant scripts,
  - add a short note to the project’s README outlining the change and its intent.
- Avoid copying large sections of METHODS or README into code comments; link to the files or summarize only what is necessary.
