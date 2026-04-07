---
trigger: always_on
---

# AKI Prediction Project scientific integrity rules

## 1. Treat this as scientific software

- Favor conservative, reviewable method changes over clever rewrites.
- Preserve the validated Catch22 paper path unless the task explicitly changes
  study design.
- Keep experimental branches clearly labeled.

## 2. Treat `METHODS.md` as authoritative

- Do not change cohort definitions, outcome definitions, waveform
  requirements, or evaluation rules without explicitly updating the study
  design.
- When a method changes, update `METHODS.md` and any affected paper-facing
  documentation.

## 3. Cohort and outcome semantics

- Preserve the shared outer cohort contract and its outcome-specific
  eligibility layers.
- Preserve KDIGO-based AKI labeling.
- Preserve patient grouping by `subjectid` across splitting and evaluation.

## 4. Leakage prevention

- The grouped split is created once in Step 03 and reused downstream.
- Learned preprocessing, HPO, calibration, and threshold selection must be fit
  on training data only.
- Reporting must consume saved artifacts and must not silently refit anything.

## 5. Feature and model changes

- Keep Catch24 feature semantics consistent unless the task explicitly changes
  the study.
- Maintain both non-windowed and windowed branches unless explicitly asked
  otherwise.
- Keep primary-model changes clearly distinguished from exploratory analyses.

## 6. Evaluation

- The primary internal validation path is patient-grouped nested CV.
- Legacy holdout evaluation may remain available but should not silently become
  the default.
- Do not present partial or mixed artifact sets as final evidence.
