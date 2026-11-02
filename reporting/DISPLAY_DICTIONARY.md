# Display dictionary

`metadata/display_dictionary.json` is the single source of truth for
publication-facing labels used in tables, figures, and interpretability
artifacts.

The JSON is loaded by `reporting.display_dictionary.DisplayDictionary`.

## Main sections

The dictionary currently covers:

- `outcomes`
- `branches`
- `model_types`
- `feature_sets`
- `features`
- `waveforms`
- `catch22_statistics`
- `catch22_aggregates`

Each entry typically provides:

- `label`
- optional `short_label`
- optional `unit`
- optional `description`

## Resolution behavior

Feature lookup follows this order:

1. explicit `features` entry
2. recognized one-hot categorical prefix
3. waveform plus Catch22 template parsing
4. raw feature name fallback

This allows the reporting stack to render human-readable labels without
hardcoding them into every report module.

## Where it is used

- `reporting.make_report`
- `reporting.cohort_flow`
- `reporting.preop_descriptives`
- `reporting.missingness_table`
- EBM explanation exports from `model_creation.step_07_train_evaluate`
- saved outer-fold explanations from `reporting.fold_explanations`

## Units and interpretation

Display labels do not convert model values. Source PT is percent, CRP is mg/dL,
aPTT is seconds, and airway pressure is hPa. ECG labels describe recorded
findings; Table 1 groups them as normal sinus rhythm/abnormal while preserving
missing/unknown categories separately.

Catch24 statistics do not automatically inherit the waveform's physical unit: a
dimensionless descriptor or sample-based timescale must not be labeled mV or
hPa. Moment features such as mean and standard deviation can have physical
units, but those require explicit registration rather than a blanket unit for
every waveform descriptor. Saved fold explanations retain raw names alongside these display
labels. LR coefficient scaling and EBM/SHAP output scales are described by the
explanation metadata rather than inferred from a feature's clinical unit.

## Maintenance guidance

- Add new feature sets, outcomes, or derived features here before regenerating
  manuscript-facing outputs.
- Keep labels concise and publication-ready.
- Review dictionary changes alongside the code and artifacts that consume them
  so labels do not drift from the underlying analysis.
