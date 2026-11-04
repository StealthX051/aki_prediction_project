# Display dictionary

[`metadata/display_dictionary.json`](../metadata/display_dictionary.json) supplies
default labels to the shared reporting modules through
`reporting.display_dictionary.DisplayDictionary`. Curated manuscript templates
can supply display-specific labels; their CSVs retain source feature IDs.

## Entries

Sections cover outcomes, branches, model types, feature sets, features, waveforms,
Catch22 statistics and window aggregates. An entry is a label string or an object
with `label` and optional `short_label`, `unit` and `description` fields.

## Resolution behavior

Feature lookup follows this order:

1. explicit `features` entry
2. recognized one-hot categorical prefix
3. waveform plus Catch22 template parsing
4. raw feature name fallback

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
every waveform descriptor. Saved fold explanations retain raw names alongside
display labels. Explanation metadata defines LR coefficient scaling and EBM/SHAP
output scales; a feature's clinical unit does not determine those scales.

## Updates

Register new feature labels and units before regenerating reports. Check the
resulting tables and figures against the source IDs and values.
