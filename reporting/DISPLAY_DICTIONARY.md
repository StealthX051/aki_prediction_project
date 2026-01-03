# Display dictionary

This repository now ships a **single source of truth** for translating internal
identifiers (feature names, outcome keys, model types, branches, feature sets)
into publication-ready labels. The dictionary lives at
`metadata/display_dictionary.json` and is loaded by
`reporting.display_dictionary.DisplayDictionary`.

## Contents and schema
The JSON file is intentionally human-editable. Each top-level section is a map
from an internal key to an object with at least a `label` and optional fields:

- `short_label`: Abbreviated label suitable for tight spaces (axes, legends).
- `unit`: Unit string to append when helpful.
- `description`: Longer human-readable description (optional).

Sections included today:

- `outcomes`: Outcome identifiers such as `any_aki` or `y_inhosp_mortality`.
- `branches`: Dataset branches (e.g., `windowed`, `non_windowed`).
- `model_types`: Model family names (`xgboost`, `ebm`, `autogluon`).
- `feature_sets`: Experiment feature set keys from the run scripts.
- `features`: Direct feature-name overrides (e.g., `preop_egfr_ckdepi_2021`).
- `waveforms`: Waveform channel names. Slash-based keys automatically map to
  their flattened underscore equivalents.
- `catch22_statistics`: Glossary for the 24 Catch22/Catch24 statistics.
- `catch22_aggregates`: Human-readable names for windowed aggregates (`mean`,
  `std`, `min`, `max`).

The schema is deliberately permissive so new sections can be added later if
needed by downstream tools.

## Python utility: `DisplayDictionary`
`reporting/display_dictionary.py` provides a lightweight helper that loads the
JSON file and exposes lookups with sensible fallbacks:

```python
from reporting.display_dictionary import load_display_dictionary

names = load_display_dictionary()

names.outcome_label("any_aki")          # -> "Any AKI"
names.branch_label("windowed")          # -> "Segmented (windowed)"

# Feature lookups prefer explicit entries, then waveform + Catch22 templates
names.feature_label("preop_egfr_ckdepi_2021", include_unit=True)
# -> "Estimated GFR (CKD-EPI 2021) (mL/min/1.73 m²)"

names.feature_label("SNUADC_PLETH_DN_HistogramMode_5_mean")
# -> "Pleth — Histogram mode (5 bins) (Mean across windows)"

# Feature-set labels merge dictionary entries with optional defaults
labels = names.feature_set_labels({"custom_set": "Custom feature set"})
```

Resolution order for feature names:
1. Exact match in the `features` section.
2. Catch22-style waveform names that follow `<waveform>_<stat>[_aggregate]`.
3. Fallback to the raw feature name when no mapping exists.

## How to extend or modify
1. Edit `metadata/display_dictionary.json` and add new keys under the relevant
   section. Keep labels concise and avoid units unless necessary.
2. If you introduce a new waveform channel or Catch22 statistic, add a new
   entry to `waveforms` or `catch22_statistics` so windowed/full features render
   cleanly.
3. For new experiment knobs (e.g., a new `feature_set` or `model_type`), add the
   human-readable label so tables and plots remain synchronized.

## Integration pointers
- **Reporting**: `results_recreation/results_analysis.py` now consumes the
  dictionary for feature-set labels and can be extended to render outcome/model
  labels the same way.
- **Interpretability**: When generating SHAP/EBM plots, prefer
  `DisplayDictionary.feature_label()` to annotate axes and tooltips.
- **Data prep**: If additional derived preoperative features are created,
  register them under `features` to keep cohort tables tidy.

Additions should be reviewed during PRs to avoid drift between the dictionary
and the data/model artifacts that reference it.
