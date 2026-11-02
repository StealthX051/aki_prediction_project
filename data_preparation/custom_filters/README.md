# Historical cohort helpers

These modules are retained for historical reference. Current `inputs.py` and
Step 01 do not import them. They are not the production waveform-filtering path;
signal filters and resampling are implemented in
[`waveform_processing.py`](../waveform_processing.py) and described in
[METHODS.md](../../METHODS.md#waveform-processing).

In particular, the historical `filter_postop_cr` uses a seven-day lab window,
and `ensure_sample_independence` selects one operation per patient. Neither
defines the current analysis. The production pipeline retains repeat operations,
groups them by patient during validation, and applies outcome-specific
eligibility with the documented 72-hour AKI ascertainment horizon.

Active cohort/outcome handling is in
[`step_01_cohort_construction.py`](../step_01_cohort_construction.py) and
[`aki_cohort.py`](../aki_cohort.py). Follow [METHODS.md](../../METHODS.md) for
scientific definitions rather than enabling these legacy helpers.
