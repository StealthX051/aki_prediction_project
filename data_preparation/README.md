# Data preparation pipeline

This directory contains the production Catch22 preprocessing stages. Run the
steps in order; each stage depends on artifacts from the previous one.

Key settings live in `data_preparation/inputs.py`; waveform rates and filters
are in `WAVEFORM_SPECS` in `waveform_processing.py`. Steps 01–05 write metadata
sidecars that are validated before artifact reuse; Step 02 records a separate
extraction fingerprint for each enabled branch.

## Stages

1. Cohort construction

```bash
python -m data_preparation.step_01_cohort_construction
```

Outputs the shared cohort CSV plus cohort-flow metadata.

2. Catch22 feature extraction

```bash
python -m data_preparation.step_02_catch_22
```

The launcher prepares both non-windowed and windowed waveform feature tables.
The standalone module defaults to whole-case extraction; enable its windowed
branch explicitly with `GENERATE_WINDOWED_FEATURES=True` in a fresh authorized
output root. Existing study features should be reused, not regenerated in place.
The manuscript retains 10-second windows with a nominal five-second stride.
The completed duration pilot does not change production extraction; see
[the duration rationale](../METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis).

3. Preoperative prep and split

```bash
python -m data_preparation.step_03_preop_prep
```

Creates the processed preoperative table plus one outcome-specific grouped
holdout split per supported outcome.

4. Intraoperative wide-table prep

```bash
python -m data_preparation.step_04_intraop_prep
```

Pivots waveform features to the wide modeling format.

5. Data merge

```bash
python -m data_preparation.step_05_data_merge
```

Creates the final non-windowed and windowed master modeling tables.

## Production signal processing

The authoritative rates, filter cutoffs and numerical rules are in
[METHODS.md](../METHODS.md#waveform-processing). Step 02 and
`waveform_processing.py` implement this sequence:

1. Load each VitalDB channel on its configured native grid. Try ECG V5 only if
   ECG II cannot be loaded, not if an already-loaded II later fails quality checks.
2. Crop the operative segment, reject empty segments or >5% nonfinite samples,
   and interpolate remaining missing samples, including nearest-endpoint filling.
3. Filter the whole operative segment with Butterworth SOS forward-backward
   filtering; resample with the centered anti-alias polyphase FIR when rates change.
4. Branch into whole-case Catch24 at 10 Hz or 10-second Catch24 windows at
   100 Hz (ECG/PPG) or 62.5 Hz (CO2/AWP). Aggregate window descriptors by
   mean/sample SD/min/max. Filters are not applied independently per window.

The windowed stride is nominally five seconds: 500 samples at 100 Hz and
312 samples (4.992 seconds) at 62.5 Hz. Flat windows are skipped; included-window
undefined descriptors propagate to their operation summaries. Partial feature
NaNs are retained for training-fold preprocessing. Finite library fallback
values are preserved; the pilot's qualification mask does not alter production.
These quality rules do not implement comprehensive spike/saturation detection.

Step 02's extraction protocol 3 fingerprints source/settings, cohort bounds,
package versions, filters, rates, anti-alias policy and missing-value handling.
Changed or unidentified extraction outputs require a fresh root. Cohort helpers
under `custom_filters/` are historical and are not the waveform-filtering path.

## Preferred validation path

For an end-to-end check, prefer the launcher:

```bash
python -m run_catch22 smoke
```

That command wires Steps 01-05 together, trims the cohort for a fast run, and
writes outputs under an isolated smoke root.
