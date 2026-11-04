# Data preparation pipeline

This directory contains the production Catch22 preprocessing stages. Run the
steps in order; each stage depends on artifacts from the previous one.

Key settings live in `data_preparation/inputs.py`; waveform rates and filters
are in `WAVEFORM_SPECS` in `waveform_processing.py`. Steps 01–05 write metadata
sidecars that are validated before artifact reuse; Step 02 records a separate
extraction fingerprint for each enabled branch.

## Stages

Run these commands in order in the activated project environment:

| Stage | Command | Output |
| --- | --- | --- |
| Cohort | `python -m data_preparation.step_01_cohort_construction` | Shared cohort and flow metadata |
| Waveform features | `python -m data_preparation.step_02_catch_22` | Catch24 features for the enabled branches |
| Preoperative data | `python -m data_preparation.step_03_preop_prep` | Preoperative table and outcome-specific patient-grouped holdout splits |
| Wide features | `python -m data_preparation.step_04_intraop_prep` | Wide waveform feature tables |
| Merge | `python -m data_preparation.step_05_data_merge` | Modeling tables for the prepared branches |

The launcher prepares both waveform branches. Standalone Step 02 defaults to
whole-case extraction; set `GENERATE_WINDOWED_FEATURES=True` to enable windows in
a fresh output root. Reuse existing study features instead of regenerating them.
The manuscript retains 10-second windows with a nominal five-second stride;
the [duration pilot](../METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis)
did not change production extraction.

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
Spike and saturation detection are not implemented.

Step 02's extraction protocol 3 fingerprints source/settings, cohort bounds,
package versions, filters, rates, anti-alias policy and missing-value handling.
Changed or unidentified extraction outputs require a fresh root. Cohort/outcome
logic is in `step_01_cohort_construction.py` and `aki_cohort.py`; the removed
`custom_filters` helpers do not define the current pipeline.

## Real-data smoke

Run this research diagnostic only when requested, with a fresh or empty
`SMOKE_ROOT`. It reads source data and contacts VitalDB:

```bash
python -m run_catch22 smoke
```

It runs Steps 01–05 on a trimmed cohort and writes to the isolated smoke root.
Use the root README’s offline verification for ordinary code checks.
