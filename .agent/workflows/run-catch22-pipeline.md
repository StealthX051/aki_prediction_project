---
description: Run the Catch22 pipeline
---

# Catch22 execution

Use [README.md](../../README.md) for the environment, source paths, study
roots and launcher commands. Scientific settings are in
[METHODS.md](../../METHODS.md); artifact contracts are in
[RESULTS.md](../../RESULTS.md). Do not duplicate commands here.

- Readiness-only requests: offline tests, CLI help, resource inventory, and
  `experiments --preset core --dry-run`; no data preparation or model fitting.
- Research execution requires a run request. Start with the authorized scope;
  use the gated core queue, which mixes model families, before supplementary
  grids. The ICU/mortality cores, 144 main supplementary configurations and
  72 broader ICU supplementary configurations completed; their isolated runs
  are recorded in the evidence register. Mortality supplements remain unstarted.
- Retain 10/5-second windows for this manuscript. The duration pilot completed
  without selecting a candidate under its full rule; no additional duration
  ablation or gate tuning is planned. Any follow-up descriptor audit uses saved
  models; its status is in the [pilot record](window-duration-pilot.md).
- Requested real-data smoke uses a fresh disposable root and live VitalDB access.
- Never overwrite historical artifacts or bypass failed provenance checks.
- Record verification and run status in
  [the evidence register](manuscript-reorientation.md).
