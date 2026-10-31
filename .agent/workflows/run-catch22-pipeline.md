---
description: Run the Catch22 pipeline
---

# Catch22 execution

Use [README.md](../../README.md) for current environment, source paths, fresh
study roots, resource budgets, and launch commands. Scientific settings are in
[METHODS.md](../../METHODS.md); artifact contracts are in
[RESULTS.md](../../RESULTS.md). Do not duplicate commands here.

- Readiness-only requests: offline tests, CLI help, resource inventory, and
  `experiments --preset core --dry-run`; no data preparation or model fitting.
- Research execution requires a run request. Start with the authorized scope;
  use the gated core queue, which mixes model families, before supplementary
  grids. The current full-budget canary and production batches remain on hold.
- Real-data smoke uses a fresh disposable root and live VitalDB access.
- Never overwrite historical artifacts or bypass failed provenance checks.
- Record verification and run status in
  [the evidence register](manuscript-reorientation.md).
