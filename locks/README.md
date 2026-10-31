# CPU environment replay

These locks capture the Linux x86_64 / CPython 3.12.16 candidate prepared on
October 30, 2026, from repository commit `b75138f`. They cover the primary
Catch24/XGBoost/EBM pipeline plus pytest. They do not recover the original GPU
environment or establish numerical equivalence of scientific results.

For everyday environment recreation, use the Conda setup in the root README.
It downloads the pinned package versions from their hosts. The captured-wheel
procedure below is optional exact binary replay; it is not needed for ordinary
development setup. CI and a separate binary archive are optional additions.

- `conda-linux-64.explicit.txt`: 27 exact Conda URLs/builds with SHA256 hashes.
- `requirements-linux-64.lock.txt`: 66 pinned pip distributions; also constrains
  ordinary installs from the root manifests.
- `requirements-linux-64.wheels.txt`: the same 66 pins with captured wheel hashes.
- `installed-python-linux-64.txt`: 70 Python distribution versions, including
  Conda-owned packaging tools. This uses package metadata rather than `pip freeze`
  so Conda's build-time `file://` paths are not treated as installable inputs.

The wheelhouse is stored outside Git. On the current CPU VM it is at
`/home/exouser/codex-handoffs/catch22-cpu-2026-10-30/wheelhouse`.
Retain this directory with the locks: `pycatch22==0.4.5` was compiled locally;
rebuilding its source can change the wheel hash. A clone containing only the
locks cannot promise exact binary replay without these wheels. Conda packages
must also remain available upstream or in the Conda cache. The pip install is
offline; Conda creation may contact conda-forge for uncached packages.

From the repository root, select a new prefix and the retained wheelhouse:

```bash
AKI_REBUILD_PREFIX="$HOME/miniforge3/envs/aki_prediction_project_rebuild_final" \
AKI_WHEELHOUSE=/home/exouser/codex-handoffs/catch22-cpu-2026-10-30/wheelhouse \
  bash scripts/cpu/rebuild-from-locks.sh
```

The destination must not exist, even as an empty directory. The script uses
`CONDA_EXE` or `conda` on PATH; set `AKI_CONDA_EXE` if needed. It never updates
locks, rebuilds wheels, overwrites an environment, or deletes a failed rebuild.
It checks wheel hashes, exact Conda builds and Python distribution versions,
then runs `pip check`, launcher help, native checks and the offline suite.
Evidence is retained in a new `/tmp/aki-cpu-rebuild.*` directory printed at
startup. Copy evidence elsewhere before temporary storage is cleaned.

For subsequent verification, activate the environment and run:

```bash
bash scripts/cpu/verify-offline.sh
```

Outputs go to a fresh `/tmp/aki-cpu-offline.*` directory. To keep evidence in a
durable location, set `AKI_VERIFY_ROOT` to a new directory whose parent exists.
Existing directories are refused because pytest clears its temporary base.
All project input/output roots are overridden inside the script, and storage
enforcement is disabled only for those synthetic validation subprocesses.
Neither script runs the smoke launcher or accesses original research storage.

Native checks use real libraries in a separate process before pytest: Catch24
names/order and values against one synthetic Windows Python 3.9 reference
(`rtol=atol=1e-10`), CPU XGBoost fit/predict/SHAP/save-load, EBM through the
repository factory (`missing='gain'`, `max_bins=1024`), Plotly HTML and synthetic
CSV/Markdown/DOCX/PDF creation. Checks verify SHAP additivity, EBM explanation
values, CSV roundtrip, Markdown/DOCX content, and PDF structure. This does not
include visual review or optional browser image exports.

The automated suite is offline; obsolete live-service diagnostics have been
removed. Real-data smoke for the three model families,
input provenance, cohort/split/schema checks and comparisons with original
results remain separate prerequisites for scientific reproduction. Do not
modify the original GPU VM or research volume to perform environment replay.

## Verification recorded October 30, 2026

The finalized scripts completed successfully in a new Conda prefix:
`aki_prediction_project_rebuild_verified`. All 27 Conda builds/checksums,
66 hashed pip wheels and 70 installed Python distribution versions matched.
Native checks passed, and the complete offline suite passed **81 tests**.
The synthetic Catch24 reference matched with maximum absolute difference
`6.661338147750939e-16`. Inherited output paths remained unused, and inherited
pytest filters and Python optimization did not weaken the checks. Existing
environment/evidence paths, including normalized paths, were refused.
Local logs and native reports are retained under the ignored directory
`outputs/cpu-validation/2026-10-30/`. No real-data smoke or cloud/storage changes
were performed; scientific result equivalence remains unverified.

The ordinary README setup was also verified in a fresh environment using package
hosts with pip caching disabled and no local wheelhouse. Installed distribution
versions matched the snapshot, native checks passed, and all 81 offline tests
passed. This is the default route for everyday recreation.

## Changing dependencies

1. Keep the validated environment and its Git revision intact.
   Work on a separate branch and create a new Conda candidate from the edited
   manifests. Install Conda packages first and pip packages last. Avoid updating
   the validated mixed Conda/pip environment in place.
2. Change direct pins and affected transitive constraints together, recording why
   each change is needed. Never install both XGBoost distributions. Treat changes
   to Python, pycatch22 or EBM as requiring additional compatibility and numerical
   checks. Update the Python-version expectation in `verify_native.py` if needed.
3. Capture the candidate's exact Conda export with `conda list --explicit --sha256`.
   Regenerate the pip-only version lock. If maintaining exact binary replay,
   capture compatible wheels and refresh the wheel hashes together.
   Regenerate the installed-distribution snapshot from package metadata. Keep
   Conda-owned packaging tools out of the pip lock; do not copy `file://` build
   paths from `pip freeze`. Review all manifest/lock differences together.
4. Recreate from the manifests in a fresh environment and run native checks and
   the complete offline suite before promoting the candidate. Version changes
   may alter predictions even when tests
   pass: use fixed, provenance-checked real-data inputs and compare features,
   splits, predictions, calibration and thresholds before replacing research runs.
5. Retain the old and new Git revisions, verification logs, input hashes and run
   settings. Keep patient data out of Git. Archiving package binaries separately
   is optional for longer-term preservation.

Replay and verification are scripted; lock refresh and off-VM binary retention
are not currently automated, and no hosted CI workflow is configured. The local
wheelhouse and Conda cache are sufficient for the exact binary rebuild verified
above. Everyday setup uses the pinned manifests and package hosts.

This procedure follows the upstream guidance for
[pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/)
and [using pip with Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
