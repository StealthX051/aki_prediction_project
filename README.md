# AKI Prediction Project

Postoperative AKI and ICU-admission prediction from VitalDB intraoperative
Catch24 waveform features and preoperative clinical data. The manuscript path
uses logistic regression, XGBoost and EBM. The completed paper grid contains
180 main-cohort, 90 broader-ICU and 18 exploratory mortality configurations.
Saved research artifacts remain separate from this code checkout.

- [METHODS.md](METHODS.md): scientific definitions and validation protocol
- [RESULTS.md](RESULTS.md): saved artifacts and reporting commands
- [Current handoff](.agent/workflows/manuscript-reorientation.md): completed runs, open work and manuscript exports
- [AGENTS.md](AGENTS.md): repository working rules

## Active code

| Location | Purpose |
| --- | --- |
| `run_catch22.py` | Canonical preparation, experiment and reporting launcher |
| `data_preparation/` | [Steps 01–05](data_preparation/README.md), cohort eligibility and waveform processing |
| `model_creation/` | Patient-grouped HPO, training, calibration and threshold selection |
| `results_recreation/` | Saved-prediction metrics, paired comparisons and bootstrap estimates |
| `reporting/` | Cohort flow, descriptives, figures, tables and reports |
| `reporting/manuscript_workspace/` | Code-only templates for the portable manuscript workspace |
| `scripts/` | CPU execution, window-duration evidence and INSPIRE linkage tools |
| `tests/` | Isolated offline regression tests and synthetic fixtures |
| `experimental/aeon/` | [Experimental Aeon commands](experimental/aeon/README.md); implementations retain their documented module paths |
| `archive/` | Superseded notebooks, experiments and utilities; excluded from the current workflow |

## Environment and verification

Linux x86_64 / Python 3.12 CPU is supported. Activate the validated environment:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate aki_prediction_project
bash scripts/cpu/verify-offline.sh
```

For a new environment:

```bash
conda env create -f environment.yml
conda activate aki_prediction_project
python -m pip install -r requirements-test.txt
```

Runtime pins are in `requirements-cpu.txt`; exact replay and dependency-change
procedures are in [locks/README.md](locks/README.md). Preserve validated environments
when experimenting. Use `xgboost-cpu`, never both XGBoost distributions. CUDA,
Aeon, AutoGluon and notebooks are outside the supported CPU profile.

The verification script checks dependencies, CLI help, native libraries and the
complete offline test suite in fresh temporary directories. It does not read
research inputs or contact VitalDB. For a focused code check, use
`python -m pytest -q tests/test_<name>.py`. Real-data smoke is a separate research
diagnostic, not a prerequisite for documentation or archive-only changes.

Run tests for the code being changed during ordinary development. For example:

```bash
# Figure/table exports and publication styling
python -m pytest -q tests/test_manuscript_exports.py tests/test_publication*.py
# Cohort definitions and patient separation
python -m pytest -q tests/test_modular_cohorts.py tests/test_patient_level_splitting.py
```

The full suite also checks retained research tools and execution safeguards; use
it for a release or environment checkpoint, rather than every small edit.

## Manuscript figures and tables

The downloadable complete workspace contains the data and editable documents.
Git retains its rebuild code, not the research data or generated ZIPs. To compare
an unpacked workspace with the current source templates:

```bash
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --check
```

To copy updated code and the shared appearance profile into that workspace:

```bash
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --apply
python /path/to/manuscript-workspace/05_CODE/rebuild_publication_workspace.py --package
```

The rebuild consumes bundled data; it fits no models or calibrators and draws
no new bootstrap samples. Existing author manuscript files are preserved.
Edit `reporting/publication_style.json` to change colors, fonts and export
settings before applying the templates. See the
[workspace source guide](reporting/manuscript_workspace/README.md) and
[shared style guide](reporting/PUBLICATION_STYLE.md).

## Existing research artifacts

On this VM, study roots are under
`/media/volume/catch22/data/aki_prediction_project/studies/`:

| Study | Contents |
| --- | --- |
| `manuscript-core-v1` | Original 36 main-core models and sealed processed features |
| `manuscript-optimized-v1` | Numerical equivalence gate, broader-ICU core and exploratory mortality core |
| `manuscript-supplementary-v1` | Full 180-configuration main grid, reusing the main core |
| `manuscript-icu-supplementary-v1` | Full 90-configuration broader-ICU grid, reusing its core |

Source clinical/laboratory CSVs are under
`/media/volume/catch22/projects/aki_prediction_project/data/raw`.
Study manifests and `control/launch-status.json` record completion; old tmux
sessions may remain after their jobs exit. Mortality supplements remain optional
and unstarted. The window-duration search is closed: retain 10-second windows
with a nominal five-second stride; see the
[recorded decision](METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis).

Git and the writing ZIPs do not contain fitted model binaries or original
waveforms. Retain the attached research volume when shelving compute. Exact
binary environment replay also needs the wheelhouse identified in
[locks/README.md](locks/README.md); ordinary environment recreation uses the pins.

## Future analyses and saved-result reporting

Use a fresh study root for an authorized new analysis. Path overrides select the
study and its processed inputs; historical artifact names and study identities
must remain intact. A read-only planning example is:

```bash
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export PROCESSED_DIR=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1/data/processed
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/new-study
python -m run_catch22 experiments --preset core --prep skip \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity \
  --reuse-from /media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1 \
  --dry-run
```

`--prep auto` prepares absent inputs or validates existing ones; `--prep skip`
only validates. `--prep force` requires an empty processed directory. Reuse and
resume require matching source, cohort, preprocessing, model and validation
fingerprints. See `python -m run_catch22 experiments --help` for configuration,
admission and reporting options. The retained optimized supervisor is documented
in the evidence register; completed studies do not need to be relaunched.

For saved-result work, select the existing study roots before running:

```bash
python -m run_catch22 experiments --preset core --report-only \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity
COHORT_SCOPE=mortality_waveform_sensitivity python -m reporting.fold_explanations
```

Reporting can recompute metrics and bootstrap estimates; the portable manuscript
rebuild above only renders their saved values. Definitive reporting uses 5,000
unstratified patient-cluster draws, seed 0. [RESULTS.md](RESULTS.md) documents
scoped artifacts, DCA, explanations and study-specific reporting commands.
Outcome definitions, train-only preprocessing/calibration and patient-grouped
validation remain governed by [METHODS.md](METHODS.md).
