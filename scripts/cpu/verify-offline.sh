#!/usr/bin/env bash
# Run inside the activated CPU Conda environment; never invokes real-data smoke.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -n "${AKI_VERIFY_ROOT:-}" ]]; then
  # Refuse existing directories, including symlinks: pytest clears basetemp.
  mkdir -- "$AKI_VERIFY_ROOT"
  evidence=$(cd -- "$AKI_VERIFY_ROOT" && pwd)
else
  evidence=$(mktemp -d /tmp/aki-cpu-offline.XXXXXXXX)
fi
mkdir -- "$evidence/tmp"
export TMPDIR="$evidence/tmp"
echo "Offline verification evidence: $evidence"
export MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMBA_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0
unset PYTEST_ADDOPTS PYTEST_PLUGINS
# Override all output/input roots for these subprocesses, regardless of caller settings.
export AKI_STORAGE_POLICY=off AKI_ARTIFACT_ROOT="$evidence/artifacts"
export DATA_DIR="$evidence/data" RAW_DIR="$evidence/no-research-data"
export INPUT_FILE="$RAW_DIR/clinical_data.csv" LAB_DATA_FILE="$RAW_DIR/lab_data.csv"
export PROCESSED_DIR="$evidence/processed" RESULTS_DIR="$evidence/results"
export PAPER_DIR="$evidence/paper" SMOKE_ROOT="$evidence/unused-smoke"
export FIGURES_DIR="$PAPER_DIR/figures" TABLES_DIR="$PAPER_DIR/tables"
export REPORTS_DIR="$PAPER_DIR/reports" XAI_DIR="$evidence/xai"
export AEON_OUT_DIR="$evidence/unused-aeon"
# Let isolated roots supply defaults instead of inheriting per-file research paths.
unset COHORT_FILE CATCH_22_FILE CATCH_22_ERROR_FILE PREOP_PROCESSED_FILE
unset CATCH_22_WINDOWED_FILE INTRAOP_WIDE_FILE INTRAOP_WIDE_WINDOWED_FILE
unset WIDE_FEATURES_FILE WIDE_FEATURES_WINDOWED_FILE MERGED_DATASET_PATH
unset RAW_SOURCE_DIR COHORT_CSV MERGED_DATASET COUNTS_FILE PROCESSED_PREOP
unset DISPLAY_DICT DISPLAY_DICTIONARY_PATH COHORT_SCOPE VALIDATION_SCHEME BRANCH_ROSTER
export AKI_SPLIT_CACHE_DIR="$evidence/split-cache"
export LOG_FILE="$evidence/imports.log" VERIFY_OUTPUT="$evidence/native"
export MPLCONFIGDIR="$evidence/matplotlib" NUMBA_CACHE_DIR="$evidence/numba-cache"
cd "$repo"
python -m pip check | tee "$evidence/pip-check.txt"
python -m run_catch22 --help > "$evidence/cli-help.txt"
python "$repo/scripts/cpu/verify_native.py" > "$evidence/native.log" 2>&1 || {
  cat "$evidence/native.log"
  exit 1
}
# No live-data diagnostics belong in the automated suite.
python -m pytest -q \
  --basetemp "$evidence/pytest-tmp" -o "cache_dir=$evidence/pytest-cache" \
  --junitxml="$evidence/pytest.xml" 2>&1 | tee "$evidence/pytest.log"
echo "Native checks and offline suite passed. Evidence: $evidence"
