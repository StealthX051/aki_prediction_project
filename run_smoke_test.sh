#!/bin/bash

# Run a compact, real-data smoke test of the full pipeline.
# Writes artifacts to an isolated directory so primary results remain untouched.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/artifact_paths.sh"

CASE_LIMIT=${CASE_LIMIT:-10}
HPO_TRIALS=${HPO_TRIALS:-2}
PYTHON_BIN=${PYTHON_BIN:-python}
PARALLEL_BACKEND=${PARALLEL_BACKEND:-processes}
USER_LOG_FILE="${LOG_FILE:-}"
SMOKE_OUTCOMES=${SMOKE_OUTCOMES:-"any_aki,icu_admission"}

read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
    echo "PYTHON_BIN must resolve to at least one command token." >&2
    exit 1
fi

run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

aki_configure_catch22_env "$ROOT_DIR" "smoke_test.log"
if [[ -z "$USER_LOG_FILE" ]]; then
    LOG_FILE="${SMOKE_ROOT}/smoke_test.log"
fi

SMOKE_DATA_DIR="$SMOKE_ROOT/data"
SMOKE_PROCESSED_DIR="$SMOKE_DATA_DIR/processed"
SMOKE_RESULTS_DIR="$SMOKE_ROOT/results/catch22/experiments"
SMOKE_PAPER_DIR="$SMOKE_ROOT/results/catch22/paper"
RAW_SOURCE_DIR=${RAW_SOURCE_DIR:-"$ROOT_DIR/data/raw"}

COHORT_PATH="$SMOKE_PROCESSED_DIR/aki_pleth_ecg_co2_awp.csv"

aki_enforce_generated_paths \
    "smoke_root::${SMOKE_ROOT}" \
    "smoke_processed_dir::${SMOKE_PROCESSED_DIR}" \
    "smoke_results_dir::${SMOKE_RESULTS_DIR}" \
    "smoke_paper_dir::${SMOKE_PAPER_DIR}" \
    "log_file::${LOG_FILE}"

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_PROCESSED_DIR" "$SMOKE_RESULTS_DIR" "$SMOKE_PAPER_DIR" "$(dirname "$LOG_FILE")"

cd "$ROOT_DIR"

export DATA_DIR="$SMOKE_DATA_DIR"
export PROCESSED_DIR="$SMOKE_PROCESSED_DIR"
export RAW_DIR="$RAW_SOURCE_DIR"
export RESULTS_DIR="$SMOKE_RESULTS_DIR"
export PAPER_DIR="$SMOKE_PAPER_DIR"
export GENERATE_WINDOWED_FEATURES="True"
export SMOKE_OUTCOMES

aki_refresh_repo_symlinks "$ROOT_DIR"

echo "=== Starting real-data smoke test ===" | tee "$LOG_FILE"
echo "Smoke root: $SMOKE_ROOT" | tee -a "$LOG_FILE"
echo "Using raw data from: $RAW_SOURCE_DIR" | tee -a "$LOG_FILE"
echo "Python runner: ${PYTHON_CMD[*]}" | tee -a "$LOG_FILE"
echo "Note: PYTHON_BIN accepts a whitespace-separated command prefix, e.g. PYTHON_BIN='conda run -n aki_prediction_project python'." | tee -a "$LOG_FILE"

echo "--- Step 01: Cohort construction ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.step_01_cohort_construction 2>&1 | tee -a "$LOG_FILE"

echo "--- Trimming cohort to $CASE_LIMIT cases ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.smoke_trim_cohort \
    --cohort-path "$COHORT_PATH" \
    --limit "$CASE_LIMIT" \
    --outcomes "$SMOKE_OUTCOMES" \
    2>&1 | tee -a "$LOG_FILE"

echo "--- Step 02: Catch22 feature extraction ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.step_02_catch_22 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 03: Preoperative prep & split ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.step_03_preop_prep 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 04: Intraoperative prep ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.step_04_intraop_prep 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 05: Data merge ---" | tee -a "$LOG_FILE"
run_python -m data_preparation.step_05_data_merge 2>&1 | tee -a "$LOG_FILE"

IFS=',' read -r -a SMOKE_OUTCOME_LIST <<< "$SMOKE_OUTCOMES"
for outcome in "${SMOKE_OUTCOME_LIST[@]}"; do
    outcome="${outcome// /}"
    [[ -z "$outcome" ]] && continue
    echo "--- Step 07: Holdout validation (smoke, ${outcome}) ---" | tee -a "$LOG_FILE"
    run_python -m model_creation.step_07_train_evaluate \
        --outcome "$outcome" \
        --branch windowed \
        --feature_set all_waveforms \
        --model_type xgboost \
        --validation-scheme holdout \
        --max-workers 1 \
        --threads-per-model 2 \
        --n-trials "$HPO_TRIALS" \
        --smoke_test 2>&1 | tee -a "$LOG_FILE"
done

echo "--- Reporting: metrics summary ---" | tee -a "$LOG_FILE"
run_python -m results_recreation.metrics_summary \
    --results-dir "$SMOKE_RESULTS_DIR" \
    --delta-mode reference \
    --reference-feature-set preop_only \
    --parallel-backend "$PARALLEL_BACKEND" \
    --n-jobs -1 \
    --bootstrap-timeout 1800 \
    --bootstrap-max-retries 2 \
    2>&1 | tee -a "$LOG_FILE"

echo "--- Reporting: build reports ---" | tee -a "$LOG_FILE"
run_python -m reporting.make_report 2>&1 | tee -a "$LOG_FILE"

echo "=== Smoke test complete ===" | tee -a "$LOG_FILE"
echo "Artifacts written under $SMOKE_ROOT" | tee -a "$LOG_FILE"
