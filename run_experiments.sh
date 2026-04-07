#!/bin/bash

# Primary experiment launcher for Catch22-based XGBoost/EBM models with optional
# data prep and staged model-family execution.

# Fail fast on errors within pipelines so tee preserves exit codes
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env if needed)
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATA_DIR}/processed}"
RAW_DIR="${RAW_DIR:-${DATA_DIR}/raw}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/catch22/experiments}"
PAPER_DIR="${PAPER_DIR:-${SCRIPT_DIR}/results/catch22/paper}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE="${LOG_FILE:-experiment_log.txt}"
PREP_MODE="${PREP_MODE:-auto}" # auto|force|skip
SMOKE_TEST_FLAG="${SMOKE_TEST_FLAG:-}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-processes}"
VALIDATION_SCHEME="${VALIDATION_SCHEME:-nested_cv}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-5}"
REPEATS="${REPEATS:-1}"
MAX_WORKERS="${MAX_WORKERS:-4}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-8}"
N_TRIALS="${N_TRIALS:-100}"
SAVE_FINAL_REFIT_FLAG="${SAVE_FINAL_REFIT_FLAG:-}"
RESUME_FLAG="${RESUME_FLAG:---resume}"

export DATA_DIR PROCESSED_DIR RAW_DIR RESULTS_DIR PAPER_DIR

read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
    echo "PYTHON_BIN must resolve to at least one command token." >&2
    exit 1
fi

run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

# Outcomes
OUTCOMES=("any_aki" "icu_admission")

# Branches
BRANCHES=("non_windowed" "windowed")

# Model types to evaluate; default covers both supported branches so reporting
# can harvest results from results/models/{model_type}/...
MODEL_TYPES=("xgboost" "ebm")

usage() {
    cat <<'EOF'
Usage: ./run_experiments.sh [--model-types xgboost,ebm] [--only-xgboost|--only-ebm] [--prep auto|force|skip] [--smoke]
                           [--validation-scheme nested_cv|holdout] [--outer-folds N] [--inner-folds N]
                           [--repeats N] [--max-workers N] [--threads-per-model N] [--n-trials N]
                           [--resume|--no-resume] [--save-final-refit]

Options:
  --model-types   Comma-separated list to run (default: xgboost,ebm)
  --only-xgboost  Shortcut for --model-types xgboost
  --only-ebm      Shortcut for --model-types ebm
  --prep          Data prep mode: auto (default), force, or skip
  --smoke         Pass --smoke_test into training/eval
  --validation-scheme  Validation scheme for step_07_train_evaluate (default: nested_cv)
  --outer-folds   Number of outer folds for nested CV (default: 5)
  --inner-folds   Number of inner folds for HPO/calibration (default: 5)
  --repeats       Number of nested-CV repeats (default: 1; values >1 currently unsupported)
  --max-workers   Parallel outer-fold workers per configuration (default: 4)
  --threads-per-model  Threads allocated to each fitted model (default: 8)
  --n-trials      Optuna trials per outer fit/refit (default: 100)
  --resume        Resume completed outer-fold checkpoints when the stored validation fingerprint matches exactly (default)
  --no-resume     Disable resume and rerun all outer folds
  --save-final-refit  Save an optional full-data refit bundle after validation (supported for nested_cv and holdout)
  -h, --help      Show this help

Environment:
  PYTHON_BIN      Whitespace-separated command prefix used for every Python invocation
                  (default: python). Example:
                  PYTHON_BIN='conda run -n aki_prediction_project python'

Behavior:
  The launcher continues through the requested grid, but if any configuration fails
  validation it skips metrics/report generation and exits nonzero after printing a
  compact failure summary.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-types)
            IFS=',' read -r -a MODEL_TYPES <<< "$2"
            shift 2
            ;;
        --model-types=*)
            IFS=',' read -r -a MODEL_TYPES <<< "${1#*=}"
            shift
            ;;
        --only-xgboost)
            MODEL_TYPES=("xgboost")
            shift
            ;;
        --only-ebm)
            MODEL_TYPES=("ebm")
            shift
            ;;
        --prep)
            PREP_MODE="$2"
            shift 2
            ;;
        --prep=*)
            PREP_MODE="${1#*=}"
            shift
            ;;
        --smoke)
            SMOKE_TEST_FLAG="--smoke_test"
            shift
            ;;
        --validation-scheme)
            VALIDATION_SCHEME="$2"
            shift 2
            ;;
        --validation-scheme=*)
            VALIDATION_SCHEME="${1#*=}"
            shift
            ;;
        --outer-folds)
            OUTER_FOLDS="$2"
            shift 2
            ;;
        --outer-folds=*)
            OUTER_FOLDS="${1#*=}"
            shift
            ;;
        --inner-folds)
            INNER_FOLDS="$2"
            shift 2
            ;;
        --inner-folds=*)
            INNER_FOLDS="${1#*=}"
            shift
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --repeats=*)
            REPEATS="${1#*=}"
            shift
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --max-workers=*)
            MAX_WORKERS="${1#*=}"
            shift
            ;;
        --threads-per-model)
            THREADS_PER_MODEL="$2"
            shift 2
            ;;
        --threads-per-model=*)
            THREADS_PER_MODEL="${1#*=}"
            shift
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --n-trials=*)
            N_TRIALS="${1#*=}"
            shift
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --no-resume)
            RESUME_FLAG=""
            shift
            ;;
        --save-final-refit)
            SAVE_FINAL_REFIT_FLAG="--save-final-refit"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Feature Sets
# 1. Primary Models
PRIMARY_SETS=(
    "preop_only"
    "pleth_only" "ecg_only" "co2_only" "awp_only"
    "all_waveforms"
    "preop_and_all_waveforms"
)

# 2. Ablation Models (Preop + Waveform A)
ABLATION_SETS_1=(
    "preop_and_pleth"
    "preop_and_ecg"
    "preop_and_co2"
    "preop_and_awp"
)

# 3. Ablation Models (Preop + All - Waveform A)
ABLATION_SETS_2=(
    "preop_and_all_minus_pleth"
    "preop_and_all_minus_ecg"
    "preop_and_all_minus_co2"
    "preop_and_all_minus_awp"
)

# Combine all feature sets
ALL_FEATURE_SETS=("${PRIMARY_SETS[@]}" "${ABLATION_SETS_1[@]}" "${ABLATION_SETS_2[@]}")

ensure_data_prepared() {
    local full_file="${PROCESSED_DIR}/aki_features_master_wide.csv"
    local windowed_file="${PROCESSED_DIR}/aki_features_master_wide_windowed.csv"

    if [[ "$PREP_MODE" == "skip" ]]; then
        echo "Skipping data prep (PREP_MODE=skip)." | tee -a "$LOG_FILE"
        return
    fi

    if [[ "$PREP_MODE" != "force" && -f "$full_file" && -f "$windowed_file" ]]; then
        echo "Reusing existing processed data at ${PROCESSED_DIR} (use --prep force to rebuild)." | tee -a "$LOG_FILE"
        return
    fi

    echo "--- Running data preparation (steps 01–05) ---" | tee -a "$LOG_FILE"
    mkdir -p "$PROCESSED_DIR"
    if [[ "$PREP_MODE" == "force" ]]; then
        echo "Forcing rebuild: removing existing processed directory at ${PROCESSED_DIR}" | tee -a "$LOG_FILE"
        rm -rf "$PROCESSED_DIR"
        mkdir -p "$PROCESSED_DIR"
    fi
    export GENERATE_WINDOWED_FEATURES="True"

    run_python -m data_preparation.step_01_cohort_construction 2>&1 | tee -a "$LOG_FILE" || { echo "Step 01 failed"; exit 1; }
    run_python -m data_preparation.step_02_catch_22 2>&1 | tee -a "$LOG_FILE" || { echo "Step 02 failed"; exit 1; }
    run_python -m data_preparation.step_03_preop_prep 2>&1 | tee -a "$LOG_FILE" || { echo "Step 03 failed"; exit 1; }
    run_python -m data_preparation.step_04_intraop_prep 2>&1 | tee -a "$LOG_FILE" || { echo "Step 04 failed"; exit 1; }
    run_python -m data_preparation.step_05_data_merge 2>&1 | tee -a "$LOG_FILE" || { echo "Step 05 failed"; exit 1; }
}

echo "Starting Experiments..." | tee -a "$LOG_FILE"
echo "Prep mode: $PREP_MODE | Models: ${MODEL_TYPES[*]} | Results dir: $RESULTS_DIR | Validation: $VALIDATION_SCHEME" | tee -a "$LOG_FILE"
echo "Python runner: ${PYTHON_CMD[*]}" | tee -a "$LOG_FILE"

ensure_data_prepared

successful_validation_runs=0
failed_validation_runs=0
FAILED_CONFIGS=()

for model_type in "${MODEL_TYPES[@]}"; do
    for outcome in "${OUTCOMES[@]}"; do
        for branch in "${BRANCHES[@]}"; do
            for feature_set in "${ALL_FEATURE_SETS[@]}"; do

                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
                echo "Running: Model=$model_type | Outcome=$outcome | Branch=$branch | FeatureSet=$feature_set" | tee -a "$LOG_FILE"
                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

                echo "  > Starting Validation run for model_type=$model_type..." | tee -a "$LOG_FILE"
                validation_args=(
                    -m model_creation.step_07_train_evaluate
                    --outcome "$outcome"
                    --branch "$branch"
                    --feature_set "$feature_set"
                    --model_type "$model_type"
                    --validation-scheme "$VALIDATION_SCHEME"
                    --outer-folds "$OUTER_FOLDS"
                    --inner-folds "$INNER_FOLDS"
                    --repeats "$REPEATS"
                    --max-workers "$MAX_WORKERS"
                    --threads-per-model "$THREADS_PER_MODEL"
                    --n-trials "$N_TRIALS"
                )
                if [[ "$model_type" == "ebm" ]]; then
                    validation_args+=(--export_ebm_explanations)
                fi
                if [[ -n "$RESUME_FLAG" ]]; then
                    validation_args+=("$RESUME_FLAG")
                fi
                if [[ -n "$SAVE_FINAL_REFIT_FLAG" ]]; then
                    validation_args+=("$SAVE_FINAL_REFIT_FLAG")
                fi
                if [[ -n "$SMOKE_TEST_FLAG" ]]; then
                    validation_args+=("$SMOKE_TEST_FLAG")
                fi

                if run_python "${validation_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
                    echo "  > Validation run complete for model_type=$model_type." | tee -a "$LOG_FILE"
                    successful_validation_runs=$((successful_validation_runs + 1))
                else
                    echo "  > Validation run FAILED for model_type=$model_type." | tee -a "$LOG_FILE"
                    failed_validation_runs=$((failed_validation_runs + 1))
                    FAILED_CONFIGS+=("model=${model_type} outcome=${outcome} branch=${branch} feature_set=${feature_set}")
                fi

            done
        done
    done
done

if (( failed_validation_runs > 0 )); then
    echo "--- Validation completed with failures; skipping metrics/report generation ---" | tee -a "$LOG_FILE"
    for failed_config in "${FAILED_CONFIGS[@]}"; do
        echo "  > FAILED: ${failed_config}" | tee -a "$LOG_FILE"
    done
    echo "Launcher finished with ${failed_validation_runs} failed validation configuration(s)." | tee -a "$LOG_FILE"
    exit 1
fi

if (( successful_validation_runs == 0 )); then
    echo "--- No validation artifacts were produced during this run; skipping metrics/report generation ---" | tee -a "$LOG_FILE"
    exit 1
fi

echo "--- Evaluating models and generating reports ---" | tee -a "$LOG_FILE"
echo "  > Running evaluation (results -> results directory)..." | tee -a "$LOG_FILE"
if ! run_python -m results_recreation.metrics_summary \
    --results-dir "$RESULTS_DIR" \
    --delta-mode reference \
    --reference-feature-set preop_only \
    --parallel-backend "$PARALLEL_BACKEND" \
    --n-jobs -1 \
    --bootstrap-timeout 1800 \
    --bootstrap-max-retries 2 \
    2>&1 | tee -a "$LOG_FILE"; then
    echo "Evaluation failed." | tee -a "$LOG_FILE"
    exit 1
fi

echo "  > Building reports from standardized artifacts..." | tee -a "$LOG_FILE"
if ! env \
    CALIBRATION_BIN_STRATEGY=quantile \
    CALIBRATION_N_BINS=10 \
    CALIBRATION_SHOW_BIN_COUNTS=true \
    CALIBRATION_SHOW_PROB_HIST=true \
    CALIBRATION_SHOW_XLIM_INSET=true \
    CALIBRATION_MAX_COUNT_ANNOTATE=30 \
    PLOT_PREFER_CALIBRATED=true \
    PLOT_N_JOBS=-2 \
    PR_SHOW_CLASS_BALANCE=false \
    "${PYTHON_CMD[@]}" -m reporting.make_report 2>&1 | tee -a "$LOG_FILE"; then
    echo "Report generation failed." | tee -a "$LOG_FILE"
    exit 1
fi

echo "All experiments finished." | tee -a "$LOG_FILE"
