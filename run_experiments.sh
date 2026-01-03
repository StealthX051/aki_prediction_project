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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE="${LOG_FILE:-experiment_log.txt}"
PREP_MODE="${PREP_MODE:-auto}" # auto|force|skip
SMOKE_TEST_FLAG="${SMOKE_TEST_FLAG:-}"

export DATA_DIR PROCESSED_DIR RAW_DIR RESULTS_DIR

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

Options:
  --model-types   Comma-separated list to run (default: xgboost,ebm)
  --only-xgboost  Shortcut for --model-types xgboost
  --only-ebm      Shortcut for --model-types ebm
  --prep          Data prep mode: auto (default), force, or skip
  --smoke         Pass --smoke_test into training/eval
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-types)
            IFS=',' read -r -a MODEL_TYPES <<< "$2"
            shift 2
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
        --smoke)
            SMOKE_TEST_FLAG="--smoke_test"
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
    export GENERATE_WINDOWED_FEATURES="True"

    "$PYTHON_BIN" -m data_preparation.step_01_cohort_construction 2>&1 | tee -a "$LOG_FILE" || { echo "Step 01 failed"; exit 1; }
    "$PYTHON_BIN" -m data_preparation.step_02_catch_22 2>&1 | tee -a "$LOG_FILE" || { echo "Step 02 failed"; exit 1; }
    "$PYTHON_BIN" -m data_preparation.step_03_preop_prep 2>&1 | tee -a "$LOG_FILE" || { echo "Step 03 failed"; exit 1; }
    "$PYTHON_BIN" -m data_preparation.step_04_intraop_prep 2>&1 | tee -a "$LOG_FILE" || { echo "Step 04 failed"; exit 1; }
    "$PYTHON_BIN" -m data_preparation.step_05_data_merge 2>&1 | tee -a "$LOG_FILE" || { echo "Step 05 failed"; exit 1; }
}

has_predictions() {
    find "$RESULTS_DIR" -path "*/predictions/test.csv" -type f -print -quit 2>/dev/null | grep -q .
}

echo "Starting Experiments..." | tee -a "$LOG_FILE"
echo "Prep mode: $PREP_MODE | Models: ${MODEL_TYPES[*]} | Results dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

ensure_data_prepared

for model_type in "${MODEL_TYPES[@]}"; do
    for outcome in "${OUTCOMES[@]}"; do
        for branch in "${BRANCHES[@]}"; do
            for feature_set in "${ALL_FEATURE_SETS[@]}"; do

                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
                echo "Running: Model=$model_type | Outcome=$outcome | Branch=$branch | FeatureSet=$feature_set" | tee -a "$LOG_FILE"
                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

                # 1. Run HPO
                echo "  > Starting HPO for model_type=$model_type..." | tee -a "$LOG_FILE"
                if "$PYTHON_BIN" -m model_creation.step_06_run_hpo --outcome "$outcome" --branch "$branch" --feature_set "$feature_set" --model_type "$model_type" 2>&1 | tee -a "$LOG_FILE"; then
                    echo "  > HPO Complete for model_type=$model_type." | tee -a "$LOG_FILE"
                else
                    echo "  > HPO FAILED for model_type=$model_type. Skipping training." | tee -a "$LOG_FILE"
                    continue
                fi

                # 2. Run Training & Evaluation
                echo "  > Starting Training & Evaluation for model_type=$model_type..." | tee -a "$LOG_FILE"
                if "$PYTHON_BIN" -m model_creation.step_07_train_evaluate \
                    --outcome "$outcome" \
                    --branch "$branch" \
                    --feature_set "$feature_set" \
                    --model_type "$model_type" \
                    $SMOKE_TEST_FLAG 2>&1 | tee -a "$LOG_FILE"; then
                    echo "  > Training & Evaluation Complete for model_type=$model_type." | tee -a "$LOG_FILE"
                else
                    echo "  > Training & Evaluation FAILED for model_type=$model_type." | tee -a "$LOG_FILE"
                fi

            done
        done
    done
done

echo "--- Evaluating models and generating reports ---" | tee -a "$LOG_FILE"

if has_predictions; then
    echo "  > Running evaluation (results -> results directory)..." | tee -a "$LOG_FILE"
    "$PYTHON_BIN" -m results_recreation.metrics_summary --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

    echo "  > Building reports from standardized artifacts..." | tee -a "$LOG_FILE"
    "$PYTHON_BIN" -m reporting.make_report 2>&1 | tee -a "$LOG_FILE"
else
    echo "  > No prediction files found; skipping metrics/report generation." | tee -a "$LOG_FILE"
fi

echo "All experiments finished." | tee -a "$LOG_FILE"
