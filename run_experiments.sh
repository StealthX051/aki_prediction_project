#!/bin/bash

# Fail fast on errors within pipelines so tee preserves exit codes
set -o pipefail

# Activate environment (if running from outside)
# source activate aki_prediction_project

# Outcomes
OUTCOMES=("any_aki" "icu_admission")

# Branches
BRANCHES=("non_windowed" "windowed")

# Model types to evaluate; default covers both supported branches so
# reporting can harvest results from results/models/{model_type}/...
MODEL_TYPES=("xgboost" "ebm")

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

# Optional flags
SMOKE_TEST_FLAG=${SMOKE_TEST_FLAG:-}

# Log file
LOG_FILE="experiment_log.txt"

echo "Starting Experiments..." | tee -a "$LOG_FILE"

for model_type in "${MODEL_TYPES[@]}"; do
    for outcome in "${OUTCOMES[@]}"; do
        for branch in "${BRANCHES[@]}"; do
            for feature_set in "${ALL_FEATURE_SETS[@]}"; do

                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
                echo "Running: Model=$model_type | Outcome=$outcome | Branch=$branch | FeatureSet=$feature_set" | tee -a "$LOG_FILE"
                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

                # 1. Run HPO
                echo "  > Starting HPO for model_type=$model_type..." | tee -a "$LOG_FILE"
                if python model_creation/step_06_run_hpo.py --outcome "$outcome" --branch "$branch" --feature_set "$feature_set" --model_type "$model_type" 2>&1 | tee -a "$LOG_FILE"; then
                    echo "  > HPO Complete for model_type=$model_type." | tee -a "$LOG_FILE"
                else
                    echo "  > HPO FAILED for model_type=$model_type. Skipping training." | tee -a "$LOG_FILE"
                    continue
                fi

                # 2. Run Training & Evaluation
                echo "  > Starting Training & Evaluation for model_type=$model_type..." | tee -a "$LOG_FILE"
                if python3 model_creation/step_07_train_evaluate.py \
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

echo "  > Running evaluation (results -> results directory)..." | tee -a "$LOG_FILE"
python -m results_recreation.metrics_summary --results-dir results 2>&1 | tee -a "$LOG_FILE"

echo "  > Building reports from standardized artifacts..." | tee -a "$LOG_FILE"
python -m reporting.make_report 2>&1 | tee -a "$LOG_FILE"

echo "All experiments finished." | tee -a "$LOG_FILE"
