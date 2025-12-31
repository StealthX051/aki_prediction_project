#!/bin/bash

# Activate environment (if running from outside)
# source activate aki_prediction_project

# Outcomes
OUTCOMES=("any_aki" "icu_admission")

# Branches
BRANCHES=("non_windowed" "windowed")

# Feature Sets
# 1. Primary Models
PRIMARY_SETS=(
    "preop_only"
    "pleth_only" "ecg_only" "co2_only" "awp_only"
    "all_waveforms"
    "preop_and_all_waveforms"
    "ventilator_only"
    "monitors_only"
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

# Log file
LOG_FILE="experiment_log.txt"

echo "Starting Experiments..." | tee -a "$LOG_FILE"

for outcome in "${OUTCOMES[@]}"; do
    for branch in "${BRANCHES[@]}"; do
        for feature_set in "${ALL_FEATURE_SETS[@]}"; do
            
            echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
            echo "Running: Outcome=$outcome | Branch=$branch | FeatureSet=$feature_set" | tee -a "$LOG_FILE"
            echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

            # 1. Run HPO
            echo "  > Starting HPO..." | tee -a "$LOG_FILE"
            if python model_creation/step_06_run_hpo.py --outcome "$outcome" --branch "$branch" --feature_set "$feature_set" 2>&1 | tee -a "$LOG_FILE"; then
                echo "  > HPO Complete." | tee -a "$LOG_FILE"
            else
                echo "  > HPO FAILED. Skipping training." | tee -a "$LOG_FILE"
                continue
            fi

            # 2. Run Training & Evaluation
            echo "  > Starting Training & Evaluation..."        # Step 7: Train and Evaluate (includes prediction generation)
        python3 model_creation/step_07_train_evaluate.py \
            --outcome "$outcome" \
            --branch "$branch" \
            --feature_set "$feature_set" \
            $SMOKE_TEST_FLAG
            
        done
    done
done

echo "All experiments finished." | tee -a "$LOG_FILE"
