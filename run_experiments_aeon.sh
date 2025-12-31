#!/bin/bash

# Activate environment (if running from outside)
# source activate aki_prediction_project

PYTHON_EXEC="/home/exouser/.conda/envs/aki_prediction_project/bin/python"

# Outcomes
OUTCOMES=("any_aki" "icu_admission")

# Models
# Start with fast ones for debugging, then heavy ones
MODELS=("minirocket") # "freshprince" is very slow for full sweep, maybe enable selectively or run separately
# MODELS+=("freshprince") # Uncomment for full run
MODELS+=("multirocket")

# Feature Sets (Channels)
# "all" -> All channels
# Singles
# Leave-One-Out (L1O) sets for ablation
FEATURE_SETS=(
    "all"
    # Singles
    "SNUADC_ECG_II"
    "SNUADC_PLETH"
    "Primus_CO2"
    "Primus_AWP"
    # Leave-One-Out (All minus X)
    "SNUADC_PLETH Primus_CO2 Primus_AWP" # Minus ECG
    "SNUADC_ECG_II Primus_CO2 Primus_AWP" # Minus PLETH
    "SNUADC_ECG_II SNUADC_PLETH Primus_AWP" # Minus CO2
    "SNUADC_ECG_II SNUADC_PLETH Primus_CO2" # Minus AWP
)

# Fusion Mode
# "true" -> --include_preop
# "false" -> (no flag)
FUSION_MODES=("true" "false")

LOG_FILE="experiment_log_aeon.txt"

echo "Starting Aeon Experiments..." | tee -a "$LOG_FILE"

for outcome in "${OUTCOMES[@]}"; do
    for model in "${MODELS[@]}"; do
        for channels in "${FEATURE_SETS[@]}"; do
            for use_preop in "${FUSION_MODES[@]}"; do
                
                # Construct flags
                CHANNEL_FLAG="--channels $channels"
                PREOP_FLAG=""
                
                # Tag naming
                if [ "$channels" == "all" ]; then
                    chan_tag="all"
                elif [[ "$channels" == *" "* ]]; then
                    # If space exists, it's a combo/ablation
                    # We can auto-gen tag or use the full string with underscores
                    chan_tag="${channels// /_}" 
                else
                    chan_tag="$channels"
                fi
                
                if [ "$use_preop" == "true" ]; then
                    PREOP_FLAG="--include_preop"
                    PREOP_TAG="fused"
                else
                    PREOP_TAG="waveonly"
                fi
                
                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
                echo "Running: Model=$model | Branch=Aeon | Channels=$chan_tag | Type=$PREOP_TAG | Outcome=$outcome" | tee -a "$LOG_FILE"
                echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

                # 1. Train & Predict
                echo "  > Starting Training..." | tee -a "$LOG_FILE"
                # $channels expands to multiple args if spaces exist
                cmd="$PYTHON_EXEC -m model_creation_aeon.step_06_aeon_train --model $model --outcome $outcome $CHANNEL_FLAG $PREOP_FLAG"
                echo "  Command: $cmd" >> "$LOG_FILE"
                
                if $cmd 2>&1 | tee -a "$LOG_FILE"; then
                    echo "  > Training Complete." | tee -a "$LOG_FILE"
                    
                    # 2. Bootstrap & Calibrate
                    # Note: Step 07 (Bootstrap) is removed. Results are handled by results_analysis.py
                    
                else
                    echo "  > Training FAILED. Skipping bootstrap." | tee -a "$LOG_FILE"
                    continue
                fi

            done
        done
    done
done

echo "All Aeon experiments finished." | tee -a "$LOG_FILE"
