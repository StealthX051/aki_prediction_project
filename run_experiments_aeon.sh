#!/bin/bash

# Activate environment (if running from outside)
# source activate aki_prediction_project

PYTHON_EXEC="/home/exouser/.conda/envs/aki_prediction_project/bin/python"

# Outcomes
OUTCOMES=("aki_label" "y_severe_aki" "y_inhosp_mortality" "y_icu_admit" "y_prolonged_los_postop")

# Models
# Start with fast ones for debugging, then heavy ones
MODELS=("minirocket") # "freshprince" is very slow for full sweep, maybe enable selectively or run separately
MODELS+=("freshprince") # Uncomment for full run
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
                    # Reconstruct Exp Name logic from step_06
                    # step_06 uses: sorted channels or just input order? 
                    # step_06 logic: chan_str = "all" if args.channels == ['all'] else "_".join(args.channels)
                    # Note: args.channels preserves order from CLI.
                    # Our bash expansion "$channels" passes them in the order defined in FEATURE_SETS string.
                    # So "SNUADC_PLETH Primus_CO2" -> args.channels=['SNUADC_PLETH', 'Primus_CO2'] -> joined with '_'
                    
                    # So we need to match that joining logic here to find the folder.
                    chan_str="${channels// /_}" 
                    exp_name="${model}_${chan_str}_${PREOP_TAG}_${outcome}"
                    pred_file="results/aeon/models/$model/$exp_name/predictions.csv"
                    
                    echo "  > Starting Bootstrap & Calibration..." | tee -a "$LOG_FILE"
                    if $PYTHON_EXEC -m model_creation_aeon.step_07_aeon_bootstrap "$pred_file" --calibrate 2>&1 | tee -a "$LOG_FILE"; then
                         echo "  > Bootstrap Complete." | tee -a "$LOG_FILE"
                    else
                         echo "  > Bootstrap FAILED (Check logs)." | tee -a "$LOG_FILE"
                    fi
                    
                else
                    echo "  > Training FAILED. Skipping bootstrap." | tee -a "$LOG_FILE"
                    continue
                fi

            done
        done
    done
done

echo "All Aeon experiments finished." | tee -a "$LOG_FILE"
