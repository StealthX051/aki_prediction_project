#!/bin/bash

# run_aeon_smoke_test.sh
# Runs a fast end-to-end smoke test of the Aeon pipeline with a small subset of data.
# Uses separate output directories to avoid overwriting real results.

# 1. Setup Environment
export AEON_OUT_DIR="outputs/aeon_smoke"
RESULTS_DIR="results/aeon_smoke"
LOG_FILE="smoke_test_aeon.log"
PYTHON_EXEC="/home/exouser/.conda/envs/aki_prediction_project/bin/python"

# Clean up previous smoke test
rm -rf "$AEON_OUT_DIR"
rm -rf "$RESULTS_DIR"
mkdir -p "$AEON_OUT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Starting Aeon Smoke Test ===" | tee "$LOG_FILE"
echo "Data Dir: $AEON_OUT_DIR" | tee -a "$LOG_FILE"
echo "Results Dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

# 2. Data Export (Limit to 50 cases)
echo "--- Step 02: Aeon Export (Limit 50) ---" | tee -a "$LOG_FILE"
if $PYTHON_EXEC -m data_preparation.step_02_aeon_export --limit 50 2>&1 | tee -a "$LOG_FILE"; then
    echo "Export Complete." | tee -a "$LOG_FILE"
else
    echo "Export Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 3. Preop Prep (Reads from AEON_OUT_DIR implicitly via inputs.py logic on inputs, but wait)
# step_04 reads PREOP_PROCESSED_FILE (production preop data) and matches it against generated waveform data in AEON_OUT_DIR
# It writes 'aki_preop_aeon.csv' to AEON_OUT_DIR.
echo "--- Step 04: Aeon Preop Prep ---" | tee -a "$LOG_FILE"
if $PYTHON_EXEC -m data_preparation.step_04_aeon_prep 2>&1 | tee -a "$LOG_FILE"; then
    echo "Preop Prep Complete." | tee -a "$LOG_FILE"
else
    echo "Preop Prep Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 4. Train Model (MiniRocket, Single Channel, Fused)
# Using MiniRocket as it's fast.
MODEL="minirocket"
CHANNELS="SNUADC_PLETH"
OUTCOME="aki_label"

echo "--- Step 06: Training ($MODEL) ---" | tee -a "$LOG_FILE"
CMD="$PYTHON_EXEC -m model_creation_aeon.step_06_aeon_train --model $MODEL --outcome $OUTCOME --channels $CHANNELS --include_preop --limit 50 --results_dir $RESULTS_DIR"
echo "Command: $CMD" | tee -a "$LOG_FILE"

if $CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo "Training Complete." | tee -a "$LOG_FILE"
else
    echo "Training Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 5. Train Model (FreshPRINCE, Single Channel, Fused)
MODEL_FP="freshprince"
echo "--- Step 06b: Training ($MODEL_FP) ---" | tee -a "$LOG_FILE"
CMD_FP="$PYTHON_EXEC -m model_creation_aeon.step_06_aeon_train --model $MODEL_FP --outcome $OUTCOME --channels $CHANNELS --include_preop --limit 50 --results_dir $RESULTS_DIR"
echo "Command: $CMD_FP" | tee -a "$LOG_FILE"

if $CMD_FP 2>&1 | tee -a "$LOG_FILE"; then
    echo "Training ($MODEL_FP) Complete." | tee -a "$LOG_FILE"
else
    echo "Training ($MODEL_FP) Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 6. Bootstrap
# Predict file path: results/aeon_smoke/models/freshprince/freshprince_SNUADC_PLETH_fused_aki_label/predictions.csv
PRED_FILE="$RESULTS_DIR/models/$MODEL_FP/${MODEL_FP}_${CHANNELS}_fused_${OUTCOME}/predictions.csv"

echo "--- Step 07: Bootstrap ---" | tee -a "$LOG_FILE"
if $PYTHON_EXEC -m model_creation_aeon.step_07_aeon_bootstrap "$PRED_FILE" 2>&1 | tee -a "$LOG_FILE"; then
    echo "Bootstrap Complete." | tee -a "$LOG_FILE"
else
    echo "Bootstrap Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Smoke Test Passed! ===" | tee -a "$LOG_FILE"
echo "Check $RESULTS_DIR for outputs."
