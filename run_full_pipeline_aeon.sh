#!/bin/bash

# run_full_pipeline_aeon.sh
# Runs the entire Aeon pipeline from Data Prep to Analysis.

LOG_FILE="pipeline_aeon.log"
PYTHON_EXEC="/home/exouser/.conda/envs/aki_prediction_project/bin/python"

echo "Starting Full Aeon Pipeline..." | tee -a "$LOG_FILE"

# 1. Data Export
echo "--- Step 02: Aeon Export ---" | tee -a "$LOG_FILE"
if $PYTHON_EXEC -m data_preparation.step_02_aeon_export 2>&1 | tee -a "$LOG_FILE"; then
    echo "Export Complete." | tee -a "$LOG_FILE"
else
    echo "Export Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 2. Preop Prep
echo "--- Step 04: Aeon Preop Prep ---" | tee -a "$LOG_FILE"
if $PYTHON_EXEC -m data_preparation.step_04_aeon_prep 2>&1 | tee -a "$LOG_FILE"; then
    echo "Preop Prep Complete." | tee -a "$LOG_FILE"
else
    echo "Preop Prep Failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# 3. Experiments
echo "--- Running Experiments ---" | tee -a "$LOG_FILE"
bash run_experiments_aeon.sh

# 4. Evaluation and Reporting
echo "--- Evaluating Aeon models and generating reports ---" | tee -a "$LOG_FILE"
${PYTHON_EXEC} -m results_recreation.metrics_summary --results-dir results 2>&1 | tee -a "$LOG_FILE"
${PYTHON_EXEC} -m reporting.make_report 2>&1 | tee -a "$LOG_FILE"

echo "Full Pipeline Finished." | tee -a "$LOG_FILE"
