#!/bin/bash

# Exit on error
set -e

echo "Starting AKI Prediction Pipeline..."

# Ensure we are in the project root
# This assumes the script is in the project root
cd "$(dirname "$0")"

echo "Step 1: Cohort Construction"
python -m data_preparation.step_01_cohort_construction

echo "Step 2: Catch-22 Feature Extraction"
python -m data_preparation.step_02_catch_22

echo "Step 3: XGBoost HPO and Training"
python -m data_preparation.step_03_xgboost_hpo

echo "Pipeline Complete!"
