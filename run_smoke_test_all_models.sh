#!/bin/bash

# Run a compact, real-data smoke test of the full pipeline for both XGBoost and EBM.
# Writes artifacts to an isolated directory so primary results remain untouched.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

SMOKE_ROOT=${SMOKE_ROOT:-"$ROOT_DIR/smoke_test_outputs"}
CASE_LIMIT=${CASE_LIMIT:-10}
HPO_TRIALS=${HPO_TRIALS:-2}
PYTHON_BIN=${PYTHON_BIN:-python}

SMOKE_DATA_DIR="$SMOKE_ROOT/data"
SMOKE_PROCESSED_DIR="$SMOKE_DATA_DIR/processed"
SMOKE_RESULTS_DIR="$SMOKE_ROOT/results"
RAW_SOURCE_DIR=${RAW_SOURCE_DIR:-"$ROOT_DIR/data/raw"}
LOG_FILE="$SMOKE_ROOT/smoke_test_all_models.log"

COHORT_PATH="$SMOKE_PROCESSED_DIR/aki_pleth_ecg_co2_awp.csv"

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_PROCESSED_DIR" "$SMOKE_RESULTS_DIR"

cd "$ROOT_DIR"

export DATA_DIR="$SMOKE_DATA_DIR"
export PROCESSED_DIR="$SMOKE_PROCESSED_DIR"
export RAW_DIR="$RAW_SOURCE_DIR"
export RESULTS_DIR="$SMOKE_RESULTS_DIR"
export GENERATE_WINDOWED_FEATURES="True"

echo "=== Starting real-data smoke test (XGBoost + EBM) ===" | tee "$LOG_FILE"
echo "Smoke root: $SMOKE_ROOT" | tee -a "$LOG_FILE"
echo "Using raw data from: $RAW_SOURCE_DIR" | tee -a "$LOG_FILE"

echo "--- Step 01: Cohort construction ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m data_preparation.step_01_cohort_construction 2>&1 | tee -a "$LOG_FILE"

echo "--- Trimming cohort to $CASE_LIMIT cases ---" | tee -a "$LOG_FILE"
$PYTHON_BIN - <<PY 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

cohort_path = Path("$COHORT_PATH")
limit = $CASE_LIMIT
cohort_path.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(cohort_path)

# Ensure we have at least 2 positive cases (for train/test) and the rest negative
pos = df[df['aki_label'] == 1]
neg = df[df['aki_label'] == 0]

n_pos = min(len(pos), max(2, limit // 2))
n_neg = limit - n_pos

df_smoke = pd.concat([pos.head(n_pos), neg.head(n_neg)])
df_smoke = df_smoke.sample(frac=1, random_state=42).reset_index(drop=True)

df_smoke.to_csv(cohort_path, index=False)
print(f"Cohort trimmed to {len(df_smoke)} rows at {cohort_path} (Pos: {n_pos}, Neg: {n_neg})")
PY

echo "--- Step 02: Catch22 feature extraction ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m data_preparation.step_02_catch_22 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 03: Preoperative prep & split ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m data_preparation.step_03_preop_prep 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 04: Intraoperative prep ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m data_preparation.step_04_intraop_prep 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 05: Data merge ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m data_preparation.step_05_data_merge 2>&1 | tee -a "$LOG_FILE"

# === XGBoost Section ===
echo "--- Step 06: Hyperparameter search (XGBoost smoke) ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m model_creation.step_06_run_hpo \
    --outcome any_aki \
    --branch windowed \
    --feature_set all_waveforms \
    --model_type xgboost \
    --n_trials "$HPO_TRIALS" \
    --smoke_test 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 07: Train & evaluate (XGBoost smoke) ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m model_creation.step_07_train_evaluate \
    --outcome any_aki \
    --branch windowed \
    --feature_set all_waveforms \
    --model_type xgboost \
    --smoke_test 2>&1 | tee -a "$LOG_FILE"

# === EBM Section ===
echo "--- Step 06: Hyperparameter search (EBM smoke) ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m model_creation.step_06_run_hpo \
    --outcome any_aki \
    --branch windowed \
    --feature_set all_waveforms \
    --model_type ebm \
    --n_trials "$HPO_TRIALS" \
    --smoke_test 2>&1 | tee -a "$LOG_FILE"

echo "--- Step 07: Train & evaluate (EBM smoke) ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m model_creation.step_07_train_evaluate \
    --outcome any_aki \
    --branch windowed \
    --feature_set all_waveforms \
    --model_type ebm \
    --export_ebm_explanations \
    --smoke_test 2>&1 | tee -a "$LOG_FILE"

echo "--- Reporting: metrics summary ---" | tee -a "$LOG_FILE"
$PYTHON_BIN -m results_recreation.metrics_summary --results-dir "$SMOKE_RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "=== Smoke test (XGBoost + EBM) complete ===" | tee -a "$LOG_FILE"
echo "Artifacts written under $SMOKE_ROOT" | tee -a "$LOG_FILE"
