#!/bin/bash
# Generate the descriptive figures/tables (demographics, cohort flow, missingness)
# using the same processed data/paths as the main Catch22 + XGBoost run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Reuse the same dirs you used for experiments; fall back to project defaults.
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATA_DIR}/processed}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
DISPLAY_DICT="${DISPLAY_DICT:-${SCRIPT_DIR}/metadata/display_dictionary.json}"

# Core inputs (override via env if your run used different filenames)
COHORT_CSV="${COHORT_CSV:-${PROCESSED_DIR}/aki_pleth_ecg_co2_awp.csv}"
MERGED_DATASET="${MERGED_DATASET:-${PROCESSED_DIR}/aki_features_master_wide.csv}"
COUNTS_FILE="${COUNTS_FILE:-${RESULTS_DIR}/metadata/cohort_flow_counts.json}"
PROCESSED_PREOP="${PROCESSED_PREOP:-${PROCESSED_DIR}/aki_preop_processed.csv}"

for path in "$COHORT_CSV" "$MERGED_DATASET" "$DISPLAY_DICT" "$COUNTS_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path"
    echo "Override via env (COHORT_CSV/MERGED_DATASET/DISPLAY_DICT/COUNTS_FILE) or regenerate artifacts."
    exit 1
  fi
done

if [[ ! -f "$PROCESSED_PREOP" ]]; then
  echo "Warning: Processed preop dataset not found at $PROCESSED_PREOP; continuous stats will use raw cohort."
else
  echo "Using processed preop dataset for continuous features: $PROCESSED_PREOP"
fi

mkdir -p "${RESULTS_DIR}/tables" "${RESULTS_DIR}/figures"
export DISPLAY_DICTIONARY_PATH="$DISPLAY_DICT"

echo "[1/3] Generating preoperative descriptive table..."
"$PYTHON_BIN" -m reporting.preop_descriptives \
  --dataset "$COHORT_CSV" \
  --processed-dataset "$PROCESSED_PREOP" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-prefix preop_descriptives

echo "[2/3] Rendering cohort flow diagram..."
"$PYTHON_BIN" -m reporting.cohort_flow \
  --counts-file "$COUNTS_FILE" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-name cohort_flow

echo "[3/3] Computing missingness table..."
"$PYTHON_BIN" -m reporting.missingness_table \
  --dataset "$MERGED_DATASET" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-prefix missingness_table

echo "Done. Outputs:"
echo "  - Demographics:    ${RESULTS_DIR}/tables/preop_descriptives.(html|tex|docx)"
echo "  - Cohort flow:      ${RESULTS_DIR}/figures/cohort_flow.(svg|png)"
echo "  - Missingness:      ${RESULTS_DIR}/tables/missingness_table.(csv|html)"
