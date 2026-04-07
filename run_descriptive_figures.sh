#!/bin/bash
# Generate the descriptive figures/tables (demographics, cohort flow, missingness)
# using the same processed data/paths as the main Catch22 + XGBoost run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/artifact_paths.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
  echo "PYTHON_BIN must resolve to at least one command token."
  exit 1
fi

run_python() {
  "${PYTHON_CMD[@]}" "$@"
}

aki_configure_catch22_env "$SCRIPT_DIR" "descriptive_figures.log"
aki_enforce_generated_paths \
  "processed_dir::${PROCESSED_DIR}" \
  "results_dir::${RESULTS_DIR}" \
  "paper_dir::${PAPER_DIR}" \
  "log_file::${LOG_FILE}"
mkdir -p "${PAPER_DIR}/tables" "${PAPER_DIR}/figures" "${PAPER_DIR}/metadata" "$(dirname "${LOG_FILE}")"
aki_refresh_repo_symlinks "$SCRIPT_DIR"

DISPLAY_DICT="${DISPLAY_DICT:-${SCRIPT_DIR}/metadata/display_dictionary.json}"

# Core inputs (override via env if your run used different filenames)
COHORT_CSV="${COHORT_CSV:-${PROCESSED_DIR}/aki_pleth_ecg_co2_awp.csv}"
MERGED_DATASET="${MERGED_DATASET:-${PROCESSED_DIR}/aki_features_master_wide.csv}"
COUNTS_FILE="${COUNTS_FILE:-${PAPER_DIR}/metadata/cohort_flow_counts.json}"
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

export DISPLAY_DICTIONARY_PATH="$DISPLAY_DICT" PAPER_DIR RESULTS_DIR

echo "[1/3] Generating preoperative descriptive table..."
run_python -m reporting.preop_descriptives \
  --dataset "$COHORT_CSV" \
  --processed-dataset "$PROCESSED_PREOP" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-prefix preop_descriptives

echo "[2/3] Rendering cohort flow diagram..."
run_python -m reporting.cohort_flow \
  --counts-file "$COUNTS_FILE" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-name cohort_flow

echo "[3/3] Computing missingness table..."
run_python -m reporting.missingness_table \
  --dataset "$MERGED_DATASET" \
  --display-dictionary "$DISPLAY_DICT" \
  --output-prefix missingness_table

echo "Done. Outputs:"
echo "  - Demographics:    ${PAPER_DIR}/tables/preop_descriptives.(html|tex|docx)"
echo "  - Cohort flow:     ${PAPER_DIR}/figures/cohort_flow.(svg|png)"
echo "  - Missingness:     ${PAPER_DIR}/tables/missingness_table.(csv|html)"
