#!/bin/bash
# Primary experiment launcher for Catch22-based XGBoost/EBM models
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Catch22 + XGBoost/EBM experiment grid..."
"${SCRIPT_DIR}/run_experiments.sh" "$@"
