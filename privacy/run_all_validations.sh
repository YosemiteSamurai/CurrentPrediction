#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_validations.sh [process]
# Example:
#   ./run_all_validations.sh 22nm_LP
#
# This validates all four run tags: baseline, dp, sl, both.

PROCESS="${1:-22nm_LP}"
DATASET="dataset_${PROCESS}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV_PY="/nfs/stak/users/renya/jonesm25/.conda/envs/currentprediction/bin/python"
PYTHON_BIN=""

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate currentprediction >/dev/null 2>&1 || true
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "$CONDA_ENV_PY" ]]; then
    PYTHON_BIN="$CONDA_ENV_PY"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

cd "$SCRIPT_DIR"

echo "Using Python: $PYTHON_BIN"
echo "Validating dataset: $DATASET"

for RUN_TAG in baseline dp sl both; do
  echo ""
  echo "=== Validating ${PROCESS}_${RUN_TAG} ==="
  "$PYTHON_BIN" validate_privacy_artifacts.py \
    --dataset "$DATASET" \
    --tag "$RUN_TAG" \
    --privacy-dir "$SCRIPT_DIR"
done

echo ""
echo "All validations completed for ${PROCESS} across baseline/dp/sl/both."
