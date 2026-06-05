#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_attacks.sh [process]
# Example:
#   ./run_all_attacks.sh 22nm_LP

PROCESS="${1:-22nm_LP}"
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

for RUN_TAG in baseline dp sl both; do
  echo ""
  echo "=== Running all attacks for ${PROCESS}_${RUN_TAG} ==="
  "$PYTHON_BIN" privacy_attack.py --process "$PROCESS" --run-tag "$RUN_TAG" --privacy_dir "$SCRIPT_DIR"
done

echo ""
echo "=== Comparing attacks across baseline/dp/sl/both ==="
"$PYTHON_BIN" compare_attacks.py --process "$PROCESS" --tags baseline dp sl both --privacy_dir "$SCRIPT_DIR" > "$SCRIPT_DIR/compare_attacks.txt"

echo ""
echo "All attacks completed for ${PROCESS} across baseline/dp/sl/both."
echo "Comparison report written to $SCRIPT_DIR/compare_attacks.txt"
