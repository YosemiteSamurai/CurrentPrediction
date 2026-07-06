#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_compare_attacks.sh [process]
# Example:
#   ./run_compare_attacks.sh 22nm_LP

PROCESS="${1:-22nm_LP}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="$SCRIPT_DIR/inputs"
mkdir -p "$INPUTS_DIR"
OUT_FILE="$INPUTS_DIR/compare_attacks.txt"  # original: OUT_FILE="$SCRIPT_DIR/compare_attacks.txt"

CONDA_ENV_PY="/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python"
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
"$PYTHON_BIN" compare_attacks.py \
  --process "$PROCESS" \
  --tags baseline dp sl both \
  --privacy_dir "$INPUTS_DIR" | tee "$OUT_FILE"

echo ""
echo "Wrote: $OUT_FILE"
