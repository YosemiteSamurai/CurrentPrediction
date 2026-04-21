#!/bin/bash
# check_datasets.sh -- Quick check for presence and size of dataset files
# Usage: bash scripts/check_datasets.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

for ds in dataset/dataset*.csv dataset/dataset*.json; do
    if [ -f "$ds" ]; then
        echo -n "$ds: "
        wc -l < "$ds"
    else
        echo "$ds: MISSING"
    fi
done
