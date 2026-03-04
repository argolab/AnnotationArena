#!/bin/bash
# For all IMPUTER runs whose name starts with easy_axis_, rebuild predictives
# (train_predictives.json, test_predictives.json) and run JK diagnostics.
# Run from repo root.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

IMPUTER_ROOT="${REPO_ROOT}/OUTPUT/IMPUTER"
GENERATED_ROOT="${REPO_ROOT}/OUTPUT/generated_data"

if [ ! -d "$IMPUTER_ROOT" ]; then
    echo "ERROR: OUTPUT/IMPUTER not found at $IMPUTER_ROOT"
    exit 1
fi

count=0
skipped=0

for run_dir in "$IMPUTER_ROOT"/easy_axis_*; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    model_pt="${run_dir}/model.pt"
    data_dir="${GENERATED_ROOT}/${run_name}"
    bundle="${data_dir}/data_bundle.json"

    if [ ! -f "$model_pt" ]; then
        echo "SKIP $run_name (no model.pt)"
        ((skipped++)) || true
        continue
    fi
    if [ ! -f "$bundle" ]; then
        echo "SKIP $run_name (no data_bundle.json at $data_dir)"
        ((skipped++)) || true
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Processing: $run_name"
    echo "=============================================="

    echo "[1/2] Rebuilding predictives..."
    python utils/evaluate_checkpoint.py \
        --model-path "$model_pt" \
        --data-dir "$data_dir" \
        --output-dir "$run_dir" || { echo "WARNING: evaluate_checkpoint failed for $run_name"; continue; }

    echo "[2/2] Running JK diagnostics (train_missing)..."
    python utils/jk_diagnostics.py \
        --data-bundle "$bundle" \
        --imputer-predictives "$run_dir" \
        --slice train_missing || { echo "WARNING: jk_diagnostics failed for $run_name"; continue; }

    echo "Done: $run_name"
    ((count++)) || true
done

echo ""
echo "=============================================="
echo "Completed: $count run(s) processed, $skipped skipped"
echo "=============================================="
