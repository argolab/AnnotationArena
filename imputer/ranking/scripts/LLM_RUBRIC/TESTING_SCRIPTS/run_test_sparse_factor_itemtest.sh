#!/bin/bash

# LOCAL — run test evaluation (best checkpoint) on all Sparse Factor ItemTest runs.
# Skips any run directory that doesn't exist yet.
# Usage: bash scripts/TESTING_SCRIPTS/run_test_sparse_factor_itemtest.sh

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

DEVICE="cpu"

SCRIPT_START=$SECONDS

# ── Run directories ───────────────────────────────────────────────────────────
RUN_DIRS=(
    "RESULTS/MARFORMER/STAN/SPARSE/Factor_225_25_9_ItemTest_10_LOCAL_NOITEMDEV_TRANS"
    "RESULTS/MARFORMER/STAN/SPARSE/Factor_225_25_9_ItemTest_20_LOCAL_NOITEMDEV_TRANS"
    "RESULTS/MARFORMER/STAN/SPARSE/Factor_225_25_9_ItemTest_50_LOCAL_NOITEMDEV_TRANS"
    "RESULTS/MARFORMER/STAN/SPARSE/Factor_225_25_9_ItemTest_100_LOCAL_NOITEMDEV_TRANS"
    "RESULTS/MARFORMER/STAN/SPARSE/Factor_225_25_9_ItemTest_Size_175_LOCAL_NOITEMDEV_TRANS"
)

echo ""
echo "============================================================"
echo " Sparse Factor ItemTest — test evaluation (best checkpoint)"
echo " Device: ${DEVICE}"
echo "============================================================"

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [ ! -d "$RUN_DIR" ]; then
        echo ""; echo "  [skip] ${RUN_DIR}  (not found)"; continue
    fi
    RUN_START=$SECONDS
    echo ""; echo "--- ${RUN_DIR} ---"; echo ""

    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR"  \
        --checkpoint best        \
        --device     "$DEVICE"

    RUN_ELAPSED=$(( SECONDS - RUN_START ))
    echo "  ↳ done in $(( RUN_ELAPSED / 60 ))m $(( RUN_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All done. Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "============================================================"
