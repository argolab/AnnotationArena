#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
# Local — evaluate best-val checkpoint of each DawidSkene_225_25_9_ItemTest
# Marformer run on the test split.
# Saves results to RESULTS/MARFORMER/STAN/SPARSE/<RUN_NAME>/TEST_RESULTS/best.json

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

RESULTS_ROOT="RESULTS/MARFORMER/STAN/SPARSE"

echo ""
echo "============================================================"
echo " DawidSkene_225_25_9_ItemTest — Test Evaluation (best ckpt)"
echo "============================================================"

for SIZE in 10 30 50 75 100 150 175 200; do
    RUN_NAME="DawidSkene_225_25_9_ItemTest_${SIZE}_CLUSTER_NOITEMDEV_TRANS"
    RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"

    if [ ! -d "$RUN_DIR" ]; then
        echo "  [SKIP] ${RUN_NAME} — run dir not found"
        continue
    fi

    echo ""
    echo "--- Size ${SIZE} | ${RUN_NAME} ---"
    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR" \
        --checkpoint best       \
        --device     cpu
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All done. Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results saved under each run's TEST_RESULTS/best.json"
echo "============================================================"
