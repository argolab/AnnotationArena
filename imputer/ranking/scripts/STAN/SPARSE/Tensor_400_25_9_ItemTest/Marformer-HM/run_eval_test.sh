#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
# Local — evaluate all saved checkpoints (best + periodic) for each
# Tensor_500_25_9_ItemTest Marformer run on the test split.
# Saves results to RESULTS/MARFORMER/STAN/SPARSE/<RUN_NAME>/TEST_RESULTS/<ckpt_stem>.json

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

RESULTS_ROOT="RESULTS/MARFORMER/STAN/SPARSE/Tensor"

echo ""
echo "============================================================"
echo " Tensor_400_25_9_ItemTest — Test Evaluation (all checkpoints)"
echo "============================================================"

for SIZE in 10 50 100 200 300; do
    RUN_NAME="Tensor_400_25_9_ItemTest_${SIZE}_NOITEMDEV_TRANS_MARFORMER-HM"
    RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"

    if [ ! -d "$RUN_DIR" ]; then
        echo "  [SKIP] ${RUN_NAME} — run dir not found"
        continue
    fi

    echo ""
    echo "--- Size ${SIZE} | ${RUN_NAME} ---"
    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR" \
        --checkpoint all        \
        --device     cpu
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All done. Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results saved under each run's TEST_RESULTS/<ckpt_stem>.json"
echo "============================================================"
