#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
# Local — evaluate all saved checkpoints (best + periodic) for each
# Saves results to RESULTS/MARFORMER/STAN/SPARSE/<RUN_NAME>/TEST_RESULTS/<ckpt_stem>.json

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

RESULTS_ROOT="RESULTS/MARFORMER/STAN/SPARSE/TensorAnn"

echo ""
echo "============================================================"
echo " Tensor_300_30_9_AnnTest — Test Evaluation (all checkpoints)"
echo "============================================================"

for SIZE in 5 10 15 20; do
    RUN_NAME="Tensor_300_30_9_AnnTest_${SIZE}_NOITEMDEV_TRANS_MARFORMER"
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
