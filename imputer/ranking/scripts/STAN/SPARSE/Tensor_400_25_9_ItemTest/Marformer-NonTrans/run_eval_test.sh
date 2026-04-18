#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
#
# WARNING — do not run this script on the login node. It invokes Python eval
# over the full test split (many checkpoints if --checkpoint all). Submit with
# Slurm, e.g. `sbatch run_eval_test.sh`, or run from an interactive compute node.
#
# Local — evaluate all saved checkpoints (best + periodic) for each
# Saves results to RESULTS/MARFORMER_CONT/STAN/SPARSE/<RUN_NAME>/TEST_RESULTS/<ckpt_stem>.json

# imputer/ranking (five levels up from .../Marformer-NonTrans)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKING_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${RANKING_ROOT}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

# Non-trans runs saved under MARFORMER_CONT (see run_size100.sh OUTPUT_ROOT)
RESULTS_ROOT="RESULTS/MARFORMER_CONT/STAN/SPARSE"

echo ""
echo "============================================================"
echo " Tensor_400_25_9_ItemTest — Test Evaluation (all checkpoints)"
echo "  RESULTS_ROOT: ${RESULTS_ROOT}"
echo "============================================================"

for SIZE in 10 50 100 200 300; do
    RUN_NAME="Tensor_400_25_9_ItemTest_${SIZE}_NOITEMDEV_NONTRANS_MARFORMER"
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
