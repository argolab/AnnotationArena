#!/bin/bash
#
# Standalone trainer launcher for:
#   z_ijk = exp(u_i + v_j) · e_k
#
# Usage:
#   bash scripts/standalone/run_exp_uv_dot_e_small_transductive.sh

set -euo pipefail

cd /home/xwang397/AA_new/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

DATA_DIR="DATA/STAN/SPARSE/MARFORMER_NOBIN/Tensor_400_25_9_ItemTest_10"
OUT_ROOT="RESULTS/standalone_exp_uv_dot_e"
RUN_NAME="tensor400_size10_exp_uplusv_dot_e"

python -u scripts/standalone/train_exp_uv_dot_e.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --embed-dim 32 \
  --epochs 200 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --batch-size 4096 \
  --device cuda \
  --seed 42
