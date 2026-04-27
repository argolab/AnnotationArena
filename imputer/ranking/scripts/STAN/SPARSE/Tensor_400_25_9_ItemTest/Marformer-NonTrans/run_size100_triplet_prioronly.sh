#!/bin/bash

# WARNING — do not run training on the login node.
# Submit with: sbatch run_size100_triplet_prioronly.sh

set -euo pipefail

cd /home/xwang397/AA_new/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

DATA_ROOT="DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest/Tensor_400_25_9_ItemTest_100"
OUTPUT_ROOT="RESULTS/MARFORMER_CONT/STAN/SPARSE"
RUN_NAME="Tensor_400_25_9_ItemTest_100_NOITEMDEV_NONTRANS_TRIPLET_PRIORONLY_MARFORMER"

python -u -m imputer.entity_mf.train \
    --data-dir "${DATA_ROOT}" \
    --run-name "${RUN_NAME}" \
    --output-root "${OUTPUT_ROOT}" \
    --seed 42 \
    --embedding-dim 80 \
    --num-layers 8 \
    --attention-heads 4 \
    --d-ff 128 \
    --num-ffn-layers 1 \
    --dropout 0.1 \
    --item-dropout-rate 1.0 \
    --annotator-dropout-rate 0.0 \
    --epochs 500 \
    --lr 2e-4 \
    --lr-schedule none \
    --lr-min 1e-5 \
    --weight-decay 0.01 \
    --masking-rate 0.15 \
    --mask-augmentations 5 \
    --masked-loss-weight 15.0 \
    --observed-loss-weight 1.0 \
    --device cuda \
    --max-item 10 \
    --type-embedding-init kaiming \
    --item-reg-weight 0.0 \
    --attribute-reg-weight 0.0 \
    --annotator-reg-weight 0.0 \
    --use-triplet-rating-base \
    --triplet-mix-mode prior_only \
    --no-per-head-rel \
    --scale-shared-rel \
    --use-pointer \
    --overwrite-existing-data

