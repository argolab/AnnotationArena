#!/bin/bash

# WARNING — do not run training on the login node.
# Submit with: sbatch run_size100_triplet_anneal.sh

#SBATCH --job-name=TEN_100_TRIPLET_ANN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=a100
#SBATCH --exclude=c001
#SBATCH --time=06:00:00

set -euo pipefail

cd /home/xwang397/AA_new/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

DATA_ROOT="DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest/Tensor_400_25_9_ItemTest_100"
OUTPUT_ROOT="RESULTS/MARFORMER_CONT/STAN/SPARSE"
RUN_NAME="Tensor_400_25_9_ItemTest_100_NOITEMDEV_NONTRANS_TRIPLET_ANNEAL_MARFORMER"

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
    --triplet-mix-mode anneal_to_average \
    --triplet-anneal-start-epoch 0 \
    --triplet-anneal-end-epoch 200 \
    --triplet-transformer-final-weight 0.5 \
    --triplet-prior-final-weight 0.5 \
    --no-per-head-rel \
    --scale-shared-rel \
    --use-pointer \
    --overwrite-existing-data

