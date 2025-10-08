#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

cd /export/fs06/psingh54/StanExps/imputer/ranking

export PYTHONPATH=.

# NORMALIZE, 4 FFN
python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4 --normalize-parameter

# NO NORMALIZE, 4 FFN 
python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4

python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4 --num_ffn_layers 8

python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4 --num_ffn_layers 8 --normalize-parameter

python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4 --num_ffn_layers 2

python imputer/run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/generated_data/run_20251007_011136 --epochs 300 --device cuda \
 --masking-rate 0.50 --transductive_learning --masked-loss-weight 15 --observed-loss-weight 1 --no-final-norm --mask-augmentations 5 --lr 5e-4 --num_ffn_layers 2 --normalize-parameter