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

python /export/fs06/psingh54/ActiveRubric-Internal/src/post_training_prabhav.py \
--results_path /export/fs06/psingh54/ActiveRubric-Internal/outputs/results_enhanced_hanna/DynamicMasking_NewSelectionLoss/TEST1/enhanced_gradient_voi_q0_human_with_embedding.json \
--output_dir /export/fs06/psingh54/ActiveRubric-Internal/outputs/results_enhanced_hanna/PostTrainingAnalysis \
--dataset hanna \
--epochs 15 \
--batch_size 32 \
--lr 1e-5 \
--runner prabhav \
--use_embedding \
--num_patterns_per_example 3 \
--visible_ratio 0.5