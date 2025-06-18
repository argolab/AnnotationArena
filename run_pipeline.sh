#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/ActiveRubric-Internal/src/activeLearnerFixed.py --examples_per_cycle 50 \
 --experiment comparison --loss_type l2 --resample_validation --run_until_exhausted \
 --dataset hanna --runner prabhav --use_embedding True --cold_start True \
 --validation_set_size 50 --active_set_size 100 --epochs_per_cycle 10 \
 --train_option dynamic_masking --gradient_top_only True --num_patterns_per_example 5 \
 --visible_ratio 0.5 --features_per_example 5 --output_path DynamicMasking_NewSelectionLoss


 #### Use if Need A100 #SBATCH --account=a100acct