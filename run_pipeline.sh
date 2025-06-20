#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

wandb login

python /export/fs06/psingh54/ActiveRubric-Internal/src/activeLearner.py \
    --examples_per_cycle 50 \
    --experiment gradient_voi_q0_human \
    --loss_type cross_entropy \
    --resample_validation \
    --cycles 15 \
    --dataset hanna \
    --runner prabhav \
    --use_embedding True \
    --cold_start True \
    --validation_set_size 50 \
    --active_set_size 100 \
    --epochs_per_cycle 5 \
    --train_option dynamic_masking \
    --gradient_top_only True \
    --num_patterns_per_example 3 \
    --visible_ratio 0.5 \
    --features_per_example 5 \
    --experiment_name TestExperiment1 \
    --log_level INFO \
    --use_wandb \
    --wandb_project active-learning-hanna \
    --wandb_entity prabhavsingh55221-johns-hopkins-university
