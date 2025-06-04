#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --gpus=1
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/ActiveRubric-Internal/src/activeLearnerNoisy.py --examples_per_cycle 50 --features_per_example 10 \
    --experiment all --loss_type l2 --resample_validation --run_until_exhausted \
    --dataset hanna --runner prabhav --use_embedding True --human_cost 2 --llm_cost 1 \
    --llm_alpha_multiplier 0.4 --human_flip_prob 0.7 --cold_start True \
    --validation_set_size 100 --active_set_size 200