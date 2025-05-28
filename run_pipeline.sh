#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=36G
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/ActiveRubric-Internal/src/activeLearnerNoisy.py --examples_per_cycle 50 --features_per_example 6 \
    --experiment all --loss_type l2 --resample_validation --run_until_exhausted --dataset hanna --runner prabhav \
    --use_embedding True --human_cost 3 --llm_cost 1 --llm_sigma 1 --human_noise 0.5 --cold_start True