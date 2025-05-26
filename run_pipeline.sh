#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/ActiveRubric-Internal/src/activeLearner_noise.py --examples_per_cycle 30 --features_per_example 3 --experiment all --loss_type l2 --resample_validation --run_until_exhausted --dataset hanna --runner prabhav --llm_sigmas 0.6 --human_corruptions 0.3