#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

cd /export/fs06/psingh54/StanExps/imputer/ranking/imputer

python run_imputer.py --data-dir /export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/random_folder --epochs 70 --transductive_learning