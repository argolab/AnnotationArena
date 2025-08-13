#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=24GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/AnnotationArena/imputer/graphical_synthetic_modeling/main.py --node-sizes 5 7 10 \
 --imputer-sizes Tiny Small Large --max-samples 2000 --test-samples 250 --start-examples 10 --increment 200 --missing-rates 0.5 --n-graphs 3