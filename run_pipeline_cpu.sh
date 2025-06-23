#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ActiveLearner
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/ActiveRubric-Internal/src/post_analysis.py \
 --model_path /export/fs06/psingh54/ActiveRubric-Internal/src/output/models/15_CYCLES_DM-3-0.5_50-Examples-5-Features_G-VOI-Q0_20250622_023141.pth \
 --experiment_name 15_CYCLES_DM-3-0.5_50-Examples-5-Features_G-VOI-Q0 --dataset hanna --runner prabhav --log_level INFO --attention_samples 200