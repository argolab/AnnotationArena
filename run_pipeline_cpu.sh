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

python src/analysis/post_rmse_analysis.py --model_path /export/fs06/psingh54/ActiveRubric-Internal/src/output/models/15_CYCLES_DM-3-0.5_50-Examples-5-Features_Comparision_gradient_voi_q0_human_20250625_141037.pth