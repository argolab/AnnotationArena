#!/bin/bash
#SBATCH --job-name=d22split
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gh104,gl110
set -euo pipefail
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source config/cluster.sh
rerun_activate_env
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer
python RERUN/analysis/split_missing_metrics.py \
    --run-dir RERUN/runs/replicate-01 --device cuda \
    -o RERUN/_d22/split.json
