#!/bin/bash
#SBATCH --job-name=TAB200
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/tab200_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/tab200_%j.err
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${RERUN_DIR}"
python analysis/tables_fixed_budget.py --device cuda
