#!/bin/bash
#SBATCH --job-name=DEVTEMP
#SBATCH --partition=h100,h200,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=8:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gl110,gh104
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/devtemp_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/devtemp_%j.err
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${RERUN_DIR}"
exec python analysis/dev_temperature_scaling.py "$@"
