#!/bin/bash
#SBATCH --job-name=CALCURV
#SBATCH --partition=l40s,h100,a100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gl110,gh104
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/calcurv_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/calcurv_%j.err
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${RERUN_DIR}"
exec python analysis/calibration_curves.py "$@"
