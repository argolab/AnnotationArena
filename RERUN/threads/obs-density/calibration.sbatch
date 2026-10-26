#!/bin/bash
#SBATCH --job-name=OBSDEN-CAL
#SBATCH --partition=med
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --comment=accept_cost
set -euo pipefail
# 24G not 120G: med has MaxMemPerCPU=4000, so a big --mem is secretly a big
# core request and the task will not schedule. See PLAN.md.
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python RERUN/threads/obs-density/calibration.py \
    --device cpu --gate -o "${RERUN_DIR}/threads/obs-density/out/calibration.json"
