#!/bin/bash
#SBATCH --job-name=capacity
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110,gh113
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
python RERUN/fig6/capacity_probe.py --run-dir "${CP_RUN}" \
    --max-items "${CP_MAXITEMS:-300,600,1200,2400,3000,5000}" -o "${CP_OUT}"
