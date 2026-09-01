#!/bin/bash
#SBATCH --job-name=grid-eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110,gh113
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
python RERUN/fig6/grid_eval.py --run-dir "${GE_RUN}" \
    --eval-max-items "${GE_EVALS:-300,600,1200}" --device cuda -o "${GE_OUT}"
