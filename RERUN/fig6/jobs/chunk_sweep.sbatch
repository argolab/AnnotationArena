#!/bin/bash
#SBATCH --job-name=chunk-sweep
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110,gh113
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
python RERUN/fig6/chunk_sweep_eval.py --run-dir "${CS_RUN}" \
    --max-items "${CS_MAXITEMS:-300,600,1200,2400}" --device cuda -o "${CS_OUT}"
