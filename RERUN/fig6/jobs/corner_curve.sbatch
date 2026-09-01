#!/bin/bash
#SBATCH --job-name=fig6-curve
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
: "${CURVE_RUNS:?}"
ARGS=(); for r in ${CURVE_RUNS}; do ARGS+=(--run-dir "$r"); done
python RERUN/fig6/corner_curve.py "${ARGS[@]}" --device cuda -o "${CURVE_OUT}"
