#!/bin/bash
#SBATCH --job-name=tmi-eval
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110,gh113
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
: "${TMI_RUNS:?}"
ARGS=(); for r in ${TMI_RUNS}; do ARGS+=(--run-dir "$r"); done
python RERUN/fig6/tmi_compare.py "${ARGS[@]}" --device cuda -o "${TMI_OUT}"
