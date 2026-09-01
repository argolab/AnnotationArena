#!/bin/bash
#SBATCH --job-name=gap-anat
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
python RERUN/fig6/gap_anatomy.py --run-dir "${ANAT_RUN}" --device cuda -o "${ANAT_OUT}"
