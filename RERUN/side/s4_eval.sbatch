#!/bin/bash
#SBATCH --job-name=S4-chunk
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
python RERUN/fig6/grid_eval.py \
    --run-dir "${RERUN_DIR}/runs/seeds-01" \
    --eval-max-items "100,200,300,400,500,600" \
    --device cpu \
    -o "${RERUN_DIR}/side/out/s4_chunk_seeds01.json"
