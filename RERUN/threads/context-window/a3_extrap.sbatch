#!/bin/bash
#SBATCH --job-name=A3-EXTRAP
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
# Score every A3 arm at a range of EVAL chunk sizes. The question is whether a
# model trained on a range of chunk sizes generalises to chunks larger than any
# it saw; a model trained at a fixed 300 or 500 has no reason to.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
mkdir -p "${RERUN_DIR}/threads/context-window/out"
for RUN in ctx-a3-s101 ctx-a3-s105; do
  for MI in 300 500 700 900 1200; do
    OUT="${RERUN_DIR}/threads/context-window/out/a3_${RUN}_mi${MI}.json"
    [ -s "${OUT}" ] && { echo "skip ${RUN} mi=${MI}"; continue; }
    echo "=== ${RUN} eval chunk ${MI}"
    python RERUN/side/s7_curves.py --run-dir "${RERUN_DIR}/runs/${RUN}" \
        --eval-max-item "${MI}" --device cpu -o "${OUT}"
  done
done
