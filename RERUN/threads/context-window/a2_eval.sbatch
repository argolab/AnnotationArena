#!/bin/bash
#SBATCH --job-name=CTX-A2-EVAL
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
# A2: score every training-chunk arm at EVERY evaluation chunk.
#
# The whole point of the matrix is that training chunk and evaluation chunk are
# separate knobs, and the September sweep found each model does best near the
# chunk it was trained at. Scoring each arm only at 500 would therefore hand the
# 500-trained arm an advantage that has nothing to do with training context. So
# every arm is read at every chunk and each is reported at its own best.
#
#   A2_RUN = run id (e.g. ctx-a2-s101)
#   A2_CHUNKS = comma-separated evaluation chunks
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
OUT_DIR="${RERUN_DIR}/threads/context-window/out"
mkdir -p "${OUT_DIR}"
IFS=',' read -r -a CHUNKS <<< "${A2_CHUNKS:-300,500,600,700,900}"
for c in "${CHUNKS[@]}"; do
    echo "=== ${A2_RUN} at evaluation chunk ${c} ==="
    python RERUN/side/s7_curves.py \
        --run-dir "${RERUN_DIR}/runs/${A2_RUN}" \
        --eval-max-item "${c}" \
        --device cpu \
        -o "${OUT_DIR}/arms_${A2_RUN}_mi${c}.json"
done
echo "done"
