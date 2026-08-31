#!/bin/bash
#SBATCH --job-name=fig6-corner
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110
set -euo pipefail

# Score every trained Domain 3 run on the fixed test corner alone, which is what
# the paper says the reported log-loss is (main.tex:1269 and the Fig. 6 caption)
# and what test.py does not do (D22). Also emits the pooled and training-cell
# numbers from the same forward pass, so the published value stays reconstructible.
#
# Read-only with respect to the run trees: writes a single JSON per run id.

RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env

: "${TARGET_RUNS:?TARGET_RUNS must be set (space separated run ids)}"
CKPT="${CORNER_CKPT:-last}"

cd "$(dirname "${RERUN_DIR}")"
for rid in ${TARGET_RUNS}; do
    echo "=== ${rid} (checkpoint=${CKPT})"
    python RERUN/analysis/split_missing_metrics.py \
        --run-dir "RERUN/runs/${rid}" \
        --checkpoint "${CKPT}" \
        --device cuda \
        -o "RERUN/fig6/out/corner_${rid}_${CKPT}.json"
done
echo "corner evaluation complete"
