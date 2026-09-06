#!/bin/bash
#SBATCH --job-name=ARM-EVAL
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
# Corner-score the tail checkpoints of every run under one arm-sweep tree.
#
# Scoring the last few periodic checkpoints rather than only the final one is
# not extra work for its own sake: averaging them cuts the paired between-arm
# sd by ~1.8x overall and ~2.1x at K=300 (measured on resid-01, job 300293),
# which is worth roughly 4x the replicates -- for zero GPU. See s7_curves.py
# for the same machinery used to answer a different question.
#
#   AE_RUN = run id (e.g. s2item-s101)
#   AE_MI  = eval chunk size (default 500, per S4)
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
mkdir -p "${RERUN_DIR}/side/out"
python RERUN/side/s7_curves.py \
    --run-dir "${RERUN_DIR}/runs/${AE_RUN}" \
    --eval-max-item "${AE_MI:-500}" \
    --device cpu \
    -o "${RERUN_DIR}/side/out/arms_${AE_RUN}.json"
