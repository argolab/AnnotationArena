#!/bin/bash
#SBATCH --job-name=SEL-DECOMP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
# Score every periodic checkpoint of the published K=4000 run ON THE CORNER, so
# the two evaluation bugs can be separated.
#
# The published point is an argmin over these same checkpoints taken on the
# POOLED test metric. Scoring each of them on the corner tells us what the
# corner loss was at the checkpoint that argmin picked -- which is the only way
# to say what the selection bug costs on its own, holding the metric fixed.
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
mkdir -p "${RERUN_DIR}/fig6/out"
rerun_run python "${RERUN_DIR}/fig6/corner_curve.py" \
    --run-dir "$(rerun_run_dir replicate-01)/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_4000_MARFORMER" \
    --device "${EVAL_DEVICE:-cuda}" \
    -o "${RERUN_DIR}/fig6/out/selection_decomp_K4000.json"
