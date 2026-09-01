#!/bin/bash
#SBATCH --job-name=tmi-smoke
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110
set -euo pipefail
# Smoke test for --test-missing-in-graph: 40 epochs at K=500, flag off then on.
# Full output is kept -- an earlier version piped through grep|head, which under
# `set -o pipefail` turned a plain argparse error into a silent 14s failure.
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1
rerun_activate_env
cd "${RANKING_ROOT}"

DATA="${RERUN_DIR}/runs/replicate-01/DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_500"
OUT="${RERUN_DIR}/runs/tmi-smoke"
for mode in off on; do
    flag=""
    [ "$mode" = "on" ] && flag="--test-missing-in-graph"
    echo "=================== test_missing_in_graph=${mode}"
    rerun_run python -m imputer.entity_mf.train \
        --data-dir "${DATA}" --output-root "${OUT}" --run-name "tmi_${mode}" \
        --embedding-dim 80 --num-layers 8 --attention-heads 4 --d-ff 128 \
        --dropout 0.1 --epochs 40 --lr 2e-4 --weight-decay 0.01 \
        --masking-rate 0.15 --mask-augmentations 5 --max-item 300 \
        --masked-loss-weight 15 --observed-loss-weight 1 \
        --item-dropout-rate 1.0 --annotator-dropout-rate 0.0 \
        --transductive-learning --seed 42 --device cuda ${flag}
done
echo "SMOKE DONE"
