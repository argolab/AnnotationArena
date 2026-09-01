#!/bin/bash
#SBATCH --job-name=tmi-full
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh104,gl110
set -euo pipefail
# Full-length (400 epoch) test of --test-missing-in-graph at two sizes, flag off
# vs on, then the train-vs-corner gap for each. 40 epochs was not enough: both
# arms sat at ~1.38, essentially chance.
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1
rerun_activate_env
cd "${RANKING_ROOT}"
OUT="${RERUN_DIR}/runs/tmi-01"
DATAROOT="${RERUN_DIR}/runs/replicate-01/DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive"
RUNS=""
for K in 500 2000; do
  for mode in off on; do
    flag=""; [ "$mode" = "on" ] && flag="--test-missing-in-graph"
    echo "=========== K=${K} test_missing_in_graph=${mode}"
    rerun_run python -m imputer.entity_mf.train \
        --data-dir "${DATAROOT}/DOMAIN3-SIGMOID_Item_T_${K}" \
        --output-root "${OUT}" --run-name "K${K}_${mode}" \
        --embedding-dim 80 --num-layers 8 --attention-heads 4 --d-ff 128 \
        --dropout 0.1 --epochs 400 --lr 2e-4 --weight-decay 0.01 \
        --masking-rate 0.15 --mask-augmentations 5 --max-item 300 \
        --masked-loss-weight 15 --observed-loss-weight 1 \
        --item-dropout-rate 1.0 --annotator-dropout-rate 0.0 \
        --transductive-learning --seed 42 --device cuda ${flag}
    RUNS="${RUNS} ${OUT}/K${K}_${mode}"
  done
done
ARGS=(); for r in ${RUNS}; do ARGS+=(--run-dir "$r"); done
python "${RERUN_DIR}/fig6/tmi_compare.py" "${ARGS[@]}" --device cuda \
    -o "${RERUN_DIR}/fig6/out/tmi_gap.json"
echo "TMI DONE"
