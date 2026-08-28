#!/bin/bash
# Run the non-item-axis training configurations short, on one checkout, and
# print wall clock plus the loss curve. Used to confirm the optimised code
# behaves like the reference on the paths the item-axis benchmark never hits:
# the annotator axis (entity dropout actually on, deviation regularisation) and
# LLM-Rubric (structured masking, soft rating_dist targets).
#
#   sanity_configs.sh <checkout-root> <epochs> [extra train.py args...]
set -euo pipefail

ROOT="$1"; EPOCHS="$2"; shift 2
RANKING="${ROOT}/imputer/ranking"
DATA="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/replicate-01/DATA"
OUT="${ROOT}/RERUN/bench/out/sanity"

set +u; eval "$(conda shell.bash hook)"; conda activate torch; set -u
export PYTHONPATH="${RANKING}:${ROOT}"
cd "${RANKING}"

run() {
    local name="$1"; shift
    rm -rf "${OUT}/${name}"
    local start=$(date +%s%N)
    python -u -m imputer.entity_mf.train --run-name run --output-root "${OUT}/${name}" \
        --epochs "${EPOCHS}" --device cuda --seed 42 --overwrite-existing-data "$@" >/dev/null
    local end=$(date +%s%N)
    printf '%-14s wall=%d.%03ds  ' "${name}" $(( (end-start)/1000000000 )) $(( ((end-start)/1000000)%1000 ))
    python - "${OUT}/${name}/run/training_history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
def ce(e):
    for key in ("combined_eval", "val_eval", "train_eval"):
        m = e.get(key, {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" in m:
            return m["xent"]
    return float("nan")
print(f"final_ce={ce(h[-1]):.4f}  curve=" + ",".join(f"{ce(e):.3f}" for e in h[::max(1, len(h)//8)]))
PY
}

COMMON=(--embedding-dim 80 --num-layers 8 --attention-heads 4 --d-ff 128
        --num-ffn-layers 1 --dropout 0.1 --lr 2e-4 --lr-schedule none --lr-min 1e-5
        --weight-decay 0.01 --masking-rate 0.15 --mask-augmentations 5
        --masked-loss-weight 15.0 --observed-loss-weight 1.0
        --type-embedding-init kaiming --no-per-head-rel --scale-shared-rel --use-pointer)

run annot_T_30 \
    --data-dir "${DATA}/STAN/DOMAIN3-SIGMOID/AnnotSplits/Transductive/DOMAIN3-SIGMOID_Annot_T_30" \
    "${COMMON[@]}" --max-item 300 --transductive-learning \
    --item-dropout-rate 0.0 --annotator-dropout-rate 0.8 \
    --annotator-reg-weight 1e-3 --item-reg-weight 0.0 --attribute-reg-weight 0.0 "$@"

run llmrubric_50 \
    --data-dir "${DATA}/LLM_RUBRIC/LLMRubric_225_25_9_50" \
    "${COMMON[@]}" --max-item 10 \
    --item-dropout-rate 0.7 --annotator-dropout-rate 0.0 \
    --llm-annotator-id 24 --human-observed-rate 0.0 --llm-input-dist "$@"
