#!/bin/bash
# One LLM-Rubric training with the real jobs/llmrubric_train.sbatch settings,
# timed, with the final val-missing CE printed. Extra args are passed through.
#   bench_llmrubric.sh <checkout-root> <size> <epochs> <tag> [extra args...]
set -euo pipefail
ROOT="$1"; SIZE="$2"; EPOCHS="$3"; TAG="$4"; shift 4
RANKING="${ROOT}/imputer/ranking"
DATA="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/replicate-01/DATA/LLM_RUBRIC"
OUT="${ROOT}/RERUN/bench/out/lr_${TAG}"
rm -rf "${OUT}"
set +u; eval "$(conda shell.bash hook)"; conda activate torch; set -u
export PYTHONPATH="${RANKING}:${ROOT}"; cd "${RANKING}"
START=$(date +%s%N)
python -u -m imputer.entity_mf.train \
    --data-dir "${DATA}/LLMRubric_225_25_9_${SIZE}" --run-name run --output-root "${OUT}" \
    --seed 42 --embedding-dim 80 --num-layers 8 --attention-heads 4 --d-ff 128 \
    --num-ffn-layers 1 --dropout 0.1 --item-dropout-rate 0.7 --epochs "${EPOCHS}" \
    --lr-schedule none --lr-min 1e-5 --weight-decay 0.01 --masking-rate 0.15 \
    --mask-augmentations 5 --masked-loss-weight 15.0 --observed-loss-weight 1.0 \
    --device cuda --llm-annotator-id 24 --human-observed-rate 0.0 --max-item 10 \
    --type-embedding-init kaiming --item-reg-weight 0.0 --attribute-reg-weight 0.0 \
    --annotator-reg-weight 0.0 --no-per-head-rel --scale-shared-rel --use-pointer \
    --llm-input-dist --overwrite-existing-data "$@" >/dev/null
END=$(date +%s%N)
printf '%-22s wall=%d.%03ds  ' "${TAG}" $(( (END-START)/1000000000 )) $(( ((END-START)/1000000)%1000 ))
python - "${OUT}/run/training_history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
def ce(e):
    for k in ("val_eval", "combined_eval", "train_eval"):
        m = e.get(k, {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" in m:
            return m["xent"]
    return float("inf")
xs = [(ce(e), e["epoch"]) for e in h]
print(f"final={xs[-1][0]:.4f}  best={min(xs)[0]:.4f}@{min(xs)[1]}")
PY
