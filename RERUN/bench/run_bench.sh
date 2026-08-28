#!/bin/bash
# Run one benchmark/profile of entity_mf training on the workbench node.
#
#   run_bench.sh [--profile] [--size 500] [--epochs 20] [--tag NAME] [extra train.py args...]
#
# Uses the replicate-01 Domain 3 splits and the exact hyperparameters from
# scripts/DOMAIN3/run_marformer_group.sh (item axis, transductive), so a
# speedup measured here transfers to the real runs.
set -euo pipefail

OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RANKING="${OPT_ROOT}/imputer/ranking"
DATA_ROOT="${OPT_ROOT}/RERUN/runs/replicate-01/DATA/STAN/DOMAIN3-SIGMOID"

PROFILE=0; SIZE=500; EPOCHS=20; TAG=""; MAX_ITEM=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE=1; shift ;;
        --size)    SIZE="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --max-item) MAX_ITEM="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
TAG="${TAG:-size${SIZE}_ep${EPOCHS}}"

OUT="${OPT_ROOT}/RERUN/bench/out/${TAG}"
rm -rf "${OUT}"; mkdir -p "${OUT}"

set +u
eval "$(conda shell.bash hook)"; conda activate torch
set -u
export PYTHONPATH="${RANKING}:${OPT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${RANKING}"

ARGS=(
  --data-dir "${DATA_ROOT}/ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_${SIZE}"
  --run-name "bench" --output-root "${OUT}"
  --seed 42 --embedding-dim 80 --num-layers 8 --attention-heads 4
  --d-ff 128 --num-ffn-layers 1 --dropout 0.1
  --item-dropout-rate 1.0 --annotator-dropout-rate 0.0
  --epochs "${EPOCHS}" --lr 2e-4 --lr-schedule none --lr-min 1e-5
  --weight-decay 0.01 --masking-rate 0.15 --mask-augmentations 5
  --masked-loss-weight 15.0 --observed-loss-weight 1.0
  --device cuda --max-item "${MAX_ITEM}" --type-embedding-init kaiming
  --item-reg-weight 0.0 --attribute-reg-weight 0.0 --annotator-reg-weight 0.0
  --transductive-learning --no-per-head-rel --scale-shared-rel --use-pointer
  --overwrite-existing-data
  "$@"
)

echo "node=$(hostname)  tag=${TAG}  size=${SIZE}  epochs=${EPOCHS}  profile=${PROFILE}"
START=$(date +%s%N)
if [[ "${PROFILE}" == "1" ]]; then
    python -u -m RERUN.bench.profile_train -- "${ARGS[@]}"
else
    python -u -m imputer.entity_mf.train "${ARGS[@]}"
fi
END=$(date +%s%N)
printf 'BENCH_WALL_SECONDS %s %d.%03d\n' "${TAG}" $(( (END-START)/1000000000 )) $(( ((END-START)/1000000)%1000 ))
python - "${OUT}/bench/training_history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
def ce(e):
    m = e.get("combined_eval", {}).get("metrics", {})
    return m.get("missing", {}).get("rating", {})
last = ce(h[-1])
print(f"BENCH_FINAL epochs={len(h)} train_masked_ce={h[-1].get('train_masked_ce'):.4f} "
      f"combined_missing_ce={last.get('xent', float('nan')):.4f} acc={last.get('acc', float('nan')):.4f}")
xs = [(ce(e).get("xent", float("inf")), e["epoch"]) for e in h]
best, best_ep = min(xs)
print(f"BENCH_BEST combined_missing_ce={best:.4f} at_epoch={best_ep}")
print("BENCH_CURVE " + " ".join(f"{e['epoch']}:{ce(e).get('xent', float('nan')):.4f}" for e in h))
PY
