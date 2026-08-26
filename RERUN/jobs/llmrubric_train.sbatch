#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#
# Marformer on the real LLM-Rubric data, one array task per training size.
#
# Hyperparameters are those of the original run
# (AA_new/.../scripts/LLM_RUBRIC/MARFORMER/TRAIN/run_train.sh). They differ from
# the Domain 3 synthetic settings in several places, so they are spelled out
# here rather than delegated to scripts/DOMAIN3/run_marformer_group.sh:
#   epochs 300 (not 400), max_item 10 (not 300), item_dropout_rate 0.7,
#   annotator_reg_weight 0, plus the real-data-only flags --llm-annotator-id 24
#   (the GPT judge rates every item, so it is always observed) and
#   --llm-input-dist.
# The original script ran only sizes 10/50/100/150/175; the paper's figures
# plot all ten, so all ten are swept here.
set -euo pipefail

if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"
fi

if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi

# This is a --gres=gpu job; a silent CPU fallback would be a bug, not a fallback.
export REQUIRE_GPU=1
rerun_activate_env

: "${RUN_ID:?RUN_ID is required}"
SIZES="${SIZES:-10,20,30,40,50,75,100,125,150,175}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
DATA_ROOT="${RUN_DIR}/DATA/LLM_RUBRIC"
OUTPUT_ROOT="${RUN_DIR}/RESULTS/MARFORMER/LLM_RUBRIC"
SPLIT="LLMRubric_225_25_9_${SIZE}"

[[ -d "${DATA_ROOT}/${SPLIT}" ]] || {
    echo "FATAL: missing bundle ${DATA_ROOT}/${SPLIT}" >&2
    echo "Run jobs/make_llmrubric_data.sh --run-id ${RUN_ID} first." >&2
    exit 1
}

echo "=============================================================="
echo " LLM-Rubric | Marformer | size=${SIZE}"
echo " DATA   : ${DATA_ROOT}/${SPLIT}"
echo " OUTPUT : ${OUTPUT_ROOT}/${SPLIT}"
echo "=============================================================="

cd "${RANKING_ROOT}"
export PYTHONUNBUFFERED=1
START=${SECONDS}

rerun_run python -u -m imputer.entity_mf.train \
    --data-dir             "${DATA_ROOT}/${SPLIT}" \
    --run-name             "${SPLIT}"              \
    --output-root          "${OUTPUT_ROOT}"        \
    --seed                 42                      \
    --embedding-dim        80                      \
    --num-layers           8                       \
    --attention-heads      4                       \
    --d-ff                 128                     \
    --num-ffn-layers       1                       \
    --dropout              0.1                     \
    --item-dropout-rate    0.7                     \
    --epochs               300                     \
    --lr                   2e-4                    \
    --lr-schedule          none                    \
    --lr-min               1e-5                    \
    --weight-decay         0.01                    \
    --masking-rate         0.15                    \
    --mask-augmentations   5                       \
    --masked-loss-weight   15.0                    \
    --observed-loss-weight 1.0                     \
    --device               cuda                    \
    --llm-annotator-id     24                      \
    --human-observed-rate  0.0                     \
    --max-item             10                      \
    --type-embedding-init  kaiming                 \
    --item-reg-weight      0.0                     \
    --attribute-reg-weight 0.0                     \
    --annotator-reg-weight 0.0                     \
    --no-per-head-rel                              \
    --scale-shared-rel                             \
    --use-pointer                                  \
    --llm-input-dist                               \
    --overwrite-existing-data

ELAPSED=$(( SECONDS - START ))
echo
echo "=============================================================="
echo " size=${SIZE} done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================================="
