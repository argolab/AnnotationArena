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
SPLIT="LLMRubric_225_25_9_${SIZE}"

# Ablation knobs. Every default reproduces the original run exactly, so an
# existing caller that sets none of these is unaffected.
MAX_ITEM="${LLMRUBRIC_MAX_ITEM:-10}"
SEED="${LLMRUBRIC_SEED:-42}"
EPOCHS="${LLMRUBRIC_EPOCHS:-300}"

# VARIANT_TAG keeps arms and seeds on distinct paths. This matters more here
# than elsewhere: the block below *deletes* the target directory before
# training, so two arms sharing a path would not merely collide, the second
# would destroy the first. The unseeded default path is kept for the plain
# replication so existing plot globs still resolve.
VARIANT_TAG="${VARIANT_TAG:-}"
if [[ -n "${VARIANT_TAG}" ]]; then
    OUTPUT_ROOT="${RUN_DIR}/RESULTS/MARFORMER/LLM_RUBRIC/${VARIANT_TAG}"
else
    OUTPUT_ROOT="${RUN_DIR}/RESULTS/MARFORMER/LLM_RUBRIC"
fi

[[ -d "${DATA_ROOT}/${SPLIT}" ]] || {
    echo "FATAL: missing bundle ${DATA_ROOT}/${SPLIT}" >&2
    echo "Run jobs/make_llmrubric_data.sh --run-id ${RUN_ID} first." >&2
    exit 1
}

# Lightning versions rather than overwrites existing checkpoints, so a
# resubmission into a directory that already holds a (possibly cancelled) run
# leaves best-*.ckpt / last-v1.ckpt from both. test.py then picks
# sorted(glob("best-*.ckpt"))[0] -- the earliest, which may be from the dead
# run. Start from a clean directory instead.
TARGET="${OUTPUT_ROOT}/${SPLIT}"
if [[ -d "${TARGET}" ]]; then
    echo "removing previous run at ${TARGET}"
    rm -rf "${TARGET}"
fi

# Speed knobs, off by default. LLM-Rubric runs at max_item=10, so a bundle is
# ~23 small chunks and batching the mask augmentations is worth ~2.6x here --
# see RERUN/reports/TRAINING_SPEED.md. It is one optimizer step per epoch
# instead of five, so raise LLMRUBRIC_LR with it (1e-3 measured best).
BATCH_AUG_FLAG=""
[[ "${LLMRUBRIC_BATCH_AUGMENTATIONS:-false}" == "true" ]] && BATCH_AUG_FLAG="--batch-augmentations"
TF32_FLAG=""
[[ "${LLMRUBRIC_TF32:-false}" == "true" ]] && TF32_FLAG="--tf32"

echo "=============================================================="
echo " LLM-Rubric | Marformer | size=${SIZE}"
echo " LR     : ${LLMRUBRIC_LR:-2e-4}  batch_aug=${LLMRUBRIC_BATCH_AUGMENTATIONS:-false}"
echo " MAXITEM: ${MAX_ITEM}   SEED: ${SEED}   EPOCHS: ${EPOCHS}"
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
    --seed                 "${SEED}"               \
    --embedding-dim        80                      \
    --num-layers           8                       \
    --attention-heads      4                       \
    --d-ff                 128                     \
    --num-ffn-layers       1                       \
    --dropout              0.1                     \
    --item-dropout-rate    0.7                     \
    --epochs               "${EPOCHS}"            \
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
    --max-item             "${MAX_ITEM}"           \
    --type-embedding-init  kaiming                 \
    --item-reg-weight      0.0                     \
    --attribute-reg-weight 0.0                     \
    --annotator-reg-weight 0.0                     \
    --no-per-head-rel                              \
    --scale-shared-rel                             \
    --use-pointer                                  \
    --llm-input-dist                               \
    --lr                   "${LLMRUBRIC_LR:-2e-4}" \
    ${BATCH_AUG_FLAG}                              \
    ${TF32_FLAG}                                   \
    --overwrite-existing-data

ELAPSED=$(( SECONDS - START ))
echo
echo "=============================================================="
echo " size=${SIZE} done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================================="
