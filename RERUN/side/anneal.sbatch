#!/bin/bash
#SBATCH --job-name=ANNEAL
#SBATCH --comment=accept_cost
#SBATCH --gres=gpu:1
# Does a terminal learning-rate decay buy anything on top of the shipped
# constant 2e-4?
#
# The shipped Domain 3 config trains at a flat lr for its whole budget
# (train_config.json: "lr_schedule": "none"). Standard practice everywhere else
# is to anneal at the end, and the usual reason it helps is that a constant lr
# leaves the iterate bouncing in a basin it could descend further. If that is
# happening here it is free accuracy, and it would eat into the ~0.09 nats
# against Stan that no configuration change has explained.
#
# Rather than retrain, this resumes each FINISHED run's final checkpoint --
# weights AND AdamW moments, see train.py:on_train_start -- and runs a short
# decay phase. That is the D of warmup-stable-decay with the WS already paid
# for, so a question that would cost a full sweep costs ~10 GPU-minutes a cell.
#
# TWO ARMS PER SIZE, AND THE CONTROL IS THE POINT. `cont_const` continues at the
# flat lr for exactly the same number of extra epochs. Without it, "the annealed
# model is better than the shipped checkpoint" is unreadable: the anneal arm has
# also had 100 more epochs of training, and that alone moves the corner metric.
# Comparing arms that differ in step count at a fixed budget is the error that
# cost this project two retractions (S8, S9). Both arms here get identical
# steps, identical data, identical init; the only difference is the lr curve.
#
# ISOLATION. Output goes to runs/anneal-s101/RESULTS, a tree created for this
# and nothing else. The source checkpoints are COPIED to runs/anneal-s101/SRC
# first and read from there, so nothing under runs/legacy is opened for writing
# at all; SRC/source_checksums.txt is the baseline to verify that afterwards.
# train.py's _new_run_dir refuses to start if its run directory already exists,
# so a re-submission cannot silently overwrite a finished arm.
#
# Walltime: K=4000 took 2225 s for 400 epochs, so 100 epochs is ~560 s and the
# smaller sizes are faster. 40 min is ~4x margin on the slowest cell; nothing
# resumes, and short jobs backfill.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
ANNEAL_ROOT="${ANNEAL_ROOT:-/weka/scratch/jhu/jeisner1/xwang397/Marformer-anneal}"
source "${RERUN_DIR}/config/cluster.sh"

# Per-job settings arrive in a conf file, not via --export: sbatch's --export
# splits its value on commas, so any setting containing one is truncated. Every
# other launcher in RERUN/ uses this pattern for the same reason.
[[ -n "${RERUN_CONF:-}" && -f "${RERUN_CONF}" ]] && source "${RERUN_CONF}"

# The --init-from flag and the DOMAIN3_INIT_ROOT plumbing live only on the
# side/anneal branch. cluster.sh has already exported PYTHONPATH and
# DOMAIN3_REPO_ROOT_OVERRIDE pointing at the MAIN tree, and it is PYTHONPATH --
# not the cwd -- that decides which train.py is imported (job 315714 learned
# this the hard way on the S1 worktree). Override both.
export RANKING_ROOT="${ANNEAL_ROOT}/imputer/ranking"
export DOMAIN3_REPO_ROOT_OVERRIDE="${RANKING_ROOT}"
export PYTHONPATH="${RANKING_ROOT}"
cd "${ANNEAL_ROOT}"
rerun_activate_env

: "${AN_SIZE:?AN_SIZE required}" "${AN_ARM:?AN_ARM required}"
RUN_ID="${AN_RUN_ID:-anneal-s101}"
DATA_RUN="${AN_DATA_RUN:-dataseed-s101}"
RUN_DIR="$(rerun_run_dir "${RUN_ID}")"

export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${DATA_RUN}")"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"

# Everything below must match the source runs' train_config.json exactly, or the
# continuation is a different model and the comparison is meaningless. These are
# the shipped Domain 3 values; the group runner defaults to all of them, and
# they are restated here so a change upstream shows up as a diff rather than as
# a silently different experiment.
export DOMAIN3_SEED=42
export DOMAIN3_MAX_ITEM=300
export DOMAIN3_EMBEDDING_DIM=80 DOMAIN3_NUM_LAYERS=8 DOMAIN3_ATTENTION_HEADS=4
export DOMAIN3_D_FF=128 DOMAIN3_NUM_FFN_LAYERS=1 DOMAIN3_DROPOUT=0.1
export DOMAIN3_DUAL_OUTER_RESIDUAL=all DOMAIN3_INNER_RESIDUAL=both
export DOMAIN3_MASKED_LOSS_WEIGHT=15.0 DOMAIN3_OBSERVED_LOSS_WEIGHT=1.0
export DOMAIN3_WEIGHT_DECAY=0.01
export DOMAIN3_VAL_PROTOCOL=none

export DOMAIN3_EPOCHS="${AN_EPOCHS:-100}"
export DOMAIN3_LR="${AN_LR:-2e-4}"
export DOMAIN3_INIT_ROOT="${RUN_DIR}/SRC"
export DOMAIN3_INIT_CKPT="${AN_CKPT:-last.ckpt}"

case "${AN_ARM}" in
    anneal_cos)  export DOMAIN3_LR_SCHEDULE=cosine DOMAIN3_LR_MIN="${AN_LR_MIN:-1e-6}" ;;
    cont_const)  export DOMAIN3_LR_SCHEDULE=none ;;
    *) echo "unknown AN_ARM: ${AN_ARM}" >&2; exit 2 ;;
esac

OUT="$(rerun_results_root "${RUN_ID}")/MARFORMER/DOMAIN3-SIGMOID/ITEM/${AN_ARM}/Marformer/seed${DOMAIN3_SEED}"
mkdir -p "${OUT}"

echo "=============================================================="
echo " anneal | K=${AN_SIZE} | arm=${AN_ARM}"
echo " epochs=${DOMAIN3_EPOCHS} lr=${DOMAIN3_LR} sched=${DOMAIN3_LR_SCHEDULE}"
echo " init  : ${DOMAIN3_INIT_ROOT}/<run>/checkpoints/${DOMAIN3_INIT_CKPT}"
echo " data  : ${DOMAIN3_DATA_ROOT}"
echo " out   : ${OUT}"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh" \
    --family-label Marformer \
    --output-root  "${OUT}" \
    --axis         item \
    --protocol     transductive \
    --sizes        "${AN_SIZE}"
