#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# The default is a GPU. A CPU-partition submission clears it with --gres=none on
# the sbatch command line, which overrides this header; see side/submit_arms.sh.
#SBATCH --comment=accept_cost
set -euo pipefail

# Train one Marformer per array task. Submitted once per (axis, protocol);
# SLURM_ARRAY_TASK_ID indexes into SIZES.
#
# Required env (set by submit_all.sh):
#   RUN_ID    run label
#   FAMILY    Marformer | MarformerNT      (only affects labels/metadata)
#   AXIS      item | annot
#   PROTOCOL  transductive | nontransductive
#   SIZES     comma-separated training sizes, one per array task
#
# Architecture and optimisation hyperparameters are NOT set here -- they live in
# scripts/DOMAIN3/run_marformer_group.sh (D=80, L=8, H=4, d_ff=128, AdamW
# lr=2e-4, wd=0.01, masking 15%, K=5 augmentations, masked loss weight 15,
# 400 epochs). Change them there so training and the paper stay in sync.

# Under sbatch the script is copied to /var/spool, so BASH_SOURCE no longer
# points into the repo. submit_all.sh sources config/cluster.sh (which exports
# RERUN_DIR) before submitting, and --export=ALL carries it in; fall back to
# path resolution only when this script is run directly.
if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"
fi

# Job parameters arrive in a file, not via --export (see write_conf in
# submit_all.sh: --export mangles comma-separated size lists and spaces).
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
# A silent CPU fallback on a GPU node is a bug (it makes a 40-minute run take
# hours and nobody notices), so a missing GPU is fatal by default. But Domain 3
# training is CPU-bound -- the GPU sits ~90% idle while Python builds the entity
# graph -- so running deliberately on a CPU partition is a legitimate throughput
# choice when the GPU queues are deep. REQUIRE_GPU=0 selects that on purpose.
export REQUIRE_GPU="${REQUIRE_GPU:-1}"
# Large training chunks sit right at the memory ceiling: max_item=1200 at K=4000
# needs ~76 GiB with backward and died 3.5 GiB short on an 80 GiB A100.
# expandable_segments reclaims fragmentation rather than reserving around it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rerun_activate_env

: "${RUN_ID:?}" "${FAMILY:?}" "${AXIS:?}" "${PROTOCOL:?}" "${SIZES:?}"

IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

AXIS_DIR="$(echo "${AXIS}" | tr '[:lower:]' '[:upper:]')"
if [[ "${PROTOCOL}" == "transductive" ]]; then
    RESULTS_SUBDIR="MARFORMER"
else
    RESULTS_SUBDIR="MARFORMER-NT"
fi
# VARIANT_TAG keeps ablation arms from colliding in the results tree. Empty for
# the plain replication, so those paths stay exactly what the plot scripts glob.
VARIANT_TAG="${VARIANT_TAG:-}"
if [[ -n "${VARIANT_TAG}" ]]; then
    OUTPUT_ROOT="$(rerun_results_root "${RUN_ID}")/${RESULTS_SUBDIR}/DOMAIN3-SIGMOID/${AXIS_DIR}/${VARIANT_TAG}"
else
    OUTPUT_ROOT="$(rerun_results_root "${RUN_ID}")/${RESULTS_SUBDIR}/DOMAIN3-SIGMOID/${AXIS_DIR}"
fi

# Point the upstream runner at this run's data and results.
export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
# DATA_RUN_ID lets an ablation reuse an existing run's generated data instead of
# regenerating it, so every arm is compared on byte-identical splits.
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${DATA_RUN_ID:-${RUN_ID}}")"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"
# FINAL RUN DEFAULTS. These are no longer the paper's values -- every one of
# them is a decision from the September sweep and the post-meeting threads, and
# an ablation that wants the old behaviour must now set the variable explicitly.
#
#   max_item 500        thread 1: 500 beats 300 by 0.016 nats at K=4000, measured
#                       twice on independent scoring machinery. Train AND eval.
#   outer residual none } thread 3, together: the standard Transformer block,
#   block proj     none } h' = h + attn + ff. -0.021 nats at K=2000. Removing
#                       either one ALONE is worse -- they only cancel as a pair.
#   value codec dual    REVERTED 2026-10-06. The typed codec is what makes the
#                       model slow and erratic at leaving the uniform plateau:
#                       it escapes at a median epoch 80 with a spread of 50-310
#                       (sd 77), against 28-40 (sd 3.2) for the hand-built
#                       stream across 42 cells and both block variants. When the
#                       selected epoch lands below the escape epoch the refit
#                       reports the uniform distribution -- 8 of 42 cells in the
#                       first final run. Collapses: 9 of 72 cells with the codec,
#                       0 of 42 without. The codec bought parity at best (B10),
#                       so there is nothing on the other side of the ledger.
export DOMAIN3_MAX_ITEM="${DOMAIN3_MAX_ITEM:-500}"
export DOMAIN3_DUAL_OUTER_RESIDUAL="${DOMAIN3_DUAL_OUTER_RESIDUAL:-none}"
export DOMAIN3_INNER_RESIDUAL="${DOMAIN3_INNER_RESIDUAL:-both}"
export DOMAIN3_BLOCK_OUTPUT_PROJECTION="${DOMAIN3_BLOCK_OUTPUT_PROJECTION:-none}"
export DOMAIN3_LOGIT_HIGH="${DOMAIN3_LOGIT_HIGH:-20.0}"
export DOMAIN3_VALUE_CODEC="${DOMAIN3_VALUE_CODEC:-dual}"
export DOMAIN3_CODEC_INIT="${DOMAIN3_CODEC_INIT:-unit}"
export DOMAIN3_CODEC_FREEZE="${DOMAIN3_CODEC_FREEZE:-false}"
export DOMAIN3_TEST_MISSING_IN_GRAPH="${DOMAIN3_TEST_MISSING_IN_GRAPH:-false}"
export DOMAIN3_RANDOM_ITEM_CHUNKS="${DOMAIN3_RANDOM_ITEM_CHUNKS:-false}"
# Architecture knobs for the capacity sweep. Unset, these are the shipped
# Domain 3 values, so every existing job is unaffected.
export DOMAIN3_EMBEDDING_DIM="${DOMAIN3_EMBEDDING_DIM:-80}"
export DOMAIN3_NUM_LAYERS="${DOMAIN3_NUM_LAYERS:-8}"
export DOMAIN3_ATTENTION_HEADS="${DOMAIN3_ATTENTION_HEADS:-4}"
export DOMAIN3_D_FF="${DOMAIN3_D_FF:-128}"
export DOMAIN3_NUM_FFN_LAYERS="${DOMAIN3_NUM_FFN_LAYERS:-1}"
export DOMAIN3_DROPOUT="${DOMAIN3_DROPOUT:-0.1}"
export DOMAIN3_SEED="${DOMAIN3_SEED:-42}"
# Validation carve (phase A) and, for phase B, a per-(size, seed) epoch budget
# looked up from the selection file written by analysis/select_epochs.py.
# S2 (delta_tau) and S5 (loss-weight normalisation) knobs. Unset => the runner
# keeps its shipped per-axis defaults, so passing these through cannot change
# any existing arm. See RERUN/FUTURE_EXPERIMENTS.md.
export DOMAIN3_ITEM_DROPOUT="${DOMAIN3_ITEM_DROPOUT:-}"
export DOMAIN3_ANNOT_DROPOUT="${DOMAIN3_ANNOT_DROPOUT:-}"
export DOMAIN3_MASKED_LOSS_WEIGHT="${DOMAIN3_MASKED_LOSS_WEIGHT:-15.0}"
export DOMAIN3_OBSERVED_LOSS_WEIGHT="${DOMAIN3_OBSERVED_LOSS_WEIGHT:-1.0}"
export DOMAIN3_LR="${DOMAIN3_LR:-2e-4}"
export DOMAIN3_LR_SCHEDULE="${DOMAIN3_LR_SCHEDULE:-none}"
# L2. The runner reads DOMAIN3_WEIGHT_DECAY, but until 2026-10-06 there was no
# export line here, so a conf setting it was silently dropped and every arm
# trained at the 0.01 default -- the same trap the block comment below records
# for DOMAIN3_BATCH_AUGMENTATIONS. The default reproduces the shipped value.
export DOMAIN3_WEIGHT_DECAY="${DOMAIN3_WEIGHT_DECAY:-0.01}"

# Speed/optimisation knobs. run_marformer_group.sh reads these, but the per-job
# conf is *sourced* rather than exported, so without these lines a conf setting
# DOMAIN3_BATCH_AUGMENTATIONS=true was silently ignored and the run used the
# sequential default -- the same failure DOMAIN3_EPOCHS had. Defaults preserve
# the shipped behaviour exactly.
export DOMAIN3_BATCH_AUGMENTATIONS="${DOMAIN3_BATCH_AUGMENTATIONS:-false}"
export DOMAIN3_TF32="${DOMAIN3_TF32:-false}"

# The per-job conf is *sourced*, which sets shell variables but does not export
# them, so anything the runner (a child process) needs must be re-exported here.
# DOMAIN3_EPOCHS was previously exported only inside the EPOCH_TABLE branch,
# which meant a conf setting it was silently ignored and the run took the
# runner's 400-epoch default.
[[ -n "${DOMAIN3_EPOCHS:-}" ]] && export DOMAIN3_EPOCHS

export DOMAIN3_VAL_PROTOCOL="${DOMAIN3_VAL_PROTOCOL:-none}"
export DOMAIN3_VAL_FRAC="${DOMAIN3_VAL_FRAC:-0.15}"
export DOMAIN3_VAL_SPLIT_SEED="${DOMAIN3_VAL_SPLIT_SEED:-7}"
if [[ -n "${EPOCH_TABLE:-}" ]]; then
    KEY="${FAMILY}|${AXIS}|${SIZE}|${DOMAIN3_SEED}"
    LOOKUP="$(python - "${EPOCH_TABLE}" "${KEY}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["epochs"].get(sys.argv[2], ""))
PY
)"
    [[ -n "${LOOKUP}" ]] || { echo "FATAL: no epoch budget for ${KEY} in ${EPOCH_TABLE}" >&2; exit 1; }
    export DOMAIN3_EPOCHS="${LOOKUP}"
    echo " EPOCHS (validated)  : ${DOMAIN3_EPOCHS}  [${KEY}]"
fi
mkdir -p "${OUTPUT_ROOT}"

# Note: run_marformer_group.sh also tees its own log into imputer/ranking/logs/
# (it sets DOMAIN3_LOG_DIR itself). The authoritative per-job log is the
# --output file that submit_all.sh points into this run's logs/ directory.

echo "=============================================================="
echo " ${FAMILY} | axis=${AXIS} | protocol=${PROTOCOL} | size=${SIZE}"
echo " RUN_ID      : ${RUN_ID}"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo " DATA_ROOT   : ${DOMAIN3_DATA_ROOT}"
echo " MAX_ITEM    : ${DOMAIN3_MAX_ITEM}"
echo " RESIDUAL    : ${DOMAIN3_DUAL_OUTER_RESIDUAL}"
echo " INNER RESID : ${DOMAIN3_INNER_RESIDUAL}"
echo " BLOCK PROJ  : ${DOMAIN3_BLOCK_OUTPUT_PROJECTION}"
echo " VALUE CODEC : ${DOMAIN3_VALUE_CODEC} (init ${DOMAIN3_CODEC_INIT}, freeze ${DOMAIN3_CODEC_FREEZE})"
echo " SEED        : ${DOMAIN3_SEED}"
echo " VAL         : ${DOMAIN3_VAL_PROTOCOL}"
echo " DELTA_TAU   : item=${DOMAIN3_ITEM_DROPOUT:-<axis default>} annot=${DOMAIN3_ANNOT_DROPOUT:-<axis default>}"
echo " LOSS W      : masked=${DOMAIN3_MASKED_LOSS_WEIGHT} observed=${DOMAIN3_OBSERVED_LOSS_WEIGHT}"
echo " LR          : ${DOMAIN3_LR} (${DOMAIN3_LR_SCHEDULE})"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh" \
    --family-label "${FAMILY}"      \
    --output-root  "${OUTPUT_ROOT}" \
    --axis         "${AXIS}"        \
    --protocol     "${PROTOCOL}"    \
    --sizes        "${SIZE}"
