#!/bin/bash
# Run the validation protocol end to end for one Domain 3 run.
#
#   ./submit_validation.sh --run-id val-01 --data-from replicate-01 \
#       --protocol block --seeds 42,43,44 [--stage search|refit|eval] \
#       [--search-epochs 2000] [--select per-run|per-size] [--smooth 5]
#
# Three phases, chained by SLURM dependency:
#
#   search  train with part of the training partition carved off as validation
#           (imputer/entity_mf/validation.py), for the full epoch budget, so the
#           whole validation curve is available. Results land under
#           <AXIS>/<variant>-search/seed<N>/.
#           --search-epochs sets that budget. It is a CAP, not a target: the
#           carve stops wherever it wants inside it, and the point of raising it
#           is only that it must not bind. W3d measured selected epochs up to
#           940 at K=4000, so the shipped 400 binds hard at large K; use 2000.
#           Default 400 for backwards compatibility with existing runs.
#   select  read those curves, take the argmin per run, write the epoch table.
#           Runs as a short CPU job so it can be a dependency of the refit.
#           --select per-run gives each run its own epoch (recommended at
#           K >= 1000, where the carve is large enough to have real signal);
#           per-size pools a size's seeds, which is safer on small carves.
#           Both smooth the curve first; --smooth 1 disables.
#   refit   retrain on the WHOLE training partition -- no carve -- for exactly
#           the selected number of epochs. The final model is the last
#           checkpoint, so nothing is ever selected on test. Results land under
#           <AXIS>/<variant>/seed<N>/.
#
# The refit's data budget equals Stan's, which is the point of refitting rather
# than shipping the early-stopped search model.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/cluster.sh"

EXCLUDE_ARG=""
[[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"

RUN_ID=""; DATA_FROM=""; PROTOCOL="block"; SEEDS="42,43,44"
STAGE="all"; VAL_FRAC="0.15"; VAL_SPLIT_SEED="7"; DRY_RUN=false
SEARCH_EPOCHS=""; SELECT_MODE=""; SMOOTH=""

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)     RUN_ID="$2"; shift 2 ;;
        --data-from)  DATA_FROM="$2"; shift 2 ;;
        --protocol)   PROTOCOL="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --stage)      STAGE="$2"; shift 2 ;;
        --val-frac)   VAL_FRAC="$2"; shift 2 ;;
        --val-split-seed) VAL_SPLIT_SEED="$2"; shift 2 ;;
        --search-epochs)  SEARCH_EPOCHS="$2"; shift 2 ;;
        --select)         SELECT_MODE="$2"; shift 2 ;;
        --smooth)         SMOOTH="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done
[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; usage 1; }
case "${PROTOCOL}" in block|mcar) ;; *) echo "bad --protocol: ${PROTOCOL}" >&2; exit 1 ;; esac
VARIANT="val${PROTOCOL}"

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
LOG_DIR="$(rerun_log_root "${RUN_ID}")"
CONF_DIR="${RUN_DIR}/conf"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CONF_DIR}" "$(rerun_results_root "${RUN_ID}")"
EPOCH_TABLE="${RUN_DIR}/${VARIANT}_epochs.json"

if [[ -n "${DATA_FROM}" && ! -e "${RUN_DIR}/DATA" ]]; then
    SRC="$(rerun_run_dir "${DATA_FROM}")/DATA"
    [[ -d "${SRC}" ]] || { echo "no DATA under run '${DATA_FROM}'" >&2; exit 1; }
    ln -sfn "${SRC}" "${RUN_DIR}/DATA"
fi

# Default sweep. VAL_MATRIX overrides it with a newline-separated list of
# family|axis|protocol|sizes rows, so a follow-up can run the protocol on a few
# cells (e.g. the extended item-NT sizes) without re-running the whole grid.
if [[ -n "${VAL_MATRIX:-}" ]]; then
    mapfile -t MARFORMER_MATRIX <<< "${VAL_MATRIX}"
else
MARFORMER_MATRIX=(
  "Marformer|item|transductive|100,200,300,400,500,600,700,800,900,1000,2000,3000,4000"
  "Marformer|annot|transductive|10,15,20,25,30,35,40,45,50"
  "MarformerNT|item|nontransductive|100,200,300,400,500,600,700,800,900,1000,2000,3000,4000"
  "MarformerNT|annot|nontransductive|10,15,20,25,30,35,40"
)
fi

write_conf() {
    local name="$1"; shift
    local path="${CONF_DIR}/${name}.env"
    : > "${path}"
    local kv
    for kv in "$@"; do printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${path}"; done
    echo "${path}"
}
submit() {
    local name="$1" conf="$2"; shift 2
    export RERUN_CONF="${conf}"
    local -a args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
                    -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
                    --export=ALL "$@" )
    if [[ "${DRY_RUN}" == true ]]; then echo "sbatch ${args[*]}" >&2; echo "DRYRUN"; return; fi
    sbatch --parsable "${args[@]}"
}
count_sizes() { local IFS=','; read -r -a a <<< "$1"; echo "${#a[@]}"; }
dep() { [[ -n "$1" && "$1" != "DRYRUN" ]] && echo "--dependency=afterok:$1" || echo ""; }

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
SEARCH_JOBS=""

# ------------------------------------------------------------------ search
if [[ "${STAGE}" == "all" || "${STAGE}" == "search" ]]; then
    for seed in "${SEED_LIST[@]}"; do
        for row in "${MARFORMER_MATRIX[@]}"; do
            IFS='|' read -r family axis protocol sizes <<< "${row}"
            n="$(count_sizes "${sizes}")"
            name="VAL-${VARIANT}-S-${family}-${axis}-s${seed}"
            conf="$(write_conf "${name}" "RUN_ID=${RUN_ID}" "FAMILY=${family}" \
                       "AXIS=${axis}" "PROTOCOL=${protocol}" "SIZES=${sizes}" \
                       "DOMAIN3_SEED=${seed}" "VARIANT_TAG=${VARIANT}-search/seed${seed}" \
                       "DOMAIN3_VAL_PROTOCOL=${PROTOCOL}" "DOMAIN3_VAL_FRAC=${VAL_FRAC}" \
                       "DOMAIN3_VAL_SPLIT_SEED=${VAL_SPLIT_SEED}" \
                       ${SEARCH_EPOCHS:+"DOMAIN3_EPOCHS=${SEARCH_EPOCHS}"} \
                       "DATA_RUN_ID=${RUN_ID}")"
            jid="$(submit "${name}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                          --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                          --array="0-$((n-1))" "${RERUN_DIR}/jobs/marformer_train.sbatch")"
            echo "search : ${name} (${n} tasks) job ${jid}"
            SEARCH_JOBS="${SEARCH_JOBS:+${SEARCH_JOBS}:}${jid}"
        done
    done
fi

# ------------------------------------------------------------------ select
SELECT_JOB=""
if [[ "${STAGE}" == "all" || "${STAGE}" == "select" ]]; then
    conf="$(write_conf "VAL-${VARIANT}-SELECT" "RUN_ID=${RUN_ID}" "VARIANT=${VARIANT}" \
               ${SELECT_MODE:+"SELECT_MODE=${SELECT_MODE}"} \
               ${SMOOTH:+"SELECT_SMOOTH=${SMOOTH}"} \
               "EPOCH_TABLE=${EPOCH_TABLE}")"
    SELECT_JOB="$(submit "VAL-${VARIANT}-SELECT" "${conf}" -p "${CPU_PARTITION}" -c 2 \
                  --mem=8G -t 00:30:00 $(dep "${SEARCH_JOBS}") \
                  "${RERUN_DIR}/jobs/select_epochs.sbatch")"
    echo "select : job ${SELECT_JOB}  -> ${EPOCH_TABLE}"
fi

# ------------------------------------------------------------------- refit
REFIT_JOBS=""
if [[ "${STAGE}" == "all" || "${STAGE}" == "refit" ]]; then
    for seed in "${SEED_LIST[@]}"; do
        for row in "${MARFORMER_MATRIX[@]}"; do
            IFS='|' read -r family axis protocol sizes <<< "${row}"
            n="$(count_sizes "${sizes}")"
            name="VAL-${VARIANT}-R-${family}-${axis}-s${seed}"
            conf="$(write_conf "${name}" "RUN_ID=${RUN_ID}" "FAMILY=${family}" \
                       "AXIS=${axis}" "PROTOCOL=${protocol}" "SIZES=${sizes}" \
                       "DOMAIN3_SEED=${seed}" "VARIANT_TAG=${VARIANT}/seed${seed}" \
                       "DOMAIN3_VAL_PROTOCOL=none" "EPOCH_TABLE=${EPOCH_TABLE}" \
                       "DATA_RUN_ID=${RUN_ID}")"
            jid="$(submit "${name}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                          --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                          --array="0-$((n-1))" $(dep "${SELECT_JOB}") \
                          "${RERUN_DIR}/jobs/marformer_train.sbatch")"
            echo "refit  : ${name} (${n} tasks) job ${jid}"
            REFIT_JOBS="${REFIT_JOBS:+${REFIT_JOBS}:}${jid}"
        done
    done
fi

# -------------------------------------------------------------------- eval
if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
    conf="$(write_conf "VAL-${VARIANT}-EVAL" "RUN_ID=${RUN_ID}")"
    jid="$(submit "VAL-${VARIANT}-EVAL" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                  --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                  $(dep "${REFIT_JOBS}") "${RERUN_DIR}/jobs/tree_eval.sbatch")"
    echo "eval   : job ${jid}"
fi

echo
echo "run dir     : ${RUN_DIR}"
echo "epoch table : ${EPOCH_TABLE}"
