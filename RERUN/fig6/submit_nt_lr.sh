#!/bin/bash
# NT item axis at reduced learning rates.
#
# D25: the non-transductive arm fits the training cells as well as the
# transductive one (0.7659 vs 0.7664 at K=4000) but its corner log-loss reverses
# past K~900, bottoming near epoch 100 and climbing ~0.14 nats by epoch 400.
# Transductive training is anchored by having the corner's observed cells as
# targets; NT has no anchor. A smaller step is the cheapest lever to try before
# changing the protocol.
#
# Baseline lr is 2e-4. Everything else is held at the replication's values.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="nt-lr-01"; DATA_FROM="fig6-01"
SIZES="1000,2000,4000"; LRS="1e-4,5e-5"; SEEDS="42"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --lrs)   LRS="$2";   shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
RUN_DIR="$(rerun_run_dir "${RUN_ID}")"; CONF_DIR="${RUN_DIR}/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
[[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${DATA_FROM}")/DATA" "${RUN_DIR}/DATA"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")
echo "run ${RUN_ID}: NT item, sizes ${SIZES}, lrs ${LRS}, seeds ${SEEDS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a LR_LIST <<< "${LRS}"
for seed in "${SEED_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    tag="lr${lr}"
    name="NTLR-${tag}-s${seed}"
    conf="${CONF_DIR}/${name}.env"; : > "${conf}"
    for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=MarformerNT" \
              "AXIS=item" "PROTOCOL=nontransductive" "SIZES=${SIZES}" \
              "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=300" \
              "DOMAIN3_LR=${lr}" "DOMAIN3_BATCH_AUGMENTATIONS=false" \
              "VARIANT_TAG=${tag}/seed${seed}"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
    done
    export RERUN_CONF="${conf}"
    args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
           -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
           --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
           -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
           "${RERUN_DIR}/jobs/marformer_train.sbatch" )
    if [[ "${DRY_RUN}" == true ]]; then echo "sbatch ${args[*]}";
    else echo "  lr=${lr} seed=${seed}: job $(sbatch --parsable "${args[@]}")"; fi
  done
done
