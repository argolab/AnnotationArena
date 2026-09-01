#!/bin/bash
# Does the transductive train-vs-corner gap survive making test items
# structurally identical to training items?
#
# Baseline (off) is the shipped construction: train_missing sits in the training
# graph as status-0 nodes, test_missing does not. With the flag on, test_missing
# is added the same way -- node positions only, no labels, no loss -- which is
# exactly the graph test.py builds at evaluation time.
#
# Goes through jobs/marformer_train.sbatch -> run_marformer_group.sh so the
# architecture matches the paper's runs. A direct `python -m
# imputer.entity_mf.train` invocation silently omits --use-pointer,
# --no-per-head-rel, --scale-shared-rel and --type-embedding-init kaiming, and
# the resulting model does not learn at all (it sits at chance, ~1.386).
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="tmi-01"; DATA_FROM="replicate-01"
SIZES="500,2000"; SEEDS="42"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
CONF_DIR="${RUN_DIR}/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
[[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${DATA_FROM}")/DATA" "${RUN_DIR}/DATA"

EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")
echo "run ${RUN_ID}: item T, sizes ${SIZES}, seeds ${SEEDS}, test_missing_in_graph off vs on"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for seed in "${SEED_LIST[@]}"; do
  for mode in off on; do
    tmig=false; [ "$mode" = "on" ] && tmig=true
    name="TMI-${mode}-s${seed}"
    conf="${CONF_DIR}/${name}.env"; : > "${conf}"
    for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=Marformer" \
              "AXIS=item" "PROTOCOL=transductive" "SIZES=${SIZES}" \
              "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=300" \
              "DOMAIN3_BATCH_AUGMENTATIONS=false" \
              "DOMAIN3_TEST_MISSING_IN_GRAPH=${tmig}" \
              "VARIANT_TAG=tmi_${mode}/seed${seed}"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
    done
    export RERUN_CONF="${conf}"
    args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
           -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
           --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
           -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
           "${RERUN_DIR}/jobs/marformer_train.sbatch" )
    if [[ "${DRY_RUN}" == true ]]; then echo "sbatch ${args[*]}";
    else echo "  ${mode} seed ${seed}: job $(sbatch --parsable "${args[@]}")"; fi
  done
done
