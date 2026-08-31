#!/bin/bash
# D1: the paper says the item graph is chunked at 500 per forward pass; the code
# trains at 300 and only *evaluates* at 500 (test.py hardcode, D20). So the
# stated protocol has never actually been run on the training side.
#
# This trains the item axis at max_item=500 on the sizes where chunking binds
# (> 300), so the footnote's "the gap is small" can be checked instead of
# assumed. Evaluation matches training, via split_missing_metrics.py reading
# max_item back from train_config.json.
#
# Separate run id, separate results tree: this must not be mistaken for, or
# mixed into, the figure's dots, which are all at 300.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="chunk500-01"; DATA_FROM="replicate-01"
SIZES="500,1000,2000,4000"; SEEDS="42,43,44"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
CONF_DIR="${RUN_DIR}/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
# Reuse replicate-01's bundles so this differs from the baseline in exactly one
# knob.
if [[ ! -e "${RUN_DIR}/DATA" ]]; then
    ln -sfn "$(rerun_run_dir "${DATA_FROM}")/DATA" "${RUN_DIR}/DATA"
fi

EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")
echo "run ${RUN_ID}: item T, max_item=500, sizes ${SIZES}, seeds ${SEEDS}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for seed in "${SEED_LIST[@]}"; do
    name="CHUNK500-item-s${seed}"
    conf="${CONF_DIR}/${name}.env"; : > "${conf}"
    for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=Marformer" \
              "AXIS=item" "PROTOCOL=transductive" "SIZES=${SIZES}" \
              "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=500" \
              "DOMAIN3_BATCH_AUGMENTATIONS=false" "VARIANT_TAG=chunk500/seed${seed}"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
    done
    export RERUN_CONF="${conf}"
    args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
           -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
           --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
           -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
           "${RERUN_DIR}/jobs/marformer_train.sbatch" )
    if [[ "${DRY_RUN}" == true ]]; then echo "sbatch ${args[*]}";
    else echo "seed ${seed}: job $(sbatch --parsable "${args[@]}")"; fi
done
