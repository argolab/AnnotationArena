#!/bin/bash
# Push the item axis to K=10000 and ask whether transfer saturates.
#
# The whole curve 1000..10000 is trained INSIDE this tree, because the 10k base
# is a fresh draw with its own corner (items 10101..10200). A dot here is not
# comparable with a dot in fig6-01; only the shape within this tree is.
#
# Training chunk 600 -- the 3-seed grid showed chunk size, not shuffling, is the
# lever that moves the corner loss, and 600 is its best measured value. Chunking
# is what keeps memory flat as K grows: the chunk is 600 items regardless, so
# K=10000 costs more STEPS, not more memory per step.
#
#   ./submit_train.sh [--sizes ...] [--arms ...] [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="pilot-10k"; SIZES="1000,2000,3000,4000,6000,8000,10000"
ARMS="Marformer,MarformerNT"; SEEDS="42"; MAX_ITEM=600; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --max-item) MAX_ITEM="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
CONF_DIR="$(rerun_run_dir "${RUN_ID}")/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")
IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
count=0
for arm in "${ARM_LIST[@]}"; do
    proto=transductive; [[ "${arm}" == "MarformerNT" ]] && proto=nontransductive
    for seed in "${SEED_LIST[@]}"; do
        # Tag by chunk size: the first pass ran at 600 and hit the memory
        # cliff above K=4000, so a 300 pass has to coexist with it rather than
        # overwrite the sizes that did finish.
        variant="mi${MAX_ITEM}"
        [[ "${seed}" != "42" ]] && variant="${variant}/seed${seed}"
        name="P10K-${arm}-s${seed}"
        conf="${CONF_DIR}/${name}.env"; : > "${conf}"
        for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=${arm}" \
                  "AXIS=item" "PROTOCOL=${proto}" "SIZES=${SIZES}" \
                  "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=${MAX_ITEM}" \
                  "DOMAIN3_RANDOM_ITEM_CHUNKS=false" \
                  "DOMAIN3_BATCH_AUGMENTATIONS=false" "VARIANT_TAG=${variant}"; do
            printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
        done
        export RERUN_CONF="${conf}"
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem=180G
               -t "24:00:00" ${EXCLUDE_ARG} --array="0-$((n-1))"
               "${RERUN_DIR}/jobs/marformer_train.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then echo "  ${name}: (dry) ${n} tasks"
        else echo "  ${name}: job $(sbatch --parsable "${args[@]}")  (${n} tasks)"; fi
        count=$((count+1))
    done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) runs"
