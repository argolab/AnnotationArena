#!/bin/bash
# Train the item axis on each freshly drawn dataset.
#
# Held fixed on purpose so the ONLY thing varying across runs is the generative
# draw: max_item 300, no shuffle, 400 epochs, model seed 42, no validation
# carve, batch_augmentations off. Those are the values behind every number in
# the corrected Figure 6, so the gap measured here is directly comparable with
# the +0.078 measured on base seed 42.
#
# Sizes stop at 1000 (Tom: "feel free to just train things at K=1000 for speed,
# which might allow for a few more data seeds"). Three sizes give the curve a
# shape; the gap is read at each of them.
#
#   ./submit_train.sh [--seeds 42,101,...] [--sizes 200,500,1000] [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

SEEDS="42,101,102,103,104,105"; SIZES="200,500,1000"
ARMS="Marformer,MarformerNT"; MODEL_SEED=42; MAX_ITEM=300; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --arms)  ARMS="$2";  shift 2 ;;
        --max-item) MAX_ITEM="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
n=$(awk -F, '{print NF}' <<< "${SIZES}")
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a ARM_LIST  <<< "${ARMS}"
count=0
for dseed in "${SEED_LIST[@]}"; do
    RUN_ID="dataseed-s${dseed}"
    DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
    if [[ ! -d "${DATA_ROOT}/ItemSplits" ]]; then
        echo "SKIP seed ${dseed}: no data at ${DATA_ROOT} (generation not finished?)" >&2
        continue
    fi
    CONF_DIR="$(rerun_run_dir "${RUN_ID}")/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
    mkdir -p "${CONF_DIR}" "${LOG_DIR}"
    for arm in "${ARM_LIST[@]}"; do
        proto=transductive; [[ "${arm}" == "MarformerNT" ]] && proto=nontransductive
        name="DSEED-${arm}-s${dseed}"
        conf="${CONF_DIR}/${name}.env"; : > "${conf}"
        # Each data seed already has its own run tree, so VARIANT_TAG stays empty
        # and the results land on the canonical paths inside that tree. (The
        # seed-collision bug that bit submit_nt_extend.sh came from several seeds
        # sharing ONE tree; that cannot happen here.)
        for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=${arm}" \
                  "AXIS=item" "PROTOCOL=${proto}" "SIZES=${SIZES}" \
                  "DOMAIN3_SEED=${MODEL_SEED}" "DOMAIN3_MAX_ITEM=${MAX_ITEM}" \
                  "DOMAIN3_RANDOM_ITEM_CHUNKS=false" \
                  "DOMAIN3_BATCH_AUGMENTATIONS=false" "VARIANT_TAG="; do
            printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
        done
        export RERUN_CONF="${conf}"
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
               -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
               "${RERUN_DIR}/jobs/marformer_train.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then echo "  ${name}: (dry) ${n} tasks"
        else echo "  ${name}: job $(sbatch --parsable "${args[@]}")  (${n} tasks)"; fi
        count=$((count+1))
    done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) runs"
