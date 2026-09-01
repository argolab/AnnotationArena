#!/bin/bash
# Chunking grid: does shuffling before chunking, and a bigger training chunk,
# help the non-transductive arm build a representation for unseen items?
#
# Tom's argument: the point of Marformer NT is to assemble a representation for
# a test item out of related variables it CAN see. Fixed contiguous chunks give
# every item the same neighbours for all 400 epochs; shuffling gives it a fresh
# neighbourhood each step, and a larger chunk gives it more of them at once.
#
# Axes
#   shuffle      --random-item-chunks off / on   (on also disables chunk caching,
#                                                 so chunks are rebuilt per step)
#   train chunk  DOMAIN3_MAX_ITEM 300 / 600 / 1200
#   eval chunk   applied afterwards by the evaluator, not here
#
# Each cell trains under the block-validation protocol so the stopping epoch is
# chosen on a dev carve rather than on test.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="grid-01"; DATA_FROM="fig6-01"
SIZES="1000,4000"; SEEDS="42"; ARMS="MarformerNT"
SHUFFLES="false,true"; TRAIN_CHUNKS="300,600,1200"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --shuffles) SHUFFLES="$2"; shift 2 ;;
        --train-chunks) TRAIN_CHUNKS="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
RUN_DIR="$(rerun_run_dir "${RUN_ID}")"; CONF_DIR="${RUN_DIR}/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
[[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${DATA_FROM}")/DATA" "${RUN_DIR}/DATA"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
IFS=',' read -r -a SH_LIST  <<< "${SHUFFLES}"
IFS=',' read -r -a TC_LIST  <<< "${TRAIN_CHUNKS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
echo "grid ${RUN_ID}: arms=${ARMS} sizes=${SIZES} shuffle=${SHUFFLES} train_chunk=${TRAIN_CHUNKS} seeds=${SEEDS}"
count=0
for arm in "${ARM_LIST[@]}"; do
  proto=transductive; [[ "${arm}" == "MarformerNT" ]] && proto=nontransductive
  for sh in "${SH_LIST[@]}"; do
    for tc in "${TC_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        tag="${arm}_sh${sh}_mi${tc}"
        name="GRID-${tag}-s${seed}"
        conf="${CONF_DIR}/${name}.env"; : > "${conf}"
        for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=${arm}" \
                  "AXIS=item" "PROTOCOL=${proto}" "SIZES=${SIZES}" \
                  "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=${tc}" \
                  "DOMAIN3_RANDOM_ITEM_CHUNKS=${sh}" \
                  "DOMAIN3_BATCH_AUGMENTATIONS=false" \
                  "VARIANT_TAG=${tag}/seed${seed}"; do
            printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
        done
        export RERUN_CONF="${conf}"
        # bigger chunks are quadratic in memory: 300->1.1GiB, 600->4GiB,
        # 1200->16GiB forward, roughly triple that with backward.
        mem=96G; [[ "${tc}" -ge 1200 ]] && mem=180G
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${mem}"
               -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
               "${RERUN_DIR}/jobs/marformer_train.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then echo "  ${tag} s${seed}: (dry)";
        else echo "  ${tag} s${seed}: job $(sbatch --parsable "${args[@]}")"; fi
        count=$((count+1))
      done
    done
  done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) runs"
