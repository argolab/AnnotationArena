#!/bin/bash
# Train the item-axis non-transductive cells the paper's figure stops short of.
#
# Marformer NT is plotted only to K=900 in the current figure (the plot script
# also caps it: ITEM_MARFORMER_NT_MAX = 500). The transductive arm runs to 4000.
# Training is ~10x faster since the optimisation work, so the NT curve can span
# the same x-range. Data for these sizes was generated in P0 into runs/fig6-01
# and checksum-verified not to disturb replicate-01's bundles.
#
# The annotator axis is NOT extended: domain3subsplit.py enforces
# train_size <= total_annots - test_annotators, and the base has J=50 with 10
# reserved for the test corner, so Annot_NT_40 is already the ceiling.
#
# Everything else is held at the replication's values, read back from
# replicate-01/.../train_config.json rather than the runner defaults:
# max_item=300, 400 epochs, AdamW lr 2e-4 wd 0.01, masking 0.15, K=5
# augmentations, masked/observed loss weight 15/1, dropout 0.1,
# item_dropout_rate 1.0. batch_augmentations stays OFF -- it changes the
# optimisation and these dots must be comparable with the existing ones.
#
#   ./submit_nt_extend.sh [--sizes 1000,2000,3000,4000] [--seeds 42,43,44] [--dry-run]
set -euo pipefail

RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="fig6-01"
SIZES="1000,2000,3000,4000"
SEEDS="42,43,44"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)  SIZES="$2";  shift 2 ;;
        --seeds)  SEEDS="$2";  shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

CONF_DIR="${RERUN_DIR}/runs/${RUN_ID}/conf"
LOG_DIR="${RERUN_DIR}/runs/${RUN_ID}/logs"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"

EXCLUDE_ARG=""
[[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"

n_sizes=$(awk -F, '{print NF}' <<< "${SIZES}")

echo "run id   : ${RUN_ID}"
echo "sizes    : ${SIZES}  (${n_sizes} per seed)"
echo "seeds    : ${SEEDS}"
echo "data     : $(rerun_data_root "${RUN_ID}")"
echo "partition: ${GPU_PARTITION}${EXCLUDE_ARG:+  ${EXCLUDE_ARG}}"
echo

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for si in "${!SEED_LIST[@]}"; do
    seed="${SEED_LIST[${si}]}"
    # Seed separation follows submit_all.sh: the FIRST seed keeps the canonical
    # untagged path that the plot scripts glob, and the rest go under seed<N>/.
    # Without this every seed writes to the same run directory and the last one
    # to finish silently wins.
    if [[ "${si}" -eq 0 ]]; then variant=""; else variant="seed${seed}"; fi
    name="FIG6-NT-item-s${seed}"
    conf="${CONF_DIR}/${name}.env"
    : > "${conf}"
    # printf %q so comma-separated lists survive; see submit_all.sh write_conf.
    for kv in "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" \
              "FAMILY=MarformerNT" "AXIS=item" "PROTOCOL=nontransductive" \
              "SIZES=${SIZES}" "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=300" \
              "DOMAIN3_BATCH_AUGMENTATIONS=false" "VARIANT_TAG=${variant}"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
    done

    # RERUN_CONF goes through the submitting environment, never as
    # --export=ALL,VAR=value -- that spec is rejected by this cluster's slurmd
    # with "user env retrieval failed requeued held".
    export RERUN_CONF="${conf}"
    args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
           -o "${LOG_DIR}/${name}_%A_%a.out"
           -e "${LOG_DIR}/${name}_%A_%a.err"
           --export=ALL
           -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
           -t "${GPU_TIME}" ${EXCLUDE_ARG}
           --array="0-$((n_sizes-1))"
           "${RERUN_DIR}/jobs/marformer_train.sbatch" )
    if [[ "${DRY_RUN}" == true ]]; then
        echo "sbatch ${args[*]}"
    else
        jid="$(sbatch --parsable "${args[@]}")"
        echo "seed ${seed}: job ${jid}  (${n_sizes} tasks)"
    fi
done
