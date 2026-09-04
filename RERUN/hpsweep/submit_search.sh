#!/bin/bash
# Capacity sweep: can a deeper or wider Marformer beat the shipped one?
#
# Phase A (this script) trains every config with a BLOCK VALIDATION CARVE:
# 15% of the training items are relabelled into a validation split at load time,
# so there is a legitimate signal to select the stopping epoch on. Without it,
# "the bigger model got a lower log-loss" would just be an argmin taken on test
# (D14), and extra capacity is exactly the setting where that lies to you.
#
# Phase B (submit_refit.sh) takes the winners' validated epochs and retrains on
# the FULL training partition, so the final numbers have the same data budget as
# every other dot in Figure 6.
#
# Held fixed: K=4000, item axis, training chunk 600, no shuffle, model seed 42,
# lr 2e-4, dropout 0.1. Only the architecture moves.
#
#   ./submit_search.sh [--arms Marformer,MarformerNT] [--tags base,deep12] [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="hp-01"; DATA_FROM="fig6-01"; SIZES="4000"; MAX_ITEM=600
ARMS="Marformer,MarformerNT"; SEEDS="42"; TAGS=""; PHASE="search"
EPOCH_TABLE=""; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms) ARMS="$2"; shift 2 ;;
        --tags) TAGS="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --max-item) MAX_ITEM="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --data-from) DATA_FROM="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --epoch-table) EPOCH_TABLE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"; CONF_DIR="${RUN_DIR}/conf"
LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
[[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${DATA_FROM}")/DATA" "${RUN_DIR}/DATA"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")

mapfile -t ROWS < <(python - "${RERUN_DIR}/hpsweep/configs.json" "${TAGS}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))["configs"]
want = [t for t in sys.argv[2].split(",") if t] if len(sys.argv) > 2 else []
for c in cfg:
    if want and c["tag"] not in want: continue
    print(f'{c["tag"]} {c["dim"]} {c["layers"]} {c["heads"]} {c["d_ff"]} {c["ffn"]}')
PY
)

# Cheap structural guard against the collision above: every seed must end up on
# a distinct path, so the tag template has to mention the seed.
if [[ "${SEEDS}" == *,* ]] && [[ "${PHASE}" == "search" ]]; then
    :  # variant below includes /seed${seed}; kept explicit so a future edit that
       # drops it fails review rather than silently overwriting results.
fi

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
echo "${RUN_ID} phase=${PHASE}: ${#ROWS[@]} configs x ${#ARM_LIST[@]} arms x ${#SEED_LIST[@]} seeds, K=${SIZES}, chunk ${MAX_ITEM}"
count=0
for row in "${ROWS[@]}"; do
    read -r tag dim layers heads dff ffn <<< "${row}"
    for arm in "${ARM_LIST[@]}"; do
        proto=transductive; [[ "${arm}" == "MarformerNT" ]] && proto=nontransductive
        for seed in "${SEED_LIST[@]}"; do
            # The seed MUST be in the path. Without it every seed writes to the
            # same run directory and the last one to finish silently wins --
            # which is exactly what happened on 2026-09-04, when a 3-seed
            # replication overwrote the single-seed results it was meant to
            # extend. Same failure as submit_nt_extend.sh and submit_chunk500.sh.
            variant="${arm}_${tag}-${PHASE}/seed${seed}"
            name="HP-${arm}-${tag}-${PHASE}-s${seed}"
            conf="${CONF_DIR}/${name}.env"; : > "${conf}"
            kvs=( "RUN_ID=${RUN_ID}" "DATA_RUN_ID=${RUN_ID}" "FAMILY=${arm}"
                  "AXIS=item" "PROTOCOL=${proto}" "SIZES=${SIZES}"
                  "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=${MAX_ITEM}"
                  "DOMAIN3_RANDOM_ITEM_CHUNKS=false"
                  "DOMAIN3_BATCH_AUGMENTATIONS=false"
                  "DOMAIN3_EMBEDDING_DIM=${dim}" "DOMAIN3_NUM_LAYERS=${layers}"
                  "DOMAIN3_ATTENTION_HEADS=${heads}" "DOMAIN3_D_FF=${dff}"
                  "DOMAIN3_NUM_FFN_LAYERS=${ffn}"
                  "VARIANT_TAG=${variant}" )
            if [[ "${PHASE}" == "search" ]]; then
                kvs+=( "DOMAIN3_VAL_PROTOCOL=block" "DOMAIN3_VAL_FRAC=0.15"
                       "DOMAIN3_VAL_SPLIT_SEED=7" )
            else
                kvs+=( "DOMAIN3_VAL_PROTOCOL=none" )
                [[ -n "${EPOCH_TABLE}" ]] && kvs+=( "EPOCH_TABLE=${EPOCH_TABLE}" )
            fi
            for kv in "${kvs[@]}"; do printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"; done
            export RERUN_CONF="${conf}"
            # 160-wide / 16-layer at chunk 600 is the memory pole; give the big
            # cells more headroom rather than discovering it as an OOM at hour 1.
            mem=96G; [[ "${dim}" -ge 128 || "${layers}" -ge 12 ]] && mem=180G
            args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
                   -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
                   --export=ALL -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${mem}"
                   -t "${GPU_TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
                   "${RERUN_DIR}/jobs/marformer_train.sbatch" )
            if [[ "${DRY_RUN}" == true ]]; then echo "  ${name} (dry, mem ${mem})"
            else echo "  ${name}: job $(sbatch --parsable "${args[@]}")"; fi
            count=$((count+1))
        done
    done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) runs"
