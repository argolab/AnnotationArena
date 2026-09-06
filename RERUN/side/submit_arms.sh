#!/bin/bash
# Generic knob sweep for the FUTURE_EXPERIMENTS side suites.
#
# An "arms file" is JSON: {"arms":[{"tag":"...", "env":{"DOMAIN3_X":"v", ...}}]}.
# Every arm is trained on the same (data draw, model seed, K) cells, so the
# comparison is PAIRED -- which is the only kind that survives Domain 3's noise
# floor (draw sd 0.085, model-seed sd 0.014; see FUTURE_EXPERIMENTS Part 0).
#
#   ./submit_arms.sh --arms-file side/arms_s2_item.json --run-id s2-item \
#       --draws dataseed-s101,dataseed-s105 --seeds 42,43,44 --sizes 300,1000
#
# Fixed-epoch comparison by default (DOMAIN3_VAL_PROTOCOL=none, 400 epochs), so
# nothing is ever selected on the corner. Pass --val block to run the search
# stage of the validation protocol instead.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

ARMS_FILE=""; RUN_ID=""; DRAWS="dataseed-s101,dataseed-s105"; SEEDS="42,43,44"
SIZES="300,1000"; AXIS="item"; FAMILIES="Marformer"; MAX_ITEM=300; VAL="none"
DRY_RUN=false; USE_CPU=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms-file) ARMS_FILE="$2"; shift 2 ;;
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --draws)     DRAWS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --axis)      AXIS="$2"; shift 2 ;;
        --families)  FAMILIES="$2"; shift 2 ;;
        --max-item)  MAX_ITEM="$2"; shift 2 ;;
        --val)       VAL="$2"; shift 2 ;;
        --cpu)       USE_CPU=true; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${ARMS_FILE}" && -n "${RUN_ID}" ]] || { echo "--arms-file and --run-id required" >&2; exit 2; }

mapfile -t ROWS < <(python - "${ARMS_FILE}" <<'PY'
import json, sys, shlex
for a in json.load(open(sys.argv[1]))["arms"]:
    env = " ".join(f"{k}={v}" for k, v in a["env"].items())
    print(f'{a["tag"]}\t{env}')
PY
)
n=$(awk -F, '{print NF}' <<< "${SIZES}")
IFS=',' read -r -a DRAW_LIST <<< "${DRAWS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a FAM_LIST  <<< "${FAMILIES}"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
# --cpu sends the sweep to the CPU partition. MEASURED 2026-09-04 AND NOT
# RECOMMENDED: 45 s/epoch at K=1000 on med, against 1.12 s/epoch on h100 --
# 40x slower, so no amount of queue availability makes it up. The "training is
# CPU-bound" benchmark says the GPU is ~90% *idle*, which is true and is why
# the GPU class barely matters; it does NOT mean the GPU is doing nothing. Kept
# only for jobs small enough that queue latency dominates (smoke tests).
if [[ "${USE_CPU}" == true ]]; then
    PART="${CPU_PARTITION}"; CPUS="${CPU_CPUS}"; MEM="${CPU_MEM}"; TIME="${CPU_TIME}"
    export REQUIRE_GPU=0
    # Overrides the --gres=gpu:1 in the job script's header.
    GRES_ARG="--gres=none"
else
    PART="${GPU_PARTITION}"; CPUS="${GPU_CPUS}"; MEM="${GPU_MEM}"; TIME="${GPU_TIME}"
    GRES_ARG="--gres=gpu:1"
fi

echo "${RUN_ID}: ${#ROWS[@]} arms x ${#DRAW_LIST[@]} draws x ${#SEED_LIST[@]} seeds x ${#FAM_LIST[@]} fam x ${n} sizes = $(( ${#ROWS[@]} * ${#DRAW_LIST[@]} * ${#SEED_LIST[@]} * ${#FAM_LIST[@]} * n )) runs"
count=0
for draw in "${DRAW_LIST[@]}"; do
    # One run tree per data draw. Draws are never mixed inside a tree: the
    # bundles differ, so a shared tree would make two arms silently incomparable.
    rid="${RUN_ID}-${draw#dataseed-}"
    RUN_DIR="$(rerun_run_dir "${rid}")"; CONF_DIR="${RUN_DIR}/conf"
    LOG_DIR="$(rerun_log_root "${rid}")"
    mkdir -p "${CONF_DIR}" "${LOG_DIR}"
    [[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${draw}")/DATA" "${RUN_DIR}/DATA"
    for row in "${ROWS[@]}"; do
        tag="${row%%$'\t'*}"; envkvs="${row#*$'\t'}"
        for fam in "${FAM_LIST[@]}"; do
            proto=transductive; [[ "${fam}" == *NT ]] && proto=nontransductive
            for seed in "${SEED_LIST[@]}"; do
                # The seed MUST be in the variant path. Omitting it has silently
                # destroyed results five times in this project (submit_search.sh,
                # submit_nt_extend.sh, submit_chunk500.sh, ...): every seed writes
                # to one directory and the last to finish wins.
                variant="${tag}/${fam}/seed${seed}"
                name="ARM-${RUN_ID}-${tag}-${fam}-s${seed}"
                conf="${CONF_DIR}/${name}.env"; : > "${conf}"
                kvs=( "RUN_ID=${rid}" "DATA_RUN_ID=${rid}" "FAMILY=${fam}"
                      "AXIS=${AXIS}" "PROTOCOL=${proto}" "SIZES=${SIZES}"
                      "DOMAIN3_SEED=${seed}" "DOMAIN3_MAX_ITEM=${MAX_ITEM}"
                      "DOMAIN3_VAL_PROTOCOL=${VAL}"
                      "VARIANT_TAG=${variant}" )
                for kv in ${envkvs}; do kvs+=( "${kv}" ); done
                for kv in "${kvs[@]}"; do printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"; done
                export RERUN_CONF="${conf}"
                args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
                       -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
                       --export=ALL -p "${PART}" -c "${CPUS}" --mem="${MEM}" ${GRES_ARG}
                       -t "${TIME}" ${EXCLUDE_ARG} --array="0-$((n-1))"
                       "${RERUN_DIR}/jobs/marformer_train.sbatch" )
                if [[ "${DRY_RUN}" == true ]]; then echo "  ${name} -> ${variant}  [${envkvs}]"
                else echo "  ${name}: job $(sbatch --parsable "${args[@]}")"; fi
                count=$((count+1))
            done
        done
    done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) runs"
