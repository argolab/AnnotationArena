#!/bin/bash
# Put work in the pool.
#
# Sweep mode -- the common one. Same interface as side/submit_arms.sh, so an
# existing arms file drops straight in:
#
#   ./enqueue.sh --arms-file F --run-id ID --draws d1,d2 --seeds 42,43,44 \
#                --sizes 300,2000 [--max-item 300] [--runner-root <worktree>/imputer/ranking]
#
# One-off mode:
#
#   ./enqueue.sh --cmd 'bash something.sh' --label 'what it is' --eta-s 900 --gpu-gb 20
#
# Enqueueing is idempotent: each task records the file its run will produce, and
# a worker that finds that file already there marks the task done instead of
# repeating it. So re-running this after a partial sweep tops up the queue rather
# than duplicating it.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
pool_dirs

ARMS_FILE=""; RUN_ID=""; DRAWS=""; SEEDS="42,43,44"; SIZES=""; AXIS="item"
MAX_ITEM=300; EPOCHS=""; RUNNER_ROOT="${RANKING_ROOT:-}"; CMD=""; LABEL=""
ETA_S=""; GPU_GB=""; PRIORITY=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms-file) ARMS_FILE="$2"; shift 2 ;;
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --draws)     DRAWS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --axis)      AXIS="$2"; shift 2 ;;
        --max-item)  MAX_ITEM="$2"; shift 2 ;;
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --runner-root) RUNNER_ROOT="$2"; shift 2 ;;
        --cmd)       CMD="$2"; shift 2 ;;
        --label)     LABEL="$2"; shift 2 ;;
        --eta-s)     ETA_S="$2"; shift 2 ;;
        --gpu-gb)    GPU_GB="$2"; shift 2 ;;
        --priority)  PRIORITY="$2"; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
source "${RERUN_DIR}/config/cluster.sh"
RUNNER_ROOT="${RUNNER_ROOT:-${RANKING_ROOT}}"

next_id() {
    local n=0
    for f in "${POOL_ROOT}"/tasks/*.json; do
        [[ -e "${f}" ]] || break
        local b; b="$(basename "${f}" .json)"; (( 10#${b} > n )) && n=$(( 10#${b} ))
    done
    printf '%05d' $(( n + 1 ))
}

add_task() {  # add_task <label> <eta_s> <gpu_gb> <done_marker> <cmd>
    local id; id="$(next_id)"
    python - "${POOL_ROOT}/tasks/${id}.json" "$1" "$2" "$3" "$4" "$5" "${PRIORITY}" <<'PY'
import json, sys
path, label, eta, gb, marker, cmd, prio = sys.argv[1:8]
json.dump({"label": label, "eta_s": int(float(eta)), "gpu_gb": float(gb),
           "done_marker": marker, "priority": int(prio), "cmd": cmd},
          open(path, "w"), indent=1)
PY
    echo "  ${id}  ${1}  (~$(( $(printf '%.0f' "$2") / 60 )) min, $(printf '%.0f' "$3") GB)"
}

if [[ -n "${CMD}" ]]; then
    add_task "${LABEL:-one-off}" "${ETA_S:-1800}" "${GPU_GB:-20}" "" "${CMD}"
    exit 0
fi
[[ -n "${ARMS_FILE}" && -n "${RUN_ID}" && -n "${DRAWS}" && -n "${SIZES}" ]] \
    || { echo "--arms-file, --run-id, --draws and --sizes are required" >&2; exit 2; }

mapfile -t ROWS < <(python - "${ARMS_FILE}" <<'PY'
import json, sys
for a in json.load(open(sys.argv[1]))["arms"]:
    print(a["tag"] + "\t" + " ".join(f"{k}={v}" for k, v in a["env"].items()))
PY
)

# Cost model, measured: seconds per epoch at chunk 300 (RERUN/FUTURE_EXPERIMENTS
# Part 0) and GB of GPU at a given chunk (threads/context-window/out/a4_memory.json).
# Both scale close to linearly in K, so interpolating between the measured points
# is good enough for a scheduling decision.
estimate() {  # estimate <K> <max_item> <epochs> -> "<eta_s> <gpu_gb>"
    python - "$1" "$2" "$3" <<'PY'
import sys
K, mi, ep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sec_at_300 = {300: 0.35, 1000: 1.12, 2000: 1.58, 4000: 3.14}
ks = sorted(sec_at_300)
if K <= ks[0]:   s = sec_at_300[ks[0]] * K / ks[0]
elif K >= ks[-1]: s = sec_at_300[ks[-1]] * K / ks[-1]
else:
    lo = max(k for k in ks if k <= K); hi = min(k for k in ks if k >= K)
    s = sec_at_300[lo] if lo == hi else sec_at_300[lo] + (sec_at_300[hi] - sec_at_300[lo]) * (K - lo) / (hi - lo)
s *= 1.0 + 0.05 * (mi - 300) / 300          # a wider chunk costs ~5% per +300
gb = 18.4 * (K / 2000) * (1.0 + 0.95 * (mi - 300) / 300)
print(int(s * ep + 120), round(max(gb, 4.0), 1))   # +120s for setup and scoring
PY
}

IFS=',' read -r -a DRAW_LIST <<< "${DRAWS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
AXIS_DIR="$(tr '[:lower:]' '[:upper:]' <<< "${AXIS}")"
EP="${EPOCHS:-400}"
echo "enqueue ${RUN_ID}:"
for draw in "${DRAW_LIST[@]}"; do
    rid="${RUN_ID}-${draw#dataseed-}"
    RUN_DIR="$(rerun_run_dir "${rid}")"; mkdir -p "$(rerun_log_root "${rid}")"
    [[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${draw}")/DATA" "${RUN_DIR}/DATA"
    for row in "${ROWS[@]}"; do
        tag="${row%%$'\t'*}"; envkvs="${row#*$'\t'}"
        # An arm may override the chunk; the estimate has to see the real value.
        mi="${MAX_ITEM}"
        [[ "${envkvs}" =~ DOMAIN3_MAX_ITEM=([0-9]+) ]] && mi="${BASH_REMATCH[1]}"
        # A dynamic chunk range costs whatever its WIDEST draw costs, not what
        # DOMAIN3_MAX_ITEM says. Estimating from the nominal value would hand a
        # 57 GB task to a worker that budgeted for 36 and OOM it mid-epoch.
        [[ "${envkvs}" =~ DOMAIN3_ITEM_CHUNK_RANGE=[0-9]+,([0-9]+) ]] && mi="${BASH_REMATCH[1]}"
        for seed in "${SEED_LIST[@]}"; do
            for size in "${SIZE_LIST[@]}"; do
                # The seed MUST stay in the variant path: without it every seed
                # writes to one directory and the last to finish wins.
                out="$(rerun_results_root "${rid}")/MARFORMER/DOMAIN3-SIGMOID/${AXIS_DIR}/${tag}/Marformer/seed${seed}"
                marker="${out}/DOMAIN3-SIGMOID_Item_T_${size}_MARFORMER/training_history.json"
                log="$(rerun_log_root "${rid}")/CELL-${tag}-s${seed}-K${size}.log"
                read -r eta gb <<< "$(estimate "${size}" "${mi}" "${EP}")"
                epenv=""; [[ -n "${EPOCHS}" ]] && epenv="DOMAIN3_EPOCHS=${EPOCHS} "
                # The arm's own settings go LAST in the env list: `env A=1 A=2` takes the
                # last assignment, so an arm that overrides DOMAIN3_MAX_ITEM has to come
                # after the default or it is silently ignored -- which would have made
                # every chunk-size arm train at the same chunk.
                # config/cluster.sh exports DOMAIN3_REPO_ROOT_OVERRIDE and PYTHONPATH pointing
                # at ITS OWN tree, and run_marformer_group.sh resolves the trainer from that
                # override -- so picking the script with --runner-root is not enough. Without
                # these two lines a task submitted from the main pool ran a side branch's
                # runner against the main checkout's train.py and died with "unrecognized
                # arguments: --value-codec". Pin both, after cluster.sh has had its say.
                cmd="source ${RERUN_DIR}/config/cluster.sh; export DOMAIN3_REPO_ROOT_OVERRIDE=${RUNNER_ROOT} PYTHONPATH=${RUNNER_ROOT}:\${PYTHONPATH}; rerun_activate_env >/dev/null; env ${epenv}DOMAIN3_DATA_ROOT=$(rerun_data_root "${rid}") DOMAIN3_MAX_ITEM=${MAX_ITEM} DOMAIN3_SEED=${seed} DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${envkvs} bash ${RUNNER_ROOT}/scripts/DOMAIN3/run_marformer_group.sh --family-label Marformer --output-root ${out} --axis ${AXIS} --protocol transductive --sizes ${size} > ${log} 2>&1"
                add_task "${rid} ${tag} s${seed} K${size}" "${eta}" "${gb}" "${marker}" "${cmd}"
            done
        done
    done
done
