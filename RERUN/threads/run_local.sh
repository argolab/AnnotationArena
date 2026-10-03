#!/bin/bash
# Run an arms sweep on a held interactive allocation instead of through the queue.
#
# Why this exists: this account is capped at roughly ten concurrent jobs, so with
# three suites in flight two of them sit entirely PENDING while a reserved H100
# is idle. This runs the same cells, through the same runner, into the same
# results tree and variant paths that side/submit_arms.sh produces -- so the
# evaluation and analysis steps cannot tell the difference.
#
#   RERUN_DIR=<worktree>/RERUN ./run_local.sh --arms-file F --run-id ID \
#       --draws d1,d2 --seeds 42,43,44 --sizes 300,1000 [--jobs 4] [--jobid J]
#
# --jobs is how many trainings share the GPU. They are small (18 GiB at K=2000,
# chunk 300) and the GPU is ~90% idle on this workload, so several at once is
# throughput, not contention. Check against threads/context-window/out/a4_memory.json
# before raising it for large K or large chunks.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

ARMS_FILE=""; RUN_ID=""; DRAWS=""; SEEDS="42,43,44"; SIZES=""; AXIS="item"
MAX_ITEM=300; JOBS=4; JOBID="${SLURM_JOB_ID:-}"; DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms-file) ARMS_FILE="$2"; shift 2 ;;
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --draws)     DRAWS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --axis)      AXIS="$2"; shift 2 ;;
        --max-item)  MAX_ITEM="$2"; shift 2 ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --jobid)     JOBID="$2"; shift 2 ;;
        --dry-run)   DRY=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${ARMS_FILE}" && -n "${RUN_ID}" && -n "${DRAWS}" && -n "${SIZES}" ]] \
    || { echo "--arms-file, --run-id, --draws and --sizes are required" >&2; exit 2; }

mapfile -t ROWS < <(python - "${ARMS_FILE}" <<'PY'
import json, sys
for a in json.load(open(sys.argv[1]))["arms"]:
    print(a["tag"] + "\t" + " ".join(f"{k}={v}" for k, v in a["env"].items()))
PY
)

CMDS="$(mktemp)"
IFS=',' read -r -a DRAW_LIST <<< "${DRAWS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
for draw in "${DRAW_LIST[@]}"; do
    rid="${RUN_ID}-${draw#dataseed-}"
    RUN_DIR="$(rerun_run_dir "${rid}")"; LOG_DIR="$(rerun_log_root "${rid}")"
    mkdir -p "${LOG_DIR}"
    [[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${draw}")/DATA" "${RUN_DIR}/DATA"
    for row in "${ROWS[@]}"; do
        tag="${row%%$'\t'*}"; envkvs="${row#*$'\t'}"
        for seed in "${SEED_LIST[@]}"; do
            for size in "${SIZE_LIST[@]}"; do
                # Same variant layout as submit_arms.sh. The seed MUST stay in
                # the path: without it every seed writes to one directory and the
                # last to finish wins, which has destroyed results here before.
                out="$(rerun_results_root "${rid}")/MARFORMER/DOMAIN3-SIGMOID/$(tr '[:lower:]' '[:upper:]' <<< "${AXIS}")/${tag}/Marformer/seed${seed}"
                log="${LOG_DIR}/LOCAL-${tag}-s${seed}-K${size}.log"
                printf '%s\n' "env ${envkvs} DOMAIN3_DATA_ROOT=$(rerun_data_root "${rid}") DOMAIN3_MAX_ITEM=${MAX_ITEM} DOMAIN3_SEED=${seed} DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash ${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh --family-label Marformer --output-root ${out} --axis ${AXIS} --protocol transductive --sizes ${size} > ${log} 2>&1" >> "${CMDS}"
            done
        done
    done
done

n=$(wc -l < "${CMDS}")
echo "${RUN_ID}: ${n} runs, ${JOBS} at a time, on ${JOBID:-this shell}"
if [[ "${DRY}" == true ]]; then head -3 "${CMDS}"; echo "..."; rm -f "${CMDS}"; exit 0; fi

# ONE srun step for the whole fan-out, not one per training. A per-command
# `srun -n1` gets cpus-per-task=1, and this workload is CPU-bound -- the GPU sits
# ~90% idle while Python builds the entity graph -- so pinning each training to a
# single core ran it at 18 s/epoch against 0.35 s/epoch with the node's cores
# available. Fifty times slower, and nothing in the logs says so.
DRIVER="${RERUN_DIR}/runs/${RUN_ID}-work"; mkdir -p "${DRIVER}"
cp "${CMDS}" "${DRIVER}/cells.txt"; rm -f "${CMDS}"
cat > "${DRIVER}/fanout.sh" <<INNER
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env >/dev/null 2>&1
< "${DRIVER}/cells.txt" xargs -P ${JOBS} -I{} bash -lc '{}'
INNER
if [[ -n "${JOBID}" ]]; then
    # --cpu-bind=none rather than -c N: the allocation's CPU mask is not a
    # contiguous range, so asking a step for N cpus-per-task fails with "CPU
    # binding outside of job step allocation". Unbound, the step's children are
    # free to use every core the allocation holds, which is the point.
    srun --jobid="${JOBID}" --overlap -n1 --cpu-bind=none bash "${DRIVER}/fanout.sh"
else
    bash "${DRIVER}/fanout.sh"
fi
echo "${RUN_ID}: all ${n} runs finished"
