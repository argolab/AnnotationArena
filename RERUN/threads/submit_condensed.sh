#!/bin/bash
# Submit an arms sweep as a FEW jobs that each run MANY trainings.
#
# side/submit_arms.sh submits one array per (arm, seed, draw), which is the right
# shape when the queue is wide. It is the wrong shape here: this account runs
# about ten jobs at once, so a 36-cell sweep spends most of its life PENDING
# while whole nodes sit idle. This instead enumerates every cell, splits the list
# into --groups pieces, and gives each piece one job that works through its share
# --workers at a time.
#
# The results tree, variant paths and env are byte-identical to submit_arms.sh,
# so evaluation and analysis cannot tell which submitter produced a run.
#
#   RERUN_DIR=<worktree>/RERUN ./submit_condensed.sh --arms-file F --run-id ID \
#       --draws d1,d2 --seeds 42,43,44 --sizes 300,2000 [--groups 2] [--workers 4]
#
# Sizing --workers: they share one GPU, so check the cell's memory against
# threads/context-window/out/a4_memory.json. At chunk 300 a K=2000 training peaks
# at 18 GB and a K=4000 at 36 GB, so four small cells fit on an 80 GB card and two
# large ones do.
set -euo pipefail
# NOTE: do not name this variable GROUPS. Bash keeps GROUPS as a built-in array of
# the caller's group ids, silently ignores assignments to it, and ${GROUPS} then
# expands to a gid -- which submitted 1,470 empty jobs the first time this ran.
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

ARMS_FILE=""; RUN_ID=""; DRAWS=""; SEEDS="42,43,44"; SIZES=""; AXIS="item"
MAX_ITEM=300; NGROUPS=2; WORKERS=4; TIME="${GPU_TIME}"; DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms-file) ARMS_FILE="$2"; shift 2 ;;
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --draws)     DRAWS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --axis)      AXIS="$2"; shift 2 ;;
        --max-item)  MAX_ITEM="$2"; shift 2 ;;
        --groups)    NGROUPS="$2"; shift 2 ;;
        --workers)   WORKERS="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
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

WORK="${RERUN_DIR}/runs/${RUN_ID}-work"; mkdir -p "${WORK}"
CELLS="${WORK}/cells.txt"; : > "${CELLS}"
IFS=',' read -r -a DRAW_LIST <<< "${DRAWS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
AXIS_DIR="$(tr '[:lower:]' '[:upper:]' <<< "${AXIS}")"
for draw in "${DRAW_LIST[@]}"; do
    rid="${RUN_ID}-${draw#dataseed-}"
    RUN_DIR="$(rerun_run_dir "${rid}")"; mkdir -p "$(rerun_log_root "${rid}")"
    [[ -e "${RUN_DIR}/DATA" ]] || ln -sfn "$(rerun_run_dir "${draw}")/DATA" "${RUN_DIR}/DATA"
    for row in "${ROWS[@]}"; do
        tag="${row%%$'\t'*}"; envkvs="${row#*$'\t'}"
        for seed in "${SEED_LIST[@]}"; do
            for size in "${SIZE_LIST[@]}"; do
                # The seed MUST stay in the variant path: without it every seed
                # writes to one directory and the last to finish wins.
                out="$(rerun_results_root "${rid}")/MARFORMER/DOMAIN3-SIGMOID/${AXIS_DIR}/${tag}/Marformer/seed${seed}"
                log="$(rerun_log_root "${rid}")/CELL-${tag}-s${seed}-K${size}.log"
                echo "env ${envkvs} DOMAIN3_DATA_ROOT=$(rerun_data_root "${rid}") DOMAIN3_MAX_ITEM=${MAX_ITEM} DOMAIN3_SEED=${seed} DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash ${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh --family-label Marformer --output-root ${out} --axis ${AXIS} --protocol transductive --sizes ${size} > ${log} 2>&1" >> "${CELLS}"
            done
        done
    done
done

n=$(wc -l < "${CELLS}")
# Round-robin rather than block split, so every group gets a mix of sizes and
# finishes at about the same time instead of one group drawing all the K=4000s.
for ((g = 0; g < NGROUPS; g++)); do
    awk -v g="${g}" -v G="${NGROUPS}" 'NR % G == g' "${CELLS}" > "${WORK}/group${g}.txt"
done
echo "${RUN_ID}: ${n} cells -> ${NGROUPS} jobs x ${WORKERS} workers"

cat > "${WORK}/runner.sbatch" <<'SB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -euo pipefail
source "${RERUN_DIR}/config/cluster.sh"
export REQUIRE_GPU=1
rerun_activate_env
echo "group ${GROUP_FILE}: $(wc -l < "${GROUP_FILE}") cells, ${WORKERS} at a time"
< "${GROUP_FILE}" xargs -P "${WORKERS}" -I{} bash -lc '{}'
echo "group complete"
SB

EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
for ((g = 0; g < NGROUPS; g++)); do
    export GROUP_FILE="${WORK}/group${g}.txt" WORKERS
    args=( "${SBATCH_ACCEPT_COST}" -J "${RUN_ID}-g${g}"
           -o "${WORK}/g${g}_%j.out" -e "${WORK}/g${g}_%j.err"
           --export=ALL -p "${GPU_PARTITION}" -c "$((WORKERS * 4))" --mem="${GPU_MEM}"
           -t "${TIME}" ${EXCLUDE_ARG} "${WORK}/runner.sbatch" )
    if [[ "${DRY}" == true ]]; then echo "  g${g}: $(wc -l < "${WORK}/group${g}.txt") cells"
    else echo "  g${g} ($(wc -l < "${WORK}/group${g}.txt") cells): job $(sbatch --parsable "${args[@]}")"; fi
done
