#!/bin/bash
# Shared definitions for the task pool.
#
# The pool exists because this account can hold only about ten Slurm jobs at
# once, while a sweep is a hundred small trainings. Submitting one job per
# training wastes the whole allocation on queueing. Instead: tasks go in a
# directory, and a handful of long-lived worker jobs pull from it until they run
# out of time. Workers can sit on different partitions, so whichever GPU class
# frees up first does the work.
#
# State is kept as files, claimed with mkdir, which is atomic on this filesystem
# and needs no daemon:
#
#   tasks/<id>.json     the work item (command, estimate, memory)
#   state/<id>.lock/    a claim; holds owner + heartbeat
#   state/<id>.done     finished successfully
#   state/<id>.failed   finished non-zero; log kept
#   logs/<id>.log       the task's output
# NOT `set -e`: worker.sh sources this, and with -e on, a task that exits
# non-zero kills the worker itself -- before it can record the failure or release
# its claim. That is exactly what happened on the first run: three workers died
# silently mid-task and left stale locks, and the pool reported "0 failed".
# Scripts that want -e set it themselves before sourcing.
set -uo pipefail

POOL_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RERUN_DIR="${RERUN_DIR:-$(cd "${POOL_LIB_DIR}/.." && pwd)}"
POOL_ROOT="${POOL_ROOT:-${RERUN_DIR}/pool/queue}"

pool_dirs() {
    mkdir -p "${POOL_ROOT}/tasks" "${POOL_ROOT}/state" "${POOL_ROOT}/logs"
}

# A claim is stale when its owner has not touched the heartbeat for this long.
# Longer than any plausible pause inside a task, short enough that a node dying
# does not strand work for a whole day.
POOL_STALE_S="${POOL_STALE_S:-1800}"

# Multiply a task's estimate by this before deciding it fits in the time left.
# Estimates come from a cost model measured on an idle GPU; a worker running
# several tasks at once is slower than that, and a task killed by the walltime
# is worse than one never started.
POOL_SAFETY="${POOL_SAFETY:-1.6}"

# Stop claiming this long before the walltime ends, so the last task can flush
# its checkpoints and the job can exit cleanly.
POOL_RESERVE_S="${POOL_RESERVE_S:-300}"

pool_now() { date +%s; }

# Seconds until this job's walltime expires. Falls back to POOL_MAX_SECONDS for
# an interactive shell with no job of its own.
pool_deadline() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local end
        end="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
               | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' | head -1)"
        if [[ -n "${end}" && "${end}" != "Unknown" ]]; then
            date -d "${end}" +%s && return 0
        fi
    fi
    echo $(( $(pool_now) + ${POOL_MAX_SECONDS:-86400} ))
}

pool_field() {  # pool_field <task.json> <key>
    python - "$1" "$2" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
PY
}
