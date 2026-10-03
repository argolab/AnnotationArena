#!/bin/bash
# Run pool workers inside the held interactive allocation.
#
# The reserved node is already paid for, so it should always be draining the
# queue. This is the same worker the Slurm jobs run; it just takes its deadline
# from the interactive job's walltime.
#
#   ./run_local_workers.sh <jobid> [workers] [gpu_gb_total]
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
JOBID="${1:?usage: run_local_workers.sh <jobid> [workers] [gpu_gb_total]}"
W="${2:-3}"; GPUGB="${3:-0}"   # 0 = let each worker size itself from the GPU
LOCAL_BUDGET=""; (( GPUGB > 0 )) && LOCAL_BUDGET=$(( GPUGB / W )) || true
END="$(scontrol show job "${JOBID}" | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' | head -1)"
SECS=$(( $(date -d "${END}" +%s) - $(date +%s) ))
# Same reason as in submit_workers.sh: never let a later edit reach a worker
# that is already executing the file.
REV="${POOL_ROOT}/rev_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${REV}"
cp "${POOL_LIB_DIR}/worker.sh" "${POOL_LIB_DIR}/lib.sh" "${POOL_LIB_DIR}/calibrate.py" "${REV}/"
cat > "${POOL_ROOT}/local_fanout.sh" <<INNER
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env >/dev/null 2>&1
# Split the node between workers: unconstrained, each torch process opens one
# thread per core and they contend rather than compute (8.6 s/epoch with six of
# them, against 0.35 s/epoch for one).
export OMP_NUM_THREADS=\$(( 24 / ${W} )) MKL_NUM_THREADS=\$(( 24 / ${W} ))
export POOL_ROOT="${POOL_ROOT}" POOL_MAX_SECONDS=${SECS}
# This allocation is held regardless, so its workers wait for work rather
# than releasing a node we keep paying for either way.
export POOL_WAIT_WHEN_EMPTY=1
for i in \$(seq 1 ${W}); do
    POOL_PER_JOB=${W} POOL_GPU_BUDGET_GB="${LOCAL_BUDGET}" POOL_WORKER_LABEL="local${JOBID}.\${i}" \\
        POOL_LIB_DIR="${REV}" bash "${REV}/worker.sh" "\${i}" &
    sleep 5
done
wait
INNER
echo "local workers on job ${JOBID}: ${W} x ${LOCAL_BUDGET:-auto} GB, $(( SECS / 3600 ))h of walltime left"
# setsid + nohup, detached from this shell. Run in the foreground (or merely
# backgrounded) these die the moment whatever launched them is reaped -- which
# silently killed three in-flight trainings and left their claims stranded until
# the heartbeat timeout. The allocation outlives any one shell, and so must the
# workers running inside it.
LOG="${POOL_ROOT}/logs/local_workers_$(date +%H%M%S).log"
setsid nohup srun --jobid="${JOBID}" --overlap -n1 --cpu-bind=none \
    bash "${POOL_ROOT}/local_fanout.sh" > "${LOG}" 2>&1 < /dev/null &
disown || true
sleep 2
echo "detached; log: ${LOG}"
