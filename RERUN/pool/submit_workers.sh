#!/bin/bash
# Submit worker jobs that drain the pool.
#
#   ./submit_workers.sh [--jobs 4] [--per-job 3] [--time 8:00:00]
#                       [--partitions h100,l40s,a100] [--gpu-gb 70]
#
# Workers are submitted to a COMMA-SEPARATED partition list, so Slurm starts each
# one wherever there is room first -- h100, a100 or l40s, whichever frees up.
# That is safe here despite the warning in config/cluster.sh: the held-job
# failures that warning describes were traced to `--export=ALL,VAR=value`, not to
# multi-partition lists, and this script passes a plain --export=ALL.
#
# --per-job is how many tasks one job runs at once. They share a GPU, so each
# worker is given gpu-gb/per-job of budget and only claims tasks that fit.
#
# They also share CPUs, and that matters more than it looks: this workload is
# CPU-bound, and torch by default opens as many threads as there are cores. Six
# unconstrained processes on a 30-core node ran at 8.6 s/epoch against 0.35
# s/epoch for one -- the threads fight rather than work. Each worker therefore
# gets OMP_NUM_THREADS = cpus / per-job.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
source "${RERUN_DIR}/config/cluster.sh"
pool_dirs

JOBS=4; PER_JOB=3; TIME="8:00:00"; PARTS="h100,l40s,a100"; GPUGB=0; CPUS=24; DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)       JOBS="$2"; shift 2 ;;
        --per-job)    PER_JOB="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --partitions) PARTS="$2"; shift 2 ;;
        --gpu-gb)     GPUGB="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --dry-run)    DRY=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

cat > "${POOL_ROOT}/worker_job.sbatch" <<'SB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -uo pipefail
source "${POOL_LIB}"
export REQUIRE_GPU=1
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
# Divide the node between the workers in this job. Without this each torch
# process opens one thread per core and they contend instead of computing.
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-24} / POOL_PER_JOB ))"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
echo "job ${SLURM_JOB_ID} on $(hostname) [${SLURM_JOB_PARTITION}]: ${POOL_PER_JOB} workers, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
for i in $(seq 1 "${POOL_PER_JOB}"); do
    POOL_GPU_BUDGET_GB="${POOL_GPU_BUDGET_GB:-}" \
    POOL_WORKER_LABEL="${SLURM_JOB_ID}.${i}" \
        bash "${POOL_WORKER}" "${i}" &
    sleep 5     # stagger, so they do not all race for the same task at t=0
done
wait
echo "job ${SLURM_JOB_ID} complete"
SB

EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
# Snapshot the worker code and run the copy. Bash reads a script by byte offset
# as it executes, so editing worker.sh while workers are running it can drop them
# into the middle of a different statement -- the project has lost three job
# batches to exactly this. A per-launch revision directory makes later edits
# unreachable by anything already in flight.
REV="${POOL_ROOT}/rev_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${REV}"
cp "${POOL_LIB_DIR}/worker.sh" "${POOL_LIB_DIR}/lib.sh" "${POOL_LIB_DIR}/calibrate.py" "${REV}/"
export POOL_LIB="${REV}/lib.sh" POOL_WORKER="${REV}/worker.sh" POOL_LIB_DIR="${REV}"
echo "  worker code snapshot: ${REV}"
export POOL_PER_JOB="${PER_JOB}" POOL_ROOT
# Only pin a budget when the caller asked for one; otherwise each worker sizes
# itself from the GPU it actually lands on (see worker.sh).
if (( GPUGB > 0 )); then export POOL_GPU_BUDGET_GB=$(( GPUGB / PER_JOB )); else unset POOL_GPU_BUDGET_GB || true; fi
echo "pool: ${JOBS} jobs x ${PER_JOB} workers on [${PARTS}], ${TIME}, $( (( GPUGB > 0 )) && echo "$(( GPUGB / PER_JOB )) GB each" || echo "budget sized from each node's GPU" )"
for ((j = 1; j <= JOBS; j++)); do
    args=( "${SBATCH_ACCEPT_COST}" -J "pool-w${j}"
           -o "${POOL_ROOT}/logs/worker_%j.out" -e "${POOL_ROOT}/logs/worker_%j.err"
           --export=ALL -p "${PARTS}" -c "${CPUS}" --mem=120G --gres=gpu:1
           -t "${TIME}" ${EXCLUDE_ARG} "${POOL_ROOT}/worker_job.sbatch" )
    if [[ "${DRY}" == true ]]; then echo "  would submit: sbatch ${args[*]}"
    else echo "  w${j}: job $(sbatch --parsable "${args[@]}")"; fi
done
