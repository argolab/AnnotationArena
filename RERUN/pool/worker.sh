#!/bin/bash
# One pool worker: claim a task, run it, repeat until the walltime runs out.
#
#   POOL_GPU_BUDGET_GB   the GPU memory this worker may use (default 70)
#   POOL_WORKER_LABEL    name in logs (default: jobid.rank)
#
# Packing rule: take the LARGEST task that still fits in the remaining time.
# Long tasks are the ones that cannot be squeezed in later, so they go first and
# short ones fill the tail. The worker exits as soon as nothing fits -- there is
# no point holding a GPU for forty minutes when the shortest remaining task needs
# an hour, and releasing it early stops the billing.
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
pool_dirs

# Ask the GPU how big it is rather than trusting a number chosen at submit time.
# Workers are deliberately submitted across partitions, so the same worker script
# lands on an 80 GB H100, a 141 GB H200 or a 44 GB L40S -- and a budget of 22 GB
# per worker, fine on an H100, put three 18 GB trainings on a 44 GB L40S and OOMed
# the third. 0.85 leaves room for the context and allocator overhead.
if [[ -z "${POOL_GPU_BUDGET_GB:-}" ]]; then
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ -n "${total_mib}" ]]; then
        BUDGET="$(awk -v m="${total_mib}" -v n="${POOL_PER_JOB:-1}" \
                  'BEGIN { printf "%d", (m / 1024) * 0.85 / n }')"
    else
        BUDGET=20
    fi
else
    BUDGET="${POOL_GPU_BUDGET_GB}"
fi
LABEL="${POOL_WORKER_LABEL:-${SLURM_JOB_ID:-local}.${1:-0}}"
DEADLINE="$(pool_deadline)"
say() { echo "[$(date +%H:%M:%S) worker ${LABEL}] $*"; }

say "up on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)); budget ${BUDGET} GB; $(( (DEADLINE - $(pool_now)) / 60 )) min of walltime"

claimed_ok=0; ran=0
while true; do
    now="$(pool_now)"
    left=$(( DEADLINE - now - POOL_RESERVE_S ))
    if (( left <= 0 )); then say "out of time"; break; fi
    # Scale the estimates by what has actually been happening on shared
    # nodes, so a worker never claims work the walltime will cut off.
    SAFETY="$(python "${POOL_LIB_DIR}/calibrate.py" "${POOL_ROOT}" "${POOL_SAFETY}" 2>/dev/null || echo "${POOL_SAFETY}")"

    # Candidates: not finished, not held by a live worker, small enough, and
    # short enough to finish. Sorted longest-first for packing.
    best=""; best_eta=-1; best_prio=999999; any_left=0; min_eta=""; blocked_mem=0
    for t in "${POOL_ROOT}"/tasks/*.json; do
        [[ -e "${t}" ]] || break
        id="$(basename "${t}" .json)"
        [[ -e "${POOL_ROOT}/state/${id}.done"   ]] && continue
        [[ -e "${POOL_ROOT}/state/${id}.failed" ]] && continue
        lock="${POOL_ROOT}/state/${id}.lock"
        if [[ -d "${lock}" ]]; then
            hb="${lock}/heartbeat"
            if [[ -e "${hb}" ]] && (( now - $(stat -c %Y "${hb}" 2>/dev/null || echo "${now}") < POOL_STALE_S )); then
                continue
            fi
            # The owner is gone (node died, job killed). Reclaim rather than
            # leaving the task stranded.
            say "reclaiming stale ${id}"
            rm -rf "${lock}"
        fi
        eta="$(pool_field "${t}" eta_s)"; eta="${eta:-1800}"
        gb="$(pool_field "${t}" gpu_gb)"; gb="${gb:-20}"
        any_left=1
        [[ -z "${min_eta}" ]] && min_eta="${eta}"
        (( eta < min_eta )) && min_eta="${eta}"
        # awk, not bc: bc is not installed on these compute nodes, and a missing
        # bc made every fit test silently succeed -- a worker with ten minutes
        # left would happily claim an hour-long task and have it killed.
        # Separate the two reasons a task can be unavailable. Time is terminal --
        # it only ever decreases, so a task that does not fit now never will.
        # Memory is not: the task may simply belong to a bigger worker, and more
        # work can arrive at any moment. Conflating them made a worker holding a
        # reserved GPU quit because the only tasks left were too big for it, and
        # sit idle through the next batch.
        verdict="$(awk -v gb="${gb}" -v bud="${BUDGET}" -v e="${eta}" -v s="${SAFETY}" -v l="${left}" \
                   'BEGIN { if (e * s > l) print "time"; else if (gb > bud) print "mem"; else print "ok" }')"
        [[ "${verdict}" == "mem"  ]] && { blocked_mem=1; continue; }
        [[ "${verdict}" == "time" ]] && continue
        prio="$(pool_field "${t}" priority)"; prio="${prio:-50}"
        # Lower priority number first (the suites that matter most), and within a
        # priority the longest task that still fits -- long tasks are the ones
        # that cannot be squeezed in later, short ones fill the tail.
        if (( prio < best_prio )) || { (( prio == best_prio )) && (( eta > best_eta )); }; then
            best_prio="${prio}"; best_eta="${eta}"; best="${id}"
        fi
    done

    if [[ -z "${best}" ]]; then
        if (( any_left == 0 )); then
            if [[ "${POOL_WAIT_WHEN_EMPTY:-0}" == "1" ]]; then
                # For a worker inside an allocation we hold anyway (the reserved
                # node), idling costs nothing and quitting costs the rest of the
                # reservation. A Slurm worker does the opposite: it exits so the
                # allocation is released.
                say "queue empty; waiting for new work (${ran} run so far)"
                sleep 120
                continue
            fi
            say "queue empty; ${ran} task(s) run"
            break
        fi
        if (( blocked_mem == 1 )); then
            # Work exists, it is just bigger than this worker's share of the GPU.
            # Wait rather than quit: more tasks arrive, and a reserved node that
            # exits here stays idle for the rest of its allocation.
            say "waiting: remaining tasks exceed this worker's ${BUDGET} GB budget (${left}s of walltime left)"
            sleep 120
            continue
        fi
        say "nothing fits in the time left: ${left}s remaining, shortest task needs ~${min_eta}s x ${SAFETY} (calibrated); exiting early"
        break
    fi

    lock="${POOL_ROOT}/state/${best}.lock"
    if ! mkdir "${lock}" 2>/dev/null; then continue; fi   # lost the race; look again
    echo "${LABEL} $(hostname) $$" > "${lock}/owner"
    touch "${lock}/heartbeat"
    ( while [[ -d "${lock}" ]]; do touch "${lock}/heartbeat" 2>/dev/null; sleep 60; done ) &
    HB=$!

    task="${POOL_ROOT}/tasks/${best}.json"
    name="$(pool_field "${task}" label)"
    marker="$(pool_field "${task}" done_marker)"
    if [[ -n "${marker}" && -e "${marker}" ]]; then
        say "${best} (${name}) already has its output; marking done without rerunning"
        touch "${POOL_ROOT}/state/${best}.done"
        kill "${HB}" 2>/dev/null; rm -rf "${lock}"; continue
    fi

    say "start ${best} (${name}), est $(( best_eta / 60 )) min"
    start="$(pool_now)"
    # Run the command from a variable, not from a file we just wrote. Writing
    # cmd.sh and immediately exec'ing it round-trips through a distributed
    # filesystem, and that window is real: one task died with rc=127 and
    # "No such file or directory" naming a script that was on disk, correctly,
    # by the time anyone looked. The copy on disk is kept for debugging only.
    cmd="$(pool_field "${task}" cmd)"
    printf '%s\n' "${cmd}" > "${lock}/cmd.sh"
    if [[ -z "${cmd}" ]]; then
        say "SKIP ${best}: task file has no command"
        rm -rf "${lock}"; continue
    fi
    set +e
    bash -c "${cmd}" > "${POOL_ROOT}/logs/${best}.log" 2>&1
    rc=$?
    dur=$(( $(pool_now) - start ))
    kill "${HB}" 2>/dev/null
    if (( rc == 0 )); then
        printf '%s %s %ss\n' "${LABEL}" "$(date -Is)" "${dur}" > "${POOL_ROOT}/state/${best}.done"
        say "done  ${best} in $(( dur / 60 ))m (estimate was $(( best_eta / 60 ))m)"
        ran=$(( ran + 1 ))
    else
        {
            printf '%s %s rc=%s %ss\n' "${LABEL}" "$(date -Is)" "${rc}" "${dur}"
            echo "--- last lines of logs/${best}.log ---"
            tail -5 "${POOL_ROOT}/logs/${best}.log" 2>/dev/null
        } > "${POOL_ROOT}/state/${best}.failed"
        say "FAILED ${best} rc=${rc} after $(( dur / 60 ))m; see logs/${best}.log"
    fi
    rm -rf "${lock}"
done
say "exit"
