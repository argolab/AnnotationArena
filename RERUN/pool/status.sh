#!/bin/bash
# What is in the pool right now.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
pool_dirs
now="$(pool_now)"
tot=0; done_=0; fail=0; run=0; wait_=0; eta_left=0
for t in "${POOL_ROOT}"/tasks/*.json; do
    [[ -e "${t}" ]] || break
    id="$(basename "${t}" .json)"; tot=$(( tot + 1 ))
    if   [[ -e "${POOL_ROOT}/state/${id}.done"   ]]; then done_=$(( done_ + 1 ))
    elif [[ -e "${POOL_ROOT}/state/${id}.failed" ]]; then fail=$(( fail + 1 ))
    elif [[ -d "${POOL_ROOT}/state/${id}.lock"   ]]; then
        run=$(( run + 1 ))
        printf '  RUNNING %s  %s  (%s, started %s min ago)\n' "${id}" \
            "$(pool_field "${t}" label)" "$(cat "${POOL_ROOT}/state/${id}.lock/owner" 2>/dev/null | cut -d' ' -f1)" \
            "$(( (now - $(stat -c %Y "${POOL_ROOT}/state/${id}.lock" 2>/dev/null || echo "${now}")) / 60 ))"
    else
        wait_=$(( wait_ + 1 ))
        eta_left=$(( eta_left + $(pool_field "${t}" eta_s) ))
    fi
done
echo "pool ${POOL_ROOT}"
echo "  ${tot} tasks: ${done_} done, ${run} running, ${wait_} waiting, ${fail} failed"
(( wait_ > 0 )) && echo "  queued work remaining: ~$(( eta_left / 3600 )) GPU-hours"
(( fail > 0 )) && { echo "  failures:"; for f in "${POOL_ROOT}"/state/*.failed; do
    [[ -e "${f}" ]] || break; id="$(basename "${f}" .failed)"
    echo "    ${id}  $(pool_field "${POOL_ROOT}/tasks/${id}.json" label)  -- $(cat "${f}")"; done; }
echo "  workers: $(squeue -u "${USER}" -h -n pool-w1,pool-w2,pool-w3,pool-w4,pool-w5,pool-w6 -o '%T' 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
