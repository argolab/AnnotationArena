#!/bin/bash
# Put failed tasks back in the queue.
#
# A task fails for one of two reasons: the work is genuinely broken, or the
# cluster hiccuped. Only a human can tell them apart from the log, so this does
# not run automatically -- but once a failure has been diagnosed and fixed,
# clearing its state file is all that is needed to have a worker pick it up
# again. Prints what it is clearing so the decision is on the record.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
n=0
for f in "${POOL_ROOT}"/state/*.failed; do
    [[ -e "${f}" ]] || break
    id="$(basename "${f}" .failed)"
    echo "requeue ${id}: $(pool_field "${POOL_ROOT}/tasks/${id}.json" label)"
    echo "   was: $(head -1 "${f}")"
    rm -f "${f}"; rm -rf "${POOL_ROOT}/state/${id}.lock"
    n=$((n+1))
done
echo "${n} task(s) requeued"
