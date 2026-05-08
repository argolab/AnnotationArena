#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

DOMAIN3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"
DOMAIN3_SCRIPT_REPO_ROOT="$(cd "${DOMAIN3_SCRIPT_DIR}/../.." && pwd)"

domain3__looks_like_repo_root() {
    local candidate="$1"
    [[ -n "${candidate}" && -d "${candidate}/imputer/entity_mf" && -d "${candidate}/scripts/DOMAIN3" ]]
}

domain3__resolve_repo_root() {
    if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" ]] && domain3__looks_like_repo_root "${DOMAIN3_REPO_ROOT_OVERRIDE}"; then
        echo "${DOMAIN3_REPO_ROOT_OVERRIDE}"
        return
    fi

    if domain3__looks_like_repo_root "${DOMAIN3_SCRIPT_REPO_ROOT}"; then
        echo "${DOMAIN3_SCRIPT_REPO_ROOT}"
        return
    fi

    if domain3__looks_like_repo_root "${DOMAIN3_DEFAULT_LOCAL_ROOT}"; then
        echo "${DOMAIN3_DEFAULT_LOCAL_ROOT}"
        return
    fi

    if domain3__looks_like_repo_root "${DOMAIN3_DEFAULT_CLUSTER_ROOT}"; then
        echo "${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
        return
    fi

    echo "${DOMAIN3_SCRIPT_REPO_ROOT}"
}

DOMAIN3_REPO_ROOT="$(domain3__resolve_repo_root)"

domain3_repo_root() {
    echo "${DOMAIN3_REPO_ROOT}"
}

domain3_enter_repo() {
    cd "${DOMAIN3_REPO_ROOT}"
}

domain3_setup_env() {
    set -euo pipefail

    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
    if command -v module >/dev/null 2>&1; then
        module load anaconda3/2024.02-1 >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate prabhav2 >/dev/null 2>&1 || true
    fi

    export PYTHONPATH="${DOMAIN3_REPO_ROOT}"
    export CUDA_LAUNCH_BLOCKING=1
    export PYTHONUNBUFFERED=1
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
    export TMPDIR="${TMPDIR:-/tmp/${USER:-user}/domain3/${SLURM_JOB_ID:-manual}}"
    export DOMAIN3_LOG_DIR="${DOMAIN3_REPO_ROOT}/logs"
    mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${DOMAIN3_LOG_DIR}"
}

domain3_choose_torch_device() {
    python - <<'PY'
try:
    import torch
except Exception:
    print("cpu")
    raise SystemExit(0)

if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
}

domain3_enable_file_logging() {
    local exp_name="$1"
    if [[ -n "${DOMAIN3_LOGGING_ACTIVE:-}" ]]; then
        return
    fi

    mkdir -p "${DOMAIN3_LOG_DIR}"
    local log_path="${DOMAIN3_LOG_DIR}/${exp_name}.out"
    export DOMAIN3_LOGGING_ACTIVE=1
    export DOMAIN3_LOG_FILE="${log_path}"
    exec > >(tee -a "${log_path}") 2>&1
}

domain3_print_block() {
    local title="$1"
    echo ""
    echo "============================================================"
    echo " ${title}"
    echo "============================================================"
}

domain3_stage_stan_file() {
    local stan_src="$1"
    local resolved_src="${stan_src}"
    if [[ "${resolved_src}" != /* ]]; then
        resolved_src="${DOMAIN3_REPO_ROOT}/${resolved_src}"
    fi

    local stage_dir="${XDG_CACHE_HOME}/domain3_stan/${SLURM_JOB_ID:-$$}"
    mkdir -p "${stage_dir}"

    local base_name
    base_name="$(basename "${resolved_src}")"
    local local_stan="${stage_dir}/${base_name}"
    cp "${resolved_src}" "${local_stan}"
    echo "${local_stan}"
}

domain3_write_json_payload() {
    local output_path="$1"
    local payload_json="$2"

    python - "$output_path" "$payload_json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
payload["written_at_utc"] = datetime.now(timezone.utc).isoformat()
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2))
PY
}
