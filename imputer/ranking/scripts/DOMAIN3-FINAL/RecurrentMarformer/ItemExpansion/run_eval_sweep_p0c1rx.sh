#!/bin/bash
# Evaluate all runs under DOMAIN3-OLD-P0C1RX (p0c1r{x}c0 sweep).

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-P0C1RX}"
GLOB="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
MAX_ITEM="${MAX_ITEM:-}"
DEVICE="${DEVICE:-cuda}"

MAX_ITEM_ARGS=()
if [ -n "$MAX_ITEM" ]; then
    MAX_ITEM_ARGS=(--max-item "$MAX_ITEM")
fi

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}"/${GLOB} )
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No runs found: ${RESULTS_ROOT}/${GLOB}"
    exit 1
fi

for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo ""
    echo "--- $(basename "$RUN_DIR") ---"
    rm -rf "${RUN_DIR}/TEST_RESULTS"

    python -u -m imputer.entity_mf.recurrent.test \
        --run-dir "$RUN_DIR" --checkpoint all --device "$DEVICE" "${MAX_ITEM_ARGS[@]}" || true
    python -u -m imputer.entity_mf.recurrent.test \
        --run-dir "$RUN_DIR" --checkpoint last --device "$DEVICE" "${MAX_ITEM_ARGS[@]}" || true

    python - "$RUN_DIR" <<'PY'
import json, math, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
test_dir = run_dir / "TEST_RESULTS"
candidates = []
for path in sorted(test_dir.glob("*.json")):
    if path.name == "best.json":
        continue
    with open(path) as f:
        data = json.load(f)
    missing = data.get("missing", {})
    ll, n = missing.get("log_loss"), missing.get("n", 0)
    if ll is None or n == 0:
        continue
    rmse = missing.get("rmse")
    candidates.append((float(ll), float(rmse) if rmse is not None else math.inf, path.name, data))
if candidates:
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, name, best = candidates[0]
    best["selected_from"] = name
    (test_dir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"  best: {name}  log_loss={candidates[0][0]:.4f}")
else:
    print("  (no valid test metrics)")
PY
done

echo ""
echo "Done. Results under ${RESULTS_ROOT}"
