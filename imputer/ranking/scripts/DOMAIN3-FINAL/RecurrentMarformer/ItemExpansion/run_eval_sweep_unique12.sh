#!/bin/bash
# Evaluate all runs under a Recurrent MF sweep root.
#
# Env:
#   RESULTS_ROOT, MAX_ITEM (optional), FULL_GRAPH=1, DEVICE
#   MAX_ITEM=300 -> writes TEST_RESULTS_MAXITEM300/ (does not overwrite TEST_RESULTS/)

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
GLOB="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
MAX_ITEM="${MAX_ITEM:-}"
FULL_GRAPH="${FULL_GRAPH:-}"
DEVICE="${DEVICE:-cuda}"

TEST_OUT_SUBDIR="$(python3 -c "
from imputer.entity_mf.recurrent.eval_paths import test_results_dir_name
mi = int('${MAX_ITEM}') if '${MAX_ITEM}' else None
print(test_results_dir_name(mi, full_graph=bool('${FULL_GRAPH}')))
")"

MAX_ITEM_ARGS=()
if [ -n "$MAX_ITEM" ]; then
    MAX_ITEM_ARGS=(--max-item "$MAX_ITEM")
fi
FULL_GRAPH_ARGS=()
if [ -n "$FULL_GRAPH" ]; then
    FULL_GRAPH_ARGS=(--full-graph)
fi

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}"/${GLOB} )
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No runs found: ${RESULTS_ROOT}/${GLOB}"
    exit 1
fi

echo "Test output subdir per run: ${TEST_OUT_SUBDIR}/"
echo "Eval max_item: ${MAX_ITEM:-train_config}  FULL_GRAPH=${FULL_GRAPH:-0}"
echo ""

for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo ""
    echo "--- $(basename "$RUN_DIR") ---"
    OUT_DIR="${RUN_DIR}/${TEST_OUT_SUBDIR}"
    rm -rf "$OUT_DIR"

    python -u -m imputer.entity_mf.recurrent.test \
        --run-dir "$RUN_DIR" --out-dir "$OUT_DIR" --checkpoint all --device "$DEVICE" \
        "${MAX_ITEM_ARGS[@]}" "${FULL_GRAPH_ARGS[@]}" || true
    python -u -m imputer.entity_mf.recurrent.test \
        --run-dir "$RUN_DIR" --out-dir "$OUT_DIR" --checkpoint latest --device "$DEVICE" \
        "${MAX_ITEM_ARGS[@]}" "${FULL_GRAPH_ARGS[@]}" || true

    python - "$RUN_DIR" "$TEST_OUT_SUBDIR" <<'PY'
import json, math, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
test_dir = run_dir / sys.argv[2]
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
