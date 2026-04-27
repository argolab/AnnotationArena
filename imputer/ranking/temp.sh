#!/usr/bin/env bash
set -euo pipefail

# Full pipeline:
# 1) regenerate tensor-nobin data with fixed generator
# 2) subset K_train down to 200/100/50/10
# 3) relabel size-10 split: train+val -> train, test -> val
# 4) print sanity stats for observed rows

ROOT="/home/xwang397/AA_new/imputer/ranking"
cd "$ROOT"

BASE_DIR="DATA/STAN/SPARSE/MARFORMER_NOBIN_D8"
RUN_NAME="Tensor_400_25_9_ItemTest_300_D8"
FULL_DIR="${BASE_DIR}/${RUN_NAME}"

# Rewrite missing-rating "value" fields from base_scores so bundles never carry
# placeholder literal zeros for masked cells (safe if something downstream reads them).
patch_tensor_nobin_missing_values() {
  local target_dir="$1"
  PYTHONPATH=. python - "$target_dir" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
bp, cp = d / "data_bundle.json", d / "configs.json"
if not bp.is_file():
    sys.exit(f"missing {bp}")
bundle = json.loads(bp.read_text())
cfg = json.loads(cp.read_text()) if cp.is_file() else {}
dg = cfg.get("datagen", cfg)
J = int(dg.get("J", 0) or 0)
if J <= 0:
    sys.exit("configs missing datagen.J; cannot patch missing rating layout")

bs = np.asarray(bundle.get("base_scores"), dtype=np.float64)
if bs.ndim != 2:
    print("patch_tensor_nobin_missing_values: skip (no base_scores matrix)", file=sys.stderr)
    sys.exit(0)


def lookup(r: dict) -> float:
    i = int(r["attribute"]) - 1
    j = int(r["annotator"]) - 1
    k = int(r["item"]) - 1
    return float(bs[i * J + j, k])


def no_zero_placeholder(z: float) -> float:
    # Missing rows are not supervised; store deterministic base score. Avoid exact
    # 0.0 so JSON never looks like an uninitialized Stan cell.
    if abs(z) < 1e-15:
        return 1e-8
    return z


missing_keys = set()
for r in bundle.get("missing_ratings", []) or []:
    r["value"] = no_zero_placeholder(lookup(r))
    missing_keys.add(
        (int(r["attribute"]), int(r["annotator"]), int(r["item"]), r.get("instance"))
    )

for r in bundle.get("all_ratings", []) or []:
    key = (int(r["attribute"]), int(r["annotator"]), int(r["item"]), r.get("instance"))
    if key in missing_keys:
        r["value"] = no_zero_placeholder(lookup(r))

bp.write_text(json.dumps(bundle, indent=2))
print("patched missing rating values -> base_scores (non-zero) in", bp)
PY
}

echo "== Step 1: regenerate full 300 split =="
PYTHONPATH=. python STAN/stan_code/scripts/generate_data.py \
  --output-dir "$BASE_DIR" \
  --run-name "$RUN_NAME" \
  --overwrite-existing-data \
  --force-stan-recompile \
  --stan-type tensor-nobin \
  --K-train 300 --K-val 50 --K-test 50 \
  --I 9 --J 25 --C 5 --D 8 \
  --sigma-u 1.0 --sigma-v 1.0 --sigma-uit 0.0 \
  --sigma-measurement 0.1 \
  --kappa 15.0 --alpha-confusion 15.0 --temperature 0.5 \
  --use-dawid-skene-noise 0 \
  --observation-protocol mcar --mcar-missing-rate 0.5 \
  --seed 42 \
  --stan-arg num_annotate_annotator=4

patch_tensor_nobin_missing_values "$FULL_DIR"

echo "== Step 2: subset train sizes =="
for K in 200 100 50 10; do
  OUT_DIR="${BASE_DIR}/Tensor_400_25_9_ItemTest_${K}_D8"
  python STAN/stan_code/scripts/subset_item_split.py \
    --input-dir "$FULL_DIR" \
    --output-dir "$OUT_DIR" \
    --train-num "$K"
  patch_tensor_nobin_missing_values "$OUT_DIR"
done

echo "== Step 3: relabel size-10 split (train+val->train, test->val) =="
python - <<'PY'
import json
import shutil
from pathlib import Path

in_dir = Path("DATA/STAN/SPARSE/MARFORMER_NOBIN_D8/Tensor_400_25_9_ItemTest_10_D8")
out_dir = Path("DATA/STAN/SPARSE/MARFORMER_NOBIN_D8/Tensor_400_25_9_ItemTest_10_D8_trainplusval_train_testasval")
out_dir.mkdir(parents=True, exist_ok=True)

bundle = json.load(open(in_dir / "data_bundle.json"))
cfg = json.load(open(in_dir / "configs.json"))
mapping = {"train": "train", "val": "train", "test": "val"}

def relabel(rows):
    out = []
    for r in rows:
        rr = dict(r)
        rr["instance"] = mapping.get(rr.get("instance"), rr.get("instance"))
        out.append(rr)
    return out

for key in ["all_ratings", "observed_ratings", "missing_ratings", "all_pairwise", "observed_pairwise", "missing_pairwise"]:
    if key in bundle and bundle[key] is not None:
        bundle[key] = relabel(bundle[key])

bundle["missing_ratings_indexes_in_test_instance"] = [
    i for i, r in enumerate(bundle.get("missing_ratings", [])) if r.get("instance") == "test"
]
bundle.setdefault("stats", {})["split_mode"] = "trainplusval_train_testasval"
cfg.setdefault("datagen", {})["split_mode"] = "trainplusval_train_testasval"
cfg["datagen"]["instance_relabel_map"] = mapping

json.dump(bundle, open(out_dir / "data_bundle.json", "w"))
json.dump(cfg, open(out_dir / "configs.json", "w"))
json.dump({"mapping": mapping}, open(out_dir / "relabel_meta.json", "w"))
if (in_dir / "stan_data.json").exists():
    shutil.copy2(in_dir / "stan_data.json", out_dir / "stan_data.json")

print("wrote", out_dir)
PY

RELABEL_DIR="DATA/STAN/SPARSE/MARFORMER_NOBIN_D8/Tensor_400_25_9_ItemTest_10_D8_trainplusval_train_testasval"
patch_tensor_nobin_missing_values "$RELABEL_DIR"

echo "== Step 4: quick sanity stats on relabeled split =="
python - <<'PY'
import json
import numpy as np
from pathlib import Path

p = Path("DATA/STAN/SPARSE/MARFORMER_NOBIN_D8/Tensor_400_25_9_ItemTest_10_D8_trainplusval_train_testasval/data_bundle.json")
b = json.load(open(p))
bs = np.array(b["base_scores"], dtype=float)

def counts(rows):
    return {k: sum(1 for r in rows if r.get("instance") == k) for k in ["train", "val", "test"]}

print("all_ratings:", counts(b.get("all_ratings", [])))
print("observed_ratings:", counts(b.get("observed_ratings", [])))
print("missing_ratings:", counts(b.get("missing_ratings", [])))

obs = b.get("observed_ratings", [])
sq = []
for r in obs:
    i = int(r["attribute"]) - 1
    j = int(r["annotator"]) - 1
    k = int(r["item"]) - 1
    pred = bs[i * 25 + j, k]
    sq.append((pred - float(r["value"])) ** 2)
print("oracle base_scores mse on observed:", float(np.mean(sq)) if sq else None)
print("done")
PY
