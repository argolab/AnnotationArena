#!/bin/bash
set -euo pipefail
source /weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/config/cluster.sh
rerun_activate_env
cd "${RANKING_ROOT}"
SP="$1"
BASE="${RUNS_ROOT}/legacy/dataseed-s101/DATA/STAN/DOMAIN3-SIGMOID/Base/DOMAIN3-SIGMOID_BASE_5000_50_9"
REF="${RUNS_ROOT}/legacy/dataseed-s101/DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_500/data_bundle.json"

common=(--input-dir "${BASE}" --run-prefix DOMAIN3-SIGMOID
        --item-trans-sizes 500 --item-nt-sizes "" --annot-trans-sizes "" --annot-nt-sizes ""
        --test-items 100 --test-annotators 10)

echo "### A. regression: no --observed-rate must reproduce the shipped split"
python STAN/stan_code/scripts/domain3subsplit.py "${common[@]}" --output-root "${SP}/noflag" >/dev/null
NEW="${SP}/noflag/ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_500/data_bundle.json"
if cmp -s "${NEW}" "${REF}"; then echo "PASS: byte-identical to shipped bundle"; else
  echo "DIFF -- comparing parsed content"; python - "$NEW" "$REF" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
for k in sorted(set(a)|set(b)):
    if a.get(k)!=b.get(k): print("  differs:",k)
PY
fi

echo
echo "### B. three rates, fixed mask seed"
for r in 0.15 0.50 0.85; do
  python STAN/stan_code/scripts/domain3subsplit.py "${common[@]}" \
      --observed-rate "$r" --mask-seed 20261017 --output-root "${SP}/o${r}" | sed -n '/remasked/,+2p'
done

echo
echo "### C. nesting + per-cell counts"
python - "${SP}" <<'PY'
import json, sys
from pathlib import Path
sp = Path(sys.argv[1])
def load(r):
    p = sp/f"o{r}"/"ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_500/data_bundle.json"
    return json.load(open(p))
key = lambda d: (d["attribute"], d["annotator"], d["item"], d["instance"])
sets, stats = {}, {}
for r in ("0.15","0.50","0.85"):
    b = load(r)
    sets[r] = {"obs": {key(x) for x in b["observed_ratings"]},
               "mis": {key(x) for x in b["missing_ratings"]}}
    s = b["stats"]; stats[r] = s
    print(f"  rate {r}: total={s['total_ratings']} "
          f"train {s['train_observed']}obs/{s['train_ratings']-s['train_observed']}mis  "
          f"test {s['test_observed']}obs/{s['test_ratings']-s['test_observed']}mis  "
          f"realised_obs={s['observation_rate']:.4f}")
a,b_,c = sets["0.15"], sets["0.50"], sets["0.85"]
print("  nesting obs 15 in 50 :", a["obs"] <= b_["obs"])
print("  nesting obs 50 in 85 :", b_["obs"] <= c["obs"])
print("  nesting mis 85 in 50 :", c["mis"] <= b_["mis"])
print("  nesting mis 50 in 15 :", b_["mis"] <= a["mis"])
core = a["mis"] & b_["mis"] & c["mis"]
core_test = {k for k in core if k[3] == "test"}
print(f"  common-core missing cells: {len(core)}  (test-instance: {len(core_test)})")
print(f"  expected core == missing@85: {len(core)==len(c['mis'])}")
PY
