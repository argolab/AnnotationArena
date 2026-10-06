"""Does the patched plot path reproduce corner.json's test-corner log-loss?

If these disagree, the figures and the tables are still different quantities and
nothing downstream is trustworthy.
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RANK))
spec = importlib.util.spec_from_file_location(
    "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
m = importlib.util.module_from_spec(spec)
sys.modules["pd3"] = m          # dataclass() resolves cls.__module__ through sys.modules
spec.loader.exec_module(m)

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
corner = {}
for r in json.loads((RUNS / "corner_metrics/corner.json").read_text())["rows"]:
    v = (r.get("test") or {}).get("log_loss")
    if v is not None:
        corner[(r["family"], str(r["axis"]).lower(), int(r["size"]))] = (v, r["test"]["n"])

CASES = [("Marformer", "item", 2000, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_2000_MARFORMER"),
         ("Marformer", "item", 300,  "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"),
         ("MarformerNT", "item", 4000, "MARFORMER-NT/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_NT_4000_MARFORMER"),
         ("Marformer", "annot", 20, "MARFORMER/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_T_20_MARFORMER")]

for pop in ("corner", "pooled"):
    m.METRIC_POPULATION = pop
    print(f"\n=== METRIC_POPULATION = {pop} ===")
    print(f"{'cell':26s}{'plot n':>8}{'plot ll':>10}{'corner n':>10}{'corner ll':>11}{'diff':>9}")
    for fam, axis, K, rel in CASES:
        rd = RUNS / "RESULTS" / rel
        probs, labels = m._marformer_probs_all_missing(rd, "last.ckpt", 500, "cuda")
        ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
        cv, cn = corner[(fam, axis, K)]
        print(f"{fam+'/'+axis+'/'+str(K):26s}{len(labels):>8}{ll:>10.4f}{cn:>10}{cv:>11.4f}{ll-cv:>+9.4f}")
