#!/usr/bin/env python
"""Would a fixed epoch budget beat F13's validation-selected stopping epoch?

At item/500 the search run reaches corner log-loss 0.7995 at epoch 400 against
1.0087 at the epoch F13 selects (105), and smECE 0.030 against 0.077 -- while the
VALIDATION curve says the opposite (0.962 at 105, 1.198 at 400). So for that cell
the carve selects an epoch the corner does not want. This asks whether that is one
cell or the rule.

Every comparison here is on the SAME search run (85% of the training partition),
so only the budget varies -- the data fraction, architecture and chunk are held
fixed. F13's selected epoch is read from the run's own validation curve, never
from the corner.

READ THIS BEFORE ACTING ON THE OUTPUT. Choosing a budget because it wins on the
corner IS selection on test -- D14, the bug F13 exists to remove. Legitimate
conclusions: (a) if a budget fixed a priori (400 is the previously published
choice) systematically beats the selected epoch, that is evidence the carve is
mis-selecting and the carve should be examined (thread S7); (b) a budget could be
chosen on the validation blocks across cells. Illegitimate: picking whichever
budget this table says is best and reporting it.

    python fixed_budget_vs_f13.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, re, statistics as st, sys
from pathlib import Path

import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
BUDGETS = (200, 400, 800)


def load_module():
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    return m


def ckpt_at(run: Path, epoch: int):
    eps = []
    for p in (run / "checkpoints").glob("periodic-epoch=*.ckpt"):
        mm = re.search(r"(\d+)\.ckpt$", p.name)
        if mm:
            eps.append((int(mm.group(1)), p.name))
    if not eps:
        return None, None
    e, name = min(eps, key=lambda t: abs(t[0] - epoch))
    return name, e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("--smece", action="store_true", help="also compute smECE (slower)")
    ap.add_argument("-o", "--out", default=str(RUNS / "reports/fixed_budget_vs_f13.json"))
    a = ap.parse_args()
    m = load_module()
    m.METRIC_POPULATION = "corner"

    sel = {}
    for c in json.loads((RUNS / "reports/curves_final-dual.json").read_text())["cells"]:
        sel[(c["family"], c["axis"], c["K"])] = c.get("sel_epoch")

    rows = []
    for fam_dir, fam in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = RUNS / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID"
        for axis_dir, axis in (("ITEM", "item"), ("ANNOT", "annot")):
            root = base / axis_dir / "valblock-search"
            if not root.is_dir():
                continue
            for run in sorted(root.rglob("*_MARFORMER")):
                mm = re.search(r"_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$", run.name)
                if not mm:
                    continue
                K = int(mm.group(3))
                s = sel.get((fam, axis, K))
                if s is None:
                    print(f"  [skip] {fam}/{axis}/{K}: no selected epoch")
                    continue
                for label, target in [("F13-selected", s)] + [(f"fixed-{b}", b) for b in BUDGETS]:
                    name, ep = ckpt_at(run, target)
                    if name is None:
                        continue
                    probs, labels = m._marformer_probs_all_missing(
                        run, name, a.max_item, a.device)
                    nll = float(-np.log(np.clip(
                        probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
                    ce = None
                    if a.smece:
                        conf, acc = m.shared._all_class_calibration(probs, labels)
                        ce = float(m.shared.rp.prepare_rel_diagram(
                            conf, acc, num_bootstrap=500, report_CE=True).get("ce", float("nan")))
                    rows.append(dict(family=fam, axis=axis, K=K, label=label,
                                     target_epoch=target, actual_epoch=ep,
                                     log_loss=nll, smece=ce, n=int(len(labels))))
                    print(f"  {fam:<12}{axis}/{K:<5} {label:<14} ep={ep:<5} "
                          f"ll={nll:.4f}" + (f" smECE={ce:.4f}" if ce is not None else ""),
                          flush=True)

    Path(a.out).write_text(json.dumps(rows, indent=2))

    idx = {(r["family"], r["axis"], r["K"], r["label"]): r for r in rows}
    cells = sorted({(r["family"], r["axis"], r["K"]) for r in rows})
    print(f"\n=== corner log-loss: fixed budget MINUS F13-selected "
          f"(negative = the fixed budget is BETTER) ===")
    hdr = f"{'budget':<12}{'all cells':>24}{'item':>24}{'annot':>24}"
    print(hdr); print("-" * len(hdr))
    for b in BUDGETS:
        line = f"{'fixed-'+str(b):<12}"
        for grp in (None, "item", "annot"):
            ds = [idx[c + (f"fixed-{b}",)]["log_loss"] - idx[c + ("F13-selected",)]["log_loss"]
                  for c in cells
                  if (grp is None or c[1] == grp)
                  and c + (f"fixed-{b}",) in idx and c + ("F13-selected",) in idx]
            if not ds:
                line += f"{'-':>24}"; continue
            mean = st.mean(ds)
            se = st.stdev(ds) / len(ds) ** 0.5 if len(ds) > 1 else 0.0
            wins = sum(1 for d in ds if d < 0)
            line += f"{mean:>+9.4f} +/-{se:>6.4f} {wins}/{len(ds)}".rjust(24)
        print(line)
    print("\n  'wins' = cells where the fixed budget beats the selected epoch.")
    print("  Reminder: this table is a DIAGNOSTIC. Choosing a budget from it is"
          "\n  selection on test. See the module docstring.")
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
