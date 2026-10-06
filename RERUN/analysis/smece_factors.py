#!/usr/bin/env python
"""Which of the four differences between the published run and the final run moved smECE?

`replicate-01` -> `final-dual` changed four things at once:

  1 architecture    outer residual + trailing projection  ->  standard block
  2 chunk max_item  300                                   ->  500
  3 length          400 epochs                            ->  ~106 (F13-selected)
  4 protocol        one run                               ->  search on 85% + refit on 100%

Factor 2 is already measured at ~0 (smece_decompose.py --eval-max-item). This
isolates the other three from checkpoints that are already on disk -- no retraining:

  1  proj-c2-s101 trained `proj_keep_add` (the OLD block) against `proj_drop_rot`
     (the standard block final-dual ships) on the same draw, seeds and sizes, so
     the architecture is the only thing that differs. Paired on (seed, K).
  3  final-dual's SEARCH run keeps periodic checkpoints out past epoch 1900, so the
     new architecture can be read at the epoch F13 picked and at much longer
     training, holding everything else fixed.
  4  the search run (85% of the training partition) and the refit (100%, same
     epoch count, fresh fit) differ only in the thing "add the dev set back and
     retrain" means.

    python smece_factors.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs")


def load_module():
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    return m


def nearest_ckpt(run_dir: Path, epoch: int) -> str | None:
    """The periodic checkpoint closest to `epoch` (they are every 25)."""
    eps = []
    for p in (run_dir / "checkpoints").glob("periodic-epoch=*.ckpt"):
        mm = re.search(r"(\d+)\.ckpt$", p.name)
        if mm:
            eps.append((int(mm.group(1)), p.name))
    if not eps:
        return None
    return min(eps, key=lambda t: abs(t[0] - epoch))[1]


def score(m, run_dir: Path, ckpt: str, pop: str, max_item: int, device: str):
    m.METRIC_POPULATION = pop
    probs, labels = m._marformer_probs_all_missing(run_dir, ckpt, max_item, device)
    conf, acc = m.shared._all_class_calibration(probs, labels)
    diag = m.shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                           report_CE=True, report_CE_std=True)
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
    return float(diag.get("ce", float("nan"))), nll, int(len(labels))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--population", default="corner", choices=["corner", "pooled"])
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("-o", "--out", default=str(RUNS / "final-dual/reports/smece_factors.json"))
    a = ap.parse_args()
    m = load_module()
    rows = []

    def add(factor, level, cell, run_dir, ckpt):
        if ckpt is None or not (run_dir / "checkpoints" / ckpt).is_file():
            print(f"  [skip] {factor}/{level}/{cell}: no {ckpt} under {run_dir}")
            return
        ce, nll, n = score(m, run_dir, ckpt, a.population, a.max_item, a.device)
        rows.append(dict(factor=factor, level=level, cell=cell, ckpt=ckpt,
                         smece=ce, log_loss=nll, n=n, run=str(run_dir)))
        print(f"  {factor:<12} {level:<22} {cell:<16} smECE={ce:.4f} ll={nll:.4f} "
              f"n={n} [{ckpt}]", flush=True)

    # ---- factor 1: architecture (old block vs standard block) ----------------
    P = RUNS / "proj-c2-s101/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
    for arm, level in (("proj_keep_add", "old block"), ("proj_drop_rot", "standard block")):
        for seed in (42, 43, 44):
            for K in (300, 2000):
                rd = P / arm / "Marformer" / f"seed{seed}" / f"DOMAIN3-SIGMOID_Item_T_{K}_MARFORMER"
                if not rd.is_dir():
                    continue
                add("architecture", level, f"item/{K} s{seed}", rd, "last.ckpt")

    # ---- factors 3 and 4: length, and search(85%) vs refit(100%) -------------
    FD = RUNS / "final-dual/RESULTS/MARFORMER/DOMAIN3-SIGMOID"
    for axis, size, stem in (("ITEM", 500, "Item_T_500"), ("ANNOT", 30, "Annot_T_30")):
        search = FD / axis / "valblock-search" / "seed42" / f"DOMAIN3-SIGMOID_{stem}_MARFORMER"
        refit = FD / axis / "valblock" / "seed42" / f"DOMAIN3-SIGMOID_{stem}_MARFORMER"
        cell = f"{axis.lower()}/{size}"
        if search.is_dir():
            for ep in (100, 400, 1000, 1900):
                ck = nearest_ckpt(search, ep)
                if ck:
                    add("length", f"search @~{ep}ep", cell, search, ck)
        if refit.is_dir():
            add("protocol", "refit 100% (last)", cell, refit, "last.ckpt")
            ck = nearest_ckpt(search, 100)
            if ck:
                add("protocol", "search 85% (~same ep)", cell, search, ck)

    Path(a.out).write_text(json.dumps(rows, indent=2))

    # ---- summaries ----------------------------------------------------------
    def pick(factor, level, cell):
        for r in rows:
            if (r["factor"], r["level"], r["cell"]) == (factor, level, cell):
                return r
        return None

    print(f"\n=== FACTOR 1: architecture (population={a.population}) ===")
    cells = sorted({r["cell"] for r in rows if r["factor"] == "architecture"})
    diffs = []
    print(f"{'cell':<18}{'old block':>11}{'standard':>11}{'diff':>10}")
    for c in cells:
        o, n = pick("architecture", "old block", c), pick("architecture", "standard block", c)
        if o and n:
            d = n["smece"] - o["smece"]; diffs.append(d)
            print(f"{c:<18}{o['smece']:>11.4f}{n['smece']:>11.4f}{d:>+10.4f}")
    if diffs:
        import statistics as st
        se = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else 0.0
        print(f"{'PAIRED MEAN':<18}{'':>11}{'':>11}{st.mean(diffs):>+10.4f}  +/-{se:.4f} (n={len(diffs)})")

    print(f"\n=== FACTOR 3: training length, new architecture, 85% search run ===")
    for c in sorted({r["cell"] for r in rows if r["factor"] == "length"}):
        line = f"{c:<14}"
        for ep in (100, 400, 1000, 1900):
            r = pick("length", f"search @~{ep}ep", c)
            line += f"{('@'+str(ep)+': '+format(r['smece'],'.4f')) if r else '--':>18}"
        print(line)

    print(f"\n=== FACTOR 4: refit on 100% vs search on 85%, same epoch ===")
    for c in sorted({r["cell"] for r in rows if r["factor"] == "protocol"}):
        s, rf = pick("protocol", "search 85% (~same ep)", c), pick("protocol", "refit 100% (last)", c)
        if s and rf:
            print(f"{c:<14} search85={s['smece']:.4f}  refit100={rf['smece']:.4f}  "
                  f"diff={rf['smece']-s['smece']:+.4f}")
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
