#!/usr/bin/env python
"""Would a temperature fitted on the validation block fix the calibration gap?

`mean_entropy_curves.py` shows that 56-95% of the dev-loss rise under
overfitting is removable by one scalar temperature, and that the model is
already overconfident AT the epoch the protocol stops on (T* 1.2-2.8, predicted
variance 38-75% of realised squared error). That is an oracle statement -- the
temperature was fitted on the split it was scored on.

This is the honest version. For each cell, at the epoch F13 selects:

  1. fit one scalar T on the VALIDATION BLOCK the protocol already carves,
  2. apply it to the test corner, which was never used to fit anything,
  3. report corner log-loss and corner smECE before and after.

Nothing here touches the test corner before step 3, so a win is a win the real
protocol could have had. smECE goes through the same
`_all_class_calibration` -> `rp.prepare_rel_diagram(..., report_CE=True)` path
the calibration panels use, so the numbers are comparable to the figures.

The refit runs (`valblock/`) have no validation block left -- it was merged back
in -- so T is fitted on the SEARCH run's block and evaluated on the search run's
corner. That is a within-run statement about recalibration, not a replacement
for the reported refit numbers.

    python dev_temperature_scaling.py --run-dir runs/final-dual --device cuda -o out.json
"""
from __future__ import annotations

import argparse, importlib.util, json, math, re, statistics as st, sys
from pathlib import Path

import numpy as np
import torch

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RERUN))
sys.path.insert(0, str(RANK))

from analysis.mean_entropy_curves import (                      # noqa: E402
    reconstruct, split_graphs, fit_temperature)
from imputer.entity_mf.data import variable_list_to_entity_graph  # noqa: E402
from imputer.entity_mf.test import _load_checkpoint               # noqa: E402

CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
EPOCH_RE = re.compile(r"periodic-epoch=(\d+)\.ckpt$")


def load_smece():
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    shared = m.shared

    def smece(probs: np.ndarray, labels: np.ndarray) -> float:
        conf, acc = shared._all_class_calibration(probs, labels)
        d = shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500, report_CE=True)
        return float(d.get("ce", float("nan")))
    return smece


def selected_epoch(run: Path) -> int | None:
    h = json.loads((run / "training_history.json").read_text())
    ce = []
    for i, e in enumerate(h):
        v = ((e.get("val_eval") or {}).get("metrics") or {}).get(
            "missing", {}).get("rating", {}).get("xent")
        if v is not None:
            ce.append((int(e.get("epoch", i)), float(v)))
    if not ce:
        return None
    sm = [(ce[i][0], st.mean(v for _, v in ce[max(0, i - 2): i + 3]))
          for i in range(len(ce))]
    return min(sm, key=lambda t: t[1])[0]


@torch.no_grad()
def logits_for(model, types, variables, score_ids, max_item, device):
    from analysis.split_missing_metrics import chunk_by_item
    C = model.types["rating"].num_classes
    L, Y = [], []
    keep = {i for v in variables if id(v) in score_ids for i in v.item_ids}
    for chunk in chunk_by_item(variables, max_item, keep_whole=keep or None):
        graph = variable_list_to_entity_graph(chunk, types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk):
            if id(var) in score_ids and var.rating_value is not None:
                L.append(params[0, idx, 1:1 + C])
                Y.append(int(var.rating_value))
    if not L:
        return None, None
    return torch.stack(L).float(), torch.tensor(Y, device=L[0].device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="valblock-search")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    smece = load_smece()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    root = Path(a.run_dir) / "RESULTS"
    rows = []

    for fam_dir, fam in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run in sorted(base.rglob("*_MARFORMER")):
            m = CELL.search(run.name)
            if not m or a.variant not in str(run):
                continue
            sel = selected_epoch(run)
            if sel is None:
                print(f"  [skip] {run.name}: no dev curve")
                continue
            ck = [(int(x.group(1)), p) for p in (run / "checkpoints").glob("periodic-epoch=*.ckpt")
                  if (x := EPOCH_RE.search(p.name))]
            if not ck:
                continue
            ep, path = min(ck, key=lambda t: abs(t[0] - sel))

            model, types, part, train_cfg, proto, _ = reconstruct(run)
            _load_checkpoint(model, path, device)
            model.to(device).eval()
            mi = train_cfg.get("max_item")
            g = {n: (v, ids) for n, v, ids in split_graphs(part, True)}
            if "dev" not in g or "corner" not in g:
                print(f"  [skip] {run.name}: missing split")
                continue

            dl, dy = logits_for(model, types, *g["dev"], mi, device)
            cl, cy = logits_for(model, types, *g["corner"], mi, device)
            if dl is None or cl is None:
                continue

            T = fit_temperature(dl, dy)                 # fitted on dev ONLY
            ce = torch.nn.functional.cross_entropy
            p0 = torch.softmax(cl, -1).cpu().numpy()
            pT = torch.softmax(cl / T, -1).cpu().numpy()
            yl = cy.cpu().numpy()
            row = dict(family=fam, axis=m.group(1).lower(), proto=m.group(2),
                       K=int(m.group(3)), sel_epoch=sel, scored_epoch=ep,
                       n_dev=int(len(dy)), n_corner=int(len(cy)), T_dev=T,
                       corner_ll=float(ce(cl, cy)), corner_ll_T=float(ce(cl / T, cy)),
                       corner_smece=smece(p0, yl), corner_smece_T=smece(pT, yl),
                       # the oracle bound: T fitted on the corner itself
                       T_corner=fit_temperature(cl, cy))
            row["corner_ll_oracle"] = float(ce(cl / row["T_corner"], cy))
            rows.append(row)
            print(f"  {fam:<12}{row['axis']}/{row['proto']}/{row['K']:<5} ep{ep:<5} "
                  f"T_dev={T:4.2f} (corner wants {row['T_corner']:4.2f})  "
                  f"ll {row['corner_ll']:.4f} -> {row['corner_ll_T']:.4f} "
                  f"({row['corner_ll_T']-row['corner_ll']:+.4f})  "
                  f"smECE {row['corner_smece']:.4f} -> {row['corner_smece_T']:.4f} "
                  f"({row['corner_smece_T']-row['corner_smece']:+.4f})", flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))

    if rows:
        dll = [r["corner_ll_T"] - r["corner_ll"] for r in rows]
        dce = [r["corner_smece_T"] - r["corner_smece"] for r in rows]
        print(f"\n{len(rows)} cells, temperature fitted on the validation block only")
        print(f"  corner log-loss {st.mean(dll):+.4f} +/- "
              f"{st.stdev(dll)/len(dll)**0.5:.4f}  better in {sum(d<0 for d in dll)}/{len(dll)}")
        print(f"  corner smECE    {st.mean(dce):+.4f} +/- "
              f"{st.stdev(dce)/len(dce)**0.5:.4f}  better in {sum(d<0 for d in dce)}/{len(dce)}")
    print(f"\nwrote {len(rows)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
