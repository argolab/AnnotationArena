#!/usr/bin/env python
"""Reliability diagrams for train-masked against held-out, same path as the paper.

Jason, 2026-10-06: "Calibration plots might be easier to understand. I'm guessing
it is less well calibrated on held-out data (wrong but still confident, because
it learned to be overconfident on training data due to overfitting)?"

This computes the diagram through the SAME call the paper's calibration panels
use -- `shared._all_class_calibration` then
`rp.prepare_rel_diagram(num_bootstrap=500, report_CE=True, report_CE_std=True)`
-- so the smECE printed here is the same statistic, on the same convention, as
the figure. ALL-CLASS, not top-label: `_all_class_calibration` flattens the
(n, C) probability matrix against the flattened one-hot outcome, so every rated
cell contributes C points to the diagram. C is 4 for Domain 3 synthetic
(num_likert_classes), so a cell contributes four ticks, one per Likert level --
most of them the near-zero probabilities of the levels the model did not pick,
which is why the all-class curve is dominated by the low-confidence end and is
not comparable term-by-term with the top-label numbers elsewhere.

Two epochs per cell: the one F13 stops at, and the end of the budget.

    python calibration_curves.py --run RUNDIR... --device cuda -o out.json
"""
from __future__ import annotations

import argparse, importlib.util, json, re, statistics as st, sys
from pathlib import Path

import numpy as np
import torch

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(RANK))

from analysis.mean_entropy_curves import reconstruct, split_graphs      # noqa: E402
from imputer.entity_mf.data import variable_list_to_entity_graph        # noqa: E402
from imputer.entity_mf.test import _load_checkpoint                     # noqa: E402

EPOCH_RE = re.compile(r"periodic-epoch=(\d+)\.ckpt$")
CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def load_shared():
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec); sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    return m.shared, m.shared.rp


def selected_epoch(run: Path) -> int | None:
    h = json.loads((run / "training_history.json").read_text())
    ce = [(int(e.get("epoch", i)),
           ((e.get("val_eval") or {}).get("metrics") or {}).get(
               "missing", {}).get("rating", {}).get("xent"))
          for i, e in enumerate(h)]
    ce = [(a, b) for a, b in ce if b is not None]
    if not ce:
        return None
    sm = [(ce[i][0], st.mean(v for _, v in ce[max(0, i - 2): i + 3])) for i in range(len(ce))]
    return min(sm, key=lambda t: t[1])[0]


@torch.no_grad()
def probs_for(model, types, variables, score_ids, max_item, device):
    from analysis.split_missing_metrics import chunk_by_item
    C = model.types["rating"].num_classes
    P, Y = [], []
    keep = {i for v in variables if id(v) in score_ids for i in v.item_ids}
    for chunk in chunk_by_item(variables, max_item, keep_whole=keep or None):
        graph = variable_list_to_entity_graph(chunk, types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk):
            if id(var) in score_ids and var.rating_value is not None:
                P.append(torch.softmax(params[0, idx, 1:1 + C].float(), -1).cpu().numpy())
                Y.append(int(var.rating_value))
    if not P:
        return None, None
    return np.stack(P), np.asarray(Y)


def thin(a, n=120):
    a = np.asarray(a, dtype=float)
    if len(a) <= n:
        return [round(float(v), 5) for v in a]
    idx = np.linspace(0, len(a) - 1, n).round().astype(int)
    return [round(float(v), 5) for v in a[idx]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="train_masked,dev,corner")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    shared, rp = load_shared()
    want = set(a.splits.split(","))
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = []

    for rd in a.run:
        run = Path(rd)
        m = CELL.search(run.name)
        cell = f"{m.group(1).lower()}/{m.group(3)}"
        print(f"\n=== {cell}", flush=True)
        model, types, part, train_cfg, _, _ = reconstruct(run)
        model.to(device).eval()
        mi = train_cfg.get("max_item")
        rate = train_cfg.get("masking_rate", 0.15)
        graphs = {n: (v, ids) for n, v, ids in split_graphs(part, rate, 0) if n in want}

        ck = [(int(x.group(1)), p) for p in (run / "checkpoints").glob("periodic-epoch=*.ckpt")
              if (x := EPOCH_RE.search(p.name))]
        ck.sort()
        sel = selected_epoch(run)
        picks = [("stop", min(ck, key=lambda t: abs(t[0] - sel))), ("end", ck[-1])]

        for when, (ep, path) in picks:
            _load_checkpoint(model, path, device); model.to(device).eval()
            for name, (variables, ids) in graphs.items():
                probs, labels = probs_for(model, types, variables, ids, mi, device)
                if probs is None:
                    continue
                conf, acc = shared._all_class_calibration(probs, labels)
                d = rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                           report_CE=True, report_CE_std=True)
                out.append(dict(
                    cell=cell, when=when, epoch=ep, split=name,
                    n_cells=int(len(labels)), n_classes=int(probs.shape[1]),
                    n_points=int(conf.size),
                    smece=float(d["ce"]), smece_ci=float(d.get("ce_ci_width", 0.0)),
                    mesh=thin(d["mesh"]), mu=thin(d["mu"]),
                    lower=thin(d["lower"]), upper=thin(d["upper"]),
                    density=thin(d["density"]),
                    # top-label summaries, for the confidence question
                    mean_conf=float(probs.max(1).mean()),
                    accuracy=float((probs.argmax(1) == labels).mean()),
                ))
                r = out[-1]
                print(f"  {when:<5} ep{ep:<5} {name:<13} "
                      f"smECE={r['smece']:.4f}+/-{r['smece_ci']:.4f}  "
                      f"n={r['n_cells']}x{r['n_classes']}={r['n_points']}  "
                      f"meanconf={r['mean_conf']:.3f} acc={r['accuracy']:.3f}", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(out))

    print(f"\nwrote {len(out)} diagrams -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
