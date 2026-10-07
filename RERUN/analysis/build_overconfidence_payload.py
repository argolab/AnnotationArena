#!/usr/bin/env python
"""Collect everything the overfitting/overconfidence artifact plots into one JSON.

Sources, all produced elsewhere so this script does no science:
  training_history.json        train + dev curves at full resolution
  meanent_*.json               entropy / mean-error / variance-ratio / T* curves
  overconfidence_summary.json  the conf-vs-pred split at the selected epoch
  dev_temp.json                validation-fitted temperature applied to the corner
  runs/l2-s101/...             the weight-decay sweep, if it has landed

    python build_overconfidence_payload.py -o analysis/out/overconf_payload.json
"""
from __future__ import annotations
import json, re, statistics as st
from pathlib import Path
import argparse

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def history(run: Path, every: int = 5) -> dict | None:
    """Train and dev curves, thinned. 2000 points per series is more than any
    screen resolves and would quadruple the page weight."""
    f = run / "training_history.json"
    if not f.is_file():
        return None
    h = json.loads(f.read_text())
    ep, tr, dv, ac = [], [], [], []
    for i, e in enumerate(h):
        if i % every and i != len(h) - 1:
            continue
        m = ((e.get("val_eval") or {}).get("metrics") or {}).get(
            "missing", {}).get("rating", {})
        if not m:
            continue
        ep.append(int(e.get("epoch", i)))
        tr.append(round(float(e.get("train_masked_ce") or 0), 4))
        dv.append(round(float(m.get("xent")), 4))
        ac.append(round(float(m.get("acc")), 4))
    return {"epoch": ep, "train": tr, "dev": dv, "dev_acc": ac} if ep else None


def sel_epoch(run: Path) -> int | None:
    h = json.loads((run / "training_history.json").read_text())
    ce = [(int(e.get("epoch", i)),
           ((e.get("val_eval") or {}).get("metrics") or {}).get(
               "missing", {}).get("rating", {}).get("xent"))
          for i, e in enumerate(h)]
    ce = [(e, v) for e, v in ce if v is not None]
    if not ce:
        return None
    sm = [(ce[i][0], st.mean(v for _, v in ce[max(0, i - 2): i + 3])) for i in range(len(ce))]
    return min(sm, key=lambda t: t[1])[0]


def l2_sweep(root: Path) -> list:
    """One row per (wd, K, seed): the dev curve's floor and where it turns."""
    out = []
    for cfgf in sorted(root.rglob("DOMAIN3-SIGMOID_Item_T_*_MARFORMER/train_config.json")):
        run = cfgf.parent
        c = json.loads(cfgf.read_text())
        h = history(run, every=5)
        if h is None or len(h["epoch"]) < 20:
            continue
        sm = [st.mean(h["dev"][max(0, i - 2): i + 3]) for i in range(len(h["dev"]))]
        i = min(range(len(sm)), key=lambda j: sm[j])
        m = CELL.search(run.name)
        out.append({
            "wd": c["training"]["weight_decay"], "K": int(m.group(3)),
            "seed": c["run"]["seed"], "n_epochs": h["epoch"][-1],
            "dev_min": round(sm[i], 4), "dev_argmin": h["epoch"][i],
            "dev_end": round(sm[-1], 4), "rise": round(sm[-1] - sm[i], 4),
            "acc_at_min": h["dev_acc"][i], "acc_end": h["dev_acc"][-1],
            "curve": h,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(RERUN / "analysis/out/overconf_payload.json"))
    a = ap.parse_args()

    base = RERUN / "runs/final-dual/RESULTS/MARFORMER/DOMAIN3-SIGMOID"
    cells = [("ITEM", "Item", k) for k in (200, 500, 1000, 2000, 4000)] + \
            [("ANNOT", "Annot", j) for j in (30, 40)]
    curves = {}
    for d, ax, k in cells:
        run = base / d / "valblock-search/seed42" / f"DOMAIN3-SIGMOID_{ax}_T_{k}_MARFORMER"
        h = history(run)
        if h:
            h["sel_epoch"] = sel_epoch(run)
            curves[f"{ax.lower()}/{k}"] = h

    signals = {}
    for f in ("meanent_small.json", "meanent_large.json"):
        p = RERUN / "analysis/out" / f
        if not p.is_file():
            continue
        for r in json.loads(p.read_text()):
            m = CELL.search(Path(r["run"]).name)
            key = f"{m.group(1).lower()}/{m.group(3)}"
            s = signals.setdefault(key, [])
            s.append({"epoch": r["epoch"], **{
                k: {kk: round(vv, 5) for kk, vv in v.items()}
                for k, v in r.items() if k in ("train_masked", "train_missing", "dev", "corner")}})
    for v in signals.values():
        v.sort(key=lambda r: r["epoch"])

    def rd(p):
        p = RERUN / "analysis/out" / p
        return json.loads(p.read_text()) if p.is_file() else None

    payload = {
        "generated": "2026-10-06",
        "run": "final-dual, valblock-search, seed 42",
        "curves": curves,
        "signals": signals,
        "summary": rd("overconfidence_summary.json"),
        "dev_temp": rd("dev_temp.json"),
        "calib": rd("calcurves.json"),
        # Only the scalars: dev_curves.json also carries 40 full 2000-epoch
        # curves, which alone is 6 MB and nothing on the page draws them.
        "divergence": [{k: v for k, v in r.items()
                        if k not in ("epoch", "train_masked_ce",
                                     "train_observed_ce", "dev_ce", "dev_acc")}
                       for r in (rd("dev_curves.json") or [])],
        "l2": l2_sweep(RERUN / "runs/l2-s101") if (RERUN / "runs/l2-s101").is_dir() else [],
    }
    Path(a.out).write_text(json.dumps(payload, separators=(",", ":")))
    n = Path(a.out).stat().st_size
    print(f"curves {len(curves)}  signals {len(signals)}  "
          f"summary {len(payload['summary'] or [])}  "
          f"dev_temp {len(payload['dev_temp'] or [])}  l2 {len(payload['l2'])}")
    print(f"wrote {a.out} ({n/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
