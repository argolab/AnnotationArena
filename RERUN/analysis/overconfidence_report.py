#!/usr/bin/env python
"""Summarise mean_entropy_curves.py output: how much of the dev-loss rise is confidence?

For each scored run, take the epoch the dev curve bottoms at (the epoch F13's
carve is trying to find) and the end of the 2000-epoch budget, and split the
rise between them:

    rise        = nll_end - nll_min                 what overfitting costs
    conf_part   = d_oc_end - d_oc_min               removable by one temperature
    pred_part   = nll_T_end - nll_T_min             left after recalibration

conf_part + pred_part == rise by construction. conf_part/rise is the headline:
near 1 means the heteroskedastic story (the model is just too sure); near 0
means the predictions themselves decayed.

    python overconfidence_report.py analysis/out/meanent_*.json
"""
from __future__ import annotations
import json, re, statistics as st, sys
from pathlib import Path

CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER")


def selected_epoch(run: Path) -> int | None:
    """The epoch F13 would stop at: argmin of the 5-smoothed dev curve.

    Taken from training_history.json at full resolution rather than from the
    50-epoch scored grid. On a small dev block (annot/30 has 3 validation
    annotators and 132 cells) the scored grid's own argmin can land on epoch 24,
    where the model is still near-uniform and merely UNDERconfident -- the
    sparse grid misses the real basin and the decomposition then measures the
    wrong interval.
    """
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


def main(paths) -> int:
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    print(f"{'cell':<14}{'ep*':>6} | {'conf':>7}{'acc':>7}{'gap':>7} | "
          f"{'cWrong':>8}{'cRight':>8} | {'nllWrong':>9}{'lossShr':>8} | "
          f"{'T*':>6}{'conf%':>7}")
    print("-" * 94)
    out = []
    for run, rs in sorted(by_run.items(), key=lambda kv: (
            CELL.search(kv[0]).group(1), int(CELL.search(kv[0]).group(3)))):
        m = CELL.search(run)
        cell = f"{m.group(1).lower()}/{m.group(2)}/{m.group(3)}"
        rs = sorted([r for r in rs if "dev" in r], key=lambda r: r["epoch"])
        if len(rs) < 3:
            continue
        sel = selected_epoch(Path(run))
        lo = (min(rs, key=lambda r: abs(r["epoch"] - sel)) if sel is not None
              else min(rs, key=lambda r: r["dev"]["nll"]))
        hi = rs[-1]
        d, e = lo["dev"], hi["dev"]
        rise = e["nll"] - d["nll"]
        conf = e["d_oc"] - d["d_oc"]
        pred = e["nll_T"] - d["nll_T"]
        pct = conf / rise * 100 if rise > 1e-9 else float("nan")
        print(f"{cell:<14}{lo['epoch']:>6} | "
              f"{d['conf']:>7.3f}{d['acc']:>7.3f}{d['conf_gap']:>+7.3f} | "
              f"{d['conf_wrong']:>8.3f}{d['conf_right']:>8.3f} | "
              f"{d['nll_wrong']:>9.3f}{d['loss_share_wrong']:>8.2f} | "
              f"{d['T_star']:>6.2f}{pct:>6.0f}%")
        print(f"{'  -> ep'+str(hi['epoch']):<14}{'':>6} | "
              f"{e['conf']:>7.3f}{e['acc']:>7.3f}{e['conf_gap']:>+7.3f} | "
              f"{e['conf_wrong']:>8.3f}{e['conf_right']:>8.3f} | "
              f"{e['nll_wrong']:>9.3f}{e['loss_share_wrong']:>8.2f} | "
              f"{e['T_star']:>6.2f}{'':>7}")
        out.append(dict(cell=cell, run=run, ep_min=lo["epoch"], ep_end=hi["epoch"],
                        rise=rise, conf=conf, pred=pred,
                        conf_frac=conf / rise if rise else None,
                        at_min=d, at_end=e))

    print("\nconf%% = share of the dev-loss rise a single temperature would remove.")
    Path("analysis/out/overconfidence_summary.json").write_text(json.dumps(out, indent=1))
    print(f"wrote analysis/out/overconfidence_summary.json ({len(out)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
