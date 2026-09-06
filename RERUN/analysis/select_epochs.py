#!/usr/bin/env python
"""
Pick each model's stopping epoch from its validation curve.

Phase A of the validation protocol trains with part of the training partition
carved off as validation (see imputer/entity_mf/validation.py). This reads those
runs' training_history.json, takes the argmin of the validation missing-rating
cross-entropy, and writes the per-(family, axis, size, seed) epoch budget that
phase B refits to on the whole training partition.

Selection is **per training size**, pooling the seeds: the validation curves of
a size's seeds are averaged epoch-by-epoch and the argmin of that mean is taken,
so all seeds of a size refit to the same budget. Per-run selection is recorded
alongside for comparison, but it is not used, and the reason is visible in the
data: on a validation set of ~200 targets the per-run argmin swings between
epoch 74 and epoch 305 for neighbouring sizes, which is noise, not signal.

For the same reason the pooled curve is smoothed with a short centred moving
average before the argmin (--smooth, default 5 epochs). This narrows the search
to the bottom of the basin instead of the single luckiest epoch in it; set
--smooth 1 to disable and see the raw choice, which is also recorded.

Sizes with no training block to carve from (Item_T_100 has K_train = 0, and
Annot_T_10 has no training annotators) cannot select on data at all; they fall
back to the median selected epoch of their own axis and are flagged.

    python select_epochs.py --run-dir RERUN/runs/val-01 --variant valblock -o epochs.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")


def val_curve(history: list) -> list[tuple[int, float]]:
    """(epoch, validation missing-rating CE), for epochs where val was scored."""
    out = []
    for e in history:
        m = e.get("val_eval", {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" in m and m.get("n", 1) > 0:
            out.append((int(e["epoch"]), float(m["xent"])))
    return out


def smooth(ys: list[float], window: int) -> list[float]:
    """Centred moving average; the ends shrink the window rather than pad it."""
    if window <= 1:
        return list(ys)
    half = window // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True,
                    help="variant tag the search runs were written under, e.g. valblock-search")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-epochs", type=int, default=25,
                    help="floor on the refit budget; a 1-epoch model is a bug, not a choice")
    ap.add_argument("--select", default="per-size", choices=["per-size", "per-run"],
                    help="per-size pools a size's seeds before taking the argmin")
    ap.add_argument("--smooth", type=int, default=5,
                    help="centred moving average applied before the argmin; 1 disables")
    args = ap.parse_args()

    root = Path(args.run_dir) / "RESULTS"
    rows = []
    for family_dir, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / family_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run_dir in sorted(base.rglob("*_MARFORMER")):
            if args.variant not in str(run_dir):
                continue
            m = RUN_RE.search(run_dir.name)
            if not m:
                continue
            hist_path = run_dir / "training_history.json"
            if not hist_path.is_file():
                continue
            with open(hist_path) as fh:
                history = json.load(fh)
            curve = val_curve(history)
            # Smoothed once here, and used for the per-run argmin below. The
            # pooled path smooths its own mean curve separately.
            curve_sm = smooth([c[1] for c in curve], args.smooth) if curve else []
            sm = next((SEED_RE.match(p.name) for p in run_dir.parents
                       if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            axis = m.group(1).lower()
            axis = "annot" if axis == "annot" else "item"
            rows.append({
                "family": family, "axis": axis, "size": int(m.group(3)), "seed": seed,
                "n_val_points": len(curve),
                # Smoothed, for the same reason the pooled curve is: the raw
                # argmin picks the single luckiest epoch in the basin. This
                # path used to take the raw argmin while --select per-size got
                # a smoothed one, so the two modes were not comparable. On the
                # W3d curves, smoothing here is worth -0.005 at K=1000 and
                # -0.001 at K=4000. --smooth 1 restores the raw behaviour.
                "per_run_epoch": (
                    curve[min(range(len(curve_sm)), key=lambda i: curve_sm[i])][0]
                    if curve else None),
                "best_val_ce": min(c[1] for c in curve) if curve else None,
                "last_val_ce": curve[-1][1] if curve else None,
                "total_epochs": len(history),
                "path": str(run_dir.relative_to(Path(args.run_dir))),
                "_curve": curve,
            })

    # Pool a size's seeds: average the validation curves epoch by epoch, then
    # take the argmin of the smoothed mean.
    pooled: dict[tuple, int | None] = {}
    for key in {(r["family"], r["axis"], r["size"]) for r in rows}:
        curves = [dict(r["_curve"]) for r in rows
                  if (r["family"], r["axis"], r["size"]) == key and r["_curve"]]
        if not curves:
            pooled[key] = None
            continue
        shared = sorted(set.intersection(*(set(c) for c in curves)))
        if not shared:
            pooled[key] = None
            continue
        means = [sum(c[e] for c in curves) / len(curves) for e in shared]
        sm = smooth(means, args.smooth)
        pooled[key] = shared[min(range(len(sm)), key=lambda i: sm[i])]

    # Fallback for cells with no validation signal, by axis.
    by_axis: dict[str, list[int]] = {}
    for (family, axis, _size), ep in pooled.items():
        if ep is not None:
            by_axis.setdefault(axis, []).append(ep)
    fallback = {a: int(statistics.median(v)) for a, v in by_axis.items()}

    table: dict[str, int] = {}
    for r in rows:
        key = (r["family"], r["axis"], r["size"])
        ep = pooled.get(key) if args.select == "per-size" else r["per_run_epoch"]
        r["pooled_epoch"] = pooled.get(key)
        r["fallback"] = ep is None
        if ep is None:
            ep = fallback.get(r["axis"])
            if ep is None:
                continue
        r["best_epoch"] = ep
        # history epochs are 0-indexed; --epochs is a count.
        r["refit_epochs"] = max(args.min_epochs, ep + 1)
        table[f"{r['family']}|{r['axis']}|{r['size']}|{r['seed']}"] = r["refit_epochs"]
        r.pop("_curve", None)
    for r in rows:
        r.pop("_curve", None)

    payload = {"variant": args.variant, "select": args.select, "smooth": args.smooth,
               "fallback_by_axis": fallback, "epochs": table, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"{len(rows)} search runs -> {len(table)} epoch budgets "
          f"(select={args.select}, smooth={args.smooth})  ({out})")
    print(f"{'family':<13}{'axis':<7}{'size':>6}{'seed':>6}{'per-run':>9}{'pooled':>8}"
          f"{'refit':>7}{'val CE':>9}{'':>3}")
    for r in sorted(rows, key=lambda r: (r["family"], r["axis"], r["size"], r["seed"])):
        flag = " *fallback" if r["fallback"] else ""
        ce = f"{r['best_val_ce']:.4f}" if r["best_val_ce"] is not None else "-"
        print(f"{r['family']:<13}{r['axis']:<7}{r['size']:>6}{r['seed']:>6}"
              f"{r['per_run_epoch'] if r['per_run_epoch'] is not None else '-':>9}"
              f"{r['pooled_epoch'] if r['pooled_epoch'] is not None else '-':>8}"
              f"{r.get('refit_epochs', '-'):>7}{ce:>9}{flag}")
    if any(r["fallback"] for r in rows):
        print("\n* no validation signal (no training block to carve); axis median used")


if __name__ == "__main__":
    main()
