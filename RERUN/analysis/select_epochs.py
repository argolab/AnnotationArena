#!/usr/bin/env python
"""
Pick each model's stopping epoch from its validation curve.

Phase A of the validation protocol trains with part of the training partition
carved off as validation (see imputer/entity_mf/validation.py). This reads those
runs' training_history.json, takes the argmin of the validation missing-rating
cross-entropy, and writes the per-(family, axis, size, seed) epoch budget that
phase B refits to on the whole training partition.

Selection is per run, not per size: that is what early stopping actually is, and
it lets the seed spread in the *selected epoch* be reported rather than hidden.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True,
                    help="variant tag the search runs were written under, e.g. valblock-search")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-epochs", type=int, default=25,
                    help="floor on the refit budget; a 1-epoch model is a bug, not a choice")
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
            sm = next((SEED_RE.match(p.name) for p in run_dir.parents
                       if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            axis = m.group(1).lower()
            axis = "annot" if axis == "annot" else "item"
            rows.append({
                "family": family, "axis": axis, "size": int(m.group(3)), "seed": seed,
                "n_val_points": len(curve),
                "best_epoch": min(curve, key=lambda t: t[1])[0] if curve else None,
                "best_val_ce": min(c[1] for c in curve) if curve else None,
                "last_val_ce": curve[-1][1] if curve else None,
                "total_epochs": len(history),
                "path": str(run_dir.relative_to(Path(args.run_dir))),
            })

    # Fallback for cells with no validation signal, by axis.
    by_axis: dict[str, list[int]] = {}
    for r in rows:
        if r["best_epoch"] is not None:
            by_axis.setdefault(r["axis"], []).append(r["best_epoch"])
    fallback = {a: int(statistics.median(v)) for a, v in by_axis.items()}

    table: dict[str, int] = {}
    for r in rows:
        ep = r["best_epoch"]
        r["fallback"] = ep is None
        if ep is None:
            ep = fallback.get(r["axis"])
            if ep is None:
                continue
            r["best_epoch"] = ep
        # history epochs are 0-indexed; --epochs is a count.
        r["refit_epochs"] = max(args.min_epochs, ep + 1)
        table[f"{r['family']}|{r['axis']}|{r['size']}|{r['seed']}"] = r["refit_epochs"]

    payload = {"variant": args.variant, "fallback_by_axis": fallback,
               "epochs": table, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"{len(rows)} search runs -> {len(table)} epoch budgets  ({out})")
    print(f"{'family':<13}{'axis':<7}{'size':>6}{'seed':>6}{'sel.ep':>8}{'refit':>7}"
          f"{'val CE':>9}{'':>3}")
    for r in sorted(rows, key=lambda r: (r["family"], r["axis"], r["size"], r["seed"])):
        flag = " *fallback" if r["fallback"] else ""
        print(f"{r['family']:<13}{r['axis']:<7}{r['size']:>6}{r['seed']:>6}"
              f"{r['best_epoch'] if r['best_epoch'] is not None else '-':>8}"
              f"{r.get('refit_epochs', '-'):>7}"
              f"{(f'{r['best_val_ce']:.4f}' if r['best_val_ce'] is not None else '-'):>9}{flag}")
    if any(r["fallback"] for r in rows):
        print("\n* no validation signal (no training block to carve); axis median used")


if __name__ == "__main__":
    main()
