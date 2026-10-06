#!/usr/bin/env python
"""Export every cell's training and validation curve as one JSON blob.

This is what the "uniform plateau" artifact plots. It pairs each cell's SEARCH
run (85% of the training partition, 2000 epochs, with a held-out validation
block) with its REFIT (100%, stopped at the selected epoch), and records the
epoch at which each one left the uniform plateau.

Escape is defined on the TRAINING curve, not validation: training loss is the
quantity available without a validation block, so a degeneracy rule built on it
applies to every cell including the two that cannot carve one -- and, more
importantly, it never touches the test corner.

    python curve_export.py --run-id final-dual --out curves.json
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path

UNIFORM = math.log(4)
ESC_MARGIN = 0.05          # training loss must fall this far below ln4 to count
DEGEN_WINDOW = 0.02        # corner this close to ln4 == never escaped
RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def train_ce(rec):
    return rec.get("train_masked_ce")


def val_ce(rec):
    ev = rec.get("val_eval") or {}
    try:
        return ev["metrics"]["missing"]["rating"]["xent"]
    except (KeyError, TypeError):
        return None


def escape_epoch(hist):
    for rec in hist:
        v = train_ce(rec)
        if v is not None and v < UNIFORM - ESC_MARGIN:
            return rec.get("epoch")
    return None


def series(hist, stride):
    e, tr, va = [], [], []
    for i, rec in enumerate(hist):
        if i % stride and i != len(hist) - 1:
            continue
        e.append(rec.get("epoch", i))
        v = train_ce(rec)
        tr.append(round(v, 4) if v is not None else None)
        v = val_ce(rec)
        va.append(round(v, 4) if v is not None else None)
    return e, tr, va


def corner_map(run_dir: Path) -> dict:
    f = run_dir / "corner_metrics" / "corner.json"
    if not f.is_file():
        return {}
    out = {}
    for row in json.loads(f.read_text()).get("rows", []):
        val = (row.get("test") or {}).get("log_loss")
        if val is None:
            continue
        out[(row["family"], str(row["axis"]).lower(), int(row["size"]))] = float(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerun-dir",
                    default="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
    ap.add_argument("--search-tag", default="valblock-search/seed42")
    ap.add_argument("--refit-tag", default="valblock/seed42")
    ap.add_argument("--search-stride", type=int, default=10)
    ap.add_argument("--refit-stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run_dir = Path(a.rerun_dir) / "runs" / a.run_id
    results = run_dir / "RESULTS"
    corners = corner_map(run_dir)
    # valblock_epochs.json keys are "family|axis|size|seed", and "rows" also
    # records whether the cell fell back to the axis median because it had no
    # validation block to carve (item/T K=100 and annot/T J=10).
    epochs_tbl, fallback = {}, {}
    et = run_dir / "valblock_epochs.json"
    if et.is_file():
        tbl = json.loads(et.read_text())
        epochs_tbl = tbl.get("epochs", {})
        for row in tbl.get("rows", []):
            fallback[(row["family"], str(row["axis"]).lower(), int(row["size"]))] = \
                bool(row.get("fallback"))

    cells = []
    for fam_dir, fam in ((results / "MARFORMER", "Marformer"),
                         (results / "MARFORMER-NT", "MarformerNT")):
        if not fam_dir.is_dir():
            continue
        for sdir in sorted(fam_dir.rglob(f"{a.search_tag}/*_MARFORMER")):
            m = RUN_RE.search(sdir.name)
            if not m:
                continue
            axis, arm, K = m.group(1).lower(), m.group(2), int(m.group(3))
            sh = sdir / "training_history.json"
            if not sh.is_file():
                continue
            hist = json.loads(sh.read_text())
            e, tr, va = series(hist, a.search_stride)

            # the refit lives at the same path with the search tag swapped
            rdir = Path(str(sdir).replace(a.search_tag, a.refit_tag))
            re_, rtr = [], []
            refit_epochs = esc_refit = None
            rh = rdir / "training_history.json"
            if rh.is_file():
                rhist = json.loads(rh.read_text())
                re_, rtr, _ = series(rhist, a.refit_stride)
                refit_epochs = len(rhist)
                esc_refit = escape_epoch(rhist)

            corner = corners.get((fam, axis, K))
            cells.append({
                "family": fam, "axis": axis, "K": K,
                "sel_epoch": epochs_tbl.get(f"{fam}|{axis}|{K}|{a.seed}"),
                "no_val_block": fallback.get((fam, axis, K), False),
                "corner": round(corner, 4) if corner is not None else None,
                "degenerate": bool(corner is not None
                                   and abs(corner - UNIFORM) < DEGEN_WINDOW),
                "e": e, "tr": tr, "va": va,
                "re": re_, "rtr": rtr,
                "refit_epochs": refit_epochs,
                "esc_search": escape_epoch(hist),
                "esc_refit": esc_refit,
            })

    order = {"item": 0, "annot": 1}
    cells.sort(key=lambda c: (order[c["axis"]], c["family"], c["K"]))
    blob = {"uniform": round(UNIFORM, 4), "stride": a.search_stride,
            "run_id": a.run_id, "cells": cells}
    Path(a.out).write_text(json.dumps(blob))
    n_deg = sum(1 for c in cells if c["degenerate"])
    n_ref = sum(1 for c in cells if c["refit_epochs"])
    print(f"{a.run_id}: {len(cells)} cells, {n_ref} with a refit, "
          f"{n_deg} degenerate -> {a.out}")
    for c in cells:
        if c["degenerate"]:
            print(f"  DEGENERATE {c['family']}/{c['axis']}/{c['K']} "
                  f"corner={c['corner']} sel={c['sel_epoch']} "
                  f"esc_search={c['esc_search']} esc_refit={c['esc_refit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
