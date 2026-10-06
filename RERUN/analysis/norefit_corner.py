#!/usr/bin/env python
"""Score each search run at the epoch the protocol selected, with no refit.

The shipped protocol carves 15% of the training partition as a validation
block, selects a stopping epoch on it, then merges the block back and trains a
FRESH model on 100% of the data for that many epochs. This asks what the corner
would say if the second stage were dropped: keep the carve, report the search
run's own checkpoint at the selected epoch, and accept that the model has seen
only 85% of the data.

The checkpoint used is the periodic one nearest the selected epoch, which is
uniform across every cell. Lightning's own best-*.ckpt is not used because the
two smallest cells (Item_T_100, Annot_T_10) have no carvable validation block
and therefore never wrote one.

    python norefit_corner.py --run-id final-s101 -o out.json
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import torch

RANKING = Path(os.environ.get("RANKING_ROOT", "imputer/ranking")).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(RANKING))
from analysis.split_missing_metrics import evaluate, summarise   # noqa: E402

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
CK_RE = re.compile(r"periodic-epoch=(\d+)\.ckpt$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rerun-dir", default=os.environ.get(
        "RERUN_DIR", "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN"))
    ap.add_argument("--epochs-file", default=None,
                    help="epoch table; defaults to the run's seed-42 backup then valblock_epochs.json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run = Path(a.rerun_dir) / "runs" / a.run_id
    tbl_path = Path(a.epochs_file) if a.epochs_file else next(
        (p for p in (run/"valblock_epochs.seed42.json.bak", run/"valblock_epochs.json") if p.is_file()), None)
    if tbl_path is None:
        sys.exit("no epoch table found")
    table = json.loads(tbl_path.read_text())["epochs"]
    device = torch.device(a.device)

    out = []
    for sub, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = run / "RESULTS" / sub / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for d in sorted(base.rglob("*/valblock-search/seed42/*_MARFORMER")):
            m = RUN_RE.search(d.name)
            if not m:
                continue
            axis, K = m.group(1).lower(), int(m.group(3))
            sel = table.get(f"{family}|{axis}|{K}|42")
            if sel is None:
                print(f"  [no selected epoch] {family} {axis} {K}", file=sys.stderr)
                continue
            cks = [(int(CK_RE.search(p.name).group(1)), p)
                   for p in (d/"checkpoints").glob("periodic-epoch=*.ckpt") if CK_RE.search(p.name)]
            if not cks:
                print(f"  [no periodic checkpoints] {family} {axis} {K}", file=sys.stderr)
                continue
            ep, ck = min(cks, key=lambda t: abs(t[0] - sel))
            trained_max_item = json.loads((d/"train_config.json").read_text())["training"].get("max_item")
            rows, _ = evaluate(d, device, ck.name, trained_max_item, corner_whole=True)
            t = summarise(rows, "test")
            if not t:
                continue
            out.append({"family": family, "axis": axis, "size": K,
                        "selected_epoch": sel, "scored_epoch": ep,
                        "test": t})
            print(f"{family:<12}{axis:<6}{K:>5}  selected {sel:>4} -> checkpoint {ep:>4}  "
                  f"corner {t['log_loss']:.4f} (n={t['n']})")
    if not out:
        sys.exit("FATAL: scored nothing")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"checkpoint": "nearest-periodic-to-selected", "rows": out}, indent=1))
    print(f"\n{len(out)} runs -> {a.out}")


if __name__ == "__main__":
    main()
