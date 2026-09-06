#!/usr/bin/env python
"""
S7: does the block validation carve pick the epoch the corner actually wants?

For every search run under a run tree, produce three epochs:

  val_argmin     the epoch the protocol would select, from this run's own
                 validation curve in training_history.json
  corner_argmin  the epoch the corner actually bottoms out at, from scoring
                 every periodic checkpoint on the corner
  last           the end of the budget, i.e. what "no early stopping" gives

and the corner log-loss at each. The question S7 asks is whether val_argmin
tracks corner_argmin. D26 found it does not on base seed 42 (~190 epochs late
on the NT arm). D28 then found that seed 42's corner is anomalously hard, which
would produce exactly that symptom without the carve being wrong at all -- so
this script is pointed at fresh draws to tell the two explanations apart.

    python s7_curves.py --run-dir RERUN/runs/s7-block-s101 -o out.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate, summarise

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")


def val_curve(run: Path) -> list[tuple[int, float]]:
    """(epoch, validation cross-entropy) from the run's own history."""
    h = json.loads((run / "training_history.json").read_text())
    epochs = h if isinstance(h, list) else h.get("epochs", h.get("history", []))
    out = []
    for i, e in enumerate(epochs):
        if not isinstance(e, dict):
            continue
        ep = e.get("epoch", i)
        # The validation metric lives under val_eval for the NT arm and under
        # combined_eval for the transductive one; take whichever is present and
        # read the missing-cell rating cross-entropy, which is what the
        # protocol selects on.
        v = None
        for key in ("val_missing_rating_ce", "val_ce", "val_rating_ce"):
            if isinstance(e.get(key), (int, float)):
                v = float(e[key]); break
        if v is None:
            blob = e.get("val_eval") or {}
            m = (blob.get("metrics") or {}).get("missing", {}).get("rating", {})
            for key in ("xent", "cross_entropy", "log_loss", "ce"):
                if isinstance(m.get(key), (int, float)):
                    v = float(m[key]); break
        if v is not None:
            out.append((int(ep), v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="run tree root (contains RESULTS/)")
    ap.add_argument("--eval-max-item", type=int, default=500)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    root = Path(a.run_dir) / "RESULTS"
    rows = []
    for fam_dir in ("MARFORMER", "MARFORMER-NT"):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run.name)
            # training_history.json is written once, at the end of training, so
            # it is the completion marker. last.ckpt is NOT: Lightning rewrites
            # it every epoch, so it exists a minute after a run starts. Scoring
            # on last.ckpt would quietly mix half-trained runs into the output.
            if not m or not (run / "training_history.json").is_file():
                continue
            sm = next((SEED_RE.match(p.name) for p in run.parents if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            ckpts = sorted(run.glob("checkpoints/periodic-epoch=*.ckpt"),
                           key=lambda p: int(re.search(r"(\d+)\.ckpt$", p.name).group(1)))
            if (run / "checkpoints" / "last.ckpt").is_file():
                ckpts.append(run / "checkpoints" / "last.ckpt")
            series = []
            for c in ckpts:
                mm = re.search(r"(\d+)\.ckpt$", c.name)
                ep = int(mm.group(1)) if mm else -1     # -1 == last
                r, _ = evaluate(run, dev, c.name, a.eval_max_item, corner_whole=True)
                t = summarise(r, "test")
                series.append({"epoch": ep, "corner": t["log_loss"], "n": t["n"]})
                print(f"{run.name:<40} s{seed} ep {ep:>4} corner {t['log_loss']:.4f}", flush=True)
            if not series:
                continue
            vc = val_curve(run)
            va = min(vc, key=lambda t: t[1])[0] if vc else None
            periodic = [s for s in series if s["epoch"] >= 0]
            ca = min(periodic, key=lambda s: s["corner"]) if periodic else None
            # corner at the epoch the protocol would have picked: nearest
            # periodic checkpoint at or before val_argmin.
            at_val = None
            if va is not None and periodic:
                cand = [s for s in periodic if s["epoch"] <= va] or periodic
                at_val = max(cand, key=lambda s: s["epoch"])
            rows.append({
                "run": str(run), "axis": m.group(1).lower(), "arm": m.group(2),
                "size": int(m.group(3)), "seed": seed,
                "val_argmin_epoch": va,
                "corner_argmin_epoch": ca["epoch"] if ca else None,
                "corner_at_val_argmin": at_val["corner"] if at_val else None,
                "corner_at_corner_argmin": ca["corner"] if ca else None,
                "corner_at_last": series[-1]["corner"],
                "series": series,
            })
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
