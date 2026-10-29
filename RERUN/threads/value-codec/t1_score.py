#!/usr/bin/env python
"""T1: score the width sweep -- every run's curves, and its test loss.

For each finished run under runs/codec-t1-*, one JSON file with:

  train      per-epoch masked training cross-entropy (training_history.json)
  dev        per-epoch cross-entropy on the mcar carve; empty at K=100, whose
             training partition has no observed cells to carve
  test       corner log-loss at every periodic checkpoint (every 25 epochs) and
             last.ckpt, scored by split_missing_metrics.evaluate at the shipped
             evaluation chunk -- the paper's population, test items whole
  e_star     the epoch the shipped selector picks from the dev curve
             (analysis.select_epochs: 5-point centred smoothing, argmin, floor of
             25), the same rule E1 graded
  test_at_e_star  the test curve linearly interpolated at e_star, as E1 does
  test_tail  mean of the last five periodic checkpoints -- the selection-free
             reading B10 and B11 used, and the only one available at K=100

Resumable and shardable: a run whose output file exists is skipped, and
--shard i/n takes every n-th run, so an array job can split the work.

    python t1_score.py --shard 0/20 [--eval-max-item 500] [--device cpu]
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import torch

RERUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RERUN))
from analysis.select_epochs import smooth, val_curve                    # noqa: E402
from analysis.split_missing_metrics import evaluate, summarise          # noqa: E402
# Only a tree with codecs can rebuild these models; fail here, not on the first
# checkpoint, if PYTHONPATH points at an older checkout.
import imputer.entity_mf.value_codecs                                    # noqa: E402,F401

SMOOTH, MIN_EPOCHS, TAIL = 5, 25, 5
OUT = RERUN / "threads/value-codec/out/t1"
RUN_RE = re.compile(r"runs/(codec-t1-[^/]+)/RESULTS/.*/ITEM/([^/]+)/Marformer/seed(\d+)/"
                    r"DOMAIN3-SIGMOID_Item_T_(\d+)_MARFORMER$")


def select(curve):
    if not curve:
        return None
    sm = smooth([c[1] for c in curve], SMOOTH)
    best = curve[min(range(len(sm)), key=lambda i: sm[i])][0]
    return max(MIN_EPOCHS, best + 1)


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if a <= x <= b:
            return ya + (yb - ya) * (x - a) / (b - a)
    return ys[-1]


def score(run: Path, device, max_item: int) -> dict:
    m = RUN_RE.search(str(run))
    history = json.loads((run / "training_history.json").read_text())
    train = [(int(r["epoch"]), float(r["train_masked_ce"])) for r in history
             if r.get("train_masked_ce") is not None]
    dev = val_curve(history)
    ckpts = sorted(run.glob("checkpoints/periodic-epoch=*.ckpt"),
                   key=lambda p: int(re.search(r"(\d+)\.ckpt$", p.name).group(1)))
    test = []
    for c in ckpts:
        ep = int(re.search(r"(\d+)\.ckpt$", c.name).group(1)) + 1   # epoch index -> epochs run
        rows, _ = evaluate(run, device, c.name, max_item, corner_whole=True)
        t = summarise(rows, "test")
        test.append({"epoch": ep, "loss": t["log_loss"], "n": t["n"]})
    rows, _ = evaluate(run, device, "last.ckpt", max_item, corner_whole=True)
    last = summarise(rows, "test")
    e_star = select(dev)
    xs, ys = [t["epoch"] for t in test], [t["loss"] for t in test]
    return {
        "run_id": m.group(1), "draw": m.group(1).split("-")[-1], "arm": m.group(2),
        "seed": int(m.group(3)), "K": int(m.group(4)),
        "codec": json.loads((run / "train_config.json").read_text())["model"].get("codecs"),
        "train": train, "dev": dev, "test": test,
        "test_last": last["log_loss"], "test_n": last["n"],
        "e_star": e_star,
        "test_at_e_star": interp(xs, ys, e_star) if e_star is not None else None,
        "test_tail": sum(ys[-TAIL:]) / len(ys[-TAIL:]),
        "test_best": min(ys), "test_best_epoch": xs[ys.index(min(ys))],
        "eval_max_item": max_item,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--eval-max-item", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    dev = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    # training_history.json is written once, at the end of training, so it is
    # the completion marker; last.ckpt is rewritten every epoch and is not.
    runs = sorted(Path(p).parent for p in glob.glob(
        str(RERUN / "runs/codec-t1-*/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/*/Marformer/seed*"
            "/DOMAIN3-SIGMOID_Item_T_*_MARFORMER/training_history.json")))
    OUT.mkdir(parents=True, exist_ok=True)
    for k, run in enumerate(runs):
        if k % n != i:
            continue
        m = RUN_RE.search(str(run))
        out = OUT / f"{m.group(1)}__{m.group(2)}__s{m.group(3)}__K{m.group(4)}.json"
        if out.exists():
            continue
        rec = score(run, dev, a.eval_max_item)
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out)
        print(f"{out.name}: tail {rec['test_tail']:.4f}  e*={rec['e_star']} "
              f"at e* {rec['test_at_e_star']}", flush=True)


if __name__ == "__main__":
    main()
