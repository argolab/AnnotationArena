#!/usr/bin/env python
"""
Score every periodic checkpoint of ONE search run on the held-out corner, so the
corner curve can be put beside the validation curve the carve actually selected
on. That comparison is the only direct evidence that the selection procedure
works: if the carve's argmin sits near the corner's argmin, selecting on 67
cells is sound; if it does not, the whole protocol is in question.

Reuses analysis/split_missing_metrics.py's own evaluate()/summarise() rather
than re-deriving evaluation -- re-deriving is how D22/D23 got into several
readers in the first place.

    python corner_curve.py --run-dir <search run dir> -o out/curve.json
"""
from __future__ import annotations

import argparse, importlib.util, json, re, sys
from pathlib import Path

RERUN = Path(__file__).resolve().parents[2]
RANK = RERUN.parent / "imputer/ranking"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="a *-search/seed<N>/<run>_MARFORMER dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=1, help="score every Nth periodic checkpoint")
    a = ap.parse_args()

    import torch
    smm = load_module("smm", RERUN / "analysis/split_missing_metrics.py")
    device = torch.device(a.device)
    run_dir = Path(a.run_dir)
    cfg_path = run_dir / "train_config.json"
    trained_max_item = json.loads(cfg_path.read_text()).get("max_item") if cfg_path.exists() else None

    ckpts = sorted(run_dir.glob("checkpoints/periodic-epoch=*.ckpt"))
    ckpts = ckpts[:: max(1, a.every)]
    print(f"{run_dir.name}: {len(ckpts)} checkpoints to score", flush=True)

    points = []
    for c in ckpts:
        epoch = int(re.search(r"epoch=(\d+)", c.name).group(1))
        rows, _ = smm.evaluate(run_dir, device, c.name, trained_max_item, corner_whole=True)
        test = smm.summarise(rows, "test")
        train = smm.summarise(rows, "train")
        if test is None:
            continue
        points.append({"epoch": epoch,
                       "corner_ll": test["log_loss"], "corner_n": test["n"],
                       "train_ll": (train or {}).get("log_loss")})
        print(f"  epoch {epoch:>5}  corner {test['log_loss']:.4f} (n={test['n']})", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"run_dir": str(run_dir), "points": points}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
