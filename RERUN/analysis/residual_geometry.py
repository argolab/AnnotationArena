#!/usr/bin/env python
"""
Weight geometry of the outer projection M = [proj_out ; W_param].

Each dual-stream layer ends with

    features += proj_out(c)        # 75 x 80
    params   += W_param(c)         #  5 x 80

Stacking those two gives a square 80 x 80 map M acting on the inner stream `c`.
The question the residual ablation raises is whether the outer add (h + Mc) is
redundant with the inner one -- i.e. whether M could just learn to contain an
identity and carry the stream itself.

This reports, per layer and per variant:
  rank, sigma_min, sigma_max, condition number   -- can M carry the stream?
  ||M||_F, cos(M, I)                             -- does M look anything like I?

    python residual_geometry.py --runs <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def load_final(run: Path):
    ckpts = sorted(run.glob("checkpoints/periodic-epoch=*.ckpt"))
    if not ckpts:
        return None, None
    last = max(ckpts, key=lambda p: int(re.search(r"(\d+)\.ckpt$", p.name).group(1)))
    sd = torch.load(last, map_location="cpu", weights_only=False)["state_dict"]
    return sd, last.name


def layer_maps(sd):
    """Yield (layer_idx, M) with M = [proj_out ; W_param] stacked, shape [80, 80]."""
    idxs = sorted({int(m.group(1))
                   for k in sd if (m := re.search(r"blocks\.(\d+)\.proj_out\.weight", k))})
    for i in idxs:
        pref = next(k.split("blocks.")[0] for k in sd if f"blocks.{i}.proj_out.weight" in k)
        P = sd[f"{pref}blocks.{i}.proj_out.weight"].float()   # [75, 80]
        W = sd[f"{pref}blocks.{i}.W_param.weight"].float()    # [5, 80]
        yield i, torch.cat([P, W], dim=0)                     # [80, 80]


def _cond(A):
    s = torch.linalg.svdvals(A)
    return (s[0] / s[-1]).item()


def describe(M):
    s = torch.linalg.svdvals(M)
    D = M.shape[0]
    I = torch.eye(D)
    fro = M.norm().item()
    cos = (M * I).sum().item() / (fro * I.norm().item() + 1e-12)
    return {
        "rank": int(torch.linalg.matrix_rank(M)),
        "dim": D,
        "s_min": s[-1].item(),
        "s_max": s[0].item(),
        "cond": (s[0] / s[-1]).item(),
        "fro": fro,
        "cos_I": cos,
        # if M were carrying the stream we'd expect ||M - I|| << ||M|| + ||I||
        "d_I": (M - I).norm().item(),
        # `none_id` folds the identity in structurally, so the map its layer
        # actually applies is (I + M), not M. Reported for every variant so the
        # columns are comparable.
        "cond_IM": _cond(I + M),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--layers", default="", help="comma list; default all")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    want = {int(x) for x in args.layers.split(",") if x.strip()}
    out = {}
    for run in args.runs:
        run = Path(run)
        sd, ck = load_final(run)
        if sd is None:
            print(f"!! no checkpoint under {run}")
            continue
        # variant name is the directory two levels above seed<N>/<run>
        variant = run.parent.parent.name
        print(f"\n=== {variant}   ({ck})")
        print(f"{'layer':>5} {'rank':>7} {'s_min':>9} {'s_max':>8} {'cond':>8} "
              f"{'||M||_F':>9} {'cos(M,I)':>9} {'||M-I||':>9} {'cond(I+M)':>10}")
        rows = []
        for i, M in layer_maps(sd):
            if want and i not in want:
                continue
            d = describe(M)
            rows.append({"layer": i, **d})
            print(f"{i:>5} {d['rank']:>4}/{d['dim']:<2} {d['s_min']:>9.4f} "
                  f"{d['s_max']:>8.4f} {d['cond']:>8.1f} {d['fro']:>9.4f} "
                  f"{d['cos_I']:>9.4f} {d['d_I']:>9.4f} {d['cond_IM']:>10.1f}")
        out[variant] = rows
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
