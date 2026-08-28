"""Wall-clock breakdown of an entity_mf training run.

Runs the real trainer (imputer.entity_mf.train.main) with a small --epochs and
wraps the functions that make up a step so we can see where the time actually
goes. Every timer synchronises CUDA first, so GPU work is charged to the call
that launched it rather than to whichever later call happened to block.

Usage:
  python -m RERUN.bench.profile_train --epochs 10 -- <normal train.py args>
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from collections import defaultdict

import torch

TOTALS: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)
_SYNC = True


def _sync() -> None:
    if _SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            _sync()
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                _sync()
                TOTALS[label] += time.perf_counter() - t0
                COUNTS[label] += 1
        return wrapper
    return deco


def install_probes() -> None:
    from imputer.entity_mf import train as T
    from imputer.entity_mf import data as D
    from imputer.entity_mf import masking as M
    from imputer.entity_mf import model as MD
    from imputer.entity_mf import eval as EV

    M.MCARMasking.mask = timed("mask/mcar")(M.MCARMasking.mask)
    M.StructuredMasking.mask = timed("mask/structured")(M.StructuredMasking.mask)
    D.EntityGraph.build_edge_masks = timed("graph/edge_mask")(D.EntityGraph.build_edge_masks)
    D.variable_list_to_entity_graph = timed("graph/build")(D.variable_list_to_entity_graph)
    MD.EntityMarformer._build_initial_streams = timed("fwd/init_streams")(
        MD.EntityMarformer._build_initial_streams)
    MD.EntityMarformer._get_k_aug = timed("fwd/k_aug")(MD.EntityMarformer._get_k_aug)
    MD.RelationalAttentionBlock.forward = timed("fwd/attn")(MD.RelationalAttentionBlock.forward)
    MD.EntityMarformer.forward = timed("fwd/total")(MD.EntityMarformer.forward)
    EV.compute_trainable_loss = timed("loss/trainable")(EV.compute_trainable_loss)
    EV.evaluate_entity_marformer_split = timed("eval/split")(EV.evaluate_entity_marformer_split)
    # train.py imported these by name, so rebind there too.
    T.compute_trainable_loss = EV.compute_trainable_loss
    T.evaluate_entity_marformer_split = EV.evaluate_entity_marformer_split
    T.variable_list_to_entity_graph = D.variable_list_to_entity_graph

    L = T.EntityMarformerLightningModule
    L.training_step = timed("step/total")(L.training_step)
    L.on_train_epoch_end = timed("epoch_end/total")(L.on_train_epoch_end)
    L._refresh_variable_tokens = staticmethod(
        timed("step/refresh_tokens")(L._refresh_variable_tokens.__func__
                                     if hasattr(L._refresh_variable_tokens, "__func__")
                                     else L._refresh_variable_tokens))
    L._build_training_chunks = timed("setup/build_chunks")(L._build_training_chunks)

    _orig_backward = torch.Tensor.backward
    torch.Tensor.backward = timed("step/backward")(_orig_backward)


def report(wall: float) -> None:
    print("\n" + "=" * 74)
    print(f"{'section':<26}{'total_s':>10}{'calls':>9}{'ms/call':>11}{'% wall':>10}")
    print("-" * 74)
    for k in sorted(TOTALS, key=lambda k: -TOTALS[k]):
        t, n = TOTALS[k], COUNTS[k]
        print(f"{k:<26}{t:>10.2f}{n:>9}{1000*t/max(n,1):>11.2f}{100*t/wall:>9.1f}%")
    print("-" * 74)
    print(f"{'WALL':<26}{wall:>10.2f}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true",
                    help="skip cuda synchronize in timers (lower overhead, blurrier attribution)")
    args, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    global _SYNC
    _SYNC = not args.no_sync

    install_probes()
    from imputer.entity_mf import train as T

    sys.argv = ["train.py"] + rest
    t0 = time.perf_counter()
    T.main()
    report(time.perf_counter() - t0)


if __name__ == "__main__":
    main()
