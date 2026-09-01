#!/usr/bin/env python
"""
Does test_observed actually receive gradient in transductive training?

Everything else about the transductive corner gap has been ruled out -- graph
presence, item parameters (they are exactly zero), label shift (0.007 nats),
context density (the gap survives matching on observed-cells-per-item), and
chunking (0.038 of 0.101; 0.063 survives a single chunk). The remaining
assumption, never directly measured, is that masking actually selects test_obs
cells so the model gets supervised on them.

This builds the training chunks exactly as training does and counts, per chunk,
how many maskable cells and how many *masked* cells belong to each partition.

    python masking_audit.py --data-dir <bundle> [--steps 20]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "imputer" / "ranking"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-item", type=int, default=300)
    ap.add_argument("--masking-rate", type=float, default=0.15)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()

    from imputer.data import DataConverter
    from imputer.entity_mf.masking import MCARMasking, MCARConfig

    # Derive sizes the way test.py does, via imputer.utils.sizes_from_configs,
    # rather than guessing key names out of configs.json (which nests them under
    # "datagen" and does not use K/J/I/C).
    from imputer.utils import sizes_from_configs
    cfg = json.loads((Path(a.data_dir) / "configs.json").read_text())
    sizes = sizes_from_configs(cfg)
    conv = DataConverter(num_attributes=sizes["num_attributes"],
                         num_annotators=sizes["num_annotators"],
                         num_items=sizes["num_items"],
                         num_likert_classes=sizes["num_likert_classes"],
                         max_rank_size=sizes.get("max_rank_size", 2))
    bundle = conv.load_bundle_data(Path(a.data_dir) / "data_bundle.json")

    train_obs = conv.create_variables_from_bundle(bundle, partition="train", status="observed")
    test_obs  = conv.create_variables_from_bundle(bundle, partition="test",  status="observed")
    maskable = list(train_obs) + list(test_obs)
    print(f"maskable pool: train_obs={len(train_obs)}  test_obs={len(test_obs)}  "
          f"test share={len(test_obs)/len(maskable):.3f}")

    # chunk exactly as training does: contiguous item-id groups
    ids = sorted({i for v in maskable for i in v.item_ids})
    groups = [set(ids[s:s+a.max_item]) for s in range(0, len(ids), a.max_item)]
    strat = MCARMasking(MCARConfig(masking_rate=a.masking_rate))
    tot = Counter(); pool = Counter()
    for ci, g in enumerate(groups):
        chunk = [v for v in maskable if any(i in g for i in v.item_ids)]
        pc = Counter(getattr(v, "instance", "?") for v in chunk)
        for k, v in pc.items(): pool[f"chunk{ci}:{k}"] += v
        for _ in range(a.steps):
            out = strat.mask(chunk)
            mc = Counter(getattr(v, "instance", "?") for v in out if getattr(v, "is_masked", False))
            for k, v in mc.items(): tot[f"chunk{ci}:{k}"] += v
    print(f"\n{'chunk:partition':<22} {'in pool':>9} {'masked/step':>12} {'rate':>7}")
    for k in sorted(pool):
        m = tot[k] / a.steps
        print(f"{k:<22} {pool[k]:>9} {m:>12.1f} {m/pool[k]:>7.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"pool": dict(pool), "masked_per_step": {k: v/a.steps for k, v in tot.items()}}, indent=2))

if __name__ == "__main__":
    main()
