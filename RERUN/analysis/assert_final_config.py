#!/usr/bin/env python
"""Assert that a launched run matches the paper spec, cell by cell.

Written because the previous final run completed all 84 cells at
delta_item=1.0 / delta_annot=0.8 when the paper says 0.7, and nothing noticed
until the run was finished. Every value below is checked against
train_config.json, which the runner writes BEFORE the first epoch -- so running
this a minute after launch costs nothing and catches a wrong knob before the
GPU-hours are spent.

    python assert_final_config.py --run-dir runs/final-v2          # exits 1 on any mismatch
    python assert_final_config.py --run-dir runs/final-v2 --watch  # poll until all cells seen

The spec is spelled out here rather than read from the launcher on purpose: a
launcher bug cannot pass its own typo through to the assertion.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")

# --- the spec, from Marformer-Paper/main.tex and the settled-architecture notes
COMMON = {
    ("training", "lr"):                    2e-4,
    ("training", "weight_decay"):          0.01,
    ("training", "masking_rate"):          0.15,
    ("training", "mask_augmentations"):    5,
    ("training", "masked_loss_weight"):    15.0,
    ("training", "observed_loss_weight"):  1.0,
    ("training", "max_item"):              500,
    ("model",    "embedding_dim"):         80,
    ("model",    "num_layers"):            8,
    ("model",    "attention_heads"):       4,
    ("model",    "d_ff"):                  128,
    ("model",    "num_ffn_layers"):        1,
    ("model",    "dropout"):               0.1,
    # settled 2026-09-18: the standard Transformer block, both pieces dropped
    ("model",    "dual_outer_residual"):   "none",
    ("model",    "block_output_projection"): "none",
    ("model",    "inner_residual"):        "both",
    ("model",    "value_codec"):           "dual",
    ("model",    "codec_init"):            "unit",
    # full Shaw et al. 2018 S3.1 relational attention, both halves
    ("model",    "rel_key"):               "learned",
    ("model",    "rel_value"):             "learned",
}
# delta_tau: 0.7 on the axis being generalised over, 0 on the other.
PER_AXIS = {
    "item":  {("training", "item_dropout_rate"): 0.7,
              ("training", "annotator_dropout_rate"): 0.0},
    "annot": {("training", "item_dropout_rate"): 0.0,
              ("training", "annotator_dropout_rate"): 0.7},
}
SEARCH = {("training", "epochs"): 2000, ("training", "val_protocol"): "block"}


def get(cfg: dict, path: tuple):
    node = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return "<missing>"
        node = node[k]
    return node


def check(cfg: dict, want: dict, where: str, bad: list, run: str):
    for path, exp in want.items():
        got = get(cfg, path)
        ok = (abs(got - exp) < 1e-9 if isinstance(exp, float) and isinstance(got, (int, float))
              else got == exp)
        if not ok:
            bad.append(f"{run}  {'.'.join(path)}: got {got!r}, spec says {exp!r}  [{where}]")


def scan(root: Path, variant: str) -> tuple[int, list]:
    bad, n = [], 0
    for f in sorted(root.rglob("DOMAIN3-SIGMOID_*_MARFORMER/train_config.json")):
        run = f.parent
        m = RUN_RE.search(run.name)
        if not m or variant not in str(run):
            continue
        axis = "item" if m.group(1) == "Item" else "annot"
        cfg = json.loads(f.read_text())
        n += 1
        label = f"{axis}/{m.group(2)}/{m.group(3)}"
        check(cfg, COMMON, "common", bad, label)
        check(cfg, PER_AXIS[axis], "per-axis", bad, label)
        if variant.endswith("search"):
            check(cfg, SEARCH, "search stage", bad, label)
    return n, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="valblock-search")
    ap.add_argument("--expect", type=int, default=0, help="cell count to wait for")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    root = Path(a.run_dir) / "RESULTS"

    while True:
        n, bad = scan(root, a.variant)
        if bad:
            print(f"\nMISMATCH -- {len(bad)} problem(s) across {n} cells:\n")
            for b in bad:
                print("  " + b)
            print("\nStop the run. Nothing downstream is worth the GPU-hours.")
            return 1
        if not a.watch or (a.expect and n >= a.expect):
            print(f"OK: {n} cells match the spec "
                  f"({len(COMMON)} common + 2 per-axis"
                  + (f" + {len(SEARCH)} search" if a.variant.endswith('search') else "") + " keys each)")
            return 0
        print(f"  {n} cells so far, all matching; waiting for {a.expect} ...", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
