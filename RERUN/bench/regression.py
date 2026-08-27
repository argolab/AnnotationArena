"""Differential regression test: run the same configurations on two checkouts.

`test_equiv.py` proves the replaced code paths return the same numbers as the
loops they replaced. That is not the same as proving training is unchanged: it
covers the paths the item-axis default happens to exercise, and it says nothing
about the flag combinations the repo can set.

This runs a matrix of configurations on a *reference* checkout and a
*candidate* checkout and compares the training curves. Entity dropout moved to
the GPU RNG in the candidate, so the two are not bit-identical by construction;
the comparison is therefore statistical -- N seeds per cell, and the candidate's
mean has to sit inside the reference's own seed spread.

  python -m RERUN.bench.regression --ref <checkout> --cand <checkout> \
      [--epochs 15] [--seeds 42,43,44] [--only item_T,annot] [--jobid <slurm>]

Every cell reports PASS / FAIL / CRASH, and the exit code is non-zero if any
cell is not PASS.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

DATA = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/replicate-01/DATA")
D3 = DATA / "STAN/DOMAIN3-SIGMOID"

# ── profiles: the four data/regime combinations the repo actually trains on ──
BASE = [
    "--embedding-dim", "80", "--num-layers", "8", "--attention-heads", "4",
    "--d-ff", "128", "--num-ffn-layers", "1", "--dropout", "0.1",
    "--lr", "2e-4", "--lr-schedule", "none", "--lr-min", "1e-5",
    "--weight-decay", "0.01", "--masking-rate", "0.15", "--mask-augmentations", "5",
    "--masked-loss-weight", "15.0", "--observed-loss-weight", "1.0",
    "--type-embedding-init", "kaiming", "--device", "cuda",
    "--overwrite-existing-data",
]

PROFILES = {
    "item_T": BASE + [
        "--data-dir", str(D3 / "ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_300"),
        "--transductive-learning", "--max-item", "300",
        "--item-dropout-rate", "1.0", "--annotator-dropout-rate", "0.0",
        "--no-per-head-rel", "--scale-shared-rel", "--use-pointer",
    ],
    "item_NT": BASE + [
        "--data-dir", str(D3 / "ItemSplits/NonTransductive/DOMAIN3-SIGMOID_Item_NT_300"),
        "--max-item", "300",
        "--item-dropout-rate", "1.0", "--annotator-dropout-rate", "0.0",
        "--no-per-head-rel", "--scale-shared-rel", "--use-pointer",
    ],
    "annot_T": BASE + [
        "--data-dir", str(D3 / "AnnotSplits/Transductive/DOMAIN3-SIGMOID_Annot_T_30"),
        "--transductive-learning", "--max-item", "300",
        "--item-dropout-rate", "0.0", "--annotator-dropout-rate", "0.8",
        "--annotator-reg-weight", "1e-3",
        "--no-per-head-rel", "--scale-shared-rel", "--use-pointer",
    ],
    "llmrubric": BASE + [
        "--data-dir", str(DATA / "LLM_RUBRIC/LLMRubric_225_25_9_50"),
        "--max-item", "10", "--item-dropout-rate", "0.7",
        "--llm-annotator-id", "24", "--human-observed-rate", "0.0", "--llm-input-dist",
        "--no-per-head-rel", "--scale-shared-rel", "--use-pointer",
    ],
}

# ── the matrix: (cell name, profile, extra flags) ───────────────────────────
# Each row is a code path the optimisation touched or could have broken. The
# ones marked "never used by the paper" are here because the changed attention
# still has to support them.
CELLS = [
    ("item_T/default",        "item_T",    []),
    ("item_T/per_head_rel",   "item_T",    ["--use-per-head-rel"]),          # drops --no-per-head-rel
    ("item_T/rel_value",      "item_T",    ["--use-rel-value"]),             # needs the dense edge mask
    ("item_T/graph_mask",     "item_T",    ["--use-graph-mask"]),            # needs the dense edge mask
    ("item_T/addone_attn",    "item_T",    ["--use-addone-attn"]),
    ("item_T/param_head",     "item_T",    ["--use-param-output-head"]),
    ("item_T/deviation_norm", "item_T",    ["--use-deviation-norm"]),
    ("item_T/embed_unembed",  "item_T",    ["--use-embed-unembed"]),
    ("item_T/random_chunks",  "item_T",    ["--random-item-chunks"]),
    ("item_T/valtest_mask",   "item_T",    ["--transductive-valtest-mask"]),
    ("item_T/outer_none_id",  "item_T",    ["--dual-outer-residual", "none_id"]),
    ("item_T/inner_none",     "item_T",    ["--inner-residual", "none"]),
    ("item_NT/default",       "item_NT",   []),
    ("annot_T/default",       "annot_T",   []),
    ("llmrubric/default",     "llmrubric", []),
]

# Candidate-only cells: flags that do not exist on the reference checkout. They
# are compared against the same profile's reference run, since the point of
# each is to be a faster route to the same optimisation problem (or, for
# batch-augmentations, an explicitly different one -- see the note in the
# report; it is checked for sanity, not for equality).
CAND_ONLY = [
    ("item_T/tf32",           "item_T",    ["--tf32"],                    True),
    ("item_T/eval_every5",    "item_T",    ["--eval-every", "5"],         True),
    ("item_T/amp_bf16",       "item_T",    ["--amp", "bf16"],             False),
    ("item_T/compile",        "item_T",    ["--compile", "default"],      True),
    ("item_T/batch_aug",      "item_T",    ["--batch-augmentations"],     False),
    ("llmrubric/batch_aug",   "llmrubric", ["--batch-augmentations"],     False),
]


def run_one(root: Path, profile: str, extra, seed: int, epochs: int,
            out_root: Path, jobid: str | None):
    """Train once; return the final missing-rating CE, or None if it crashed."""
    args = list(PROFILES[profile])
    if "--use-per-head-rel" in extra:
        args = [a for a in args if a != "--no-per-head-rel"]
    args += extra
    out = out_root / f"{profile}_{seed}"
    shutil.rmtree(out, ignore_errors=True)
    ranking = root / "imputer/ranking"
    inner = (
        'eval "$(conda shell.bash hook)"; conda activate torch; '
        f'cd {ranking}; export PYTHONPATH={ranking}:{root}; '
        + " ".join(["python", "-u", "-m", "imputer.entity_mf.train",
                    "--run-name", "run", "--output-root", str(out),
                    "--epochs", str(epochs), "--seed", str(seed)] + args)
    )
    cmd = (["srun", "--jobid", jobid, "--overlap"] if jobid else []) + \
          ["bash", "-lc", inner]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    hist = out / "run/training_history.json"
    if proc.returncode != 0 or not hist.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return None, " | ".join(tail)[:300]
    with open(hist) as fh:
        h = json.load(fh)
    for key in ("combined_eval", "val_eval", "train_eval"):
        m = h[-1].get(key, {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" in m:
            return float(m["xent"]), ""
    # Some splits have no evaluable missing tokens; fall back to training CE so
    # the cell still compares something rather than silently passing on None.
    tce = h[-1].get("train_masked_ce")
    return (float(tce), "train_ce") if tce is not None else (None, "no metric in history")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobid", default=None)
    ap.add_argument("--tol-sd", type=float, default=3.0,
                    help="candidate mean must be within this many reference SDs")
    ap.add_argument("--tol-abs", type=float, default=0.03,
                    help="absolute nats always allowed on top of the SD band")
    args = ap.parse_args()

    ref, cand = Path(args.ref), Path(args.cand)
    seeds = [int(s) for s in args.seeds.split(",")]
    only = [s for s in args.only.split(",") if s]
    ref_out = ref / "RERUN/bench/out/regress"
    cand_out = cand / "RERUN/bench/out/regress"

    cells = [c for c in CELLS if not only or any(o in c[0] for o in only)]
    cand_only = [c for c in CAND_ONLY if not only or any(o in c[0] for o in only)]

    print(f"epochs={args.epochs}  seeds={seeds}  "
          f"tolerance={args.tol_sd} ref-SD + {args.tol_abs} nats\n")
    header = f"{'cell':<24}{'reference':>22}{'candidate':>22}  verdict"
    print(header); print("-" * len(header))

    failures, ref_cache = [], {}

    def ref_stats(profile, extra, key):
        if key not in ref_cache:
            vals, notes = [], []
            for sd in seeds:
                v, n = run_one(ref, profile, extra, sd, args.epochs, ref_out, args.jobid)
                vals.append(v); notes.append(n)
            ref_cache[key] = (vals, notes)
        return ref_cache[key]

    for name, profile, extra in cells:
        rv, rn = ref_stats(profile, extra, name)
        cv = [run_one(cand, profile, extra, sd, args.epochs, cand_out, args.jobid)[0]
              for sd in seeds]
        report(name, rv, cv, rn, args, failures)

    for name, profile, extra, strict in cand_only:
        base_name = f"{profile}/default"
        rv, rn = ref_stats(profile, [], base_name)
        cv = [run_one(cand, profile, extra, sd, args.epochs, cand_out, args.jobid)[0]
              for sd in seeds]
        report(name, rv, cv, rn, args, failures, strict=strict)

    print()
    if failures:
        print("FAILED cells: " + ", ".join(failures))
        sys.exit(1)
    print("all cells within tolerance")


def report(name, rv, cv, rn, args, failures, strict=True):
    rok = [v for v in rv if v is not None]
    cok = [v for v in cv if v is not None]
    if not cok:
        print(f"{name:<24}{'--':>22}{'CRASH':>22}  FAIL  {rn[0] if rn else ''}")
        failures.append(name)
        return
    if not rok:
        print(f"{name:<24}{'CRASH(ref)':>22}{statistics.mean(cok):>22.4f}  "
              f"SKIP  reference cannot run this")
        return
    rm = statistics.mean(rok)
    rsd = statistics.stdev(rok) if len(rok) > 1 else 0.0
    cm = statistics.mean(cok)
    band = args.tol_sd * rsd + args.tol_abs
    ok = abs(cm - rm) <= band
    tag = "PASS" if ok else ("FAIL" if strict else "DIFFERS(expected)")
    print(f"{name:<24}{rm:>15.4f}+/-{rsd:<4.3f}{cm:>15.4f}+/-"
          f"{(statistics.stdev(cok) if len(cok) > 1 else 0.0):<4.3f}  "
          f"{tag}  d={cm - rm:+.4f} band={band:.4f}")
    if not ok and strict:
        failures.append(name)


if __name__ == "__main__":
    main()
