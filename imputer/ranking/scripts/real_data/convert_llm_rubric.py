#!/usr/bin/env python3
"""
Convert raw TSV annotation files → data_bundle.json with rating distributions,
configs.json, and stan_data.json — mirroring the DataGenConfig / to_stan_data() pattern.

Steps (single pass):
  1. Convert human + LLM TSVs into the data_bundle format.
  2. Attach a 'rating_dist' to every rating record:
       - Human annotators → one-hot distribution (length C) over value
       - LLM annotator    → 4-class probability vector from the TSV
  3. Save configs.json (all CLI args + derived dimensions) and stan_data.json.

Usage:
    python make_data_bundle_dist.py \\
        --human-tsv path/to/human.tsv \\
        --llm-tsv   path/to/llm.tsv \\
        --output-dir runs/my_run \\
        --stan-type factored-dot-product \\
        --D 64 --sigma-annotator 0.5 --sigma-measurement 0.1 --kappa 2.0 --temperature 0.5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Set

# ── Stan type definitions (mirrors configs.py) ────────────────────────────────

STAN_TYPE_REQUIRED: Dict[str, Set[str]] = {
    "discrete": {"M", "S", "sigma_measurement", "kappa", "temperature"},
    "normal-noise-dot-product": {
        "D", "d_annotator", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
        "use_factored_annotator", "derive_thresholds_from_annotator",
    },
    "factored-dot-product": {
        "D", "d_annotator", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
        "use_factored_annotator", "derive_thresholds_from_annotator",
    },
    "tensor": {
        "D", "factor_decay", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
        "use_log_scores", "use_logistic_link", "use_normal_loadings",
    },
}

# Default values per stan_type for type-specific fields
STAN_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "discrete": {
        "M": 6,
        "S": 3,
        "sigma_measurement": 0.1,
        "kappa": 2.0,
        "temperature": 0.5,
    },
    "normal-noise-dot-product": {
        "D": 64,
        "d_annotator": None,          # defaults to D if not set
        "sigma_annotator": 0.5,
        "sigma_measurement": 0.1,
        "kappa": 2.0,
        "temperature": 0.5,
        "use_factored_annotator": 0,
        "derive_thresholds_from_annotator": 0,
    },
    "factored-dot-product": {
        "D": 64,
        "d_annotator": None,          # defaults to D if not set
        "sigma_annotator": 0.5,
        "sigma_measurement": 0.1,
        "kappa": 2.0,
        "temperature": 0.5,
        "use_factored_annotator": 1,
        "derive_thresholds_from_annotator": 1,
    },
    "tensor": {
        "D": 64,
        "factor_decay": 0.5,
        "sigma_annotator": 0.5,
        "sigma_measurement": 0.1,
        "kappa": 2.0,
        "temperature": 0.5,
        "use_log_scores": 0,
        "use_logistic_link": 0,
        "use_normal_loadings": 0,
    },
}

Q_COLS    = [f"Q{i}" for i in range(9)]
PROB_COLS = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]


# ── Stan data builder (mirrors DataGenConfig.to_stan_data) ────────────────────

def build_stan_data(cfg: dict, K_train: int, K_test: int, J: int) -> dict:
    """
    Construct the stan_data.json dict from the resolved config.
    Mirrors DataGenConfig.to_stan_data() exactly.
    """
    stan_type = cfg["stan_type"]
    base = {
        "K_train":                  K_train,
        "K_test":                   K_test,
        "I":                        cfg["I"],
        "J":                        J,
        "C":                        cfg["C"],
        "enable_pairwise_rankings": 0,   # real-data pipeline: no pairwise
        "pairwise_cap_per_item":    cfg["pairwise_cap_per_item"],
    }

    bool_int_keys = {
        "use_factored_annotator", "derive_thresholds_from_annotator",
        "use_log_scores", "use_logistic_link", "use_normal_loadings",
    }
    for key in sorted(STAN_TYPE_REQUIRED[stan_type]):
        val = cfg[key]
        base[key] = int(val) if key in bool_int_keys else val

    # Compatibility shims (matching DataGenConfig.to_stan_data)
    if stan_type == "discrete":
        base.setdefault("D", 1)
        base.setdefault("sigma_annotator", 0.1)
    if stan_type == "tensor":
        base["d_annotator"] = base["D"]

    return base


# ── TSV loaders ───────────────────────────────────────────────────────────────

def load_human(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.drop(columns=[c for c in df.columns if c.startswith("DQ")])
    df = df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")
    return df


def load_llm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df[df["criterion"].isin(Q_COLS)].copy()


# ── Item mapping ──────────────────────────────────────────────────────────────

def split_test_items(human_df: pd.DataFrame, llm_df: pd.DataFrame, n_test: int, seed: int) -> set:
    llm_ids = set(llm_df["text_id"].unique())
    all_ids = sorted(set(human_df["text_id"].unique()) & llm_ids)
    rng = np.random.default_rng(seed)
    return set(rng.choice(all_ids, size=n_test, replace=False).tolist())


def build_item_map(human_df: pd.DataFrame, llm_df: pd.DataFrame, test_ids: set) -> dict:
    llm_ids = set(llm_df["text_id"].unique())
    all_ids = sorted(set(human_df["text_id"].unique()) & llm_ids)
    train_sorted = sorted(tid for tid in all_ids if tid not in test_ids)
    test_sorted  = sorted(tid for tid in all_ids if tid in test_ids)
    return {tid: idx + 1 for idx, tid in enumerate(train_sorted + test_sorted)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def one_hot(value_1indexed: int, length: int) -> list:
    dist = [0.0] * length
    idx = value_1indexed - 1
    if 0 <= idx < length:
        dist[idx] = 1.0
    return dist


def build_llm_dist_lookup(llm_df: pd.DataFrame, item_map: dict) -> dict:
    lookup = {}
    for _, row in llm_df.iterrows():
        text_id = row["text_id"]
        if text_id not in item_map:
            continue
        item_id = item_map[text_id]
        attr_id = int(row["criterion"][1:]) + 1
        lookup[(item_id, attr_id)] = [float(row[c]) for c in PROB_COLS]
    return lookup


def make_record(attribute, annotator, item_id, value, instance, rating_dist) -> dict:
    return {
        "attribute":   attribute,
        "annotator":   annotator,
        "item":        item_id,
        "value":       value,
        "instance":    instance,
        "rating_dist": rating_dist,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_stan_arg(s: str) -> tuple:
    if "=" not in s:
        raise ValueError(f"Invalid --stan-arg: expected KEY=VALUE, got {s!r}")
    key, value = s.split("=", 1)
    key, value = key.strip(), value.strip()
    for cast in (int, float):
        try:
            return (key, cast(value))
        except ValueError:
            pass
    return (key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TSV annotations → data_bundle.json + configs.json + stan_data.json"
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument("--human-tsv", type=Path,
                        default=Path("scripts/real_data/human_judges_real_convs_FIXED_ANON.tsv"))
    parser.add_argument("--llm-tsv", type=Path,
                        default=Path("scripts/real_data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("OUTPUT/generated_data/real_data"),
                        help="Directory where data_bundle.json, configs.json, stan_data.json are saved")

    # ── Core data dimensions ──────────────────────────────────────────────────
    parser.add_argument("--n-test", type=int, default=25,
                        help="Number of test items (default: 25)")
    parser.add_argument("--C", type=int, default=4,
                        help="Number of rating categories (default: 4)")
    parser.add_argument("--I", type=int, default=9,
                        help="Number of attributes / questions (default: 9, i.e. Q0–Q8)")
    parser.add_argument("--pairwise-cap-per-item", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # ── Stan type ─────────────────────────────────────────────────────────────
    parser.add_argument("--stan-type", type=str, default="factored-dot-product",
                        choices=list(STAN_TYPE_REQUIRED),
                        help="Stan model type (default: factored-dot-product)")

    # ── Type-specific args (shared across types; unused ones are ignored) ─────
    parser.add_argument("--D", type=int, default=8,
                        help="Embedding dimension (default from stan-type: 64)")
    parser.add_argument("--d-annotator", type=int, default=None,
                        help="Annotator embedding dim (default: D)")
    parser.add_argument("--sigma-annotator", type=float, default=None)
    parser.add_argument("--sigma-measurement", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None,
                        help="Dirichlet concentration for rating thresholds")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--factor-decay", type=float, default=None,
                        help="CP factor decay (tensor only)")
    parser.add_argument("--M", type=int, default=6, help="(discrete only)")
    parser.add_argument("--S", type=int, default=3, help="(discrete only)")
    parser.add_argument("--derive-thresholds-from-annotator", action="store_true", default=False)

    # ── Tensor misspecification flags ─────────────────────────────────────────
    parser.add_argument("--use-log-scores", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-logistic-link", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-normal-loadings", type=int, default=1, choices=[0, 1])

    # ── Escape hatch for any remaining stan field ─────────────────────────────
    parser.add_argument("--stan-arg", action="append", default=None, metavar="KEY=VALUE",
                        help="Override any config field. E.g. --stan-arg M=8")

    return parser.parse_args()


def resolve_config(args) -> dict:
    """
    Merge per-type defaults with CLI overrides → flat config dict.
    Keys from --stan-arg take highest priority.
    """
    stan_type = args.stan_type
    cfg = dict(STAN_TYPE_DEFAULTS[stan_type])   # start from per-type defaults

    # CLI overrides (only when explicitly provided / non-None)
    cli_overrides = {
        "D":                               args.D,
        "d_annotator":                     args.d_annotator,
        "sigma_annotator":                 args.sigma_annotator,
        "sigma_measurement":               args.sigma_measurement,
        "kappa":                           args.kappa,
        "temperature":                     args.temperature,
        "factor_decay":                    args.factor_decay,
        "M":                               args.M,
        "S":                               args.S,
        "derive_thresholds_from_annotator": 1 if args.derive_thresholds_from_annotator else None,
        "use_log_scores":                  args.use_log_scores,
        "use_logistic_link":               args.use_logistic_link,
        "use_normal_loadings":             args.use_normal_loadings,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    # --stan-arg overrides (highest priority)
    if args.stan_arg:
        for s in args.stan_arg:
            k, v = _parse_stan_arg(s)
            cfg[k] = v

    # d_annotator falls back to D for embedding-based types
    if cfg.get("d_annotator") is None and "D" in cfg:
        cfg["d_annotator"] = cfg["D"]

    # Tensor: misspecification flags default to 0 if still unset
    if stan_type == "tensor":
        for flag in ("use_log_scores", "use_logistic_link", "use_normal_loadings"):
            cfg.setdefault(flag, 0)

    # Validate all required fields for this stan_type are present
    required = STAN_TYPE_REQUIRED[stan_type]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(
            f"For stan_type={stan_type!r}, required fields are missing: {missing}. "
            f"Pass them via CLI args or --stan-arg KEY=VALUE."
        )

    # Attach core (non-type-specific) fields
    cfg["stan_type"]            = stan_type
    cfg["C"]                    = args.C
    cfg["I"]                    = args.I
    cfg["n_test"]               = args.n_test
    cfg["pairwise_cap_per_item"] = args.pairwise_cap_per_item
    cfg["seed"]                 = args.seed
    cfg["human_tsv"]            = str(args.human_tsv)
    cfg["llm_tsv"]              = str(args.llm_tsv)
    cfg["output_dir"]           = str(args.output_dir)

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = resolve_config(args)

    C      = cfg["C"]
    n_test = cfg["n_test"]
    seed   = cfg["seed"]

    print("Loading TSVs...")
    human_df = load_human(args.human_tsv)
    llm_df   = load_llm(args.llm_tsv)

    test_ids        = split_test_items(human_df, llm_df, n_test, seed)
    item_map        = build_item_map(human_df, llm_df, test_ids)
    llm_dist_lookup = build_llm_dist_lookup(llm_df, item_map)

    max_human_ann    = int(human_df["annotator_id"].max()) + 1   # 1-indexed
    llm_annotator_id = max_human_ann + 1
    J                = llm_annotator_id                          # total annotator count

    K_train = sum(1 for tid in item_map if tid not in test_ids)
    K_test  = len(test_ids)

    print(f"  Total items:        {len(item_map)}  (train={K_train}, test={K_test})")
    print(f"  Human annotators:   1–{max_human_ann}")
    print(f"  LLM annotator ID:   {llm_annotator_id}")
    print(f"  LLM dist entries:   {len(llm_dist_lookup)}")

    all_ratings      = []
    observed_ratings = []
    missing_ratings  = []

    # ── Human ratings ────────────────────────────────────────────────────────
    for _, row in human_df.iterrows():
        text_id = row["text_id"]
        if text_id not in item_map:
            continue
        item_id  = item_map[text_id]
        instance = "test" if text_id in test_ids else "train"
        ann_id   = int(row["annotator_id"]) + 1

        for q_col in Q_COLS:
            attribute = int(q_col[1:]) + 1
            raw_val   = row[q_col]
            raw_int   = 0 if pd.isna(raw_val) else int(raw_val)
            value     = raw_int if raw_int != 0 else 1   # DQ / NaN → 1

            record = make_record(
                attribute   = attribute,
                annotator   = ann_id,
                item_id     = item_id,
                value       = value,
                instance    = instance,
                rating_dist = one_hot(value, C),
            )
            all_ratings.append(record)
            (missing_ratings if instance == "test" else observed_ratings).append(record)

    # ── LLM ratings ──────────────────────────────────────────────────────────
    for _, row in llm_df.iterrows():
        text_id = row["text_id"]
        if text_id not in item_map:
            continue
        item_id   = item_map[text_id]
        instance  = "test" if text_id in test_ids else "train"
        attribute = int(row["criterion"][1:]) + 1
        probs     = [float(row[c]) for c in PROB_COLS]
        value     = int(np.argmax(probs)) + 1

        rating_dist = llm_dist_lookup.get((item_id, attribute), one_hot(value, C))

        record = make_record(
            attribute   = attribute,
            annotator   = llm_annotator_id,
            item_id     = item_id,
            value       = value,
            instance    = instance,
            rating_dist = rating_dist,
        )
        all_ratings.append(record)
        observed_ratings.append(record)   # LLM always observed

    # ── Missing ratings index ─────────────────────────────────────────────────
    missing_keys = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    missing_ratings_indexes_in_test_instance = [
        i for i, rec in enumerate(all_ratings)
        if rec["instance"] == "test"
        and (rec["attribute"], rec["annotator"], rec["item"]) in missing_keys
    ]

    # ── Stats ─────────────────────────────────────────────────────────────────
    train_ratings  = [r for r in all_ratings      if r["instance"] == "train"]
    test_ratings   = [r for r in all_ratings      if r["instance"] == "test"]
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    test_observed  = [r for r in observed_ratings if r["instance"] == "test"]
    total_ratings  = len(all_ratings)

    stats = {
        "K_train":                K_train,
        "K_test":                 K_test,
        "total_items":            len(item_map),
        "total_possible_ratings": total_ratings,
        "total_ratings":          total_ratings,
        "observed_ratings":       len(observed_ratings),
        "missing_ratings":        len(missing_ratings),
        "train_ratings":          len(train_ratings),
        "test_ratings":           len(test_ratings),
        "train_observed":         len(train_observed),
        "test_observed":          len(test_observed),
        "total_pairwise":         0,
        "observed_pairwise":      0,
        "missing_pairwise":       0,
        "train_pairwise":         0,
        "test_pairwise":          0,
        "observation_rate":       len(observed_ratings) / total_ratings if total_ratings else 0.0,
        "train_observation_rate": len(train_observed) / len(train_ratings) if train_ratings else 0.0,
        "test_observation_rate":  len(test_observed)  / len(test_ratings)  if test_ratings  else 0.0,
        "protocol":               "random_split",
        "mcar_missing_rate":      None,
    }

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for lst_name, lst in [("all_ratings", all_ratings), ("observed_ratings", observed_ratings),
                           ("missing_ratings", missing_ratings)]:
        n_missing_dist = sum(1 for r in lst if "rating_dist" not in r)
        assert n_missing_dist == 0, f"{n_missing_dist} records in {lst_name} missing rating_dist"

    llm_sample = next((r for r in observed_ratings if r["annotator"] == llm_annotator_id), None)
    if llm_sample:
        print(f"  Sample LLM dist:    {llm_sample['rating_dist']}  "
              f"(item={llm_sample['item']}, attr={llm_sample['attribute']})")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # data_bundle.json
    bundle = {
        "embeddings":            None,
        "mean_preferences":      None,
        "annotator_preferences": None,
        "rating_probs":          None,
        "rating_cumprobs":       None,
        "rating_thresholds_z":   None,
        "base_scores":           None,
        "all_ratings":           all_ratings,
        "all_pairwise":          [],
        "observed_ratings":      observed_ratings,
        "missing_ratings":       missing_ratings,
        "observed_pairwise":     [],
        "missing_pairwise":      [],
        "missing_ratings_indexes_in_test_instance": missing_ratings_indexes_in_test_instance,
        "stats":                 stats,
        "train_posterior_rating_probs": None,
        "test_posterior_rating_probs":  None,
    }
    with open(out_dir / "data_bundle.json", "w") as f:
        json.dump(bundle, f)
    print(f"\nSaved data_bundle.json → {out_dir / 'data_bundle.json'}")

    # configs.json — mirrors the "datagen" key from DataGenConfig
    configs = {
        "datagen": {
            # Core dimensions (derived from data)
            "K_train":                  K_train,
            "K_test":                   K_test,
            "I":                        cfg["I"],
            "J":                        J,
            "C":                        C,
            # Protocol / misc
            "enable_pairwise_rankings": False,
            "pairwise_cap_per_item":    cfg["pairwise_cap_per_item"],
            "observation_protocol":     "random_split",
            "mcar_missing_rate":        None,
            "pairwise_observation_rate": 1.0,
            "seed":                     seed,
            "stan_type":                cfg["stan_type"],
            # Type-specific fields (only those required for this stan_type)
            **{k: cfg[k] for k in sorted(STAN_TYPE_REQUIRED[cfg["stan_type"]])},
            # Provenance
            "human_tsv":                cfg["human_tsv"],
            "llm_tsv":                  cfg["llm_tsv"],
        }
    }
    with open(out_dir / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Saved configs.json     → {out_dir / 'configs.json'}")

    # stan_data.json — mirrors DataGenConfig.to_stan_data()
    stan_data = build_stan_data(cfg, K_train, K_test, J)
    with open(out_dir / "stan_data.json", "w") as f:
        json.dump(stan_data, f, indent=2)
    print(f"Saved stan_data.json   → {out_dir / 'stan_data.json'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    llm_ct   = sum(1 for r in all_ratings if r["annotator"] == llm_annotator_id)
    human_ct = total_ratings - llm_ct
    print(f"\nDone.")
    print(f"  Train items:    {K_train}")
    print(f"  Test items:     {K_test}")
    print(f"  Annotators:     {J}  ({max_human_ann} human + 1 LLM)")
    print(f"  Total ratings:  {total_ratings}  ({human_ct} human one-hot, {llm_ct} LLM soft)")
    print(f"  Observed:       {len(observed_ratings)}")
    print(f"  Missing:        {len(missing_ratings)}  (human ratings in test items)")


if __name__ == "__main__":
    main()