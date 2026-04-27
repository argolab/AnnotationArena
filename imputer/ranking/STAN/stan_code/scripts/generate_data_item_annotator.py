#!/usr/bin/env python3
"""Generate synthetic data with both item and annotator generalization splits.

This wraps the existing item-split generator, then repartitions ratings by both
item group and annotator group:
  - train if item=train and annotator=train
  - val   if no test entity is involved and at least one entity is val
  - test  if item=test or annotator=test

The saved configs keep the usual item-split fields (K_train/K_val/K_test, J)
so existing inference code remains compatible. Annotator partition metadata is
stored under J_train_split/J_val_split/J_test_split.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from STAN.stan_code.pipeline.configs import DataGenConfig, STAN_TYPE_REQUIRED
from STAN.stan_code.pipeline.data_gen import generate_data
from STAN.stan_code.pipeline.io import new_run_dir, save_bundle, save_configs, save_json
from STAN.stan_code.pipeline.bundle import GroundTruthBundle


def _parse_stan_arg(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --stan-arg: expected KEY=VALUE, got {s!r}")
    key, value = s.split("=", 1)
    key = key.strip()
    value = value.strip()
    if value.lower() in ("true", "1"):
        return key, True
    if value.lower() in ("false", "0"):
        return key, False
    try:
        return key, int(value)
    except ValueError:
        pass
    try:
        return key, float(value)
    except ValueError:
        pass
    return key, value


def _item_group(item_id: int, k_train: int, k_val: int) -> str:
    if item_id <= k_train:
        return "train"
    if item_id <= k_train + k_val:
        return "val"
    return "test"


def _annotator_group(annotator_id: int, j_train: int, j_val: int) -> str:
    if annotator_id <= j_train:
        return "train"
    if annotator_id <= j_train + j_val:
        return "val"
    return "test"


def _combine_instance(item_group: str, annotator_group: str) -> str:
    if item_group == "test" or annotator_group == "test":
        return "test"
    if item_group == "val" or annotator_group == "val":
        return "val"
    return "train"


def _relabel_rating_rows(
    rows: List[Dict[str, Any]],
    *,
    k_train: int,
    k_val: int,
    j_train: int,
    j_val: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        item_group = _item_group(int(new_row["item"]), k_train, k_val)
        ann_group = _annotator_group(int(new_row["annotator"]), j_train, j_val)
        new_row["instance"] = _combine_instance(item_group, ann_group)
        out.append(new_row)
    return out


def _relabel_pairwise_rows(
    rows: List[Dict[str, Any]],
    *,
    k_train: int,
    k_val: int,
    j_train: int,
    j_val: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        item_groups = [_item_group(int(item_id), k_train, k_val) for item_id in new_row["items"]]
        ann_group = _annotator_group(int(new_row["annotator"]), j_train, j_val)
        label = ann_group
        for g in item_groups:
            label = _combine_instance(g, label)
        new_row["instance"] = label
        out.append(new_row)
    return out


def _stats_from_bundle(
    *,
    config: DataGenConfig,
    j_train: int,
    j_val: int,
    j_test: int,
    all_ratings: List[Dict[str, Any]],
    observed_ratings: List[Dict[str, Any]],
    missing_ratings: List[Dict[str, Any]],
    all_pairwise: List[Dict[str, Any]],
    observed_pairwise: List[Dict[str, Any]],
    missing_pairwise: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _count(rows: List[Dict[str, Any]], instance: str) -> int:
        return sum(1 for row in rows if row["instance"] == instance)

    train_ratings = _count(all_ratings, "train")
    val_ratings = _count(all_ratings, "val")
    test_ratings = _count(all_ratings, "test")
    train_obs = _count(observed_ratings, "train")
    val_obs = _count(observed_ratings, "val")
    test_obs = _count(observed_ratings, "test")

    return {
        "K_train": config.K_train,
        "K_val": config.K_val,
        "K_test": config.K_test,
        "J": config.J,
        "J_train_split": j_train,
        "J_val_split": j_val,
        "J_test_split": j_test,
        "I": config.I,
        "C": config.C,
        "total_items": config.K_train + config.K_val + config.K_test,
        "total_possible_ratings": len(all_ratings),
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": train_ratings,
        "val_ratings": val_ratings,
        "test_ratings": test_ratings,
        "train_observed": train_obs,
        "val_observed": val_obs,
        "test_observed": test_obs,
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": _count(all_pairwise, "train"),
        "val_pairwise": _count(all_pairwise, "val"),
        "test_pairwise": _count(all_pairwise, "test"),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0.0,
        "train_observation_rate": train_obs / train_ratings if train_ratings else 0.0,
        "val_observation_rate": val_obs / val_ratings if val_ratings else 0.0,
        "test_observation_rate": test_obs / test_ratings if test_ratings else 0.0,
    }


def _repartition_bundle(
    bundle: GroundTruthBundle,
    *,
    config: DataGenConfig,
    j_train: int,
    j_val: int,
    j_test: int,
) -> Dict[str, Any]:
    all_ratings = _relabel_rating_rows(
        bundle.all_ratings,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    observed_ratings = _relabel_rating_rows(
        bundle.observed_ratings,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    missing_ratings = _relabel_rating_rows(
        bundle.missing_ratings,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    all_pairwise = _relabel_pairwise_rows(
        bundle.all_pairwise,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    observed_pairwise = _relabel_pairwise_rows(
        bundle.observed_pairwise,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    missing_pairwise = _relabel_pairwise_rows(
        bundle.missing_pairwise,
        k_train=config.K_train,
        k_val=config.K_val,
        j_train=j_train,
        j_val=j_val,
    )
    missing_test_indexes = [i for i, row in enumerate(missing_ratings) if row["instance"] == "test"]

    bundle_dict: Dict[str, Any] = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_test_indexes,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": _stats_from_bundle(
            config=config,
            j_train=j_train,
            j_val=j_val,
            j_test=j_test,
            all_ratings=all_ratings,
            observed_ratings=observed_ratings,
            missing_ratings=missing_ratings,
            all_pairwise=all_pairwise,
            observed_pairwise=observed_pairwise,
            missing_pairwise=missing_pairwise,
        ),
    }

    optional_keys = [
        "embeddings",
        "mean_preferences",
        "annotator_preferences",
        "rating_probs",
        "rating_cumprobs",
        "rating_thresholds_z",
        "base_scores",
        "train_posterior_rating_probs",
        "val_posterior_rating_probs",
        "test_posterior_rating_probs",
    ]
    for key in optional_keys:
        value = getattr(bundle, key, None)
        if value is not None:
            bundle_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value

    if bundle.extra_ground_truth:
        for key, value in bundle.extra_ground_truth.items():
            bundle_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value

    return bundle_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data with both item and annotator split axes")
    parser.add_argument("--output-dir", type=str, default="OUTPUT/generated_data")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stan-type", type=str, default="tensor",
                        choices=["normal-noise-dot-product", "factored-dot-product", "discrete", "tensor", "dawid-skene"])
    parser.add_argument("--stan-file", type=str, default=None)
    parser.add_argument("--overwrite-existing-data", action="store_true")
    parser.add_argument(
        "--force-stan-recompile",
        action="store_true",
        help="Force cmdstanpy to recompile the Stan generator (use after editing .stan files).",
    )
    parser.add_argument("--K-train", type=int, required=True)
    parser.add_argument("--K-test", type=int, required=True)
    parser.add_argument("--K-val", type=int, required=True)
    parser.add_argument("--J-train", type=int, required=True)
    parser.add_argument("--J-val", type=int, required=True)
    parser.add_argument("--J-test", type=int, required=True)
    parser.add_argument("--I", type=int, required=True)
    parser.add_argument("--J", type=int, default=None,
                        help="Optional total annotator count; defaults to J_train + J_val + J_test")
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--C", type=int, default=5)
    parser.add_argument("--enable-pairwise-rankings", action="store_true")
    parser.add_argument("--pairwise-cap-per-item", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observation-protocol", type=str, default="mcar",
                        choices=["tie_breaking", "mcar", "extended_rankings"])
    parser.add_argument("--mcar-missing-rate", type=float, default=0.5)
    parser.add_argument("--pairwise-observation-rate", type=float, default=1.0)
    parser.add_argument("--sigma-annotator", type=float, default=0.5)
    parser.add_argument("--sigma-measurement", type=float, default=0.1)
    parser.add_argument("--kappa", "--alpha-dirichlet", dest="kappa", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--stan-arg", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--d-annotator", type=int, default=None)
    parser.add_argument("--derive-thresholds-from-annotator", action="store_true", default=False)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--sigma-u", type=float, default=1.0)
    parser.add_argument("--sigma-v", type=float, default=1.0)
    parser.add_argument("--sigma-uit", type=float, default=0.5)
    parser.add_argument("--use-dawid-skene-noise", type=int, default=0, choices=[0, 1])
    parser.add_argument("--factor-decay", type=float, default=None)
    parser.add_argument("--alpha-confusion", type=float, default=10.0)
    args = parser.parse_args()

    j_total = args.J if args.J is not None else args.J_train + args.J_val + args.J_test
    if j_total != args.J_train + args.J_val + args.J_test:
        raise ValueError("J must equal J_train + J_val + J_test for item-annotator split generation")

    stan_arg: Dict[str, Any] = {}
    for arg in args.stan_arg or []:
        k, v = _parse_stan_arg(arg)
        stan_arg[k] = v

    defaults = {
        "D": args.D,
        "d_annotator": args.d_annotator if args.d_annotator is not None else args.D,
        "sigma_annotator": args.sigma_annotator,
        "sigma_measurement": args.sigma_measurement,
        "kappa": args.kappa,
        "temperature": args.temperature,
        "factor_decay": args.factor_decay,
        "use_log_scores": 0,
        "use_logistic_link": 0,
        "use_normal_loadings": 0,
        "alpha_confusion": args.alpha_confusion,
        "T": args.T,
        "sigma_u": args.sigma_u,
        "sigma_v": args.sigma_v,
        "sigma_uit": args.sigma_uit,
        "use_dawid_skene_noise": args.use_dawid_skene_noise,
    }
    if args.stan_type == "normal-noise-dot-product":
        defaults["use_factored_annotator"] = 0
        defaults["derive_thresholds_from_annotator"] = 0
    elif args.stan_type == "factored-dot-product":
        defaults["use_factored_annotator"] = 1
        defaults["derive_thresholds_from_annotator"] = 1 if args.derive_thresholds_from_annotator else 0
    elif args.stan_type == "dawid-skene":
        defaults["use_factored_annotator"] = 1
        defaults["derive_thresholds_from_annotator"] = 1 if args.derive_thresholds_from_annotator else 0
    elif args.stan_type == "tensor":
        defaults["derive_thresholds_from_annotator"] = 1 if args.derive_thresholds_from_annotator else 0

    required = STAN_TYPE_REQUIRED[args.stan_type]
    type_kwargs = {k: stan_arg.get(k, defaults.get(k)) for k in required}
    for key in list(type_kwargs):
        if type_kwargs[key] is None and key in required:
            raise ValueError(f"For stan_type={args.stan_type!r}, required field {key!r} is missing")

    config = DataGenConfig(
        K_train=args.K_train,
        K_test=args.K_test,
        K_val=args.K_val,
        I=args.I,
        J=j_total,
        C=args.C,
        enable_pairwise_rankings=args.enable_pairwise_rankings,
        pairwise_cap_per_item=args.pairwise_cap_per_item,
        observation_protocol=args.observation_protocol,
        mcar_missing_rate=args.mcar_missing_rate,
        pairwise_observation_rate=args.pairwise_observation_rate,
        seed=args.seed,
        stan_type=args.stan_type,
        **type_kwargs,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        potential = output_path / args.run_name
        if potential.exists() and args.overwrite_existing_data:
            print(f"\033[91mWARNING: Overwriting existing data directory: {potential}\033[0m")
            shutil.rmtree(potential)
    run_dir = new_run_dir(output_path, run_name=args.run_name)

    print(f"Generating item+annotator split data with config: {config}")
    print(f"  Item split: train={args.K_train}, val={args.K_val}, test={args.K_test}")
    print(f"  Annotator split: train={args.J_train}, val={args.J_val}, test={args.J_test}")
    print(f"Output directory: {run_dir}")

    bundle = generate_data(
        config,
        stan_file=args.stan_file,
        force_stan_recompile=bool(args.force_stan_recompile),
    )
    bundle_dict = _repartition_bundle(
        bundle,
        config=config,
        j_train=args.J_train,
        j_val=args.J_val,
        j_test=args.J_test,
    )

    datagen_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__.copy()
    datagen_dict["J_train_split"] = args.J_train
    datagen_dict["J_val_split"] = args.J_val
    datagen_dict["J_test_split"] = args.J_test
    save_configs(run_dir, datagen=datagen_dict)

    stan_data = config.to_stan_data()
    stan_data["J_train_split"] = args.J_train
    stan_data["J_val_split"] = args.J_val
    stan_data["J_test_split"] = args.J_test
    save_json(stan_data, run_dir / "stan_data.json")
    save_bundle(run_dir, bundle_dict)

    s = bundle_dict["stats"]
    print("\nData generation complete!")
    print(f"Total ratings: {s['total_ratings']}")
    print(f"Observed ratings: {s['observed_ratings']}  ({s['observation_rate']:.2%})")
    print(f"  Train: {s['train_observed']}/{s['train_ratings']} ({s['train_observation_rate']:.2%})")
    print(f"  Val:   {s['val_observed']}/{s['val_ratings']} ({s['val_observation_rate']:.2%})")
    print(f"  Test:  {s['test_observed']}/{s['test_ratings']} ({s['test_observation_rate']:.2%})")


if __name__ == "__main__":
    main()
