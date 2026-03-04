#!/usr/bin/env python3
"""
Bundle rewrite service for missingness patterns.

Rewrites observed_ratings/missing_ratings in an existing data bundle according to
specified missingness patterns, enabling controlled experiments without regenerating
data from Stan.

Usage:
    # MCAR-IID: 50% missing
    PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
        --input-dir OUTPUT/generated_data/base_run \\
        --output-dir OUTPUT/generated_data/base_run_mcar50 \\
        --pattern mcar_iid --missing-rate 0.5 --seed 42

    # Balanced-degree MCAR: ensure min degrees, then fill to 50% missing
    PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
        --input-dir OUTPUT/generated_data/base_run \\
        --output-dir OUTPUT/generated_data/base_run_balanced_mcar50 \\
        --pattern balanced_degree_mcar \\
        --missing-rate 0.5 \\
        --min-degree-annotator 3 --min-degree-item 3 \\
        --seed 42

    # Annotator block: hide 30% of annotators' ratings at 80% missing rate
    PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
        --input-dir OUTPUT/generated_data/base_run \\
        --output-dir OUTPUT/generated_data/base_run_annotator_block \\
        --pattern annotator_block \\
        --missing-rate 0.5 \\
        --block-fraction-annotator 0.3 \\
        --block-missing-fraction 0.8 \\
        --seed 42

    # Item block: hide 20% of items' ratings at 70% missing rate
    PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
        --input-dir OUTPUT/generated_data/base_run \\
        --output-dir OUTPUT/generated_data/base_run_item_block \\
        --pattern item_block \\
        --missing-rate 0.5 \\
        --block-fraction-item 0.2 \\
        --block-missing-fraction 0.7 \\
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import numpy as np

from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import save_bundle, save_json


def compute_stats(
    all_ratings: List[Dict[str, Any]],
    observed_ratings: List[Dict[str, Any]],
    missing_ratings: List[Dict[str, Any]],
    bundle: GroundTruthBundle,
) -> Dict[str, Any]:
    """Compute updated stats dictionary."""
    train_ratings = [r for r in all_ratings if r["instance"] == "train"]
    test_ratings = [r for r in all_ratings if r["instance"] == "test"]
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    test_observed = [r for r in observed_ratings if r["instance"] == "test"]
    
    stats = dict(bundle.stats)
    stats.update({
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_observed": len(train_observed),
        "test_observed": len(test_observed),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_observed) / len(train_ratings) if train_ratings else 0,
        "test_observation_rate": len(test_observed) / len(test_ratings) if test_ratings else 0,
    })
    return stats


def pattern_mcar_iid(
    all_ratings: List[Dict[str, Any]],
    missing_rate: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    MCAR-IID: randomly sample observed subset uniformly over (i,j,k).
    
    Args:
        all_ratings: Full universe of ratings
        missing_rate: Target fraction to mark as missing (0.0-1.0)
        seed: Random seed
    
    Returns:
        (observed_ratings, missing_ratings)
    """
    rng = random.Random(seed)
    observed_rate = 1.0 - missing_rate
    
    # Shuffle and sample
    ratings_copy = all_ratings.copy()
    rng.shuffle(ratings_copy)
    
    n_observed = int(len(ratings_copy) * observed_rate)
    observed_ratings = ratings_copy[:n_observed]
    missing_ratings = ratings_copy[n_observed:]
    
    return observed_ratings, missing_ratings


def pattern_balanced_degree_mcar(
    all_ratings: List[Dict[str, Any]],
    missing_rate: float,
    min_degree_annotator: int,
    min_degree_item: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Balanced-degree MCAR: enforce minimum degrees, then fill to target rate.
    
    First ensures each annotator appears in >= min_degree_annotator observed ratings
    and each item appears in >= min_degree_item observed ratings (in train),
    then samples additional edges uniformly until reaching target observed fraction.
    
    Args:
        all_ratings: Full universe of ratings
        missing_rate: Target fraction to mark as missing
        min_degree_annotator: Minimum train-observed degree for each annotator
        min_degree_item: Minimum train-observed degree for each item
        seed: Random seed
    
    Returns:
        (observed_ratings, missing_ratings)
    """
    rng = random.Random(seed)
    
    # Separate train and test
    train_ratings = [r for r in all_ratings if r["instance"] == "train"]
    test_ratings = [r for r in all_ratings if r["instance"] == "test"]
    
    # Track degrees in train bipartite graph
    annotator_degree = defaultdict(int)
    item_degree = defaultdict(int)
    observed_set: Set[Tuple[int, int, int, str]] = set()  # (i, j, k, instance)
    
    # Phase 1: Enforce minimum degrees on train ratings
    train_shuffled = train_ratings.copy()
    rng.shuffle(train_shuffled)
    
    for rating in train_shuffled:
        i, j, k = rating["attribute"], rating["annotator"], rating["item"]
        key = (i, j, k, "train")
        
        if (annotator_degree[j] < min_degree_annotator or 
            item_degree[k] < min_degree_item):
            observed_set.add(key)
            annotator_degree[j] += 1
            item_degree[k] += 1
    
    # Phase 2: Fill remaining train ratings to target rate
    target_train_observed = int(len(train_ratings) * (1.0 - missing_rate))
    remaining_train = [r for r in train_shuffled 
                       if (r["attribute"], r["annotator"], r["item"], "train") not in observed_set]
    rng.shuffle(remaining_train)
    
    current_train_observed = len([k for k in observed_set if k[3] == "train"])
    needed_train = max(0, target_train_observed - current_train_observed)
    
    for rating in remaining_train[:needed_train]:
        key = (rating["attribute"], rating["annotator"], rating["item"], "train")
        observed_set.add(key)
        annotator_degree[rating["annotator"]] += 1
        item_degree[rating["item"]] += 1
    
    # Phase 3: Sample test ratings uniformly to match overall rate
    test_shuffled = test_ratings.copy()
    rng.shuffle(test_shuffled)
    
    target_total_observed = int(len(all_ratings) * (1.0 - missing_rate))
    current_observed = len(observed_set)
    needed_test = max(0, target_total_observed - current_observed)
    
    for rating in test_shuffled[:needed_test]:
        key = (rating["attribute"], rating["annotator"], rating["item"], "test")
        observed_set.add(key)
    
    # Build final lists
    observed_ratings = [
        r for r in all_ratings
        if (r["attribute"], r["annotator"], r["item"], r["instance"]) in observed_set
    ]
    missing_ratings = [
        r for r in all_ratings
        if (r["attribute"], r["annotator"], r["item"], r["instance"]) not in observed_set
    ]
    
    return observed_ratings, missing_ratings


def pattern_annotator_block(
    all_ratings: List[Dict[str, Any]],
    missing_rate: float,
    block_fraction_annotator: float,
    block_missing_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Annotator-block missingness: hide most ratings from a subset of annotators.
    
    Chooses a fraction of annotators and marks block_missing_fraction of their ratings
    as missing, while keeping the complement relatively well observed to match
    overall missing_rate.
    
    Args:
        all_ratings: Full universe of ratings
        missing_rate: Target overall missing fraction
        block_fraction_annotator: Fraction of annotators to block (0.0-1.0)
        block_missing_fraction: Fraction of blocked annotators' ratings to hide (0.0-1.0)
        seed: Random seed
    
    Returns:
        (observed_ratings, missing_ratings)
    """
    rng = random.Random(seed)
    
    # Get unique annotators
    annotators = sorted(set(r["annotator"] for r in all_ratings))
    n_block = max(1, int(len(annotators) * block_fraction_annotator))
    blocked_annotators = set(rng.sample(annotators, n_block))
    
    # Partition ratings by annotator
    blocked_ratings = [r for r in all_ratings if r["annotator"] in blocked_annotators]
    complement_ratings = [r for r in all_ratings if r["annotator"] not in blocked_annotators]
    
    # Hide block_missing_fraction of blocked annotators' ratings
    blocked_shuffled = blocked_ratings.copy()
    rng.shuffle(blocked_shuffled)
    n_block_missing = int(len(blocked_shuffled) * block_missing_fraction)
    blocked_missing = blocked_shuffled[:n_block_missing]
    blocked_observed = blocked_shuffled[n_block_missing:]
    
    # Compute target overall missing count
    target_missing = int(len(all_ratings) * missing_rate)
    remaining_missing_needed = max(0, target_missing - len(blocked_missing))
    
    # Sample from complement to reach target rate
    complement_shuffled = complement_ratings.copy()
    rng.shuffle(complement_shuffled)
    complement_missing = complement_shuffled[:remaining_missing_needed]
    complement_observed = complement_shuffled[remaining_missing_needed:]
    
    # Combine
    observed_ratings = blocked_observed + complement_observed
    missing_ratings = blocked_missing + complement_missing
    
    return observed_ratings, missing_ratings


def pattern_item_block(
    all_ratings: List[Dict[str, Any]],
    missing_rate: float,
    block_fraction_item: float,
    block_missing_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Item-block missingness: hide most ratings touching a subset of items.
    
    Chooses a fraction of items and marks block_missing_fraction of ratings touching
    those items as missing, while keeping the complement relatively well observed.
    
    Args:
        all_ratings: Full universe of ratings
        missing_rate: Target overall missing fraction
        block_fraction_item: Fraction of items to block (0.0-1.0)
        block_missing_fraction: Fraction of blocked items' ratings to hide (0.0-1.0)
        seed: Random seed
    
    Returns:
        (observed_ratings, missing_ratings)
    """
    rng = random.Random(seed)
    
    # Get unique items
    items = sorted(set(r["item"] for r in all_ratings))
    n_block = max(1, int(len(items) * block_fraction_item))
    blocked_items = set(rng.sample(items, n_block))
    
    # Partition ratings by item
    blocked_ratings = [r for r in all_ratings if r["item"] in blocked_items]
    complement_ratings = [r for r in all_ratings if r["item"] not in blocked_items]
    
    # Hide block_missing_fraction of blocked items' ratings
    blocked_shuffled = blocked_ratings.copy()
    rng.shuffle(blocked_shuffled)
    n_block_missing = int(len(blocked_shuffled) * block_missing_fraction)
    blocked_missing = blocked_shuffled[:n_block_missing]
    blocked_observed = blocked_shuffled[n_block_missing:]
    
    # Compute target overall missing count
    target_missing = int(len(all_ratings) * missing_rate)
    remaining_missing_needed = max(0, target_missing - len(blocked_missing))
    
    # Sample from complement to reach target rate
    complement_shuffled = complement_ratings.copy()
    rng.shuffle(complement_shuffled)
    complement_missing = complement_shuffled[:remaining_missing_needed]
    complement_observed = complement_shuffled[remaining_missing_needed:]
    
    # Combine
    observed_ratings = blocked_observed + complement_observed
    missing_ratings = blocked_missing + complement_missing
    
    return observed_ratings, missing_ratings


def rewrite_bundle(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    pattern_params: Dict[str, Any],
    seed: int,
) -> None:
    """Main rewrite function."""
    # Load bundle
    bundle_path = input_dir / "data_bundle.json"
    configs_path = input_dir / "configs.json"
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    if not configs_path.exists():
        raise FileNotFoundError(f"Configs not found: {configs_path}")
    
    with open(bundle_path, 'r') as f:
        bundle_dict = json.load(f)
    
    bundle = GroundTruthBundle.from_dict(bundle_dict)
    
    # Apply pattern
    all_ratings = bundle.all_ratings.copy()
    
    if pattern == "mcar_iid":
        observed_ratings, missing_ratings = pattern_mcar_iid(
            all_ratings,
            pattern_params["missing_rate"],
            seed,
        )
    elif pattern == "balanced_degree_mcar":
        observed_ratings, missing_ratings = pattern_balanced_degree_mcar(
            all_ratings,
            pattern_params["missing_rate"],
            pattern_params["min_degree_annotator"],
            pattern_params["min_degree_item"],
            seed,
        )
    elif pattern == "annotator_block":
        observed_ratings, missing_ratings = pattern_annotator_block(
            all_ratings,
            pattern_params["missing_rate"],
            pattern_params["block_fraction_annotator"],
            pattern_params["block_missing_fraction"],
            seed,
        )
    elif pattern == "item_block":
        observed_ratings, missing_ratings = pattern_item_block(
            all_ratings,
            pattern_params["missing_rate"],
            pattern_params["block_fraction_item"],
            pattern_params["block_missing_fraction"],
            seed,
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Update stats
    stats = compute_stats(all_ratings, observed_ratings, missing_ratings, bundle)
    stats["rewrite_protocol"] = pattern
    stats["rewrite_params"] = pattern_params
    stats["rewrite_seed"] = seed
    
    # Update missing_ratings_indexes_in_test_instance
    missing_ratings_indexes_in_test_instance = [
        i for i, r in enumerate(missing_ratings) if r["instance"] == "test"
    ]
    
    # Create new bundle dict
    bundle_dict_new = {
        "embeddings": bundle.embeddings.tolist(),
        "mean_preferences": bundle.mean_preferences.tolist(),
        "annotator_preferences": bundle.annotator_preferences.tolist(),
        "rating_probs": bundle.rating_probs.tolist(),
        "rating_cumprobs": bundle.rating_cumprobs.tolist(),
        "rating_thresholds_z": bundle.rating_thresholds_z.tolist(),
        "base_scores": bundle.base_scores.tolist(),
        "all_ratings": all_ratings,
        "all_pairwise": bundle.all_pairwise,  # Preserve pairwise as-is
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "observed_pairwise": bundle.observed_pairwise,  # Preserve pairwise as-is
        "missing_pairwise": bundle.missing_pairwise,  # Preserve pairwise as-is
        "stats": stats,
    }
    
    # Add optional fields if present
    if bundle.log_lik_ratings_obs is not None:
        bundle_dict_new["log_lik_ratings_obs"] = bundle.log_lik_ratings_obs
    if bundle.log_lik_ratings_missing is not None:
        bundle_dict_new["log_lik_ratings_missing"] = bundle.log_lik_ratings_missing
    if bundle.log_lik_rankings_obs is not None:
        bundle_dict_new["log_lik_rankings_obs"] = bundle.log_lik_rankings_obs
    if bundle.log_lik_rankings_missing is not None:
        bundle_dict_new["log_lik_rankings_missing"] = bundle.log_lik_rankings_missing
    if bundle.train_posterior_rating_probs is not None:
        bundle_dict_new["train_posterior_rating_probs"] = bundle.train_posterior_rating_probs.tolist()
    if bundle.test_posterior_rating_probs is not None:
        bundle_dict_new["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs.tolist()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy configs.json
    import shutil
    shutil.copy(configs_path, output_dir / "configs.json")
    
    # Save rewritten bundle
    save_bundle(output_dir, bundle_dict_new)
    
    # Write README snippet
    readme_path = output_dir / "README_REWRITE.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Bundle Rewrite\n\n")
        f.write(f"**Pattern**: {pattern}\n")
        f.write(f"**Parameters**: {pattern_params}\n")
        f.write(f"**Seed**: {seed}\n\n")
        f.write(f"**Realized stats**:\n")
        f.write(f"- Observed: {stats['observed_ratings']} ({stats['observation_rate']:.3f})\n")
        f.write(f"- Missing: {stats['missing_ratings']} ({1-stats['observation_rate']:.3f})\n")
        f.write(f"- Train observed rate: {stats['train_observation_rate']:.3f}\n")
        f.write(f"- Test observed rate: {stats['test_observation_rate']:.3f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite data bundle with new missingness pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MCAR-IID: 50%% missing
  PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
      --input-dir OUTPUT/generated_data/base_run \\
      --output-dir OUTPUT/generated_data/base_run_mcar50 \\
      --pattern mcar_iid --missing-rate 0.5 --seed 42

  # Balanced-degree MCAR
  PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
      --input-dir OUTPUT/generated_data/base_run \\
      --output-dir OUTPUT/generated_data/base_run_balanced_mcar50 \\
      --pattern balanced_degree_mcar \\
      --missing-rate 0.5 \\
      --min-degree-annotator 3 --min-degree-item 3 \\
      --seed 42

  # Annotator block
  PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
      --input-dir OUTPUT/generated_data/base_run \\
      --output-dir OUTPUT/generated_data/base_run_annotator_block \\
      --pattern annotator_block \\
      --missing-rate 0.5 \\
      --block-fraction-annotator 0.3 \\
      --block-missing-fraction 0.8 \\
      --seed 42

  # Item block
  PYTHONPATH=. python stan/scripts/rewrite_missingness.py \\
      --input-dir OUTPUT/generated_data/base_run \\
      --output-dir OUTPUT/generated_data/base_run_item_block \\
      --pattern item_block \\
      --missing-rate 0.5 \\
      --block-fraction-item 0.2 \\
      --block-missing-fraction 0.7 \\
      --seed 42
        """
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory containing data_bundle.json and configs.json")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for rewritten bundle")
    parser.add_argument("--pattern", type=str, required=True,
                       choices=["mcar_iid", "balanced_degree_mcar", "annotator_block", "item_block"],
                       help="Missingness pattern")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Pattern-specific arguments
    parser.add_argument("--missing-rate", type=float, required=True,
                       help="Target missing rate (0.0-1.0)")
    parser.add_argument("--min-degree-annotator", type=int, default=1,
                       help="[balanced_degree_mcar] Minimum train-observed degree per annotator")
    parser.add_argument("--min-degree-item", type=int, default=1,
                       help="[balanced_degree_mcar] Minimum train-observed degree per item")
    parser.add_argument("--block-fraction-annotator", type=float, default=0.3,
                       help="[annotator_block] Fraction of annotators to block")
    parser.add_argument("--block-missing-fraction", type=float, default=0.8,
                       help="[annotator_block/item_block] Fraction of blocked ratings to hide")
    parser.add_argument("--block-fraction-item", type=float, default=0.2,
                       help="[item_block] Fraction of items to block")
    
    args = parser.parse_args()
    
    # Validate missing rate
    if not 0.0 <= args.missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0.0, 1.0], got {args.missing_rate}")
    
    # Build pattern params
    pattern_params = {"missing_rate": args.missing_rate}
    
    if args.pattern == "balanced_degree_mcar":
        pattern_params["min_degree_annotator"] = args.min_degree_annotator
        pattern_params["min_degree_item"] = args.min_degree_item
    elif args.pattern == "annotator_block":
        pattern_params["block_fraction_annotator"] = args.block_fraction_annotator
        pattern_params["block_missing_fraction"] = args.block_missing_fraction
    elif args.pattern == "item_block":
        pattern_params["block_fraction_item"] = args.block_fraction_item
        pattern_params["block_missing_fraction"] = args.block_missing_fraction
    
    # Run rewrite
    rewrite_bundle(
        Path(args.input_dir),
        Path(args.output_dir),
        args.pattern,
        pattern_params,
        args.seed,
    )
    
    print(f"✓ Rewritten bundle saved to {args.output_dir}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Missing rate: {args.missing_rate}")


if __name__ == "__main__":
    main()
