#!/usr/bin/env python3
"""
Verification script to check evaluation set consistency between Stan and Imputer.

This script analyzes a data bundle to verify:
1. Missing ratings distribution (train vs test instance)
2. Cross-instance annotations
3. Whether Stan and Imputer would evaluate the same set
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_bundle(bundle_path: str):
    """Analyze a data bundle for evaluation set consistency."""
    
    print("=" * 80)
    print("EVALUATION SET CONSISTENCY VERIFICATION")
    print("=" * 80)
    print()
    
    with open(bundle_path, 'r') as f:
        bundle = json.load(f)
    
    # Extract dimensions
    K_train = bundle["stats"]["K_train"]
    K_test = bundle["stats"]["K_test"]
    I = bundle["stats"].get("I", None)  # May not be in stats
    J = bundle["stats"].get("J", None)
    
    # Try to get from configs.json if available
    bundle_dir = Path(bundle_path).parent
    configs_path = bundle_dir / "configs.json"
    if configs_path.exists():
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        datagen = configs.get("datagen", {})
        I = datagen.get("I", I)
        J = datagen.get("J", J)
    
    print(f"Dimensions:")
    print(f"  K_train: {K_train}")
    print(f"  K_test: {K_test}")
    print(f"  I (attributes): {I}")
    print(f"  J (annotators): {J}")
    print()
    
    # Analyze missing ratings
    missing_ratings = bundle["missing_ratings"]
    print(f"Total missing ratings: {len(missing_ratings)}")
    
    # Partition by instance
    train_missing = [r for r in missing_ratings if r["instance"] == "train"]
    test_missing = [r for r in missing_ratings if r["instance"] == "test"]
    
    print(f"  Train instance missing: {len(train_missing)}")
    print(f"  Test instance missing: {len(test_missing)}")
    print()
    
    # Analyze test missing ratings for cross-instance annotations
    print("Test Instance Missing Ratings Analysis:")
    print("-" * 80)
    
    if J is not None:
        # Assuming tie_breaking protocol: train annotators = 1..2J/3, test annotators = J/3+1..J
        train_annotator_end = (2 * J) // 3
        test_annotator_start = (J // 3) + 1
        
        print(f"  Annotator partition (assuming tie_breaking protocol):")
        print(f"    Train annotators: 1..{train_annotator_end}")
        print(f"    Test annotators: {test_annotator_start}..{J}")
        print()
        
        # Count cross-instance annotations in test missing
        train_annotator_test_item = []
        test_annotator_test_item = []
        train_annotator_train_item = []  # Shouldn't exist in test_missing
        test_annotator_train_item = []   # Shouldn't exist in test_missing
        
        for r in test_missing:
            item = r["item"]
            annotator = r["annotator"]
            is_test_item = item > K_train
            
            if annotator <= train_annotator_end:
                if is_test_item:
                    train_annotator_test_item.append(r)
                else:
                    train_annotator_train_item.append(r)
            else:
                if is_test_item:
                    test_annotator_test_item.append(r)
                else:
                    test_annotator_train_item.append(r)
        
        print(f"  Test missing ratings breakdown:")
        print(f"    Train annotator + Test item: {len(train_annotator_test_item)} ⚠️ CROSS-INSTANCE")
        print(f"    Test annotator + Test item: {len(test_annotator_test_item)} ✓ SAME-INSTANCE")
        print(f"    Train annotator + Train item: {len(train_annotator_train_item)} ❌ INCONSISTENT (shouldn't be in test_missing)")
        print(f"    Test annotator + Train item: {len(test_annotator_train_item)} ❌ INCONSISTENT (shouldn't be in test_missing)")
        print()
        
        if train_annotator_train_item or test_annotator_train_item:
            print("  ⚠️  WARNING: Found test instance missing ratings with train items!")
            print("     This suggests a data generation bug.")
            print()
    else:
        # Just analyze by item ranges
        test_missing_test_items = [r for r in test_missing if r["item"] > K_train]
        test_missing_train_items = [r for r in test_missing if r["item"] <= K_train]
        
        print(f"  Test missing with test items (item > {K_train}): {len(test_missing_test_items)}")
        print(f"  Test missing with train items (item <= {K_train}): {len(test_missing_train_items)} ❌ INCONSISTENT")
        print()
    
    # Check missing_ratings_indexes_in_test_instance
    if "missing_ratings_indexes_in_test_instance" in bundle:
        test_indices = bundle["missing_ratings_indexes_in_test_instance"]
        print(f"Stan's test instance indices: {len(test_indices)}")
        
        # Verify they match
        manual_test_indices = [i for i, r in enumerate(missing_ratings) if r["instance"] == "test"]
        if set(test_indices) == set(manual_test_indices):
            print("  ✓ Matches manual count of test instance missing ratings")
        else:
            print("  ❌ MISMATCH with manual count!")
            print(f"     Bundle indices: {len(test_indices)}")
            print(f"     Manual count: {len(manual_test_indices)}")
        print()
    else:
        print("  ⚠️  WARNING: missing_ratings_indexes_in_test_instance not found in bundle")
        print()
    
    # Analyze observed ratings for comparison
    observed_ratings = bundle["observed_ratings"]
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    test_observed = [r for r in observed_ratings if r["instance"] == "test"]
    
    print("Observed Ratings (for comparison):")
    print(f"  Train instance observed: {len(train_observed)}")
    print(f"  Test instance observed: {len(test_observed)}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Evaluation Set Consistency:")
    print(f"  ✓ Imputer will evaluate on: {len(test_missing)} test instance missing ratings")
    if "missing_ratings_indexes_in_test_instance" in bundle:
        print(f"  ✓ Stan will evaluate on: {len(bundle['missing_ratings_indexes_in_test_instance'])} test instance missing ratings")
        if len(test_missing) == len(bundle['missing_ratings_indexes_in_test_instance']):
            print("  ✅ SETS MATCH - Stan and Imputer evaluate the same test missing ratings")
        else:
            print("  ❌ SETS DON'T MATCH - Potential bug!")
    print()
    
    if J is not None and (train_annotator_test_item or test_annotator_test_item):
        print("Cross-Instance Annotations:")
        print(f"  Test missing includes {len(train_annotator_test_item)} cross-instance ratings")
        print("  (Train annotator rating test items - this is CORRECT for tie_breaking protocol)")
        print()
    
    print("Next Steps:")
    print("  1. Verify that Stan's n_missing_ratings metric matches len(test_missing)")
    print("  2. Verify that Imputer's missing_metrics['num_rating_evaluations'] matches len(test_missing)")
    print("  3. Check if cross-instance annotations are intentional for your observation protocol")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_evaluation_sets.py <path_to_data_bundle.json>")
        print()
        print("Example:")
        print("  python verify_evaluation_sets.py OUTPUT/generated_data/debug_repeat_test_annotator/data_bundle.json")
        sys.exit(1)
    
    bundle_path = sys.argv[1]
    if not Path(bundle_path).exists():
        print(f"Error: File not found: {bundle_path}")
        sys.exit(1)
    
    analyze_bundle(bundle_path)

