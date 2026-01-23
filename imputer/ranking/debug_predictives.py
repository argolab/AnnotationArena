#!/usr/bin/env python3
"""
Debug script for testing predictives evaluation.
Run this with the VS Code debugger to step through the code.
"""

import json
import sys
from pathlib import Path
import cmdstanpy
import numpy as np

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from stan.pipeline.predictives import evaluate_predictives, extract_predictives_from_fit
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir

def main():
    # Configuration - update these paths as needed
    mcmc_dir = Path("OUTPUT/domain_model/runs/updated_ofat_small_center_D8_sa05_sm01_kp100_uc0_smar_stan4c")
    data_bundle_path = Path("OUTPUT/generated_data/updated_ofat_small_center_D8_sa05_sm01_kp100_uc0_smar/data_bundle.json")
    
    print(f"Loading data bundle from {data_bundle_path}")
    with open(data_bundle_path, 'r') as f:
        bundle_data = json.load(f)
    
    bundle = GroundTruthBundle.from_dict(bundle_data)
    print(f"Loaded bundle:")
    print(f"  - Missing ratings: {len(bundle.missing_ratings)}")
    print(f"  - Missing ratings (test): {len(bundle.missing_ratings_indexes_in_test_instance)}")
    print(f"  - Observed ratings: {len(bundle.observed_ratings)}")
    
    # Show first few missing ratings
    print(f"\nFirst 5 missing ratings:")
    for i, r in enumerate(bundle.missing_ratings[:5]):
        print(f"  [{i}] instance={r['instance']}, attribute={r['attribute']}, annotator={r['annotator']}, item={r['item']}, value={r['value']}")
    
    # Show test instance indices
    print(f"\nTest instance missing rating indices: {bundle.missing_ratings_indexes_in_test_instance[:10]}...")
    print(f"Test instance missing ratings:")
    for idx in bundle.missing_ratings_indexes_in_test_instance[:5]:
        r = bundle.missing_ratings[idx]
        print(f"  [{idx}] instance={r['instance']}, value={r['value']}")
    
    # Load MCMC fit
    print(f"\nLoading MCMC results from {mcmc_dir}")
    csv_files = list(mcmc_dir.glob("domain_model-*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    fit = cmdstanpy.from_csv([str(f) for f in csv_files])
    print(f"MCMC fit loaded: {len(fit.stan_variables()['log_lik_ratings_obs'])} samples")
    
    # Extract predictives
    print("\nExtracting predictives...")
    predictives = extract_predictives_from_fit(fit)
    print(f"Predictives shape:")
    print(f"  - missing_rating_predictions: {predictives['missing_rating_predictions'].shape}")
    print(f"  - missing_rating_probs: {predictives['missing_rating_probs'].shape}")
    
    # Show first few predictions
    print(f"\nFirst 5 predictions (first sample):")
    for i in range(min(5, predictives['missing_rating_predictions'].shape[1])):
        pred = predictives['missing_rating_predictions'][0, i]
        probs = predictives['missing_rating_probs'][0, i]
        if i < len(bundle.missing_ratings):
            gt = bundle.missing_ratings[i]['value']
            print(f"  [{i}] pred={pred}, gt={gt}, probs={probs}")
    
    # Get config
    configs_path = data_bundle_path.parent / "configs.json"
    with open(configs_path, 'r') as f:
        configs_data = json.load(f)
    datagen_config = configs_data["datagen"]
    config = {
        "K_train": datagen_config["K_train"],
        "K_test": datagen_config["K_test"],
        "I": datagen_config["I"],
        "J": datagen_config["J"],
        "D": datagen_config["D"],
        "C": datagen_config["C"],
        "temperature": datagen_config.get("temperature", 1.0),
    }
    
    print(f"\nConfiguration: {config}")
    
    # Evaluate predictives - SET BREAKPOINT HERE
    print("\nEvaluating predictives...")
    print("=" * 60)
    print("SET BREAKPOINT ON NEXT LINE TO DEBUG")
    print("=" * 60)
    results = evaluate_predictives(fit, bundle, config)
    
    # Print results
    print(f"\nResults:")
    print(f"  - rating_missing_accuracy: {results.metrics['rating_missing_accuracy']:.4f}")
    print(f"  - rating_missing_log_likelihood: {results.metrics['rating_missing_log_likelihood']:.4f}")
    print(f"  - rating_missing_mae: {results.metrics['rating_missing_mae']:.4f}")
    print(f"  - n_missing_ratings: {results.metrics['n_missing_ratings']}")
    
    # Detailed analysis
    print(f"\nDetailed analysis:")
    missing_indices = bundle.missing_ratings_indexes_in_test_instance
    test_predictions = predictives['missing_rating_predictions'][:, missing_indices]
    test_probs = predictives['missing_rating_probs'][:, missing_indices]
    test_gt = [bundle.missing_ratings[i]['value'] for i in missing_indices]
    
    print(f"  - Test missing count: {len(missing_indices)}")
    print(f"  - Test predictions shape: {test_predictions.shape}")
    
    # Compute accuracy manually
    posterior_mode = np.zeros(len(missing_indices), dtype=int)
    for i in range(len(missing_indices)):
        unique, counts = np.unique(test_predictions[:, i], return_counts=True)
        posterior_mode[i] = unique[np.argmax(counts)]
    
    accuracy_manual = np.mean(posterior_mode == np.array(test_gt))
    print(f"  - Manual accuracy: {accuracy_manual:.4f}")
    
    # Show first 10 predictions vs ground truth
    print(f"\nFirst 10 predictions vs ground truth:")
    for i in range(min(10, len(missing_indices))):
        idx = missing_indices[i]
        pred_mode = posterior_mode[i]
        gt = test_gt[i]
        match = "✓" if pred_mode == gt else "✗"
        print(f"  [{i}] pred={pred_mode}, gt={gt} {match}")

if __name__ == "__main__":
    main()






