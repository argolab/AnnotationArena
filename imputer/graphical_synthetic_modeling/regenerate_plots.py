#!/usr/bin/env python3
"""
Script to regenerate plots from saved experiment results.

Usage: python regenerate_plots.py
"""

import pickle
import json
from pathlib import Path
from utils.paper_plots import create_paper_plots

def main():
    # Load the saved results
    results_dir = Path("outputs/results")

    # Find the latest results pickle file
    pickle_files = list(results_dir.glob("results_*.pkl"))
    if not pickle_files:
        print("No results pickle files found in outputs/results/")
        return

    latest_pickle = max(pickle_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_pickle}")

    with open(latest_pickle, 'rb') as f:
        results = pickle.load(f)

    # Load config
    config_path = results_dir / "experiment_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded {len(results)} experiment configurations")
    print(f"Config: {config['node_sizes']} nodes, {config['missing_rates']} missing rates")

    # Create output directory for plots
    plots_dir = Path("outputs/plots/regenerated")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Process each missing rate
    for missing_rate in config['missing_rates']:
        print(f"\nRegenerating plots for missing rate: {missing_rate}")

        # Filter results for this missing rate
        filtered_results = {}
        for key, value in results.items():
            if len(key) == 3 and key[2] == missing_rate:
                # Convert back to original key format (n_nodes, policy_imputer)
                original_key = (key[0], key[1])
                filtered_results[original_key] = value

        if filtered_results:
            print(f"Found {len(filtered_results)} configurations for missing rate {missing_rate}")

            # Create the paper plots
            create_paper_plots(
                results=filtered_results,
                output_dir=str(plots_dir / f"missing_rate_{missing_rate}"),
                missing_rate=missing_rate
            )
            print(f"Plots saved to: {plots_dir / f'missing_rate_{missing_rate}'}")
        else:
            print(f"No results found for missing rate: {missing_rate}")

    print(f"\nAll plots regenerated in: {plots_dir}")

if __name__ == "__main__":
    main()