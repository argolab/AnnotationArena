import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_results(results_path):
    """Load all experiment results from JSON files."""
    experiment_results = {}
    
    # List of possible experiment files - UPDATED to include entropy experiments
    experiment_files = [
        "gradient_all.json", 
        "random_all.json", 
        "random_5.json", 
        "gradient_sequential.json", 
        "gradient_voi.json", 
        "gradient_fast_voi.json",
        "entropy_all.json",  # Added new entropy experiment
        "entropy_5.json"     # Added new entropy experiment
    ]
    
    for filename in experiment_files:
        filepath = os.path.join(results_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    experiment_name = filename.split('.')[0]
                    experiment_results[experiment_name] = json.load(f)
                    print(f"Loaded {experiment_name} results")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return experiment_results

def plot_observe_all_experiments(results_dict, save_path):
    """Plot 'observe all' experiments (gradient_all, random_all, entropy_all)."""
    # Filter only the relevant experiments - UPDATED to include entropy_all
    observe_all_results = {k: v for k, v in results_dict.items() if k in ['gradient_all', 'random_all', 'entropy_all']}
    
    if not observe_all_results:
        print("No 'observe all' experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # UPDATED to include entropy_all
    colors = {
        'gradient_all': 'red',
        'random_all': 'green',
        'entropy_all': 'blue'  # New color for entropy
    }
    
    # UPDATED to include entropy_all
    markers = {
        'gradient_all': 'o',
        'random_all': '^',
        'entropy_all': 's'  # New marker for entropy
    }
    
    for strategy, results in observe_all_results.items():
        # Plot expected loss on test set (solid line)
        if 'test_expected_losses' in results:
            expected_losses = results['test_expected_losses']
            cycles = list(range(len(expected_losses)))
            
            ax.plot(cycles, expected_losses, 
                    linestyle='-', 
                    color=colors[strategy],
                    marker=markers[strategy], 
                    label=f"{strategy} (Expected)",
                    linewidth=2,
                    markersize=8)
            
            # Plot annotated loss on test set (dotted line)
            if 'test_annotated_losses' in results:
                annotated_losses = results['test_annotated_losses']
                
                # Find where the lines diverge
                for c in range(1, len(cycles)):
                    if abs(expected_losses[c] - annotated_losses[c]) > 0.01:
                        # Draw a vertical line to show the drop
                        ax.plot([cycles[c], cycles[c]], 
                                [expected_losses[c], annotated_losses[c]],
                                linestyle='-', 
                                color=colors[strategy],
                                linewidth=1,
                                alpha=0.6)
                
                ax.plot(cycles, annotated_losses, 
                        linestyle=':', 
                        color=colors[strategy],
                        marker=markers[strategy], 
                        label=f"{strategy} (Annotated)",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    ax.set_title('Loss on Test Set: Observe All Experiments', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_expected_losses', [0])) for r in observe_all_results.values()])
    ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 'observe all' plot to {save_path}")

def plot_observe_5_experiments(results_dict, save_path):
    """Plot 'observe 5' experiments (random_5, gradient_sequential, gradient_voi, entropy_5)."""
    # Filter only the relevant experiments - UPDATED to include entropy_5
    observe_5_results = {k: v for k, v in results_dict.items() 
                         if k in ['random_5', 'gradient_sequential', 'gradient_voi', 'entropy_5']}
    
    if not observe_5_results:
        print("No 'observe 5' experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # UPDATED to include entropy_5
    colors = {
        'random_5': 'purple',
        'gradient_sequential': 'orange',
        'gradient_voi': 'brown',
        'entropy_5': 'darkgreen'  # New color for entropy
    }
    
    # UPDATED to include entropy_5
    markers = {
        'random_5': 'D',
        'gradient_sequential': 's',
        'gradient_voi': '*',
        'entropy_5': 'X'  # New marker for entropy
    }
    
    for strategy, results in observe_5_results.items():
        # Plot expected loss on test set (solid line)
        if 'test_expected_losses' in results:
            expected_losses = results['test_expected_losses']
            cycles = list(range(len(expected_losses)))
            
            ax.plot(cycles, expected_losses, 
                    linestyle='-', 
                    color=colors[strategy],
                    marker=markers[strategy], 
                    label=f"{strategy} (Expected)",
                    linewidth=2,
                    markersize=8)
            
            # Plot annotated loss on test set (dotted line)
            if 'test_annotated_losses' in results:
                annotated_losses = results['test_annotated_losses']
                
                # Find where the lines diverge
                for c in range(1, len(cycles)):
                    if abs(expected_losses[c] - annotated_losses[c]) > 0.01:
                        # Draw a vertical line to show the drop
                        ax.plot([cycles[c], cycles[c]], 
                                [expected_losses[c], annotated_losses[c]],
                                linestyle='-', 
                                color=colors[strategy],
                                linewidth=1,
                                alpha=0.6)
                
                ax.plot(cycles, annotated_losses, 
                        linestyle=':', 
                        color=colors[strategy],
                        marker=markers[strategy], 
                        label=f"{strategy} (Annotated)",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    ax.set_title('Loss on Test Set: Observe 5 Experiments', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_expected_losses', [0])) for r in observe_5_results.values()])
    ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 'observe 5' plot to {save_path}")

def plot_voi_comparison(results_dict, save_path):
    """Plot comparison of gradient_voi and gradient_fast_voi."""
    # Filter only the relevant experiments
    voi_results = {k: v for k, v in results_dict.items() 
                   if k in ['gradient_voi', 'gradient_fast_voi']}
    
    if not voi_results:
        print("No VOI comparison experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        'gradient_voi': 'brown',
        'gradient_fast_voi': 'blue'
    }
    
    markers = {
        'gradient_voi': '*',
        'gradient_fast_voi': 'o'
    }
    
    for strategy, results in voi_results.items():
        # Plot expected loss on test set (solid line)
        if 'test_expected_losses' in results:
            expected_losses = results['test_expected_losses']
            cycles = list(range(len(expected_losses)))
            
            ax.plot(cycles, expected_losses, 
                    linestyle='-', 
                    color=colors[strategy],
                    marker=markers[strategy], 
                    label=f"{strategy} (Expected)",
                    linewidth=2,
                    markersize=8)
            
            # Plot annotated loss on test set (dotted line)
            if 'test_annotated_losses' in results:
                annotated_losses = results['test_annotated_losses']
                
                # Find where the lines diverge
                for c in range(1, len(cycles)):
                    if abs(expected_losses[c] - annotated_losses[c]) > 0.01:
                        # Draw a vertical line to show the drop
                        ax.plot([cycles[c], cycles[c]], 
                                [expected_losses[c], annotated_losses[c]],
                                linestyle='-', 
                                color=colors[strategy],
                                linewidth=1,
                                alpha=0.6)
                
                ax.plot(cycles, annotated_losses, 
                        linestyle=':', 
                        color=colors[strategy],
                        marker=markers[strategy], 
                        label=f"{strategy} (Annotated)",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    ax.set_title('Loss on Test Set: VOI Comparison', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_expected_losses', [0])) for r in voi_results.values()])
    ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved VOI comparison plot to {save_path}")

def plot_feature_counts(results_dict, save_path):
    """Plot the number of features annotated per cycle for each method."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # UPDATED to include entropy methods
    colors = {
        'gradient_all': 'red',
        'random_all': 'green',
        'random_5': 'purple',
        'gradient_sequential': 'orange',
        'gradient_voi': 'brown',
        'gradient_fast_voi': 'blue',
        'entropy_all': 'darkblue',  # New color for entropy_all
        'entropy_5': 'darkgreen'    # New color for entropy_5
    }
    
    # UPDATED to include entropy methods
    markers = {
        'gradient_all': 'o',
        'random_all': '^',
        'random_5': 'D',
        'gradient_sequential': 's',
        'gradient_voi': '*',
        'gradient_fast_voi': 'X',
        'entropy_all': 's',  # New marker for entropy_all
        'entropy_5': 'P'     # New marker for entropy_5
    }
    
    for strategy, results in results_dict.items():
        if 'features_annotated' in results:
            features = results['features_annotated']
            cycles = list(range(len(features)))
            
            ax.plot(cycles, features, 
                    linestyle='-', 
                    color=colors.get(strategy, 'black'),
                    marker=markers.get(strategy, 'o'), 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax.set_title('Features Annotated Per Cycle', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Number of Features', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature counts plot to {save_path}")

def create_plots():
    base_path = "outputs"
    results_path = os.path.join(base_path, "results")
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    experiment_results = load_experiment_results(results_path)
    
    if not experiment_results:
        print("No experiment results found.")
        return
    
    plot_observe_all_experiments(experiment_results, os.path.join(plots_path, "observe_all_experiments.png"))
    plot_observe_5_experiments(experiment_results, os.path.join(plots_path, "observe_5_experiments.png"))
    plot_voi_comparison(experiment_results, os.path.join(plots_path, "voi_comparison.png"))
    plot_feature_counts(experiment_results, os.path.join(plots_path, "feature_counts.png"))
    
    print(f"All plots saved to {plots_path}")