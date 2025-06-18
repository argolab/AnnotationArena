import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_results(results_path):
    """Load all experiment results from JSON files."""
    experiment_results = {}
    
    # Load all JSON files in the directory
    for filename in os.listdir(results_path):
        if filename.endswith('.json'):
            filepath = os.path.join(results_path, filename)
            try:
                with open(filepath, 'r') as f:
                    experiment_name = filename.split('.')[0]
                    experiment_results[experiment_name] = json.load(f)
                    print(f"Loaded {experiment_name} results")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return experiment_results

def plot_cold_start_embedding_comparison(results_dict, save_path):
    """Plot comparison between combine_cold_start_with_embedding and gradient_voi_cold_start_with_embedding_cross."""
    # Filter only the relevant experiments
    embedding_results = {k: v for k, v in results_dict.items() 
                        if k in ['gradient_random_cold_start_with_embedding', 'gradient_voi_cold_start_with_embedding', "random_5_cold_start_with_embedding"]}
    
    if not embedding_results:
        print("No cold start embedding experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        'gradient_random_cold_start_with_embedding': 'orange',
        'gradient_voi_cold_start_with_embedding': 'brown',
        'random_5_cold_start_with_embedding': "cyan"
    }
    
    markers = {
        'gradient_random_cold_start_with_embedding': 's',
        'gradient_voi_cold_start_with_embedding': '*',
        'random_5_cold_start_with_embedding': 's'
    }
    
    for strategy, results in embedding_results.items():
        # Plot annotated loss on test set only
        if 'test_annotated_losses' in results:
            annotated_losses = results['test_annotated_losses']
            cycles = list(range(len(annotated_losses)))
            
            ax.plot(cycles, annotated_losses, 
                    linestyle='-', 
                    color=colors[strategy],
                    marker=markers[strategy], 
                    label=f"{strategy}",
                    linewidth=2,
                    markersize=8)
    
    ax.set_title('Loss on Test Set: Cold Start Embedding Comparison', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Annotated Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_annotated_losses', [0])) for r in embedding_results.values()])
    ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cold start embedding comparison plot to {save_path}")

def extract_costs_per_cycle(experiment_results):
    """
    Extract costs per cycle for each experiment from observation history.
    
    Args:
        experiment_results: Dictionary of experiment results with observation_history
        
    Returns:
        Dictionary of experiment -> cycle -> cost mappings
        Dictionary of experiment -> cumulative cost per cycle
    """
    cycle_costs = {}
    cumulative_costs = {}
    
    for experiment_name, results in experiment_results.items():
        if "voi" not in experiment_name:
            continue
        if 'observation_history' not in results:
            print(f"No observation history found for {experiment_name}")
            continue
        
        observations = results['observation_history']
        
        # Find the total number of cycles
        if observations:
            num_cycles = 10  # Ceiling division by 10
        else:
            num_cycles = 0
        
        # Initialize costs for each cycle
        cycle_costs[experiment_name] = [0.0] * num_cycles
        
        # Calculate cost for each cycle
        for obs in observations:
            timestamp = obs.get('timestamp', 0)
            cycle = int(timestamp // ((len(observations) + 1) / 10))
            
            variable_id = obs.get('variable_id', '')
            cost = obs.get('cost', 0.0)
            
            # Apply special cost for position_7
            if '_position_7' in variable_id:
                cost = 1.5
            
            cycle_costs[experiment_name][cycle] += cost
        
        # Calculate cumulative costs
        cum_costs = [0.0] * num_cycles
        for i in range(num_cycles):
            if i == 0:
                cum_costs[i] = cycle_costs[experiment_name][i]
            else:
                cum_costs[i] = cum_costs[i-1] + cycle_costs[experiment_name][i]
        
        cumulative_costs[experiment_name] = cum_costs
    
    return cycle_costs, cumulative_costs

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

def plot_observe_5_experiments(results_dict, save_path, feature_num):
    """Plot 'observe 5' experiments (random_5, gradient_sequential, gradient_voi, entropy_5)."""
    # Filter only the relevant experiments - UPDATED to include entropy_5
    observe_5_results = {k: v for k, v in results_dict.items() 
                         if k in ['random_5', 'gradient_sequential', 'gradient_voi', 'entropy_5', "gradient_random"]}
    
    if not observe_5_results:
        print("No 'observe 5' experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # UPDATED to include entropy_5
    colors = {
        'random_5': 'purple',
        'gradient_sequential': 'orange',
        'gradient_voi': 'brown',
        'entropy_5': 'darkgreen',
        'gradient_random': "cyan"  # New color for entropy
    }
    
    # UPDATED to include entropy_5
    markers = {
        'random_5': 'D',
        'gradient_sequential': 's',
        'gradient_voi': '*',
        'entropy_5': 'X',  # New marker for entropy
        "gradient_random": "v"
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
                    label=f"{strategy.replace('5', str(feature_num))} (Expected)",
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
                        label=f"{strategy.replace('5', str(feature_num))} (Annotated)",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    ax.set_title('Loss on Test Set: Observe 5 Experiments', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_annotated_losses', [0])) for r in observe_5_results.values()])
    ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 'observe 5' plot to {save_path}")

def plot_cold_start_experiments(results_dict, save_path, feature_num):
    """Plot 'observe 5' experiments (random_5, gradient_sequential, gradient_voi, entropy_5)."""
    # Filter only the relevant experiments - UPDATED to include entropy_5
    observe_5_results = {k: v for k, v in results_dict.items() 
                         if k in ['random_5_cold_start', 'gradient_voi_cold_start', 'entropy_5_cold_start', "gradient_random_cold_start"]}
    
    if not observe_5_results:
        print("No 'observe 5' experiment results found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # UPDATED to include entropy_5
    colors = {
        'random_5_cold_start': 'purple',
        'gradient_voi_cold_start': 'brown',
        'entropy_5_cold_start': 'darkgreen',
        'gradient_random_cold_start': "cyan"  # New color for entropy
    }
    
    # UPDATED to include entropy_5
    markers = {
        'random_5_cold_start': 'D',
        'gradient_voi_cold_start': '*',
        'entropy_5_cold_start': 'X',  # New marker for entropy
        "gradient_random_cold_start": "v"
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
                    label=f"{strategy.replace('5', str(feature_num))} (Expected)",
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
                        label=f"{strategy.replace('5', str(feature_num))} (Annotated)",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    ax.set_title('Loss on Test Set: Observe 5 Experiments', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    max_loss = max([max(r.get('test_annotated_losses', [0])) for r in observe_5_results.values()])
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

def plot_top_only_comparison(results_dict, save_path):
    """Plot comparison between regular and top_only experiments."""
    # Find pairs of experiments with regular and top_only versions
    regular_exp = [k for k in results_dict.keys() if not k.endswith('_top_only')]
    top_only_exp = [k for k in results_dict.keys() if k.endswith('_top_only')]
    
    # Match pairs of experiments
    pairs = []
    for reg in regular_exp:
        top_only = f"{reg}_top_only"
        if top_only in top_only_exp:
            pairs.append((reg, top_only))
    
    if not pairs:
        print("No matching regular and top_only experiment pairs found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define consistent colors for experiment types
    base_colors = {
        'gradient_all': 'red',
        'gradient_sequential': 'orange',
        'gradient_voi': 'brown',
        'gradient_fast_voi': 'blue',
        'gradient_random': 'cyan'  # <- New entry added
    }
    
    # Create plot with solid lines for regular and dashed lines for top_only
    for reg, top in pairs:
        base_name = reg.split('_top_only')[0]
        color = base_colors.get(reg, 'black')
        
        # Plot regular version (solid line)
        if 'test_annotated_losses' in results_dict[reg]:
            expected_losses = results_dict[reg]['test_annotated_losses']
            cycles = list(range(len(expected_losses)))
            
            ax.plot(cycles, expected_losses, 
                    linestyle='-', 
                    color=color,
                    marker='o', 
                    label=f"{reg} (Regular)",
                    linewidth=2,
                    markersize=8)
        
        # Plot top_only version (dashed line)
        if 'test_annotated_losses' in results_dict[top]:
            expected_losses = results_dict[top]['test_annotated_losses']
            cycles = list(range(len(expected_losses)))
            
            ax.plot(cycles, expected_losses, 
                    linestyle='--', 
                    color=color,
                    marker='s', 
                    label=f"{top} (Top Only)",
                    linewidth=2,
                    markersize=8)
    
    ax.set_title('Regular vs Top Only Experiments Comparison', fontsize=16)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('Annotated Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0 and have some headroom
    all_losses = []
    for pair in pairs:
        for exp in pair:
            if exp in results_dict and 'test_expected_losses' in results_dict[exp]:
                all_losses.extend(results_dict[exp]['test_expected_losses'])
    
    if all_losses:
        max_loss = max(all_losses)
        ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Regular vs Top Only comparison plot to {save_path}")

def plot_loss_vs_cost(results_dict, costs_dict, save_path, feature_num):
    """Plot loss vs cumulative cost for all experiments."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors and markers for each experiment type
    colors = {
        'gradient_all': 'red',
        'random_all': 'green',
        'random_5': 'purple',
        'gradient_voi_cold_start_with_embedding': 'orange',
        'gradient_voi_cold_start_with_embedding_cross': 'brown',
        'gradient_voi_cold_start': 'blue',
        'entropy_all': 'pink',
        'entropy_5': 'darkgreen',
        'gradient_random': 'cyan'
    }
    
    markers = {
        'gradient_all': 'o',
        'random_all': '^',
        'random_5': 'D',
        'gradient_voi_cold_start_with_embedding': 's',
        'gradient_voi_cold_start_with_embedding_cross': '*',
        'gradient_voi_cold_start': 'X',
        'entropy_all': 's',
        'entropy_5': 'P',
        'gradient_random': 'v'
    }
    
    # Define line styles
    linestyles = {
        exp: '--' if '_top_only' in exp else '-' for exp in results_dict.keys()
    }
    
    for experiment_name, results in results_dict.items():
        if 'test_annotated_losses' not in results or experiment_name not in costs_dict:
            continue
        
        expected_losses = results['test_annotated_losses']
        cumulative_costs = costs_dict[experiment_name]
        
        # Make sure we have the same number of points
        num_points = min(len(expected_losses), len(cumulative_costs))
        
        ax.plot(cumulative_costs[:num_points], expected_losses[:num_points], 
                linestyle=linestyles.get(experiment_name, '-'), 
                color=colors.get(experiment_name, 'black'),
                marker=markers.get(experiment_name, 'o'), 
                label=experiment_name.replace("5", str(feature_num)),
                linewidth=2,
                markersize=8)
    
    ax.set_title('Hanna Loss vs Cumulative Cost', fontsize=16)
    ax.set_xlabel('Cumulative Cost', fontsize=14)
    ax.set_ylabel('Annotated Loss', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Loss vs Cost plot to {save_path}")

def plot_observation_patterns(experiment_results, save_path):
    """
    Plot comparison of reobservation patterns across cycles between gradient_voi_cold_start_with_embedding 
    and gradient_voi_cold_start experiments. Shows how frequently each experiment reobserves 
    previously seen examples in each cycle.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import numpy as np
    
    # Define the experiments to compare
    target_experiments = ['gradient_voi_cold_start_with_embedding', 'gradient_voi_cold_start']
    
    # Filter experiments
    filtered_results = {k: v for k, v in experiment_results.items() 
                       if k in target_experiments and 'observation_history' in v}
    
    if len(filtered_results) < 2:
        print(f"Need both experiments for comparison. Found: {list(filtered_results.keys())}")
        return
    
    # Dictionary to store reobservation patterns for each experiment
    experiment_patterns = {}
    
    for experiment_name, results in filtered_results.items():
        observations = results['observation_history']
        
        if not observations:
            continue
            
        # Divide observations into 10 cycles based on timestamp
        max_timestamp = max(obs.get('timestamp', 0) for obs in observations)
        cycle_size = max_timestamp / 10.0
        
        # Track examples seen in each cycle and reobservation rates
        cycle_examples = defaultdict(set)  # cycle -> set of examples seen
        cycle_reobs_counts = defaultdict(int)  # cycle -> count of reobservations
        cycle_total_counts = defaultdict(int)  # cycle -> total observations
        previously_seen = set()  # all examples seen in previous cycles
        
        for obs in observations:
            timestamp = obs.get('timestamp', 0)
            variable_id = obs.get('variable_id', '')
            
            # Determine cycle (0-9)
            cycle = min(int(timestamp / cycle_size), 9)
            
            # Extract example number
            example_num = None
            if 'example_' in variable_id:
                try:
                    example_num = variable_id.split('example_')[1].split('_position_')[0]
                    example_num = int(example_num)
                except (IndexError, ValueError):
                    continue
            
            if example_num is not None:
                cycle_total_counts[cycle] += 1
                
                # Check if this example was seen in previous cycles
                if example_num in previously_seen:
                    cycle_reobs_counts[cycle] += 1
                
                # Add to current cycle's examples
                cycle_examples[cycle].add(example_num)
                
                # Update previously seen examples for future cycles
                previously_seen.add(example_num)
        
        # Calculate reobservation rates for each cycle
        reobs_rates = []
        total_obs_per_cycle = []
        unique_obs_per_cycle = []
        
        for cycle in range(10):
            total = cycle_total_counts[cycle]
            reobs = cycle_reobs_counts[cycle]
            unique = len(cycle_examples[cycle])
            
            total_obs_per_cycle.append(total)
            unique_obs_per_cycle.append(unique)
            
            if total > 0:
                reobs_rates.append(reobs / total * 100)  # Convert to percentage
            else:
                reobs_rates.append(0)
        
        experiment_patterns[experiment_name] = {
            'reobs_rates': reobs_rates,
            'total_obs': total_obs_per_cycle,
            'unique_obs': unique_obs_per_cycle,
            'cycle_examples': cycle_examples
        }
    
    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    cycles = list(range(10))
    
    colors = {'gradient_voi_cold_start_with_embedding': 'orange', 
              'gradient_voi_cold_start': 'brown'}
    markers = {'gradient_voi_cold_start_with_embedding': 'o', 
               'gradient_voi_cold_start': 's'}
    
    # Plot 1: Reobservation rates over cycles
    for exp_name, patterns in experiment_patterns.items():
        ax1.plot(cycles, patterns['reobs_rates'], 
                color=colors[exp_name], marker=markers[exp_name], 
                label=exp_name, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Reobservation Rate (%)', fontsize=12)
    ax1.set_title('Reobservation Rate Evolution Across Cycles', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cycles)
    
    # Plot 2: Total observations per cycle
    width = 0.35
    x = np.array(cycles)
    
    exp_names = list(experiment_patterns.keys())
    bars1 = ax2.bar(x - width/2, experiment_patterns[exp_names[0]]['total_obs'], 
                    width, label=exp_names[0], alpha=0.7, color=colors[exp_names[0]])
    bars2 = ax2.bar(x + width/2, experiment_patterns[exp_names[1]]['total_obs'], 
                    width, label=exp_names[1], alpha=0.7, color=colors[exp_names[1]])
    
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Total Observations', fontsize=12)
    ax2.set_title('Total Observations Per Cycle', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cycles)
    
    # Plot 3: Unique examples per cycle
    for exp_name, patterns in experiment_patterns.items():
        ax3.plot(cycles, patterns['unique_obs'], 
                color=colors[exp_name], marker=markers[exp_name], 
                label=exp_name, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Cycle', fontsize=12)
    ax3.set_ylabel('Unique Examples Observed', fontsize=12)
    ax3.set_title('Unique Examples Per Cycle', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(cycles)
    
    # Plot 4: Cumulative unique examples
    for exp_name, patterns in experiment_patterns.items():
        cumulative_unique = []
        seen_examples = set()
        
        for cycle in range(10):
            seen_examples.update(patterns['cycle_examples'][cycle])
            cumulative_unique.append(len(seen_examples))
        
        ax4.plot(cycles, cumulative_unique, 
                color=colors[exp_name], marker=markers[exp_name], 
                label=exp_name, linewidth=2, markersize=8)
    
    ax4.set_xlabel('Cycle', fontsize=12)
    ax4.set_ylabel('Cumulative Unique Examples', fontsize=12)
    ax4.set_title('Cumulative Unique Examples Across Cycles', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(cycles)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    print(f"\nReobservation Pattern Analysis:")
    print("=" * 50)
    
    for exp_name, patterns in experiment_patterns.items():
        print(f"\n{exp_name}:")
        print(f"  Average reobservation rate: {np.mean(patterns['reobs_rates']):.2f}%")
        print(f"  Max reobservation rate: {np.max(patterns['reobs_rates']):.2f}% (Cycle {np.argmax(patterns['reobs_rates'])})")
        print(f"  Total unique examples seen: {len(set().union(*patterns['cycle_examples'].values()))}")
        
        # Show cycle-by-cycle breakdown
        print(f"  Cycle-by-cycle breakdown:")
        for cycle in range(10):
            reobs_rate = patterns['reobs_rates'][cycle]
            total_obs = patterns['total_obs'][cycle]
            unique_obs = patterns['unique_obs'][cycle]
            print(f"    Cycle {cycle}: {reobs_rate:.1f}% reobs rate ({total_obs} total, {unique_obs} unique)")
    
    # Compare the two experiments
    exp1, exp2 = list(experiment_patterns.keys())
    rates1 = experiment_patterns[exp1]['reobs_rates']
    rates2 = experiment_patterns[exp2]['reobs_rates']
    
    print(f"\nComparison:")
    print(f"  {exp1} vs {exp2}:")
    for cycle in range(10):
        diff = rates1[cycle] - rates2[cycle]
        print(f"    Cycle {cycle}: {diff:+.1f}% difference")
    
    print(f"\nSaved observation patterns plot to {save_path}")

def create_plots(feature_num):
    base_path = "../outputs"
    results_path = os.path.join(base_path, "results")
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    experiment_results = load_experiment_results(results_path)
    
    if not experiment_results:
        print("No experiment results found.")
        return
    
    # Extract costs per cycle
    _, cumulative_costs = extract_costs_per_cycle(experiment_results)
    
    # Create original plots
    '''plot_observe_all_experiments(experiment_results, os.path.join(plots_path, "observe_all_experiments.png"))
    plot_observe_5_experiments(experiment_results, os.path.join(plots_path, f"observe_{feature_num}_experiments.png"), feature_num)
    plot_voi_comparison(experiment_results, os.path.join(plots_path, "voi_comparison.png"))
    plot_feature_counts(experiment_results, os.path.join(plots_path, "feature_counts.png"))
    plot_top_only_comparison(experiment_results, os.path.join(plots_path, "top_only_comparison.png"))
    plot_cold_start_experiments(experiment_results, os.path.join(plots_path, "cold_start_experiments.png"), feature_num)'''
    #plot_top_only_comparison(experiment_results, os.path.join(plots_path, "top_only_comparison.png"))
    # Create new cost-based plot
    #plot_loss_vs_cost(experiment_results, cumulative_costs, os.path.join(plots_path, "loss_vs_cost.png"), feature_num)
    plot_cold_start_embedding_comparison(experiment_results, os.path.join(plots_path, "cold_start_embedding_comparison.png"))
    #plot_observation_patterns(experiment_results, os.path.join(plots_path, "observation_cost.png"))
    
    print(f"All plots saved to {plots_path}")

if __name__ == "__main__":
    feature_num = 3
    create_plots(feature_num)