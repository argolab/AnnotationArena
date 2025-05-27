import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']

def load_noisy_experiment_results(results_path):
    """Load experiment results from JSON files with noisy data tracking."""
    experiment_results = {}
    
    for filename in os.listdir(results_path):
        if filename.endswith('.json') and 'noisy_' in filename and 'combined' not in filename:
            filepath = os.path.join(results_path, filename)
            try:
                with open(filepath, 'r') as f:
                    experiment_name = filename.replace('noisy_', '').replace('.json', '')
                    experiment_results[experiment_name] = json.load(f)
                    print(f"Loaded {experiment_name} noisy results")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return experiment_results

def analyze_detailed_selections(observation_history, dataset_path):
    """
    Analyze selections by parsing observation history and dataset structure.
    Returns breakdown by cycle: original_human, corrupted_human, original_llm, corrupted_llm
    """
    # Load dataset to get is_noisy and annotator info
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Create mapping of example_idx to data
    dataset_map = {i: entry for i, entry in enumerate(dataset)}
    
    # Group observations by cycles (approximately every 200 selections based on your experiment)
    selections_per_cycle = 200  # Adjust based on your experiment
    
    breakdown_per_cycle = []
    cycle_selections = {
        'original_human': 0,
        'corrupted_human': 0, 
        'original_llm': 0,
        'corrupted_llm': 0
    }
    
    for i, obs in enumerate(observation_history):
        # Parse variable_id: "example_{idx}_position_{pos}"
        var_id = obs['variable_id']
        parts = var_id.split('_')
        example_idx = int(parts[1])
        position_idx = int(parts[3])
        
        # Get entry data
        if example_idx in dataset_map:
            entry = dataset_map[example_idx]
            
            # Check if position exists
            if position_idx < len(entry.get('annotators', [])):
                is_llm = entry['annotators'][position_idx] == -1
                is_noisy = entry.get('is_noisy', [False] * len(entry['annotators']))[position_idx]
                
                # Categorize selection
                if is_llm:
                    if is_noisy:
                        cycle_selections['corrupted_llm'] += 1
                    else:
                        cycle_selections['original_llm'] += 1
                else:  # Human
                    if is_noisy:
                        cycle_selections['corrupted_human'] += 1
                    else:
                        cycle_selections['original_human'] += 1
        
        # End of cycle check
        if (i + 1) % selections_per_cycle == 0 or i == len(observation_history) - 1:
            if any(cycle_selections.values()):
                breakdown_per_cycle.append(cycle_selections.copy())
            cycle_selections = {
                'original_human': 0,
                'corrupted_human': 0,
                'original_llm': 0, 
                'corrupted_llm': 0
            }
    
    return breakdown_per_cycle

def plot_variable_type_breakdown_percentages(results_dict, dataset_paths, save_path):
    """
    Plot percentage breakdown of variable types selected per cycle.
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(8*len(results_dict), 6))
    if len(results_dict) == 1:
        axes = [axes]
    
    # Clean colors
    colors = {
        'original_human': '#27AE60',    # Green
        'corrupted_human': '#E74C3C',   # Red  
        'original_llm': '#3498DB',      # Blue
        'corrupted_llm': '#F39C12'      # Orange
    }
    
    labels = {
        'original_human': 'Original Human',
        'corrupted_human': 'Corrupted Human',
        'original_llm': 'Original LLM', 
        'corrupted_llm': 'Corrupted LLM'
    }
    
    strategies = list(results_dict.keys())
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        results = results_dict[strategy]
        
        # Get breakdown from observation history
        obs_history = results.get('observation_history', [])
        if not obs_history:
            continue
            
        dataset_path = dataset_paths.get(strategy, dataset_paths['default'])
        breakdown_per_cycle = analyze_detailed_selections(obs_history, dataset_path)
        
        if not breakdown_per_cycle:
            continue
            
        n_cycles = len(breakdown_per_cycle)
        cycles = list(range(n_cycles))
        
        # Convert to percentages
        percentages = {var_type: [] for var_type in colors.keys()}
        
        for cycle_data in breakdown_per_cycle:
            total = sum(cycle_data.values())
            if total > 0:
                for var_type in colors.keys():
                    percentages[var_type].append((cycle_data[var_type] / total) * 100)
            else:
                for var_type in colors.keys():
                    percentages[var_type].append(0)
        
        # Create grouped bars
        x = np.arange(n_cycles)
        width = 0.2
        positions = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
        
        for i, (var_type, values) in enumerate(percentages.items()):
            ax.bar(x + positions[i], values, width, 
                   label=labels[var_type], 
                   color=colors[var_type], alpha=0.8)
        
        ax.set_title(f'{strategy.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Acquisition Cycle', fontsize=12)
        ax.set_ylabel('Percentage of Selections', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i}' for i in cycles])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Variable Type Selection Breakdown by Cycle\n(Percentage Distribution)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved variable breakdown percentages to {save_path}")

def plot_performance_vs_selection_overlay(results_dict, save_path):
    """
    Single plot with dual y-axes showing performance and selection quality.
    """
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    colors = {'gradient_voi': '#1f77b4', 'random_random': '#ff7f0e'}
    markers = {'gradient_voi': 'o', 'random_random': 's'}
    
    for strategy, results in results_dict.items():
        color = colors.get(strategy, '#333333')
        marker = markers.get(strategy, 'o')
        strategy_name = strategy.replace('_', ' ').title()
        
        if 'test_annotated_losses' in results and 'selection_ratios_per_cycle' in results:
            test_losses = results['test_annotated_losses']
            ratios = results['selection_ratios_per_cycle']
            cumulative_samples = np.cumsum(results.get('features_annotated', []))
            
            # Performance on left axis
            min_len_perf = min(len(test_losses), len(cumulative_samples))
            if min_len_perf > 0:
                ax1.plot(cumulative_samples[:min_len_perf], test_losses[:min_len_perf],
                        color=color, linestyle='-', linewidth=1, 
                        marker=marker, markersize=8, alpha=0.9,
                        label=f'{strategy_name} - Test Loss')
            
            # Selection quality on right axis  
            min_len_qual = min(len(ratios), len(cumulative_samples))
            if min_len_qual > 0:
                ax2.plot(cumulative_samples[:min_len_qual], ratios[:min_len_qual],
                        color=color, linestyle='--', linewidth=2,
                        marker=marker, markersize=8, alpha=0.7,
                        label=f'{strategy_name} - Noise Ratio')
    
    # Random baseline
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.8, linewidth=2,
                label='Random Baseline')
    
    # Configure axes - ALL BLACK
    ax1.set_xlabel('Cumulative Samples Acquired', fontsize=13, color='black')
    ax1.set_ylabel('Test Loss', fontsize=13, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Proportion of Noisy Variables', fontsize=13, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 1)
    
    # Combined legend - CLEAN
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
    plt.title('Model Performance vs Selection Quality', 
              fontsize=15, fontweight='bold', pad=20, color='black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance vs selection overlay to {save_path}")

def plot_noise_selection_summary(results_dict, save_path):
    """
    Clean summary showing total selections and final ratios.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'original': '#27AE60', 'noisy': '#E74C3C'}
    
    strategies = list(results_dict.keys())
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    # Data extraction
    original_totals = []
    noisy_totals = []
    final_ratios = []
    
    for strategy in strategies:
        results = results_dict[strategy]
        orig_total = sum(results.get('original_selections_per_cycle', []))
        noisy_total = sum(results.get('noisy_selections_per_cycle', []))
        total = orig_total + noisy_total
        ratio = noisy_total / total if total > 0 else 0
        
        original_totals.append(orig_total)
        noisy_totals.append(noisy_total)
        final_ratios.append(ratio)
    
    # Plot 1: Total counts
    ax1.bar(x_pos - width/2, original_totals, width, label='Original Variables', 
            color=colors['original'], alpha=0.8)
    ax1.bar(x_pos + width/2, noisy_totals, width, label='Noisy Variables', 
            color=colors['noisy'], alpha=0.8)
    
    ax1.set_xlabel('Selection Strategy', fontsize=12)
    ax1.set_ylabel('Total Variables Selected', fontsize=12)
    ax1.set_title('Total Selection Counts', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise ratios
    bars = ax2.bar(x_pos, final_ratios, color=colors['noisy'], alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2,
                label='Random Baseline')
    
    # Add value labels
    for bar, ratio in zip(bars, final_ratios):
        height = bar.get_height()
        ax2.annotate(f'{ratio:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Selection Strategy', fontsize=12)
    ax2.set_ylabel('Proportion of Noisy Variables', fontsize=12)
    ax2.set_title('Final Noise Selection Ratio', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise selection summary to {save_path}")

def plot_learning_curves(results_dict, save_path):
    """
    Learning curves showing noise avoidance over time.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'gradient_voi': '#1f77b4', 'random_random': '#ff7f0e'}
    markers = {'gradient_voi': 'o', 'random_random': 's'}
    
    for strategy, results in results_dict.items():
        if 'selection_ratios_per_cycle' in results:
            ratios = results['selection_ratios_per_cycle']
            cumulative_samples = np.cumsum(results.get('features_annotated', []))
            
            if len(ratios) > 0 and len(cumulative_samples) > 0:
                min_len = min(len(ratios), len(cumulative_samples))
                ax.plot(cumulative_samples[:min_len], ratios[:min_len], 
                       marker=markers.get(strategy, 'o'), 
                       color=colors.get(strategy, 'gray'),
                       linewidth=3, markersize=8, alpha=0.8,
                       label=strategy.replace('_', ' ').title())
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2,
               label='Random Baseline')
    
    ax.set_xlabel('Cumulative Samples Acquired', fontsize=13)
    ax.set_ylabel('Proportion of Noisy Variables Selected', fontsize=13)
    ax.set_title('Learning to Avoid Noisy Variables Over Time\n(Lower is Better)', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")

def create_noisy_plots():
    """Main function to create all noisy experiment plots."""
    base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
    results_path = os.path.join(base_path, "results_noisy_hanna")
    data_path = os.path.join(base_path, "data")
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    # Load results (NO COMBINED JSON)
    experiment_results = load_noisy_experiment_results(results_path)
    
    if not experiment_results:
        print("No noisy experiment results found.")
        return
    
    # Dataset paths
    dataset_paths = {
        'gradient_voi': os.path.join(data_path, "active_pool.json"),
        'random_random': os.path.join(data_path, "active_pool.json"),
        'default': os.path.join(data_path, "active_pool.json")
    }
    
    # Create clean plots
    plot_noise_selection_summary(
        experiment_results, 
        os.path.join(plots_path, "noise_selection_summary.png")
    )
    
    plot_learning_curves(
        experiment_results,
        os.path.join(plots_path, "learning_curves.png")
    )
    
    plot_performance_vs_selection_overlay(
        experiment_results,
        os.path.join(plots_path, "performance_vs_selection.png")
    )
    
    plot_variable_type_breakdown_percentages(
        experiment_results,
        dataset_paths,
        os.path.join(plots_path, "variable_breakdown_percentages.png")
    )
    
    print(f"All clean plots saved to {plots_path}")

if __name__ == "__main__":
    create_noisy_plots()