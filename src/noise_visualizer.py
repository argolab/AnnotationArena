# import os
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Set publication-ready style
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("husl")

# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['Times New Roman']

# def load_noisy_experiment_results(results_path):
#     """Load experiment results from JSON files with noisy data tracking."""
#     experiment_results = {}
    
#     for filename in os.listdir(results_path):
#         if filename.endswith('.json') and 'noisy_' in filename and 'combined' not in filename:
#             filepath = os.path.join(results_path, filename)
#             try:
#                 with open(filepath, 'r') as f:
#                     experiment_name = filename.replace('noisy_', '').replace('.json', '')
#                     experiment_results[experiment_name] = json.load(f)
#                     print(f"Loaded {experiment_name} noisy results")
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
    
#     return experiment_results

# def analyze_detailed_selections(observation_history, dataset_path):
#     """
#     Analyze selections by parsing observation history and dataset structure.
#     Returns breakdown by cycle: original_human, corrupted_human, original_llm, corrupted_llm
#     """
#     # Load dataset to get is_noisy and annotator info
#     with open(dataset_path, 'r') as f:
#         dataset = json.load(f)
    
#     # Create mapping of example_idx to data
#     dataset_map = {i: entry for i, entry in enumerate(dataset)}
    
#     # Group observations by cycles (approximately every 200 selections based on your experiment)
#     selections_per_cycle = 200  # Adjust based on your experiment
    
#     breakdown_per_cycle = []
#     cycle_selections = {
#         'original_human': 0,
#         'corrupted_human': 0, 
#         'original_llm': 0,
#         'corrupted_llm': 0
#     }
    
#     for i, obs in enumerate(observation_history):
#         # Parse variable_id: "example_{idx}_position_{pos}"
#         var_id = obs['variable_id']
#         parts = var_id.split('_')
#         example_idx = int(parts[1])
#         position_idx = int(parts[3])
        
#         # Get entry data
#         if example_idx in dataset_map:
#             entry = dataset_map[example_idx]
            
#             # Check if position exists
#             if position_idx < len(entry.get('annotators', [])):
#                 is_llm = entry['annotators'][position_idx] == -1
#                 is_noisy = entry.get('is_noisy', [False] * len(entry['annotators']))[position_idx]
                
#                 # Categorize selection
#                 if is_llm:
#                     if is_noisy:
#                         cycle_selections['corrupted_llm'] += 1
#                     else:
#                         cycle_selections['original_llm'] += 1
#                 else:  # Human
#                     if is_noisy:
#                         cycle_selections['corrupted_human'] += 1
#                     else:
#                         cycle_selections['original_human'] += 1
        
#         # End of cycle check
#         if (i + 1) % selections_per_cycle == 0 or i == len(observation_history) - 1:
#             if any(cycle_selections.values()):
#                 breakdown_per_cycle.append(cycle_selections.copy())
#             cycle_selections = {
#                 'original_human': 0,
#                 'corrupted_human': 0,
#                 'original_llm': 0, 
#                 'corrupted_llm': 0
#             }
    
#     return breakdown_per_cycle

# def plot_variable_type_breakdown_percentages(results_dict, dataset_paths, save_path):
#     """
#     Plot percentage breakdown of variable types selected per cycle.
#     """
#     fig, axes = plt.subplots(1, len(results_dict), figsize=(8*len(results_dict), 6))
#     if len(results_dict) == 1:
#         axes = [axes]
    
#     # Clean colors
#     colors = {
#         'original_human': '#27AE60',    # Green
#         'corrupted_human': '#E74C3C',   # Red  
#         'original_llm': '#3498DB',      # Blue
#         'corrupted_llm': '#F39C12'      # Orange
#     }
    
#     labels = {
#         'original_human': 'Original Human',
#         'corrupted_human': 'Corrupted Human',
#         'original_llm': 'Original LLM', 
#         'corrupted_llm': 'Corrupted LLM'
#     }
    
#     strategies = list(results_dict.keys())
    
#     for idx, strategy in enumerate(strategies):
#         ax = axes[idx]
#         results = results_dict[strategy]
        
#         # Get breakdown from observation history
#         obs_history = results.get('observation_history', [])
#         if not obs_history:
#             continue
            
#         dataset_path = dataset_paths.get(strategy, dataset_paths['default'])
#         breakdown_per_cycle = analyze_detailed_selections(obs_history, dataset_path)
        
#         if not breakdown_per_cycle:
#             continue
            
#         n_cycles = len(breakdown_per_cycle)
#         cycles = list(range(n_cycles))
        
#         # Convert to percentages
#         percentages = {var_type: [] for var_type in colors.keys()}
        
#         for cycle_data in breakdown_per_cycle:
#             total = sum(cycle_data.values())
#             if total > 0:
#                 for var_type in colors.keys():
#                     percentages[var_type].append((cycle_data[var_type] / total) * 100)
#             else:
#                 for var_type in colors.keys():
#                     percentages[var_type].append(0)
        
#         # Create grouped bars
#         x = np.arange(n_cycles)
#         width = 0.2
#         positions = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
        
#         for i, (var_type, values) in enumerate(percentages.items()):
#             ax.bar(x + positions[i], values, width, 
#                    label=labels[var_type], 
#                    color=colors[var_type], alpha=0.8)
        
#         ax.set_title(f'{strategy.replace("_", " ").title()}', 
#                     fontsize=14, fontweight='bold')
#         ax.set_xlabel('Acquisition Cycle', fontsize=12)
#         ax.set_ylabel('Percentage of Selections', fontsize=12)
#         ax.set_xticks(x)
#         ax.set_xticklabels([f'{i}' for i in cycles])
#         ax.set_ylim(0, 100)
#         ax.legend(fontsize=10, loc='best')
#         ax.grid(True, alpha=0.3, axis='y')
    
#     plt.suptitle('Variable Type Selection Breakdown by Cycle\n(Percentage Distribution)', 
#                  fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved variable breakdown percentages to {save_path}")

# def plot_performance_vs_selection_overlay(results_dict, save_path):
#     """
#     Single plot with dual y-axes showing performance and selection quality.
#     """
    
#     fig, ax1 = plt.subplots(figsize=(12, 8))
#     ax2 = ax1.twinx()
    
#     colors = {'gradient_voi': '#1f77b4', 'random_random': '#ff7f0e'}
#     markers = {'gradient_voi': 'o', 'random_random': 's'}
    
#     for strategy, results in results_dict.items():
#         color = colors.get(strategy, '#333333')
#         marker = markers.get(strategy, 'o')
#         strategy_name = strategy.replace('_', ' ').title()
        
#         if 'test_annotated_losses' in results and 'selection_ratios_per_cycle' in results:
#             test_losses = results['test_annotated_losses']
#             ratios = results['selection_ratios_per_cycle']
#             cumulative_samples = np.cumsum(results.get('features_annotated', []))
            
#             # Performance on left axis
#             min_len_perf = min(len(test_losses), len(cumulative_samples))
#             if min_len_perf > 0:
#                 ax1.plot(cumulative_samples[:min_len_perf], test_losses[:min_len_perf],
#                         color=color, linestyle='-', linewidth=1, 
#                         marker=marker, markersize=8, alpha=0.9,
#                         label=f'{strategy_name} - Test Loss')
            
#             # Selection quality on right axis  
#             min_len_qual = min(len(ratios), len(cumulative_samples))
#             if min_len_qual > 0:
#                 ax2.plot(cumulative_samples[:min_len_qual], ratios[:min_len_qual],
#                         color=color, linestyle='--', linewidth=2,
#                         marker=marker, markersize=8, alpha=0.7,
#                         label=f'{strategy_name} - Noise Ratio')
    
#     # Random baseline
#     ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.8, linewidth=2,
#                 label='Random Baseline')
    
#     # Configure axes - ALL BLACK
#     ax1.set_xlabel('Cumulative Samples Acquired', fontsize=13, color='black')
#     ax1.set_ylabel('Test Loss', fontsize=13, color='black')
#     ax1.tick_params(axis='y', labelcolor='black')
#     ax1.tick_params(axis='x', labelcolor='black')
#     ax1.grid(True, alpha=0.3)
    
#     ax2.set_ylabel('Proportion of Noisy Variables', fontsize=13, color='black')
#     ax2.tick_params(axis='y', labelcolor='black')
#     ax2.set_ylim(0, 1)
    
#     # Combined legend - CLEAN
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
#     plt.title('Model Performance vs Selection Quality', 
#               fontsize=15, fontweight='bold', pad=20, color='black')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Saved performance vs selection overlay to {save_path}")

# def plot_noise_selection_summary(results_dict, save_path):
#     """
#     Clean summary showing total selections and final ratios.
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
#     colors = {'original': '#27AE60', 'noisy': '#E74C3C'}
    
#     strategies = list(results_dict.keys())
#     x_pos = np.arange(len(strategies))
#     width = 0.35
    
#     # Data extraction
#     original_totals = []
#     noisy_totals = []
#     final_ratios = []
    
#     for strategy in strategies:
#         results = results_dict[strategy]
#         orig_total = sum(results.get('original_selections_per_cycle', []))
#         noisy_total = sum(results.get('noisy_selections_per_cycle', []))
#         total = orig_total + noisy_total
#         ratio = noisy_total / total if total > 0 else 0
        
#         original_totals.append(orig_total)
#         noisy_totals.append(noisy_total)
#         final_ratios.append(ratio)
    
#     # Plot 1: Total counts
#     ax1.bar(x_pos - width/2, original_totals, width, label='Original Variables', 
#             color=colors['original'], alpha=0.8)
#     ax1.bar(x_pos + width/2, noisy_totals, width, label='Noisy Variables', 
#             color=colors['noisy'], alpha=0.8)
    
#     ax1.set_xlabel('Selection Strategy', fontsize=12)
#     ax1.set_ylabel('Total Variables Selected', fontsize=12)
#     ax1.set_title('Total Selection Counts', fontsize=14, fontweight='bold')
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Noise ratios
#     bars = ax2.bar(x_pos, final_ratios, color=colors['noisy'], alpha=0.8)
#     ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2,
#                 label='Random Baseline')
    
#     # Add value labels
#     for bar, ratio in zip(bars, final_ratios):
#         height = bar.get_height()
#         ax2.annotate(f'{ratio:.3f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
#     ax2.set_xlabel('Selection Strategy', fontsize=12)
#     ax2.set_ylabel('Proportion of Noisy Variables', fontsize=12)
#     ax2.set_title('Final Noise Selection Ratio', fontsize=14, fontweight='bold')
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
#     ax2.set_ylim(0, 1)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved noise selection summary to {save_path}")

# def plot_learning_curves(results_dict, save_path):
#     """
#     Learning curves showing noise avoidance over time.
#     """
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     colors = {'gradient_voi': '#1f77b4', 'random_random': '#ff7f0e'}
#     markers = {'gradient_voi': 'o', 'random_random': 's'}
    
#     for strategy, results in results_dict.items():
#         if 'selection_ratios_per_cycle' in results:
#             ratios = results['selection_ratios_per_cycle']
#             cumulative_samples = np.cumsum(results.get('features_annotated', []))
            
#             if len(ratios) > 0 and len(cumulative_samples) > 0:
#                 min_len = min(len(ratios), len(cumulative_samples))
#                 ax.plot(cumulative_samples[:min_len], ratios[:min_len], 
#                        marker=markers.get(strategy, 'o'), 
#                        color=colors.get(strategy, 'gray'),
#                        linewidth=3, markersize=8, alpha=0.8,
#                        label=strategy.replace('_', ' ').title())
    
#     ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2,
#                label='Random Baseline')
    
#     ax.set_xlabel('Cumulative Samples Acquired', fontsize=13)
#     ax.set_ylabel('Proportion of Noisy Variables Selected', fontsize=13)
#     ax.set_title('Learning to Avoid Noisy Variables Over Time\n(Lower is Better)', 
#                 fontsize=15, fontweight='bold')
#     ax.legend(fontsize=12)
#     ax.grid(True, alpha=0.3)
#     ax.set_ylim(0, 1)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved learning curves to {save_path}")

# def create_noisy_plots():
#     """Main function to create all noisy experiment plots."""
#     base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
#     results_path = os.path.join(base_path, "results_noisy_hanna")
#     data_path = os.path.join(base_path, "data")
#     plots_path = os.path.join(results_path, "plots")
#     os.makedirs(plots_path, exist_ok=True)
    
#     # Load results (NO COMBINED JSON)
#     experiment_results = load_noisy_experiment_results(results_path)
    
#     if not experiment_results:
#         print("No noisy experiment results found.")
#         return
    
#     # Dataset paths
#     dataset_paths = {
#         'gradient_voi': os.path.join(data_path, "active_pool.json"),
#         'random_random': os.path.join(data_path, "active_pool.json"),
#         'default': os.path.join(data_path, "active_pool.json")
#     }
    
#     # Create clean plots
#     plot_noise_selection_summary(
#         experiment_results, 
#         os.path.join(plots_path, "noise_selection_summary.png")
#     )
    
#     plot_learning_curves(
#         experiment_results,
#         os.path.join(plots_path, "learning_curves.png")
#     )
    
#     plot_performance_vs_selection_overlay(
#         experiment_results,
#         os.path.join(plots_path, "performance_vs_selection.png")
#     )
    
#     plot_variable_type_breakdown_percentages(
#         experiment_results,
#         dataset_paths,
#         os.path.join(plots_path, "variable_breakdown_percentages.png")
#     )
    
#     print(f"All clean plots saved to {plots_path}")

# if __name__ == "__main__":
#     create_noisy_plots()

"""
Comprehensive analysis script for multi-level noise experiments.
Compares gradient_voi vs random_random with effective noise calculation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_experiment_data(result_path, dataset_path):
    """Load experiment results and corresponding dataset."""
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    return results, dataset

def parse_variable_id(variable_id):
    """Parse variable_id to extract example and position indices."""
    parts = variable_id.split('_')
    example_idx = int(parts[1])
    position_idx = int(parts[3])
    return example_idx, position_idx

def is_effectively_noisy(answers, true_answers, comparison_type='argmax'):
    """
    Determine if a variable is effectively noisy by comparing distributions.
    
    Args:
        answers: Potentially noisy distribution
        true_answers: Clean distribution  
        comparison_type: 'argmax' or 'kl_divergence'
    
    Returns:
        bool: True if effectively noisy
    """
    answers = np.array(answers)
    true_answers = np.array(true_answers)
    
    if comparison_type == 'argmax':
        return np.argmax(answers) != np.argmax(true_answers)
    
    elif comparison_type == 'kl_divergence':
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        answers_norm = answers + eps
        true_answers_norm = true_answers + eps
        
        # Normalize
        answers_norm = answers_norm / np.sum(answers_norm)
        true_answers_norm = true_answers_norm / np.sum(true_answers_norm)
        
        # KL divergence
        kl_div = np.sum(true_answers_norm * np.log(true_answers_norm / answers_norm))
        return kl_div > 0.1  # Threshold for "effectively different"

def analyze_selections(results, dataset):
    """
    Analyze selections to determine effective noise levels.
    
    Returns:
        dict: Analysis results with noise breakdowns
    """
    observation_history = results.get('observation_history', [])
    
    analysis = {
        'total_selections': len(observation_history),
        'noise_label_breakdown': defaultdict(int),
        'effective_noise_breakdown': defaultdict(int),
        'cycle_analysis': [],
        'position_details': []
    }
    
    # Create dataset lookup
    dataset_map = {i: entry for i, entry in enumerate(dataset)}
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx not in dataset_map:
            continue
            
        entry = dataset_map[example_idx]
        
        if position_idx >= len(entry.get('answers', [])):
            continue
            
        # Get noise label information
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['answers']))[position_idx]
        is_llm = entry['annotators'][position_idx] == -1
        
        # Categorize by noise label
        if is_llm:
            if noise_type == 'original':
                label_category = 'original_llm'
            elif 'low' in noise_type:
                label_category = 'llm_low'
            elif 'medium' in noise_type:
                label_category = 'llm_medium'
            elif 'heavy' in noise_type:
                label_category = 'llm_heavy'
            else:
                label_category = 'unknown_llm'
        else:
            if noise_type == 'original':
                label_category = 'original_human'
            elif 'low' in noise_type:
                label_category = 'human_low'
            elif 'medium' in noise_type:
                label_category = 'human_medium'
            elif 'heavy' in noise_type:
                label_category = 'human_heavy'
            else:
                label_category = 'unknown_human'
        
        analysis['noise_label_breakdown'][label_category] += 1
        
        # Determine effective noise
        answers = entry['answers'][position_idx]
        true_answers = entry.get('true_answers', [answers] * len(entry['answers']))[position_idx]
        
        effectively_noisy = is_effectively_noisy(answers, true_answers, 'argmax')
        
        if effectively_noisy:
            effective_category = f"effectively_noisy_{label_category}"
        else:
            effective_category = f"effectively_clean_{label_category}"
            
        analysis['effective_noise_breakdown'][effective_category] += 1
        
        # Store detailed information
        analysis['position_details'].append({
            'variable_id': variable_id,
            'example_idx': example_idx,
            'position_idx': position_idx,
            'noise_label': noise_type,
            'is_llm': is_llm,
            'label_category': label_category,
            'effectively_noisy': effectively_noisy,
            'effective_category': effective_category,
            'timestamp': obs.get('timestamp', 0)
        })
    
    return analysis

def plot_noise_comparison(gradient_analysis, random_analysis, save_path):
    """Plot comparison of noise selections between strategies."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Noise Label Breakdown
    noise_categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy',
                       'original_human', 'human_low', 'human_medium', 'human_heavy']
    
    gradient_counts = [gradient_analysis['noise_label_breakdown'][cat] for cat in noise_categories]
    random_counts = [random_analysis['noise_label_breakdown'][cat] for cat in noise_categories]
    
    x = np.arange(len(noise_categories))
    width = 0.35
    
    ax1.bar(x - width/2, gradient_counts, width, label='Gradient VOI', alpha=0.8, color='blue')
    ax1.bar(x + width/2, random_counts, width, label='Random', alpha=0.8, color='orange')
    
    ax1.set_title('Noise Label Breakdown', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Selection Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels([cat.replace('_', '\n') for cat in noise_categories], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Effective Noise Analysis
    effective_categories = set(gradient_analysis['effective_noise_breakdown'].keys()) | \
                          set(random_analysis['effective_noise_breakdown'].keys())
    effective_categories = sorted(list(effective_categories))
    
    gradient_effective = [gradient_analysis['effective_noise_breakdown'][cat] for cat in effective_categories]
    random_effective = [random_analysis['effective_noise_breakdown'][cat] for cat in effective_categories]
    
    x_eff = np.arange(len(effective_categories))
    
    ax2.bar(x_eff - width/2, gradient_effective, width, label='Gradient VOI', alpha=0.8, color='blue')
    ax2.bar(x_eff + width/2, random_effective, width, label='Random', alpha=0.8, color='orange')
    
    ax2.set_title('Effective Noise Breakdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Selection Count')
    ax2.set_xticks(x_eff)
    ax2.set_xticklabels([cat.replace('_', '\n').replace('effectively ', '') for cat in effective_categories], 
                       rotation=45, fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Noise Effectiveness Ratio
    def calculate_noise_ratio(analysis):
        total_noisy_labeled = sum(count for cat, count in analysis['noise_label_breakdown'].items() 
                                 if 'low' in cat or 'medium' in cat or 'heavy' in cat)
        total_effectively_noisy = sum(count for cat, count in analysis['effective_noise_breakdown'].items() 
                                    if 'effectively_noisy' in cat)
        
        return total_effectively_noisy / max(total_noisy_labeled, 1)
    
    gradient_ratio = calculate_noise_ratio(gradient_analysis)
    random_ratio = calculate_noise_ratio(random_analysis)
    
    strategies = ['Gradient VOI', 'Random']
    ratios = [gradient_ratio, random_ratio]
    
    bars = ax3.bar(strategies, ratios, color=['blue', 'orange'], alpha=0.8)
    ax3.set_title('Effective Noise Ratio\n(Effectively Noisy / Labeled as Noisy)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax3.annotate(f'{ratio:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4. Clean vs Noisy Selection Preference
    def calculate_clean_preference(analysis):
        total_clean = sum(count for cat, count in analysis['noise_label_breakdown'].items() 
                         if 'original' in cat)
        total_selections = analysis['total_selections']
        return total_clean / max(total_selections, 1)
    
    gradient_clean_pref = calculate_clean_preference(gradient_analysis)
    random_clean_pref = calculate_clean_preference(random_analysis)
    
    clean_prefs = [gradient_clean_pref, random_clean_pref]
    
    bars = ax4.bar(strategies, clean_prefs, color=['blue', 'orange'], alpha=0.8)
    ax4.set_title('Clean Variable Preference\n(Original / Total Selections)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Proportion')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, pref in zip(bars, clean_prefs):
        height = bar.get_height()
        ax4.annotate(f'{pref:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise comparison to {save_path}")

def plot_learning_curves(gradient_results, random_results, save_path):
    """Plot learning curves comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test loss comparison
    gradient_test_loss = gradient_results.get('test_annotated_losses', [])
    random_test_loss = random_results.get('test_annotated_losses', [])
    
    max_cycles = max(len(gradient_test_loss), len(random_test_loss))
    cycles = list(range(max_cycles))
    
    if gradient_test_loss:
        ax1.plot(cycles[:len(gradient_test_loss)], gradient_test_loss, 
                'b-', marker='o', linewidth=3, markersize=6, label='Gradient VOI')
    
    if random_test_loss:
        ax1.plot(cycles[:len(random_test_loss)], random_test_loss, 
                'r-', marker='s', linewidth=3, markersize=6, label='Random')
    
    ax1.set_title('Test Loss Over Cycles', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training loss comparison
    gradient_train_loss = gradient_results.get('training_losses', [])
    random_train_loss = random_results.get('training_losses', [])
    
    if gradient_train_loss:
        ax2.plot(cycles[:len(gradient_train_loss)], gradient_train_loss, 
                'b--', marker='o', linewidth=3, markersize=6, label='Gradient VOI')
    
    if random_train_loss:
        ax2.plot(cycles[:len(random_train_loss)], random_train_loss, 
                'r--', marker='s', linewidth=3, markersize=6, label='Random')
    
    ax2.set_title('Training Loss Over Cycles', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")

def generate_summary_report(gradient_analysis, random_analysis, gradient_results, random_results):
    """Generate text summary of the analysis."""
    
    print("="*80)
    print("MULTI-LEVEL NOISE EXPERIMENT ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n1. SELECTION TOTALS:")
    print(f"   Gradient VOI: {gradient_analysis['total_selections']} selections")
    print(f"   Random:       {random_analysis['total_selections']} selections")
    
    print(f"\n2. CLEAN VARIABLE PREFERENCE:")
    gradient_clean = sum(count for cat, count in gradient_analysis['noise_label_breakdown'].items() 
                        if 'original' in cat)
    random_clean = sum(count for cat, count in random_analysis['noise_label_breakdown'].items() 
                      if 'original' in cat)
    
    gradient_clean_ratio = gradient_clean / max(gradient_analysis['total_selections'], 1)
    random_clean_ratio = random_clean / max(random_analysis['total_selections'], 1)
    
    print(f"   Gradient VOI: {gradient_clean:4d} ({gradient_clean_ratio:.1%}) clean selections")
    print(f"   Random:       {random_clean:4d} ({random_clean_ratio:.1%}) clean selections")
    
    print(f"\n3. EFFECTIVE NOISE ANALYSIS:")
    gradient_effective_noisy = sum(count for cat, count in gradient_analysis['effective_noise_breakdown'].items() 
                                  if 'effectively_noisy' in cat)
    random_effective_noisy = sum(count for cat, count in random_analysis['effective_noise_breakdown'].items() 
                                if 'effectively_noisy' in cat)
    
    gradient_effective_ratio = gradient_effective_noisy / max(gradient_analysis['total_selections'], 1)
    random_effective_ratio = random_effective_noisy / max(random_analysis['total_selections'], 1)
    
    print(f"   Gradient VOI: {gradient_effective_noisy:4d} ({gradient_effective_ratio:.1%}) effectively noisy")
    print(f"   Random:       {random_effective_noisy:4d} ({random_effective_ratio:.1%}) effectively noisy")
    
    print(f"\n4. FINAL PERFORMANCE:")
    gradient_final_loss = gradient_results.get('test_annotated_losses', [0])[-1]
    random_final_loss = random_results.get('test_annotated_losses', [0])[-1]
    
    print(f"   Gradient VOI final test loss: {gradient_final_loss:.4f}")
    print(f"   Random final test loss:       {random_final_loss:.4f}")
    
    improvement = ((random_final_loss - gradient_final_loss) / random_final_loss) * 100
    print(f"   Improvement: {improvement:+.1f}%")
    
    print(f"\n5. NOISE LEVEL BREAKDOWN (Gradient VOI):")
    for category in sorted(gradient_analysis['noise_label_breakdown'].keys()):
        count = gradient_analysis['noise_label_breakdown'][category]
        ratio = count / max(gradient_analysis['total_selections'], 1)
        print(f"   {category:15s}: {count:4d} ({ratio:.1%})")
    
    print(f"\n6. NOISE LEVEL BREAKDOWN (Random):")
    for category in sorted(random_analysis['noise_label_breakdown'].keys()):
        count = random_analysis['noise_label_breakdown'][category]
        ratio = count / max(random_analysis['total_selections'], 1)
        print(f"   {category:15s}: {count:4d} ({ratio:.1%})")

def main():
    """Main analysis function."""
    
    # File paths
    base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_1"
    gradient_result_path = os.path.join(base_path, "multilevel_noisy_gradient_voi_with_embedding.json")
    random_result_path = os.path.join(base_path, "multilevel_noisy_random_random_with_embedding.json")
    dataset_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/data/active_pool.json"
    
    # Load data
    print("Loading experiment data...")
    gradient_results, dataset = load_experiment_data(gradient_result_path, dataset_path)
    random_results, _ = load_experiment_data(random_result_path, dataset_path)
    
    # Analyze selections
    print("Analyzing gradient VOI selections...")
    gradient_analysis = analyze_selections(gradient_results, dataset)
    
    print("Analyzing random selections...")
    random_analysis = analyze_selections(random_results, dataset)
    
    # Create output directory
    output_dir = os.path.join(base_path, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_noise_comparison(gradient_analysis, random_analysis, 
                         os.path.join(output_dir, "noise_comparison.png"))
    
    plot_learning_curves(gradient_results, random_results,
                        os.path.join(output_dir, "learning_curves.png"))
    
    # Generate summary report
    generate_summary_report(gradient_analysis, random_analysis, 
                           gradient_results, random_results)
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}")

if __name__ == "__main__":
    main()