import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_data(gradient_path, random_path, dataset_path):
    with open(gradient_path, 'r') as f:
        gradient_results = json.load(f)
    with open(random_path, 'r') as f:
        random_results = json.load(f)
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return gradient_results, random_results, dataset

def extract_noise_selections_per_cycle(results, noise_type='llm', max_cycles=None):
    """Extract noise selections by type (llm or human)"""
    cycles = len(results.get('selection_breakdown_per_cycle', []))
    if max_cycles:
        cycles = min(cycles, max_cycles)
    
    if noise_type == 'llm':
        categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy']
        labels = ['Original', 'Low Noise', 'Medium Noise', 'Heavy Noise']
    else:  # human
        categories = ['original_human', 'human_noisy'] 
        labels = ['Original', 'Noisy']
    
    cycle_data = []
    for cycle_idx in range(cycles):
        if cycle_idx < len(results['selection_breakdown_per_cycle']):
            breakdown = results['selection_breakdown_per_cycle'][cycle_idx]
            total = sum(breakdown.get(cat, 0) for cat in categories)
            if total > 0:
                percentages = {cat: (breakdown.get(cat, 0) / total) * 100 for cat in categories}
            else:
                percentages = {cat: 0 for cat in categories}
            cycle_data.append(percentages)
        else:
            cycle_data.append({cat: 0 for cat in categories})
    
    return cycle_data, categories, labels

def parse_variable_id(variable_id):
    parts = variable_id.split('_')
    example_idx = int(parts[1])
    position_idx = int(parts[3])
    return example_idx, position_idx

def plot_noise_dynamics_separate(gradient_results, noise_type='llm', max_cycles=None, save_path=None):
    """Plot noise dynamics for either LLM or Human separately"""
    gradient_data, categories, labels = extract_noise_selections_per_cycle(
        gradient_results, noise_type, max_cycles
    )
    
    cycles = range(len(gradient_data))
    
    if noise_type == 'llm':
        colors = ['#2E8B57', '#FF8C00', '#DC143C', '#8B0000']  # Green, Orange, Red, Dark Red
        title = 'LLM Noise Selection Dynamics (Gradient VOI)'
    else:
        colors = ['#4169E1', '#B22222']  # Royal Blue, Fire Brick
        title = 'Human Noise Selection Dynamics (Gradient VOI)'
    
    plt.figure(figsize=(10, 6))
    
    for cat, color, label in zip(categories, colors, labels):
        percentages = [cycle.get(cat, 0) for cycle in gradient_data]
        plt.plot(cycles, percentages, color=color, label=label, linewidth=2.5, marker='o', markersize=4)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('% of Selections (Normalized)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_clean_selections_vs_performance(gradient_results, random_results, max_cycles=None, save_path=None):
    """Plot clean selections vs performance with improved styling"""
    gradient_cycles = len(gradient_results.get('selection_breakdown_per_cycle', []))
    random_cycles = len(random_results.get('selection_breakdown_per_cycle', []))
    min_cycles = min(gradient_cycles, random_cycles)
    if max_cycles:
        min_cycles = min(min_cycles, max_cycles)
    
    # Extract clean selections (LLM + Human original)
    gradient_clean = []
    gradient_test_loss = gradient_results.get('test_annotated_losses', [])[:min_cycles]
    
    for cycle_idx in range(min_cycles):
        breakdown = gradient_results['selection_breakdown_per_cycle'][cycle_idx]
        clean_count = breakdown.get('original_llm', 0) + breakdown.get('original_human', 0)
        gradient_clean.append(clean_count)
    
    random_clean = []
    random_test_loss = random_results.get('test_annotated_losses', [])[:min_cycles]
    
    for cycle_idx in range(min_cycles):
        breakdown = random_results['selection_breakdown_per_cycle'][cycle_idx]
        clean_count = breakdown.get('original_llm', 0) + breakdown.get('original_human', 0)
        random_clean.append(clean_count)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
    ax2 = ax1.twinx()
    
    cycles = np.arange(min_cycles)
    width = 0.35
    
    # Improved colors and transparency
    bars1 = ax1.bar(cycles - width/2, gradient_clean, width, alpha=0.6, 
                   color='#1f77b4', label='Gradient VOI Clean Selections', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(cycles + width/2, random_clean, width, alpha=0.6, 
                   color='#ff7f0e', label='Random Clean Selections', edgecolor='black', linewidth=0.5)
    
    # Performance lines
    line1 = ax2.plot(range(len(gradient_test_loss)), gradient_test_loss, '#d62728', 
                    linewidth=2.5, marker='o', markersize=5, label='Gradient VOI Test Loss')
    line2 = ax2.plot(range(len(random_test_loss)), random_test_loss, '#2ca02c', 
                    linewidth=2.5, marker='s', markersize=5, label='Random Test Loss')
    
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Clean Selections Count', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax1.set_title('Clean Selections vs Test Loss Performance', fontsize=14, fontweight='bold')
    
    ax1.set_xticks(cycles)
    ax1.set_xticklabels([str(i) for i in cycles])
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_learning_curve_percent(gradient_results, random_results, max_cycles=None, save_path=None):
    """Plot learning curve showing percentage of clean selections"""
    gradient_data, _, _ = extract_noise_selections_per_cycle(gradient_results, 'llm', max_cycles)
    random_data, _, _ = extract_noise_selections_per_cycle(random_results, 'llm', max_cycles)
    
    def moving_average(data, window=3):
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    min_cycles = min(len(gradient_data), len(random_data))
    
    # Calculate percentage of clean selections (LLM + Human)
    gradient_clean_pct = []
    for cycle_idx in range(min_cycles):
        breakdown = gradient_results['selection_breakdown_per_cycle'][cycle_idx]
        total_selections = sum(breakdown.values())
        clean_selections = breakdown.get('original_llm', 0) + breakdown.get('original_human', 0)
        clean_pct = (clean_selections / total_selections * 100) if total_selections > 0 else 0
        gradient_clean_pct.append(clean_pct)
    
    random_clean_pct = []
    for cycle_idx in range(min_cycles):
        breakdown = random_results['selection_breakdown_per_cycle'][cycle_idx]
        total_selections = sum(breakdown.values())
        clean_selections = breakdown.get('original_llm', 0) + breakdown.get('original_human', 0)
        clean_pct = (clean_selections / total_selections * 100) if total_selections > 0 else 0
        random_clean_pct.append(clean_pct)
    
    gradient_ma = moving_average(gradient_clean_pct)
    random_ma = moving_average(random_clean_pct)
    
    plt.figure(figsize=(10, 6))
    
    cycles = range(len(gradient_ma))
    plt.plot(cycles, gradient_ma, '#1f77b4', linewidth=2.5, marker='o', markersize=5, label='Gradient VOI')
    plt.plot(cycles, random_ma, '#ff7f0e', linewidth=2.5, marker='s', markersize=5, label='Random')
    
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('% Clean Selections (3-cycle MA)', fontsize=12)
    plt.title('Learning Curve: Clean Selection Percentage', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_validation_metrics(gradient_results, random_results, max_cycles=None, save_path=None):
    """Plot validation metrics over time"""
    gradient_val_metrics = gradient_results.get('val_metrics', [])
    random_val_metrics = random_results.get('val_metrics', [])
    
    if max_cycles:
        gradient_val_metrics = gradient_val_metrics[:max_cycles+1]  # +1 because cycle 0 exists
        random_val_metrics = random_val_metrics[:max_cycles+1]
    
    min_cycles = min(len(gradient_val_metrics), len(random_val_metrics))
    cycles = range(min_cycles)
    
    # Extract specific metrics
    metrics_to_plot = ['rmse', 'pearson', 'spearman']
    metric_labels = ['RMSE', 'Pearson Correlation', 'Spearman Correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        gradient_metric = [m.get(metric, 0) for m in gradient_val_metrics]
        random_metric = [m.get(metric, 0) for m in random_val_metrics]
        
        axes[i].plot(cycles, gradient_metric, '#1f77b4', linewidth=2.5, marker='o', 
                    markersize=5, label='Gradient VOI')
        axes[i].plot(cycles, random_metric, '#ff7f0e', linewidth=2.5, marker='s', 
                    markersize=5, label='Random')
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Cycle', fontsize=10)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    # Plot validation loss
    gradient_val_loss = gradient_results.get('val_losses', [])[:min_cycles]
    random_val_loss = random_results.get('val_losses', [])[:min_cycles]
    
    axes[3].plot(cycles, gradient_val_loss, '#1f77b4', linewidth=2.5, marker='o', 
                markersize=5, label='Gradient VOI')
    axes[3].plot(cycles, random_val_loss, '#ff7f0e', linewidth=2.5, marker='s', 
                markersize=5, label='Random')
    axes[3].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Cycle', fontsize=10)
    axes[3].set_ylabel('Loss', fontsize=10)
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_question_selection_patterns(results, dataset, max_cycles=None):
    """Analyze question selection patterns for clean selections"""
    observation_history = results.get('observation_history', [])
    
    # Track clean selections by question
    question_counts = defaultdict(int)
    total_clean_selections = 0
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx >= len(dataset):
            continue
            
        entry = dataset[example_idx]
        if position_idx >= len(entry.get('annotators', [])):
            continue
        
        # Check if this is a clean (original) selection
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['annotators']))[position_idx]
        if noise_type == 'original':
            question_idx = entry['questions'][position_idx]
            question_counts[question_idx] += 1
            total_clean_selections += 1
    
    return question_counts, total_clean_selections

def plot_question_selection_patterns(gradient_results, random_results, dataset, max_cycles=None, save_path=None):
    """Plot question selection patterns for clean selections"""
    grad_counts, grad_total = analyze_question_selection_patterns(gradient_results, dataset, max_cycles)
    rand_counts, rand_total = analyze_question_selection_patterns(random_results, dataset, max_cycles)
    
    # Get all questions (0-6 for HANNA)
    all_questions = list(range(7))
    
    grad_percentages = [(grad_counts[q] / grad_total * 100) if grad_total > 0 else 0 for q in all_questions]
    rand_percentages = [(rand_counts[q] / rand_total * 100) if rand_total > 0 else 0 for q in all_questions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gradient VOI
    bars1 = ax1.bar(all_questions, grad_percentages, alpha=0.7, color='#1f77b4', 
                   edgecolor='black', linewidth=0.5)
    ax1.set_title('Question Selection Distribution\n(Gradient VOI - Clean Selections)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Question Index', fontsize=11)
    ax1.set_ylabel('% of Clean Selections', fontsize=11)
    ax1.set_xticks(all_questions)
    ax1.set_xticklabels([f'Q{i}' for i in all_questions])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars1, grad_percentages)):
        if pct > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Random
    bars2 = ax2.bar(all_questions, rand_percentages, alpha=0.7, color='#ff7f0e', 
                   edgecolor='black', linewidth=0.5)
    ax2.set_title('Question Selection Distribution\n(Random - Clean Selections)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Question Index', fontsize=11)
    ax2.set_ylabel('% of Clean Selections', fontsize=11)
    ax2.set_xticks(all_questions)
    ax2.set_xticklabels([f'Q{i}' for i in all_questions])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars2, rand_percentages)):
        if pct > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main(max_cycles=15):
    gradient_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_both/multilevel_noisy_gradient_voi_with_embedding.json"
    random_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_both/multilevel_noisy_random_random_with_embedding.json"
    dataset_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/data/active_pool.json"
    
    output_dir = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    gradient_results, random_results, dataset = load_data(gradient_path, random_path, dataset_path)
    
    # 1. LLM Noise Dynamics (Gradient VOI only)
    plot_noise_dynamics_separate(gradient_results, 'llm', max_cycles, 
                                os.path.join(output_dir, "llm_noise_dynamics_gradient.png"))
    
    # 2. Human Noise Dynamics (Gradient VOI only) 
    plot_noise_dynamics_separate(gradient_results, 'human', max_cycles,
                                os.path.join(output_dir, "human_noise_dynamics_gradient.png"))
    
    # 3. Clean selections vs performance
    plot_clean_selections_vs_performance(gradient_results, random_results, max_cycles,
                                       os.path.join(output_dir, "clean_vs_performance.png"))
    
    # 4. Learning curve (percentage)
    plot_learning_curve_percent(gradient_results, random_results, max_cycles,
                               os.path.join(output_dir, "learning_curve_percent.png"))
    
    # 5. Validation metrics
    plot_validation_metrics(gradient_results, random_results, max_cycles,
                           os.path.join(output_dir, "validation_metrics.png"))
    
    # 6. Question selection patterns
    plot_question_selection_patterns(gradient_results, random_results, dataset, max_cycles,
                                   os.path.join(output_dir, "question_selection_patterns.png"))
    
    print(f"All plots saved to {output_dir}")
    print(f"Plots limited to first {max_cycles} cycles")

if __name__ == "__main__":
    main(max_cycles=9)  # Change this parameter as needed