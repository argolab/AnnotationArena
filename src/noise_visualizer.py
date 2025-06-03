"""
Analysis script for multi-level noise experiments comparing gradient_voi vs random_random.
Tracks effective noise detection based on argmax differences and KL divergence.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_experiment_data(result_path, dataset_path):
    """Load experiment results and dataset."""
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

def compute_kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two probability distributions."""
    p = np.array(p) + eps
    q = np.array(q) + eps
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

def is_effectively_noisy(answers, true_answers, is_llm, kl_threshold=0.1):
    """
    Determine if a variable is effectively noisy.
    For humans: argmax differs
    For LLMs: argmax differs OR KL divergence > threshold
    """
    answers = np.array(answers)
    true_answers = np.array(true_answers)
    
    # Check argmax difference
    argmax_differs = np.argmax(answers) != np.argmax(true_answers)
    
    if not is_llm:
        # For humans, only check argmax
        return argmax_differs
    else:
        # For LLMs, check both argmax and KL divergence
        if argmax_differs:
            return True
        
        kl_div = compute_kl_divergence(true_answers, answers)
        return kl_div > kl_threshold

def analyze_selections_per_cycle(results, dataset, use_effective_detection=True):
    """Analyze selections per cycle with optional effective noise detection."""
    observation_history = results.get('observation_history', [])
    features_per_cycle = results.get('features_annotated', [])
    
    if not features_per_cycle:
        return {}
    
    # Create dataset lookup
    dataset_map = {i: entry for i, entry in enumerate(dataset)}
    
    # Group selections by cycle
    cycle_data = []
    current_cycle = 0
    cycle_selections = []
    cumulative_selections = 0
    
    for obs in observation_history:
        # Determine cycle based on cumulative features
        while (current_cycle < len(features_per_cycle) and 
               cumulative_selections + len(cycle_selections) >= sum(features_per_cycle[:current_cycle+1])):
            # Process current cycle
            if cycle_selections:
                cycle_data.append(process_cycle_selections(cycle_selections, dataset_map, current_cycle, use_effective_detection))
            
            cumulative_selections += len(cycle_selections)
            cycle_selections = []
            current_cycle += 1
        
        cycle_selections.append(obs)
    
    # Process final cycle
    if cycle_selections:
        cycle_data.append(process_cycle_selections(cycle_selections, dataset_map, current_cycle, use_effective_detection))
    
    return {
        'cycle_data': cycle_data,
        'total_cycles': len(cycle_data)
    }

def process_cycle_selections(cycle_selections, dataset_map, cycle_num, use_effective_detection=True):
    """Process selections for a single cycle."""
    cycle_stats = {
        'cycle': cycle_num,
        'total_selections': len(cycle_selections),
        'categories': defaultdict(int),
        'human_categories': defaultdict(int),
        'llm_categories': defaultdict(int)
    }
    
    for obs in cycle_selections:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx not in dataset_map:
            continue
            
        entry = dataset_map[example_idx]
        
        if position_idx >= len(entry.get('answers', [])):
            continue
        
        # Get variable information
        answers = entry['answers'][position_idx]
        true_answers = entry.get('true_answers', [answers] * len(entry['answers']))[position_idx]
        noise_label = entry.get('noise_info', ['unknown'] * len(entry['answers']))[position_idx]
        is_llm = entry['annotators'][position_idx] == -1
        
        if use_effective_detection:
            # Use effective noise detection (for gradient_voi)
            effectively_noisy = is_effectively_noisy(answers, true_answers, is_llm)
            
            if effectively_noisy:
                # Extract noise level from label
                if 'low' in noise_label:
                    category = 'low'
                elif 'medium' in noise_label:
                    category = 'medium'
                elif 'heavy' in noise_label:
                    category = 'heavy'
                else:
                    category = 'other_noisy'
            else:
                category = 'original'
        else:
            # Use basic label counting (for random)
            if noise_label == 'original':
                category = 'original'
            elif 'low' in noise_label:
                category = 'low'
            elif 'medium' in noise_label:
                category = 'medium'
            elif 'heavy' in noise_label:
                category = 'heavy'
            else:
                category = 'other'
        
        # Update counts
        cycle_stats['categories'][category] += 1
        
        if is_llm:
            cycle_stats['llm_categories'][category] += 1
        else:
            cycle_stats['human_categories'][category] += 1
    
    return cycle_stats

def plot_learning_curves(gradient_results, random_results, save_path):
    """Plot test loss learning curves."""
    plt.figure(figsize=(8, 6))
    
    gradient_losses = gradient_results.get('test_annotated_losses', [])
    random_losses = random_results.get('test_annotated_losses', [])
    
    max_cycles = max(len(gradient_losses), len(random_losses))
    cycles = list(range(max_cycles))
    
    if gradient_losses:
        plt.plot(cycles[:len(gradient_losses)], gradient_losses, 
                'b-', linewidth=2, label='Gradient VOI')
    
    if random_losses:
        plt.plot(cycles[:len(random_losses)], random_losses, 
                'r-', linewidth=2, label='Random')
    
    plt.xlabel('Cycle')
    plt.ylabel('Test Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")

def plot_noise_selections_over_time(gradient_analysis, random_analysis, save_path):
    """Plot effective noise selections over cycles."""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    gradient_cycles = gradient_analysis.get('cycle_data', [])
    random_cycles = random_analysis.get('cycle_data', [])
    
    max_cycles = max(len(gradient_cycles), len(random_cycles))
    cycles = list(range(max_cycles))
    
    # Calculate cumulative noisy selections
    gradient_noisy = []
    random_noisy = []
    
    gradient_cum = 0
    random_cum = 0
    
    for i in range(max_cycles):
        if i < len(gradient_cycles):
            cycle_noisy = sum(count for cat, count in gradient_cycles[i]['categories'].items() 
                            if cat != 'original')
            gradient_cum += cycle_noisy
        gradient_noisy.append(gradient_cum)
        
        if i < len(random_cycles):
            cycle_noisy = sum(count for cat, count in random_cycles[i]['categories'].items() 
                            if cat != 'original')
            random_cum += cycle_noisy
        random_noisy.append(random_cum)
    
    plt.plot(cycles, gradient_noisy, 'b-', linewidth=2, label='Gradient VOI')
    plt.plot(cycles, random_noisy, 'r-', linewidth=2, label='Random')
    
    plt.xlabel('Cycle')
    plt.ylabel('Cumulative Noisy Selections')
    plt.title('Noisy Selection Over Time (VOI: Effective, Random: Labels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise selections over time to {save_path}")

def plot_total_selection_comparison(gradient_analysis, random_analysis, save_path):
    """Plot total selection counts by category."""
    plt.figure(figsize=(8, 6))
    
    # Calculate total counts
    categories = ['original', 'low', 'medium', 'heavy']
    
    gradient_totals = [0, 0, 0, 0]
    random_totals = [0, 0, 0, 0]
    
    for cycle_data in gradient_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            gradient_totals[i] += cycle_data['categories'][cat]
    
    for cycle_data in random_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            random_totals[i] += cycle_data['categories'][cat]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, gradient_totals, width, label='Gradient VOI', alpha=0.8)
    plt.bar(x + width/2, random_totals, width, label='Random', alpha=0.8)
    
    plt.xlabel('Effective Noise Category')
    plt.ylabel('Total Selections')
    plt.title('Selection Counts (VOI: Effective, Random: Labels)')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved total selection comparison to {save_path}")

def plot_noise_ratio_over_time(gradient_analysis, random_analysis, save_path):
    """Plot ratio of noisy selections over cycles."""
    plt.figure(figsize=(10, 6))
    
    gradient_cycles = gradient_analysis.get('cycle_data', [])
    random_cycles = random_analysis.get('cycle_data', [])
    
    max_cycles = max(len(gradient_cycles), len(random_cycles))
    cycles = list(range(max_cycles))
    
    gradient_ratios = []
    random_ratios = []
    
    for i in range(max_cycles):
        if i < len(gradient_cycles):
            total = gradient_cycles[i]['total_selections']
            noisy = sum(count for cat, count in gradient_cycles[i]['categories'].items() 
                       if cat != 'original')
            gradient_ratios.append(noisy / max(total, 1))
        else:
            gradient_ratios.append(0)
        
        if i < len(random_cycles):
            total = random_cycles[i]['total_selections']
            noisy = sum(count for cat, count in random_cycles[i]['categories'].items() 
                       if cat != 'original')
            random_ratios.append(noisy / max(total, 1))
        else:
            random_ratios.append(0)
    
    plt.plot(cycles, gradient_ratios, 'b-', linewidth=2, label='Gradient VOI')
    plt.plot(cycles, random_ratios, 'r-', linewidth=2, label='Random')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
    
    plt.xlabel('Cycle')
    plt.ylabel('Proportion of Noisy Selections')
    plt.title('Noise Ratio Over Time (VOI: Effective, Random: Labels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise ratio over time to {save_path}")

def plot_human_llm_breakdown(gradient_analysis, random_analysis, save_path):
    """Plot separate analysis for human vs LLM selections."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['original', 'low', 'medium', 'heavy']
    
    # Human breakdown
    gradient_human = [0, 0, 0, 0]
    random_human = [0, 0, 0, 0]
    
    for cycle_data in gradient_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            gradient_human[i] += cycle_data['human_categories'][cat]
    
    for cycle_data in random_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            random_human[i] += cycle_data['human_categories'][cat]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, gradient_human, width, label='Gradient VOI', alpha=0.8)
    ax1.bar(x + width/2, random_human, width, label='Random', alpha=0.8)
    ax1.set_xlabel('Effective Noise Category')
    ax1.set_ylabel('Human Selections')
    ax1.set_title('Human Annotations (VOI: Effective, Random: Labels)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # LLM breakdown
    gradient_llm = [0, 0, 0, 0]
    random_llm = [0, 0, 0, 0]
    
    for cycle_data in gradient_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            gradient_llm[i] += cycle_data['llm_categories'][cat]
    
    for cycle_data in random_analysis.get('cycle_data', []):
        for i, cat in enumerate(categories):
            random_llm[i] += cycle_data['llm_categories'][cat]
    
    ax2.bar(x - width/2, gradient_llm, width, label='Gradient VOI', alpha=0.8)
    ax2.bar(x + width/2, random_llm, width, label='Random', alpha=0.8)
    ax2.set_xlabel('Effective Noise Category')
    ax2.set_ylabel('LLM Selections')
    ax2.set_title('LLM Annotations (VOI: Effective, Random: Labels)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved human/LLM breakdown to {save_path}")

def generate_summary_report(gradient_analysis, random_analysis, gradient_results, random_results):
    """Generate summary statistics."""
    print("="*80)
    print("MULTI-LEVEL NOISE EXPERIMENT ANALYSIS")
    print("="*80)
    print("Note: Gradient VOI uses effective noise detection")
    print("      Random uses basic label counting")
    print("="*80)
    
    # Calculate totals
    gradient_totals = defaultdict(int)
    random_totals = defaultdict(int)
    
    for cycle_data in gradient_analysis.get('cycle_data', []):
        for cat, count in cycle_data['categories'].items():
            gradient_totals[cat] += count
    
    for cycle_data in random_analysis.get('cycle_data', []):
        for cat, count in cycle_data['categories'].items():
            random_totals[cat] += count
    
    print(f"\nSELECTION TOTALS:")
    print(f"{'Category':<15} {'Gradient VOI':<12} {'Random':<12}")
    print(f"{'(Effective)':<15} {'(Effective)':<12} {'(Labels)':<12}")
    print("-" * 45)
    for cat in ['original', 'low', 'medium', 'heavy']:
        print(f"{cat:<15} {gradient_totals[cat]:<12} {random_totals[cat]:<12}")
    
    gradient_total = sum(gradient_totals.values())
    random_total = sum(random_totals.values())
    
    gradient_clean_ratio = gradient_totals['original'] / max(gradient_total, 1)
    random_clean_ratio = random_totals['original'] / max(random_total, 1)
    
    print(f"\nCLEAN SELECTION RATIOS:")
    print(f"Gradient VOI (effective): {gradient_clean_ratio:.3f}")
    print(f"Random (labels):          {random_clean_ratio:.3f}")
    
    final_gradient_loss = gradient_results.get('test_annotated_losses', [0])[-1]
    final_random_loss = random_results.get('test_annotated_losses', [0])[-1]
    
    print(f"\nFINAL TEST LOSSES:")
    print(f"Gradient VOI: {final_gradient_loss:.4f}")
    print(f"Random:       {final_random_loss:.4f}")
    
    if final_random_loss > 0:
        improvement = ((final_random_loss - final_gradient_loss) / final_random_loss) * 100
        print(f"Improvement:  {improvement:+.1f}%")

def main():
    """Main analysis function."""
    
    # File paths
    base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_1"
    gradient_result_path = os.path.join(base_path, "multilevel_noisy_gradient_voi_with_embedding.json")
    random_result_path = os.path.join(base_path, "multilevel_noisy_random_random_with_embedding.json")
    dataset_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/data/active_pool.json"
    
    # Output directory
    output_dir = os.path.join(base_path, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    gradient_results, dataset = load_experiment_data(gradient_result_path, dataset_path)
    random_results, _ = load_experiment_data(random_result_path, dataset_path)
    
    print("Analyzing selections...")
    # Use effective detection for gradient_voi, basic labels for random
    gradient_analysis = analyze_selections_per_cycle(gradient_results, dataset, use_effective_detection=True)
    random_analysis = analyze_selections_per_cycle(random_results, dataset, use_effective_detection=False)
    
    print("Generating plots...")
    
    # Create individual plots
    plot_learning_curves(gradient_results, random_results,
                        os.path.join(output_dir, "learning_curves.png"))
    
    plot_noise_selections_over_time(gradient_analysis, random_analysis,
                                   os.path.join(output_dir, "noise_selections_over_time.png"))
    
    plot_total_selection_comparison(gradient_analysis, random_analysis,
                                   os.path.join(output_dir, "total_selection_comparison.png"))
    
    plot_noise_ratio_over_time(gradient_analysis, random_analysis,
                              os.path.join(output_dir, "noise_ratio_over_time.png"))
    
    plot_human_llm_breakdown(gradient_analysis, random_analysis,
                            os.path.join(output_dir, "human_llm_breakdown.png"))
    
    # Generate summary report
    generate_summary_report(gradient_analysis, random_analysis, 
                           gradient_results, random_results)
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}")

if __name__ == "__main__":
    main()