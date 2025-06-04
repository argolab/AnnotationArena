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

def extract_llm_selections_per_cycle(results):
    cycles = len(results.get('selection_breakdown_per_cycle', []))
    llm_categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy']
    
    cycle_data = []
    for cycle_idx in range(cycles):
        if cycle_idx < len(results['selection_breakdown_per_cycle']):
            breakdown = results['selection_breakdown_per_cycle'][cycle_idx]
            total_llm = sum(breakdown.get(cat, 0) for cat in llm_categories)
            if total_llm > 0:
                percentages = {cat: (breakdown.get(cat, 0) / total_llm) * 100 for cat in llm_categories}
            else:
                percentages = {cat: 0 for cat in llm_categories}
            cycle_data.append(percentages)
        else:
            cycle_data.append({cat: 0 for cat in llm_categories})
    
    return cycle_data

def parse_variable_id(variable_id):
    parts = variable_id.split('_')
    example_idx = int(parts[1])
    position_idx = int(parts[3])
    return example_idx, position_idx

def analyze_argmax_for_gradient(results, dataset):
    observation_history = results.get('observation_history', [])
    
    same_argmax_count = 0
    different_argmax_count = 0
    argmax_distances = []
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx >= len(dataset):
            continue
            
        entry = dataset[example_idx]
        if position_idx >= len(entry.get('annotators', [])):
            continue
            
        is_llm = entry['annotators'][position_idx] == -1
        if not is_llm:
            continue
            
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['annotators']))[position_idx]
        if noise_type == 'original':
            continue
            
        noisy_answer = entry['answers'][position_idx]
        true_answer = entry['true_answers'][position_idx]
        
        noisy_argmax = np.argmax(noisy_answer)
        true_argmax = np.argmax(true_answer)
        
        if noisy_argmax == true_argmax:
            same_argmax_count += 1
        else:
            different_argmax_count += 1
            distance = abs(noisy_argmax - true_argmax)
            argmax_distances.append(distance)
    
    total_noisy_llm = same_argmax_count + different_argmax_count
    same_percentage = (same_argmax_count / total_noisy_llm * 100) if total_noisy_llm > 0 else 0
    
    return same_percentage, argmax_distances

def plot_llm_noise_dynamics(gradient_data, random_data, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    min_cycles = min(len(gradient_data), len(random_data))
    cycles = range(min_cycles)
    categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy']
    colors = ['green', 'orange', 'red', 'darkred']
    labels = ['Original', 'Low Noise', 'Medium Noise', 'Heavy Noise']
    
    for i, (cat, color, label) in enumerate(zip(categories, colors, labels)):
        gradient_values = [gradient_data[c].get(cat, 0) for c in cycles]
        ax1.plot(cycles, gradient_values, color=color, label=label, linewidth=2)
    
    ax1.set_title('Gradient VOI')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Percentage of LLM Selections')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (cat, color, label) in enumerate(zip(categories, colors, labels)):
        random_values = [random_data[c].get(cat, 0) for c in cycles]
        ax2.plot(cycles, random_values, color=color, label=label, linewidth=2)
    
    ax2.set_title('Random')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Percentage of LLM Selections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_clean_selections_vs_performance(gradient_results, random_results, save_path):
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    
    gradient_cycles = len(gradient_results.get('selection_breakdown_per_cycle', []))
    random_cycles = len(random_results.get('selection_breakdown_per_cycle', []))
    min_cycles = min(gradient_cycles, random_cycles)
    
    gradient_clean = []
    gradient_test_loss = gradient_results.get('test_annotated_losses', [])[:min_cycles]
    
    for cycle_idx in range(min_cycles):
        breakdown = gradient_results['selection_breakdown_per_cycle'][cycle_idx]
        clean_count = breakdown.get('original_llm', 0)
        gradient_clean.append(clean_count)
    
    random_clean = []
    random_test_loss = random_results.get('test_annotated_losses', [])[:min_cycles]
    
    for cycle_idx in range(min_cycles):
        breakdown = random_results['selection_breakdown_per_cycle'][cycle_idx]
        clean_count = breakdown.get('original_llm', 0)
        random_clean.append(clean_count)
    
    cycles = np.arange(min_cycles)
    width = 0.35
    
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(cycles - width/2, gradient_clean, width, alpha=0.5, color='blue', label='Gradient VOI Clean Selections')
    bars2 = ax1.bar(cycles + width/2, random_clean, width, alpha=0.5, color='orange', label='Random Clean Selections')
    
    line1 = ax2.plot(range(len(gradient_test_loss)), gradient_test_loss, 'b-', linewidth=1, marker='o', markersize=4, label='Gradient VOI Test Loss')
    line2 = ax2.plot(range(len(random_test_loss)), random_test_loss, 'r-', linewidth=1, marker='s', markersize=4, label='Random Test Loss')
    
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Clean LLM Selections')
    ax2.set_ylabel('Test Loss')
    ax1.set_title('Clean LLM Selections vs Test Loss Performance')
    
    ax1.set_xticks(cycles)
    ax1.set_xticklabels([str(i) for i in cycles])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_argmax_analysis(same_percentage, argmax_distances, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(['Same Argmax', 'Different Argmax'], [same_percentage, 100 - same_percentage], 
            color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Percentage')
    ax1.set_title('Argmax Comparison for Noisy LLM Selections')
    ax1.set_ylim(0, 100)
    
    if argmax_distances:
        ax2.hist(argmax_distances, bins=range(1, 6), alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Argmax Distance')
        ax2.set_ylabel('Count')
        ax2.set_title('Distance When Argmax Differs')
        ax2.set_xticks(range(1, 5))
    else:
        ax2.text(0.5, 0.5, 'No different argmax cases', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distance When Argmax Differs')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curve(gradient_data, random_data, save_path):
    def moving_average(data, window=3):
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    min_cycles = min(len(gradient_data), len(random_data))
    
    gradient_clean_pct = []
    for cycle_idx in range(min_cycles - 1):
        cycle = gradient_data[cycle_idx]
        total_llm = sum(cycle.get(cat, 0) for cat in ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy'])
        if total_llm > 0:
            clean_pct = cycle.get('original_llm', 0)
        else:
            clean_pct = 0
        gradient_clean_pct.append(clean_pct)
    
    random_clean_pct = []
    for cycle_idx in range(min_cycles - 1):
        cycle = random_data[cycle_idx]
        total_llm = sum(cycle.get(cat, 0) for cat in ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy'])
        if total_llm > 0:
            clean_pct = cycle.get('original_llm', 0)
        else:
            clean_pct = 0
        random_clean_pct.append(clean_pct)
    
    gradient_ma = moving_average(gradient_clean_pct)
    random_ma = moving_average(random_clean_pct)
    
    plt.figure(figsize=(8, 5))
    
    cycles = range(len(gradient_ma))
    plt.plot(cycles, gradient_ma, 'b-', linewidth=2, label='Gradient VOI')
    plt.plot(cycles, random_ma, 'r-', linewidth=2, label='Random')
    
    plt.xlabel('Cycle')
    plt.ylabel('Clean LLM Selections (3-cycle MA)')
    plt.title('Learning Curve for Clean Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_selection_heatmap(gradient_results, random_results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    llm_categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy']
    category_labels = ['Original', 'Low Noise', 'Medium Noise', 'Heavy Noise']
    
    gradient_cycles = len(gradient_results.get('selection_breakdown_per_cycle', []))
    random_cycles = len(random_results.get('selection_breakdown_per_cycle', []))
    min_cycles = min(gradient_cycles, random_cycles)
    
    gradient_matrix = np.zeros((len(llm_categories), min_cycles - 1))
    
    for cycle_idx in range(min_cycles - 1):
        breakdown = gradient_results['selection_breakdown_per_cycle'][cycle_idx]
        for cat_idx, cat in enumerate(llm_categories):
            gradient_matrix[cat_idx, cycle_idx] = breakdown.get(cat, 0)
    
    random_matrix = np.zeros((len(llm_categories), min_cycles - 1))
    
    for cycle_idx in range(min_cycles - 1):
        breakdown = random_results['selection_breakdown_per_cycle'][cycle_idx]
        for cat_idx, cat in enumerate(llm_categories):
            random_matrix[cat_idx, cycle_idx] = breakdown.get(cat, 0)
    
    im1 = ax1.imshow(gradient_matrix, cmap='Reds', aspect='auto')
    ax1.set_title('Gradient VOI')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Noise Level')
    ax1.set_yticks(range(len(category_labels)))
    ax1.set_yticklabels(category_labels)
    plt.colorbar(im1, ax=ax1, label='Selection Count')
    
    im2 = ax2.imshow(random_matrix, cmap='Reds', aspect='auto')
    ax2.set_title('Random')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Noise Level')
    ax2.set_yticks(range(len(category_labels)))
    ax2.set_yticklabels(category_labels)
    plt.colorbar(im2, ax=ax2, label='Selection Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def detailed_argmax_analysis(results, dataset):
    observation_history = results.get('observation_history', [])
    
    noise_categories = ['llm_low', 'llm_medium', 'llm_heavy']
    analysis = {}
    
    for category in noise_categories:
        analysis[category] = {
            'same_argmax': 0,
            'different_argmax': 0,
            'distances': [],
            'examples': []
        }
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx >= len(dataset):
            continue
            
        entry = dataset[example_idx]
        if position_idx >= len(entry.get('annotators', [])):
            continue
            
        is_llm = entry['annotators'][position_idx] == -1
        if not is_llm:
            continue
            
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['annotators']))[position_idx]
        if noise_type not in noise_categories:
            continue
            
        noisy_answer = entry['answers'][position_idx]
        true_answer = entry['true_answers'][position_idx]
        
        noisy_argmax = np.argmax(noisy_answer)
        true_argmax = np.argmax(true_answer)
        
        example_data = {
            'example_idx': example_idx,
            'position_idx': position_idx,
            'noisy_answer': noisy_answer,
            'true_answer': true_answer,
            'noisy_argmax': noisy_argmax,
            'true_argmax': true_argmax,
            'question': entry['questions'][position_idx]
        }
        
        if noisy_argmax == true_argmax:
            analysis[noise_type]['same_argmax'] += 1
            if len(analysis[noise_type]['examples']) < 3:
                analysis[noise_type]['examples'].append(example_data)
        else:
            analysis[noise_type]['different_argmax'] += 1
            distance = abs(noisy_argmax - true_argmax)
            analysis[noise_type]['distances'].append(distance)
            if len(analysis[noise_type]['examples']) < 6:
                analysis[noise_type]['examples'].append(example_data)
    
    return analysis

def print_detailed_argmax_analysis(analysis):
    print("\n" + "="*80)
    print("DETAILED ARGMAX ANALYSIS BY NOISE LEVEL")
    print("="*80)
    
    for category in ['llm_low', 'llm_medium', 'llm_heavy']:
        data = analysis[category]
        total = data['same_argmax'] + data['different_argmax']
        
        if total == 0:
            continue
            
        same_pct = (data['same_argmax'] / total) * 100
        
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Total selections: {total}")
        print(f"  Same argmax: {data['same_argmax']} ({same_pct:.1f}%)")
        print(f"  Different argmax: {data['different_argmax']} ({100-same_pct:.1f}%)")
        
        if data['distances']:
            avg_distance = np.mean(data['distances'])
            print(f"  Average distance when different: {avg_distance:.2f}")
            distance_counts = {}
            for d in data['distances']:
                distance_counts[d] = distance_counts.get(d, 0) + 1
            print(f"  Distance distribution: {distance_counts}")
        
        print(f"\n  Examples (first 6):")
        for i, example in enumerate(data['examples'][:6]):
            status = "SAME" if example['noisy_argmax'] == example['true_argmax'] else "DIFF"
            print(f"    {i+1}. [{status}] Q{example['question']} | "
                  f"Noisy argmax: {example['noisy_argmax']} | True argmax: {example['true_argmax']}")
            print(f"       Noisy dist: {[f'{x:.3f}' for x in example['noisy_answer']]}")
            print(f"       True dist:  {[f'{x:.3f}' for x in example['true_answer']]}")

def plot_detailed_argmax_comparison(analysis, save_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    categories = ['llm_low', 'llm_medium', 'llm_heavy']
    category_labels = ['Low Noise', 'Medium Noise', 'Heavy Noise']
    colors = ['orange', 'red', 'darkred']
    
    same_percentages = []
    total_counts = []
    
    for category in categories:
        data = analysis[category]
        total = data['same_argmax'] + data['different_argmax']
        total_counts.append(total)
        
        if total > 0:
            same_pct = (data['same_argmax'] / total) * 100
            same_percentages.append(same_pct)
        else:
            same_percentages.append(0)
    
    ax1.bar(category_labels, same_percentages, color=colors, alpha=0.7)
    ax1.set_ylabel('Percentage Same Argmax')
    ax1.set_title('Same Argmax Rate by Noise Level')
    ax1.set_ylim(0, 100)
    
    for i, (pct, count) in enumerate(zip(same_percentages, total_counts)):
        ax1.text(i, pct + 2, f'{pct:.1f}%\n(n={count})', ha='center', va='bottom')
    
    ax2.bar(category_labels, total_counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Selections')
    ax2.set_title('Number of Selections by Noise Level')
    
    all_distances = []
    distance_labels = []
    distance_colors = []
    
    for i, category in enumerate(categories):
        distances = analysis[category]['distances']
        if distances:
            all_distances.extend(distances)
            distance_labels.extend([category_labels[i]] * len(distances))
            distance_colors.extend([colors[i]] * len(distances))
    
    if all_distances:
        unique_distances = sorted(set(all_distances))
        distance_counts = {cat: {d: 0 for d in unique_distances} for cat in category_labels}
        
        for dist, label in zip(all_distances, distance_labels):
            distance_counts[label][dist] += 1
        
        x = np.arange(len(unique_distances))
        width = 0.25
        
        for i, (cat_label, color) in enumerate(zip(category_labels, colors)):
            counts = [distance_counts[cat_label][d] for d in unique_distances]
            ax3.bar(x + i*width, counts, width, label=cat_label, color=color, alpha=0.7)
        
        ax3.set_xlabel('Argmax Distance')
        ax3.set_ylabel('Count')
        ax3.set_title('Distance Distribution When Argmax Differs')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(unique_distances)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No different argmax cases', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Distance Distribution When Argmax Differs')
    
    avg_distances = []
    for category in categories:
        distances = analysis[category]['distances']
        if distances:
            avg_distances.append(np.mean(distances))
        else:
            avg_distances.append(0)
    
    ax4.bar(category_labels, avg_distances, color=colors, alpha=0.7)
    ax4.set_ylabel('Average Distance')
    ax4.set_title('Average Argmax Distance When Different')
    
    for i, avg_dist in enumerate(avg_distances):
        if avg_dist > 0:
            ax4.text(i, avg_dist + 0.05, f'{avg_dist:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_argmax_for_human(results, dataset):
    observation_history = results.get('observation_history', [])
    
    same_argmax_count = 0
    different_argmax_count = 0
    argmax_distances = []
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx >= len(dataset):
            continue
            
        entry = dataset[example_idx]
        if position_idx >= len(entry.get('annotators', [])):
            continue
            
        is_llm = entry['annotators'][position_idx] == -1
        if is_llm:
            continue
            
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['annotators']))[position_idx]
        if noise_type == 'original':
            continue
            
        noisy_answer = entry['answers'][position_idx]
        true_answer = entry['true_answers'][position_idx]
        
        noisy_argmax = np.argmax(noisy_answer)
        true_argmax = np.argmax(true_answer)
        
        if noisy_argmax == true_argmax:
            same_argmax_count += 1
        else:
            different_argmax_count += 1
            distance = abs(noisy_argmax - true_argmax)
            argmax_distances.append(distance)
    
    total_noisy_human = same_argmax_count + different_argmax_count
    same_percentage = (same_argmax_count / total_noisy_human * 100) if total_noisy_human > 0 else 0
    
    return same_percentage, argmax_distances

def detailed_argmax_analysis_human(results, dataset):
    observation_history = results.get('observation_history', [])
    
    noise_categories = ['human_low', 'human_medium', 'human_heavy']
    analysis = {}
    
    for category in noise_categories:
        analysis[category] = {
            'same_argmax': 0,
            'different_argmax': 0,
            'distances': [],
            'examples': []
        }
    
    for obs in observation_history:
        variable_id = obs['variable_id']
        example_idx, position_idx = parse_variable_id(variable_id)
        
        if example_idx >= len(dataset):
            continue
            
        entry = dataset[example_idx]
        if position_idx >= len(entry.get('annotators', [])):
            continue
            
        is_llm = entry['annotators'][position_idx] == -1
        if is_llm:
            continue
            
        noise_type = entry.get('noise_info', ['unknown'] * len(entry['annotators']))[position_idx]
        if noise_type not in noise_categories:
            continue
            
        noisy_answer = entry['answers'][position_idx]
        true_answer = entry['true_answers'][position_idx]
        
        noisy_argmax = np.argmax(noisy_answer)
        true_argmax = np.argmax(true_answer)
        
        example_data = {
            'example_idx': example_idx,
            'position_idx': position_idx,
            'noisy_answer': noisy_answer,
            'true_answer': true_answer,
            'noisy_argmax': noisy_argmax,
            'true_argmax': true_argmax,
            'question': entry['questions'][position_idx],
            'annotator': entry['annotators'][position_idx]
        }
        
        if noisy_argmax == true_argmax:
            analysis[noise_type]['same_argmax'] += 1
            if len(analysis[noise_type]['examples']) < 3:
                analysis[noise_type]['examples'].append(example_data)
        else:
            analysis[noise_type]['different_argmax'] += 1
            distance = abs(noisy_argmax - true_argmax)
            analysis[noise_type]['distances'].append(distance)
            if len(analysis[noise_type]['examples']) < 6:
                analysis[noise_type]['examples'].append(example_data)
    
    return analysis

def print_detailed_argmax_analysis_human(analysis):
    print("\n" + "="*80)
    print("DETAILED HUMAN ARGMAX ANALYSIS BY NOISE LEVEL")
    print("="*80)
    
    for category in ['human_low', 'human_medium', 'human_heavy']:
        data = analysis[category]
        total = data['same_argmax'] + data['different_argmax']
        
        if total == 0:
            continue
            
        same_pct = (data['same_argmax'] / total) * 100
        
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Total selections: {total}")
        print(f"  Same argmax: {data['same_argmax']} ({same_pct:.1f}%)")
        print(f"  Different argmax: {data['different_argmax']} ({100-same_pct:.1f}%)")
        
        if data['distances']:
            avg_distance = np.mean(data['distances'])
            print(f"  Average distance when different: {avg_distance:.2f}")
            distance_counts = {}
            for d in data['distances']:
                distance_counts[d] = distance_counts.get(d, 0) + 1
            print(f"  Distance distribution: {distance_counts}")
        
        print(f"\n  Examples (first 6):")
        for i, example in enumerate(data['examples'][:6]):
            status = "SAME" if example['noisy_argmax'] == example['true_argmax'] else "DIFF"
            print(f"    {i+1}. [{status}] Q{example['question']} Annotator{example['annotator']} | "
                  f"Noisy argmax: {example['noisy_argmax']} | True argmax: {example['true_argmax']}")
            print(f"       Noisy dist: {[f'{x:.3f}' for x in example['noisy_answer']]}")
            print(f"       True dist:  {[f'{x:.3f}' for x in example['true_answer']]}")

def main():
    gradient_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_both/multilevel_noisy_gradient_voi_with_embedding.json"
    random_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/experiment_both/multilevel_noisy_random_random_with_embedding.json"
    dataset_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/data/active_pool.json"
    
    output_dir = "/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_multilevel_noisy_hanna/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    gradient_results, random_results, dataset = load_data(gradient_path, random_path, dataset_path)
    
    gradient_llm_data = extract_llm_selections_per_cycle(gradient_results)
    random_llm_data = extract_llm_selections_per_cycle(random_results)
    
    plot_llm_noise_dynamics(gradient_llm_data, random_llm_data, 
                           os.path.join(output_dir, "llm_noise_dynamics.png"))
    
    plot_clean_selections_vs_performance(gradient_results, random_results,
                                       os.path.join(output_dir, "clean_vs_performance.png"))
    
    same_pct_llm, distances_llm = analyze_argmax_for_gradient(gradient_results, dataset)
    plot_argmax_analysis(same_pct_llm, distances_llm,
                        os.path.join(output_dir, "argmax_analysis.png"))
    
    plot_learning_curve(gradient_llm_data, random_llm_data,
                       os.path.join(output_dir, "learning_curve.png"))
    
    plot_selection_heatmap(gradient_results, random_results,
                          os.path.join(output_dir, "selection_heatmap.png"))
    
    print(f"All plots saved to {output_dir}")
    print(f"LLM Argmax analysis: {same_pct_llm:.1f}% of noisy LLM selections had same argmax as clean")
    print(f"LLM Distance distribution when different: {len(distances_llm)} cases")

    detailed_analysis_llm = detailed_argmax_analysis(gradient_results, dataset)
    print_detailed_argmax_analysis(detailed_analysis_llm)
    plot_detailed_argmax_comparison(detailed_analysis_llm,
                                   os.path.join(output_dir, "detailed_argmax_analysis.png"))
    
    same_pct_human, distances_human = analyze_argmax_for_human(gradient_results, dataset)
    print(f"\nHuman Argmax analysis: {same_pct_human:.1f}% of noisy human selections had same argmax as clean")
    print(f"Human Distance distribution when different: {len(distances_human)} cases")
    
    detailed_analysis_human = detailed_argmax_analysis_human(gradient_results, dataset)
    print_detailed_argmax_analysis_human(detailed_analysis_human)

if __name__ == "__main__":
    main()