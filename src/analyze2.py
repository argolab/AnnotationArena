import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import torch
import glob
from collections import defaultdict
from tqdm import tqdm

# Path configuration - adjust these to your environment
base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
plots_path = os.path.join(base_path, "plots", "badge_gradient_comparison")

# Create output directory
os.makedirs(plots_path, exist_ok=True)

class AnnotationDataset:
    """Minimal loader for the dataset"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def get_data_entry(self, idx):
        return self.data[idx]
    
    def get_all_entries(self):
        return self.data

def parse_variable_id(variable_id):
    """Parse variable ID to extract example and position indices"""
    parts = variable_id.split('_')
    example_idx = int(parts[1])
    position_idx = int(parts[3])
    return example_idx, position_idx

def load_badge_gradient_results():
    """Load only badge_all and gradient_all result files"""
    strategy_results = {}
    
    # Define target strategies
    target_strategies = ["badge_all", "gradient_all"]
    
    for strategy in target_strategies:
        file_path = os.path.join(results_path, f"{strategy}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                strategy_results[strategy] = json.load(f)
                print(f"Loaded {strategy} results")
    
    return strategy_results

def extract_features(example_data):
    """
    Extract meaningful features from example data
    Returns numpy array of features
    """
    # Basic features from input, annotators, and questions
    inputs = np.array(example_data['input'])
    annotators = np.array(example_data['annotators'])
    questions = np.array(example_data['questions'])
    
    # Extract mask bits and answer distributions
    mask_bits = inputs[:, 0]
    answer_distributions = inputs[:, 1:]
    
    # Compute aggregated features
    features = []
    
    # Distribution statistics per question type
    question_types = np.unique(questions)
    for q_type in question_types:
        q_indices = np.where(questions == q_type)[0]
        
        if len(q_indices) > 0:
            # Add average distribution for this question type
            q_dists = answer_distributions[q_indices]
            avg_dist = np.mean(q_dists, axis=0)
            features.extend(avg_dist)
            
            # Add proportion of masked values for this question type
            masked_prop = np.mean(mask_bits[q_indices])
            features.append(masked_prop)
    
    # Add human vs LLM annotation proportions
    human_prop = np.mean(annotators >= 0)
    features.append(human_prop)
    
    # Add known questions proportion
    known_prop = np.mean(example_data['known_questions'])
    features.append(known_prop)
    
    return np.array(features)

def track_selections_by_cycle(strategy_results):
    """
    Track example selections by cycle for badge_all and gradient_all
    Returns dict mapping strategy -> cycle -> selected example indices
    """
    selections = {}
    
    for strategy, results in strategy_results.items():
        if 'observation_history' not in results:
            print(f"Warning: No observation history found for {strategy}")
            continue
        
        # Group observations by cycle
        cycle_selections = defaultdict(set)
        example_observations = defaultdict(list)
        
        # Extract selected examples from observation history
        for obs in results['observation_history']:
            example_idx, _ = parse_variable_id(obs['variable_id'])
            example_observations[example_idx].append(obs['timestamp'])
        
        # Map timestamps to cycles (estimate)
        features_per_cycle = results.get('features_annotated', [250])[0]
        
        for example_idx, timestamps in example_observations.items():
            # Use min timestamp to associate example with earliest cycle
            min_timestamp = min(timestamps)
            cycle = min(int(min_timestamp // features_per_cycle), len(results.get('examples_annotated', [])) - 1)
            cycle_selections[cycle].add(example_idx)
        
        selections[strategy] = dict(cycle_selections)
    
    return selections

def compute_cluster_metrics(feature_vectors, selections, n_clusters=3):
    """
    Compute cluster quality metrics for selected examples
    
    Args:
        feature_vectors: Feature vectors for all examples
        selections: List of selected example indices
        n_clusters: Number of clusters to use for metrics
        
    Returns:
        dict: Dictionary of cluster quality metrics
    """
    if len(selections) < n_clusters:
        return {
            "silhouette_score": 0,
            "calinski_harabasz_score": 0,
            "diversity_score": 0
        }
    
    # Extract features for selected examples
    selected_features = feature_vectors[selections]
    
    # Fit KMeans to determine clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(selected_features)
    
    # Compute silhouette score (higher is better, range -1 to 1)
    try:
        silhouette = silhouette_score(selected_features, cluster_labels)
    except:
        silhouette = 0
    
    # Compute Calinski-Harabasz Index (higher is better)
    try:
        calinski_harabasz = calinski_harabasz_score(selected_features, cluster_labels)
    except:
        calinski_harabasz = 0
    
    # Compute a diversity score (average distance to center)
    center = np.mean(selected_features, axis=0)
    distances = np.sqrt(np.sum((selected_features - center) ** 2, axis=1))
    diversity_score = np.mean(distances)
    
    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "diversity_score": diversity_score
    }

def create_side_by_side_tsne_animation(feature_vectors, selections_by_strategy, save_path):
    """
    Create animated GIF of side-by-side 2D t-SNE visualizations for badge_all and gradient_all
    
    Args:
        feature_vectors: Feature vectors for all examples
        selections_by_strategy: Dict mapping strategy -> cycle -> selected examples
        save_path: Path to save the animation
    """
    # Prepare t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # Reduce dimensionality
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Determine max cycles across both strategies
    max_cycles = 0
    for strategy, selections_by_cycle in selections_by_strategy.items():
        if selections_by_cycle:
            max_cycles = max(max_cycles, max(selections_by_cycle.keys()) + 1)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Set titles
        ax1.set_title(f'BADGE Strategy - Cycle {frame}', fontsize=16)
        ax2.set_title(f'Gradient Strategy - Cycle {frame}', fontsize=16)
        
        # Plot background points in both subplots
        ax1.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
        ax2.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
        
        # Plot selections for badge_all
        if 'badge_all' in selections_by_strategy:
            badge_selections = []
            for i in range(frame + 1):
                if i in selections_by_strategy['badge_all']:
                    cycle_selections = list(selections_by_strategy['badge_all'][i])
                    badge_selections.extend(cycle_selections)
                    
                    # Plot this cycle's selections
                    ax1.scatter(
                        embedded[cycle_selections, 0],
                        embedded[cycle_selections, 1],
                        c=f'C{i % 10}',
                        label=f'Cycle {i}',
                        s=80, alpha=0.8, edgecolors='black'
                    )
            
            # Compute and display metrics
            if badge_selections:
                metrics = compute_cluster_metrics(feature_vectors, badge_selections)
                ax1.text(
                    0.05, 0.95,
                    f"Diversity: {metrics['diversity_score']:.3f}\n"
                    f"Silhouette: {metrics['silhouette_score']:.3f}",
                    transform=ax1.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Plot selections for gradient_all
        if 'gradient_all' in selections_by_strategy:
            gradient_selections = []
            for i in range(frame + 1):
                if i in selections_by_strategy['gradient_all']:
                    cycle_selections = list(selections_by_strategy['gradient_all'][i])
                    gradient_selections.extend(cycle_selections)
                    
                    # Plot this cycle's selections
                    ax2.scatter(
                        embedded[cycle_selections, 0],
                        embedded[cycle_selections, 1],
                        c=f'C{i % 10}',
                        label=f'Cycle {i}',
                        s=80, alpha=0.8, edgecolors='black'
                    )
            
            # Compute and display metrics
            if gradient_selections:
                metrics = compute_cluster_metrics(feature_vectors, gradient_selections)
                ax2.text(
                    0.05, 0.95,
                    f"Diversity: {metrics['diversity_score']:.3f}\n"
                    f"Silhouette: {metrics['silhouette_score']:.3f}",
                    transform=ax2.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Add legends
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=range(max_cycles),
        blit=True, interval=1000
    )
    
    # Save animation
    anim.save(save_path, writer='pillow', fps=1, dpi=120)
    plt.close()
    
    print(f"Created side-by-side animation at {save_path}")

def create_cumulative_metrics_plot(feature_vectors, selections_by_strategy, save_path):
    """
    Create plot showing how diversity and other metrics evolve over cycles
    
    Args:
        feature_vectors: Feature vectors for all examples
        selections_by_strategy: Dict mapping strategy -> cycle -> selected examples
        save_path: Path to save the plot
    """
    # Determine max cycles across both strategies
    max_cycles = 0
    for strategy, selections_by_cycle in selections_by_strategy.items():
        if selections_by_cycle:
            max_cycles = max(max_cycles, max(selections_by_cycle.keys()) + 1)
    
    # Prepare data structures for metrics
    metrics_by_strategy = {
        strategy: {
            'diversity': np.zeros(max_cycles),
            'silhouette': np.zeros(max_cycles),
            'calinski_harabasz': np.zeros(max_cycles)
        } for strategy in selections_by_strategy
    }
    
    # Compute metrics for each cycle
    for strategy, selections_by_cycle in selections_by_strategy.items():
        for cycle in range(max_cycles):
            # Collect all examples selected up to this cycle
            all_selected = []
            for i in range(cycle + 1):
                if i in selections_by_cycle:
                    all_selected.extend(list(selections_by_cycle[i]))
            
            if all_selected:
                metrics = compute_cluster_metrics(feature_vectors, all_selected)
                metrics_by_strategy[strategy]['diversity'][cycle] = metrics['diversity_score']
                metrics_by_strategy[strategy]['silhouette'][cycle] = metrics['silhouette_score']
                metrics_by_strategy[strategy]['calinski_harabasz'][cycle] = metrics['calinski_harabasz_score']
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot diversity scores
    for strategy, metrics in metrics_by_strategy.items():
        label = 'BADGE' if strategy == 'badge_all' else 'Gradient'
        color = 'blue' if strategy == 'badge_all' else 'red'
        
        axs[0].plot(range(max_cycles), metrics['diversity'], marker='o', linestyle='-', 
                   label=label, color=color)
        axs[1].plot(range(max_cycles), metrics['silhouette'], marker='s', linestyle='-', 
                   label=label, color=color)
        axs[2].plot(range(max_cycles), metrics['calinski_harabasz'], marker='^', linestyle='-', 
                   label=label, color=color)
    
    # Add titles and labels
    axs[0].set_title('Diversity Score by Cycle', fontsize=14)
    axs[0].set_ylabel('Diversity (Higher = More Diverse)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    axs[1].set_title('Silhouette Score by Cycle', fontsize=14)
    axs[1].set_ylabel('Silhouette Score (-1 to 1, Higher = Better)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    axs[2].set_title('Calinski-Harabasz Score by Cycle', fontsize=14)
    axs[2].set_xlabel('Cycle')
    axs[2].set_ylabel('Calinski-Harabasz Score (Higher = Better)')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Created cumulative metrics plot at {save_path}")

def main():
    print("Starting BADGE vs Gradient comparison analysis...")
    
    # Load active pool dataset
    active_pool = AnnotationDataset(os.path.join(data_path, "active_pool.json"))
    active_pool_data = active_pool.get_all_entries()
    
    # Load badge_all and gradient_all results
    strategy_results = load_badge_gradient_results()
    print(f"Loaded {len(strategy_results)} strategies: {', '.join(strategy_results.keys())}")
    
    if len(strategy_results) < 2:
        print("Warning: Could not find both badge_all and gradient_all result files.")
        print("Make sure these files exist in the results directory.")
        return
    
    # Extract features for all examples
    print("Extracting features...")
    feature_vectors = []
    for example_data in tqdm(active_pool_data):
        features = extract_features(example_data)
        feature_vectors.append(features)
    feature_vectors = np.array(feature_vectors)
    print(f"Extracted features with shape: {feature_vectors.shape}")
    
    # Track selections by cycle for each strategy
    selections_by_strategy = track_selections_by_cycle(strategy_results)
    
    # Create side-by-side t-SNE animation
    print("Creating side-by-side t-SNE animation...")
    create_side_by_side_tsne_animation(
        feature_vectors,
        selections_by_strategy,
        save_path=os.path.join(plots_path, "badge_vs_gradient_tsne.gif")
    )
    
    # Create cumulative metrics plot
    print("Creating cumulative metrics plot...")
    create_cumulative_metrics_plot(
        feature_vectors,
        selections_by_strategy,
        save_path=os.path.join(plots_path, "badge_vs_gradient_metrics.png")
    )
    
    print(f"Analysis complete! Results saved to {plots_path}")

if __name__ == "__main__":
    main()