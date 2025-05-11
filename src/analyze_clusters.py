import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.stats import entropy
import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity

# Path configuration - adjust these to your environment
base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
plots_path = os.path.join(base_path, "plots")
os.makedirs(plots_path, exist_ok=True)

def analyze_clustering_by_cycle(feature_vectors, selections_by_strategy, n_clusters=3):
    """Perform K-means clustering on selections for each cycle and strategy"""
    clustering_results = {}
    
    for strategy, selections_by_cycle in selections_by_strategy.items():
        clustering_results[strategy] = {}
        
        # Collect all selections for this strategy
        all_selected = []
        for cycle in selections_by_cycle:
            all_selected.extend(list(selections_by_cycle[cycle]))
        
        if len(all_selected) < n_clusters:
            continue
            
        # Run clustering on all selections to establish global clusters
        global_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        global_kmeans.fit(feature_vectors[all_selected])
        
        # For each cycle, analyze distribution across these global clusters
        for cycle, selected in selections_by_cycle.items():
            selected_list = list(selected)
            if len(selected_list) < 2:
                continue
                
            selected_features = feature_vectors[selected_list]
            
            # Assign to global clusters
            cluster_labels = global_kmeans.predict(selected_features)
            
            # Calculate cluster distribution
            cluster_distribution = np.bincount(cluster_labels, minlength=n_clusters) / len(cluster_labels)
            
            # Calculate silhouette score if possible
            if len(np.unique(cluster_labels)) > 1 and len(cluster_labels) > 1:
                silhouette = silhouette_score(selected_features, cluster_labels)
            else:
                silhouette = 0
                
            clustering_results[strategy][cycle] = {
                'cluster_distribution': cluster_distribution,
                'silhouette_score': silhouette,
                'labels': cluster_labels,
                'selected_indices': selected_list
            }
    
    return clustering_results, global_kmeans.cluster_centers_

def fit_gaussian_mixture(feature_vectors, selections_by_strategy, n_components=3):
    """Fit Gaussian Mixture Models to selection distributions"""
    gmm_results = {}
    
    for strategy, selections_by_cycle in selections_by_strategy.items():
        # Collect all selected examples for this strategy
        all_selected = []
        for cycle in selections_by_cycle:
            all_selected.extend(list(selections_by_cycle[cycle]))
            
        if len(all_selected) < n_components * 2:  # Need enough data
            continue
            
        # Fit GMM to all selections
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        selected_features = feature_vectors[all_selected]
        gmm.fit(selected_features)
        
        # For each cycle, compute the probability distribution over components
        cycle_distributions = {}
        for cycle, selected in selections_by_cycle.items():
            selected_list = list(selected)
            if len(selected_list) < 2:
                continue
                
            cycle_features = feature_vectors[selected_list]
            probs = gmm.predict_proba(cycle_features)
            avg_probs = np.mean(probs, axis=0)
            
            cycle_distributions[cycle] = {
                'component_distribution': avg_probs,
                'log_likelihood': gmm.score(cycle_features)
            }
            
        gmm_results[strategy] = {
            'gmm': gmm,
            'cycle_distributions': cycle_distributions,
            'means': gmm.means_,
            'covariances': gmm.covariances_
        }
    
    return gmm_results

def analyze_cycle_transitions(feature_vectors, selections_by_strategy, clustering_results):
    """Analyze how selection transitions between regions across cycles"""
    transition_results = {}
    
    for strategy in selections_by_strategy:
        if strategy not in clustering_results:
            continue
            
        strategy_clusters = clustering_results[strategy]
        cycles = sorted(strategy_clusters.keys())
        
        if len(cycles) < 2:
            continue
            
        # Calculate distribution distances between consecutive cycles
        distances = []
        for i in range(len(cycles)-1):
            cycle1 = cycles[i]
            cycle2 = cycles[i+1]
            
            if cycle1 not in strategy_clusters or cycle2 not in strategy_clusters:
                continue
                
            dist1 = strategy_clusters[cycle1]['cluster_distribution']
            dist2 = strategy_clusters[cycle2]['cluster_distribution']
            
            # Jensen-Shannon divergence (symmetric KL divergence)
            js_distance = jensenshannon(dist1, dist2)
            
            # Earth Mover's Distance (1D Wasserstein)
            emd = wasserstein_distance(dist1, dist2)
            
            distances.append({
                'from_cycle': cycle1,
                'to_cycle': cycle2,
                'js_distance': js_distance,
                'emd': emd
            })
            
        transition_results[strategy] = distances
    
    return transition_results

def visualize_cluster_evolution(clustering_results, feature_vectors, tsne_embeddings, save_path=None):
    """Visualize how clusters evolve over cycles"""
    strategies = list(clustering_results.keys())
    
    for strategy in strategies:
        cycles = sorted(clustering_results[strategy].keys())
        
        if not cycles:
            continue
            
        n_cycles = len(cycles)
        n_cols = min(3, n_cycles)
        n_rows = (n_cycles + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols)
        
        for i, cycle in enumerate(cycles):
            ax = plt.subplot(gs[i//n_cols, i%n_cols])
            
            # Plot all points in light gray
            ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                      c='lightgray', alpha=0.2, s=5)
            
            # Get data for this cycle
            cycle_data = clustering_results[strategy][cycle]
            selected = cycle_data['selected_indices']
            labels = cycle_data['labels']
            
            # Plot points colored by cluster
            ax.scatter(tsne_embeddings[selected, 0], tsne_embeddings[selected, 1],
                      c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
            
            ax.set_title(f'Cycle {cycle} - Silhouette: {cycle_data["silhouette_score"]:.3f}')
            
        plt.suptitle(f'Cluster Evolution for {strategy}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{strategy}_cluster_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_density_maps(feature_vectors, selections_by_strategy, tsne_embeddings, save_path=None):
    """Generate density maps using Kernel Density Estimation"""
    strategies = list(selections_by_strategy.keys())
    
    # Create a figure with subplots for each strategy
    fig, axes = plt.subplots(1, len(strategies), figsize=(7*len(strategies), 6))
    if len(strategies) == 1:
        axes = [axes]
    
    for i, strategy in enumerate(strategies):
        ax = axes[i]
        
        # Collect all selections for this strategy
        all_selected = []
        for cycle in selections_by_strategy[strategy]:
            all_selected.extend(list(selections_by_strategy[strategy][cycle]))
            
        if not all_selected:
            continue
            
        # Get embedded coordinates for selected points
        selected_coords = tsne_embeddings[all_selected]
        
        # Create a meshgrid for the KDE
        x_min, x_max = selected_coords[:, 0].min() - 10, selected_coords[:, 0].max() + 10
        y_min, y_max = selected_coords[:, 1].min() - 10, selected_coords[:, 1].max() + 10
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                             np.linspace(y_min, y_max, 100))
        xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Fit the KDE
        kde = KernelDensity(bandwidth=5.0, kernel='gaussian')
        kde.fit(selected_coords)
        
        # Score samples and reshape
        z = np.exp(kde.score_samples(xy_sample))
        z = z.reshape(xx.shape)
        
        # Plot density map
        ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                  c='lightgray', alpha=0.1, s=5)
        ax.contourf(xx, yy, z, levels=50, cmap='viridis', alpha=0.8)
        ax.scatter(selected_coords[:, 0], selected_coords[:, 1], 
                  c='red', alpha=0.6, s=20, edgecolors='black')
        
        ax.set_title(f'Density Map - {strategy}', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/density_maps.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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

def load_results():
    """Load all result files"""
    strategy_results = {}
    result_files = glob.glob(os.path.join(results_path, "*.json"))
    
    for file_path in result_files:
        if "combined" in file_path:
            continue
        
        with open(file_path, 'r') as f:
            strategy_name = os.path.basename(file_path).replace(".json", "")
            strategy_results[strategy_name] = json.load(f)
    
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
    Track example selections by cycle for each strategy
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

def compute_diversity_metrics(feature_vectors, selected_indices):
    """Compute diversity metrics for selected examples"""
    if len(selected_indices) <= 1:
        return {
            'avg_cosine_distance': 0,
            'std_cosine_distance': 0,
            'feature_entropy': 0,
            'spatial_spread': 0
        }
    
    selected_features = feature_vectors[selected_indices]
    
    # Compute pairwise cosine similarities
    sim_matrix = cosine_similarity(selected_features)
    
    # Convert to distances: distance = 1 - similarity
    dist_matrix = 1 - sim_matrix
    
    # Compute metrics
    # Exclude self-similarities (diagonal)
    mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
    avg_distance = np.mean(dist_matrix[mask])
    std_distance = np.std(dist_matrix[mask])
    
    # Feature entropy - normalize features first
    normalized = selected_features / (np.sum(selected_features, axis=1)[:, np.newaxis] + 1e-10)
    feature_entropy = np.mean(entropy(normalized.T))
    
    # Spatial spread - mean euclidean distance from centroid
    centroid = np.mean(selected_features, axis=0)
    spatial_spread = np.mean(np.linalg.norm(selected_features - centroid, axis=1))
    
    return {
        'avg_cosine_distance': avg_distance,
        'std_cosine_distance': std_distance,
        'feature_entropy': feature_entropy,
        'spatial_spread': spatial_spread
    }

def visualize_selections(feature_vectors, selections_by_cycle, strategy, save_path=None):
    """Visualize selections for a strategy using t-SNE"""
    # Prepare data for t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # First reduce dimensionality with PCA to make t-SNE more stable
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot all examples as background in light gray
    plt.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
    
    # Plot selected examples by cycle with different colors
    cmap = plt.cm.jet
    cycle_indices = sorted(selections_by_cycle.keys())
    
    for i, cycle in enumerate(cycle_indices):
        if cycle not in selections_by_cycle:
            continue
        
        selected = list(selections_by_cycle[cycle])
        if not selected:
            continue
        
        color = cmap(i / max(1, len(cycle_indices) - 1))
        plt.scatter(
            embedded[selected, 0], 
            embedded[selected, 1], 
            c=[color], 
            label=f'Cycle {cycle}',
            s=80, alpha=0.8, edgecolors='black'
        )
    
    plt.title(f'Selection Visualization for {strategy} (t-SNE)', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return embedded

def create_cycle_animation(feature_vectors, selections_by_cycle, strategy, save_path=None):
    """Create an animation showing selection evolution over cycles"""
    # Prepare data for t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # Reduce dimensionality
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def init():
        ax.clear()
        ax.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
        ax.set_title(f'Selection Evolution for {strategy}', fontsize=16)
        return []
    
    def update(frame):
        cycle = frame
        ax.clear()
        ax.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
        
        # Plot all selections up to current cycle
        cmap = plt.cm.jet
        all_selected = []
        
        for i in range(cycle + 1):
            if i in selections_by_cycle:
                selected = list(selections_by_cycle[i])
                all_selected.extend(selected)
                
                color = cmap(i / max(1, len(selections_by_cycle) - 1))
                ax.scatter(
                    embedded[selected, 0], 
                    embedded[selected, 1], 
                    c=[color], 
                    label=f'Cycle {i}',
                    s=80, alpha=0.8, edgecolors='black'
                )
        
        ax.set_title(f'Selection Evolution for {strategy} - Cycle {cycle}', fontsize=16)
        ax.legend(loc='best')
        return []
    
    cycles = max(selections_by_cycle.keys()) + 1
    anim = FuncAnimation(
        fig, update, frames=range(cycles), 
        init_func=init, blit=True, interval=1000
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1, dpi=100)
        plt.close()
    else:
        plt.show()

def plot_diversity_trends(strategy_names, diversity_by_cycle):
    """Plot how diversity metrics change over cycles for different strategies"""
    metrics = ['avg_cosine_distance', 'spatial_spread', 'feature_entropy']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for strategy in strategy_names:
            if strategy not in diversity_by_cycle:
                continue
                
            cycles = sorted(diversity_by_cycle[strategy].keys())
            values = [diversity_by_cycle[strategy][cycle][metric] 
                      for cycle in cycles if metric in diversity_by_cycle[strategy][cycle]]
            
            if values:
                ax.plot(cycles[:len(values)], values, marker='o', label=strategy)
        
        ax.set_title(f'Evolution of {metric.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'diversity_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading datasets and results...")
    # Load active pool dataset
    active_pool = AnnotationDataset(os.path.join(data_path, "active_pool.json"))
    active_pool_data = active_pool.get_all_entries()
    
    # Load experiment results
    strategy_results = load_results()
    print(f"Found {len(strategy_results)} strategies: {', '.join(strategy_results.keys())}")
    
    print("Extracting features...")
    # Extract features for all examples
    feature_vectors = []
    for example_data in tqdm(active_pool_data):
        features = extract_features(example_data)
        feature_vectors.append(features)
    feature_vectors = np.array(feature_vectors)
    print(f"Extracted features with shape: {feature_vectors.shape}")
    
    # Track which examples were selected in each cycle for each strategy
    print("Tracking selections by cycle...")
    selections_by_strategy = track_selections_by_cycle(strategy_results)
    
    # Calculate diversity metrics for each strategy and cycle
    print("Computing diversity metrics...")
    diversity_by_strategy = {}
    for strategy, selections_by_cycle in selections_by_strategy.items():
        diversity_by_strategy[strategy] = {}
        for cycle, selected_indices in selections_by_cycle.items():
            diversity_by_strategy[strategy][cycle] = compute_diversity_metrics(
                feature_vectors, list(selected_indices)
            )
    
    # Visualize selections for each strategy
    print("Generating visualizations...")
    for strategy, selections_by_cycle in selections_by_strategy.items():
        print(f"Visualizing {strategy}...")
        embedded = visualize_selections(
            feature_vectors, 
            selections_by_cycle, 
            strategy, 
            save_path=os.path.join(plots_path, f'{strategy}_tsne.png')
        )
        
        # Create animation of selection evolution
        create_cycle_animation(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(plots_path, f'{strategy}_evolution.gif')
        )
    
    # Plot diversity trends
    print("Plotting diversity trends...")
    plot_diversity_trends(strategy_results.keys(), diversity_by_strategy)
    
    # Save diversity metrics to CSV
    diversity_data = []
    for strategy in diversity_by_strategy:
        for cycle in diversity_by_strategy[strategy]:
            metrics = diversity_by_strategy[strategy][cycle]
            row = {'strategy': strategy, 'cycle': cycle}
            row.update(metrics)
            diversity_data.append(row)
    
    diversity_df = pd.DataFrame(diversity_data)
    diversity_df.to_csv(os.path.join(plots_path, 'diversity_metrics.csv'), index=False)
    
    print(f"Analysis complete! Results saved to {plots_path}")

if __name__ == "__main__":
    main()