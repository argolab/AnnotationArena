import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Path configuration - adjust these to your environment
'''base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
plots_path = os.path.join(base_path, "plots")'''

base_path = "../outputs"
data_path = os.path.join(base_path, "data_hanna")
results_path = os.path.join(base_path, "results_l2")
plots_path = os.path.join(results_path, "plots")

# Create separate folders for different visualization types
viz_folders = {
    '2d_tsne': os.path.join(plots_path, '2d_tsne'),
    '3d_tsne': os.path.join(plots_path, '3d_tsne'),
    'animations': os.path.join(plots_path, 'animations'),
    'kde': os.path.join(plots_path, 'kde'),
    'loss_viz': os.path.join(plots_path, 'loss_viz'),
    'parallel_coords': os.path.join(plots_path, 'parallel_coords')
}

for folder in viz_folders.values():
    os.makedirs(folder, exist_ok=True)

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

def create_2d_tsne_animation(feature_vectors, selections_by_cycle, strategy, save_path):
    """Create animated GIF of 2D t-SNE visualization evolving over cycles"""
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
    
    anim.save(save_path, writer='pillow', fps=1, dpi=100)
    plt.close()
    
    return embedded

def create_3d_tsne_animation(feature_vectors, selections_by_cycle, strategy, save_path):
    """Create animated GIF of 3D t-SNE visualization evolving over cycles"""
    # Prepare data for t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # Reduce dimensionality
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create animation
    fig = plt.figure(figsize=(12, 10))
    
    def update(frame):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all examples as background in light gray
        ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                  c='lightgray', alpha=0.2, s=10)
        
        # Plot selected examples up to current cycle
        cmap = plt.cm.jet
        all_selected = []
        
        for i in range(frame + 1):
            if i in selections_by_cycle:
                selected = list(selections_by_cycle[i])
                all_selected.extend(selected)
                
                color = cmap(i / max(1, len(selections_by_cycle) - 1))
                ax.scatter(
                    embedded[selected, 0], 
                    embedded[selected, 1], 
                    embedded[selected, 2], 
                    c=[color], 
                    label=f'Cycle {i}',
                    s=80, alpha=0.8, edgecolors='black'
                )
        
        ax.set_title(f'3D Selection Evolution for {strategy} - Cycle {frame}', fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.view_init(elev=30, azim=45 + frame * 15)  # Rotate view for better animation
        plt.tight_layout()
        
        return []
    
    cycles = max(selections_by_cycle.keys()) + 1
    frames = []
    
    # Save individual frames for GIF creation
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    for frame in range(cycles):
        update(frame)
        frame_path = os.path.join(temp_dir, f'frame_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
    
    plt.close()
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return embedded

def create_rotating_3d_tsne(feature_vectors, selections_by_cycle, strategy, save_path):
    """Create a 3D t-SNE visualization with rotating view"""
    # Prepare data for t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # Reduce dimensionality
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    def update(frame):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all examples as background in light gray
        ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                  c='lightgray', alpha=0.2, s=10)
        
        # Plot all selected examples
        cmap = plt.cm.jet
        cycle_indices = sorted(selections_by_cycle.keys())
        
        for i, cycle in enumerate(cycle_indices):
            selected = list(selections_by_cycle[cycle])
            if not selected:
                continue
            
            color = cmap(i / max(1, len(cycle_indices) - 1))
            ax.scatter(
                embedded[selected, 0], 
                embedded[selected, 1], 
                embedded[selected, 2], 
                c=[color], 
                label=f'Cycle {cycle}',
                s=80, alpha=0.8, edgecolors='black'
            )
        
        ax.set_title(f'3D Selection Visualization for {strategy}', fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Rotate view for animation
        ax.view_init(elev=30, azim=frame * 10)
        plt.tight_layout()
        
        return []
    
    # Save individual frames for GIF creation
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    for frame in range(36):  # 36 frames for 360-degree rotation
        update(frame)
        frame_path = os.path.join(temp_dir, f'rotate_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
    
    plt.close()
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return embedded

def create_kde_animation(feature_vectors, selections_by_cycle, strategy, save_path):
    """Create animated GIF of KDE density evolution over cycles"""
    # Prepare data for t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    
    # Reduce dimensionality
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Create animation frames
    fig = plt.figure(figsize=(12, 10))
    
    def update(frame):
        plt.clf()
        ax = fig.add_subplot(111)
        
        # Plot all examples as background in light gray
        ax.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.1, s=5)
        
        # Collect all selections up to current cycle
        all_selected = []
        for i in range(frame + 1):
            if i in selections_by_cycle:
                all_selected.extend(list(selections_by_cycle[i]))
        
        if all_selected:
            # Get embedded coordinates for selected points
            selected_coords = embedded[all_selected]
            
            # Create a meshgrid for the KDE
            x_min, x_max = embedded[:, 0].min() - 10, embedded[:, 0].max() + 10
            y_min, y_max = embedded[:, 1].min() - 10, embedded[:, 1].max() + 10
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
            contour = ax.contourf(xx, yy, z, levels=50, cmap='viridis', alpha=0.7)
            plt.colorbar(contour, ax=ax)
            
            # Plot selected points
            current_cycle_selected = list(selections_by_cycle.get(frame, []))
            if current_cycle_selected:
                current_coords = embedded[current_cycle_selected]
                ax.scatter(current_coords[:, 0], current_coords[:, 1], 
                          c='red', alpha=0.8, s=40, edgecolors='black',
                          label=f'Cycle {frame}')
            
            # Plot previous selections
            prev_selected = []
            for i in range(frame):
                if i in selections_by_cycle:
                    prev_selected.extend(list(selections_by_cycle[i]))
            
            if prev_selected:
                prev_coords = embedded[prev_selected]
                ax.scatter(prev_coords[:, 0], prev_coords[:, 1], 
                          c='blue', alpha=0.4, s=20, edgecolors='black',
                          label='Previous Cycles')
        
        ax.set_title(f'Density Evolution for {strategy} - Cycle {frame}', fontsize=16)
        ax.legend(loc='best')
        plt.tight_layout()
        
        return []
    
    # Save individual frames for GIF creation
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    cycles = max(selections_by_cycle.keys()) + 1
    frames = []
    
    for frame in range(cycles):
        update(frame)
        frame_path = os.path.join(temp_dir, f'kde_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
    
    plt.close()
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return embedded

def create_loss_animation(strategy_results, strategy, save_path):
    """Create animated GIF showing loss curves evolving over cycles"""
    if strategy not in strategy_results:
        return
        
    # Get loss curves
    expected_losses = strategy_results[strategy].get('test_expected_losses', [])
    annotated_losses = strategy_results[strategy].get('test_annotated_losses', [])
    
    if not expected_losses:
        return
    
    # Create animation frames
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    def update(frame):
        ax.clear()
        
        # Plot expected loss curve up to current frame
        cycles = list(range(frame + 1))
        current_expected_losses = expected_losses[:frame + 1]
        current_annotated_losses = annotated_losses[:frame + 1]
        
        ax.plot(cycles, current_expected_losses, 'b-', marker='o', label='Expected Loss')
        ax.plot(cycles, current_annotated_losses, 'r--', marker='s', label='Annotated Loss')
        
        ax.set_title(f'Loss Evolution for {strategy} - Cycle {frame}', fontsize=14)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to keep visualization stable
        ax.set_ylim(0, max(expected_losses) * 1.1)
        
        # Set x-axis limits
        ax.set_xlim(0, len(expected_losses) - 1)
        
        ax.legend()
        plt.tight_layout()
        
        return []
    
    # Save individual frames for GIF creation
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    cycles = len(expected_losses)
    frames = []
    
    for frame in range(cycles):
        update(frame)
        frame_path = os.path.join(temp_dir, f'loss_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
    
    plt.close()
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)

def create_loss_gradient_animation(feature_vectors, selections_by_cycle, strategy, expected_losses, save_path):
    """Create animated GIF showing loss gradient visualization evolving over cycles"""
    # Create t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors)-1))
    pca = PCA(n_components=min(50, feature_vectors.shape[1]))
    reduced_features = pca.fit_transform(feature_vectors)
    embedded = tsne.fit_transform(reduced_features)
    
    # Calculate loss gradients
    loss_gradients = []
    for i in range(len(expected_losses) - 1):
        loss_gradients.append(expected_losses[i] - expected_losses[i+1])
    loss_gradients.append(0)  # Last cycle has no gradient
    
    max_gradient = max(max(loss_gradients), 0.001) * 1.1  # Ensure non-zero
    
    # Create animation frames
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    cycles = len(expected_losses)
    frames = []
    
    for frame in range(cycles):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all examples as background in light gray
        ax.scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.3, s=10)
        
        # Create a normalization for consistent color scaling
        import matplotlib.colors as colors
        norm = colors.Normalize(vmin=0, vmax=max_gradient)
        
        # Keep track of the scatter plot for colorbar
        scatter = None
        
        # Plot selections colored by gradient up to current frame
        for i in range(frame + 1):
            if i not in selections_by_cycle or i >= len(loss_gradients):
                continue
                
            selected = list(selections_by_cycle[i])
            if not selected:
                continue
            
            # Use gradient color - the higher the gradient, the more impact
            gradient = max(0, loss_gradients[i])  # Ensure non-negative
            
            scatter = ax.scatter(
                embedded[selected, 0], 
                embedded[selected, 1], 
                c=[gradient] * len(selected),  # Uniform color for a cycle
                label=f'Cycle {i} (Δ{gradient:.4f})',
                s=80, alpha=0.8, edgecolors='black',
                cmap='viridis_r',  # Reversed colormap to make higher values stand out
                norm=norm
            )
        
        # Add colorbar only once if we have data to plot
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Loss Reduction')
        
        ax.set_title(f'Selection Impact for {strategy} - Cycle {frame}', fontsize=16)
        ax.legend(loc='best')
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'gradient_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        frames.append(frame_path)
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)

def create_parallel_coordinates_animation(feature_vectors, selections_by_strategy, strategy, save_path):
    """Create animated GIF of parallel coordinates visualization evolving over cycles"""
    from pandas.plotting import parallel_coordinates
    
    # Get the selections for this strategy
    selections_by_cycle = selections_by_strategy[strategy]
    
    # Get feature names (or create generic ones)
    n_features = feature_vectors.shape[1]
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Create DataFrame with all examples
    df_all = pd.DataFrame(feature_vectors, columns=feature_names)
    df_all['Type'] = 'Background'
    
    # Save individual frames for GIF creation
    temp_dir = os.path.join(viz_folders['animations'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    cycles = max(selections_by_cycle.keys()) + 1
    frames = []
    
    for frame in range(cycles):
        plt.figure(figsize=(15, 8))
        
        # Copy the DataFrame for this frame
        df = df_all.copy()
        
        # Add data for cycles up to current frame
        for i in range(frame + 1):
            if i in selections_by_cycle:
                selected_list = list(selections_by_cycle[i])
                for idx in selected_list:
                    df.loc[idx, 'Type'] = f'Cycle {i}'
        
        # Sample background points to avoid overcrowding
        bg_indices = df[df['Type'] == 'Background'].index
        if len(bg_indices) > 100:
            sample_indices = np.random.choice(bg_indices, 100, replace=False)
            keep_indices = list(sample_indices) + list(df[df['Type'] != 'Background'].index)
            df = df.loc[keep_indices]
        
        # Create plot
        parallel_coordinates(df, 'Type', colormap=plt.cm.jet)
        plt.title(f'Parallel Coordinates Plot - {strategy} - Cycle {frame}', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'parallel_{frame:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        frames.append(frame_path)
    
    # Create GIF from frames using pillow
    from PIL import Image
    images = [Image.open(f) for f in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)

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
    
    # Create visualizations for each strategy
    for strategy, selections_by_cycle in selections_by_strategy.items():
        print(f"Creating visualizations for {strategy}...")
        
        # Create 2D t-SNE animation
        print(f"  - Creating 2D t-SNE animation...")
        create_2d_tsne_animation(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(viz_folders['animations'], f'{strategy}_2d_tsne.gif')
        )
        
        # Create 3D t-SNE animation
        print(f"  - Creating 3D t-SNE animation...")
        create_3d_tsne_animation(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(viz_folders['animations'], f'{strategy}_3d_tsne.gif')
        )
        
        # Create rotating 3D t-SNE visualization
        print(f"  - Creating rotating 3D t-SNE visualization...")
        create_rotating_3d_tsne(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(viz_folders['animations'], f'{strategy}_3d_rotating.gif')
        )
        
        # Create KDE density animation
        print(f"  - Creating KDE density animation...")
        create_kde_animation(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(viz_folders['animations'], f'{strategy}_kde.gif')
        )
        
        # Create loss curve animation
        if strategy in strategy_results:
            print(f"  - Creating loss curve animation...")
            create_loss_animation(
                strategy_results,
                strategy,
                save_path=os.path.join(viz_folders['animations'], f'{strategy}_loss.gif')
            )
            
            # Create loss gradient animation
            print(f"  - Creating loss gradient animation...")
            expected_losses = strategy_results[strategy].get('test_expected_losses', [])
            if expected_losses:
                create_loss_gradient_animation(
                    feature_vectors,
                    selections_by_cycle,
                    strategy,
                    expected_losses,
                    save_path=os.path.join(viz_folders['animations'], f'{strategy}_loss_gradient.gif')
                )
        
        # Create parallel coordinates animation
        print(f"  - Creating parallel coordinates animation...")
        create_parallel_coordinates_animation(
            feature_vectors,
            selections_by_cycle,
            strategy,
            save_path=os.path.join(viz_folders['animations'], f'{strategy}_parallel_coords.gif')
        )
    
    print(f"Analysis complete! Results saved to {plots_path}")

if __name__ == "__main__":
    main()