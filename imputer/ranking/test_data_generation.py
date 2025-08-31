#!/usr/bin/env python3
"""
Test script for basic Stan data generation model.

This script tests the synthetic data generation using Stan to ensure
the mathematical framework is correctly implemented.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cmdstanpy as stan
    logger.info("Using cmdstanpy for Stan interface")
except ImportError:
    try:
        import pystan
        logger.info("Using pystan for Stan interface")
    except ImportError:
        logger.error("Neither cmdstanpy nor pystan available. Please install one of them:")
        logger.error("pip install cmdstanpy  # Recommended")
        logger.error("pip install pystan     # Alternative")
        exit(1)

def test_basic_data_generation():
    """Test basic synthetic data generation with Stan."""
    
    logger.info("Starting basic data generation test...")
    
    # Set parameters for testing all annotation types
    params = {
        'K': 10,    # 10 items
        'I': 5,     # 2 attributes (e.g., "quality", "relevance")
        'J': 5,     # 3 annotators
        'D': 32,     # 5-dimensional embeddings
        'C': 5,     # 5-point rating scale (1-5)
        
        # Query generation parameters
        'N_ratings': 30,        # 30 unary rating queries
        'N_comparisons': 20,    # 20 pairwise comparison queries  
        'N_rankings': 15,       # 15 ranking queries
        'ranking_size': 4,      # Rank 4 items at a time
        
        # Hyperparameters
        'sigma_annotator': 0.3,     # Moderate annotator differences
        'sigma_measurement': 0.1,   # Low measurement noise
        'alpha_dirichlet': 2.0,     # Mild preference for uniform bins
        'temperature': 0.5,         # Temperature for Plackett-Luce rankings
    }
    
    logger.info(f"Parameters: {params}")
    
    try:
        # Try cmdstanpy first (more stable)
        logger.info("Attempting to use cmdstanpy...")
        model_path = Path(__file__).parent / "models" / "data_generator.stan"
        
        # Compile model
        model = stan.CmdStanModel(stan_file=str(model_path))
        logger.info("Stan model compiled successfully")
        
        # Generate data (using generated quantities)
        # For generated quantities, we use sampling with adaptation disabled
        fit = model.sample(
            data=params,
            chains=1,
            iter_sampling=1,
            iter_warmup=0,
            adapt_engaged=False,  # Disable adaptation when no warmup
            fixed_param=True      # This tells Stan to only evaluate generated quantities
        )
        
        logger.info("Data generation completed successfully")
        
        # Extract generated data - True model components
        embeddings = fit.stan_variable('embeddings')[0]  # Shape: (K, D)
        mean_preferences = fit.stan_variable('mean_preferences')[0]  # Shape: (I, D)
        annotator_preferences = fit.stan_variable('annotator_preferences')[0]  # Shape: (I*J, D)
        rating_probs = fit.stan_variable('rating_probs')[0]  # Shape: (I*J, C)
        base_scores = fit.stan_variable('base_scores')[0]  # Shape: (I*J, K)
        
        # Extract annotation data
        rating_attributes = fit.stan_variable('rating_attributes')[0]
        rating_annotators = fit.stan_variable('rating_annotators')[0]  
        rating_items = fit.stan_variable('rating_items')[0]
        rating_values = fit.stan_variable('rating_values')[0]
        
        comparison_attributes = fit.stan_variable('comparison_attributes')[0]
        comparison_annotators = fit.stan_variable('comparison_annotators')[0]
        comparison_item_a = fit.stan_variable('comparison_item_a')[0]
        comparison_item_b = fit.stan_variable('comparison_item_b')[0]
        comparison_results = fit.stan_variable('comparison_results')[0]
        
        ranking_attributes = fit.stan_variable('ranking_attributes')[0]
        ranking_annotators = fit.stan_variable('ranking_annotators')[0]
        ranking_item_sets = fit.stan_variable('ranking_item_sets')[0]
        ranking_orders = fit.stan_variable('ranking_orders')[0]
        
        # Validate dimensions
        assert embeddings.shape == (params['K'], params['D']), f"Wrong embeddings shape: {embeddings.shape}"
        assert mean_preferences.shape == (params['I'], params['D']), f"Wrong mean_preferences shape: {mean_preferences.shape}"
        assert annotator_preferences.shape == (params['I']*params['J'], params['D']), f"Wrong annotator_preferences shape: {annotator_preferences.shape}"
        assert base_scores.shape == (params['I']*params['J'], params['K']), f"Wrong base_scores shape: {base_scores.shape}"
        
        # Validate annotation data dimensions
        assert len(rating_values) == params['N_ratings'], f"Wrong number of ratings: {len(rating_values)}"
        assert len(comparison_results) == params['N_comparisons'], f"Wrong number of comparisons: {len(comparison_results)}"
        assert ranking_item_sets.shape == (params['N_rankings'], params['ranking_size']), f"Wrong ranking sets shape: {ranking_item_sets.shape}"
        
        logger.info("✓ All dimensions match expected shapes")
        
        # Validate data properties
        logger.info("Validating data properties...")
        
        # Check that embeddings are roughly N(0,1)
        embed_mean = np.mean(embeddings)
        embed_std = np.std(embeddings)
        logger.info(f"Embeddings: mean={embed_mean:.3f} (should be ~0), std={embed_std:.3f} (should be ~1)")
        
        # Check that rating probabilities sum to 1
        prob_sums = np.sum(rating_probs, axis=1)
        logger.info(f"Rating probabilities sum: min={np.min(prob_sums):.6f}, max={np.max(prob_sums):.6f} (should be ~1.0)")
        
        # Check annotation data ranges
        logger.info(f"Rating values range: [{np.min(rating_values)}, {np.max(rating_values)}] (should be in [1, {params['C']}])")
        logger.info(f"Comparison results: {np.sum(comparison_results)} wins for item A out of {len(comparison_results)} comparisons")
        logger.info(f"Generated {len(ranking_attributes)} rankings of {params['ranking_size']} items each")
        
        # Create annotation data structure for saving
        annotation_data = {
            'ratings': {
                'attributes': rating_attributes.tolist(),
                'annotators': rating_annotators.tolist(), 
                'items': rating_items.tolist(),
                'values': rating_values.tolist()
            },
            'comparisons': {
                'attributes': comparison_attributes.tolist(),
                'annotators': comparison_annotators.tolist(),
                'item_a': comparison_item_a.tolist(),
                'item_b': comparison_item_b.tolist(), 
                'results': comparison_results.tolist()
            },
            'rankings': {
                'attributes': ranking_attributes.tolist(),
                'annotators': ranking_annotators.tolist(),
                'item_sets': ranking_item_sets.tolist(),
                'orders': ranking_orders.tolist()
            }
        }
        
        # Create summary plots and save data
        create_summary_plots(params, embeddings, base_scores, annotation_data, rating_probs)
        save_generated_data(params, embeddings, annotator_preferences, base_scores, annotation_data)
        
        logger.info("✓ Basic data generation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
        return False

def create_summary_plots(params, embeddings, base_scores, annotation_data, rating_probs):
    """Create summary plots of generated data."""
    
    logger.info("Creating summary plots...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Mixed Annotation Types - Synthetic Data Generation Summary', fontsize=16)
    
    # Plot 1: Embedding distribution
    axes[0,0].hist(embeddings.flatten(), bins=30, alpha=0.7, density=True)
    axes[0,0].set_title('Item Embeddings Distribution')
    axes[0,0].set_xlabel('Embedding Value')
    axes[0,0].set_ylabel('Density')
    
    # Plot 2: Base scores distribution
    axes[0,1].hist(base_scores.flatten(), bins=30, alpha=0.7, density=True)
    axes[0,1].set_title('Base Scores Distribution')
    axes[0,1].set_xlabel('Score Value')
    axes[0,1].set_ylabel('Density')
    
    # Plot 3: Rating values distribution 
    rating_values_int = np.array(annotation_data['ratings']['values'], dtype=int)
    rating_counts = np.bincount(rating_values_int, minlength=params['C']+1)[1:]
    axes[0,2].bar(range(1, params['C']+1), rating_counts)
    axes[0,2].set_title(f'Unary Ratings Distribution (N={len(annotation_data["ratings"]["values"])})')
    axes[0,2].set_xlabel('Rating Category')
    axes[0,2].set_ylabel('Count')
    
    # Plot 4: Pairwise comparison results
    comp_results = annotation_data['comparisons']['results']
    comp_counts = [len(comp_results) - sum(comp_results), sum(comp_results)]
    axes[1,0].bar(['B > A', 'A > B'], comp_counts)
    axes[1,0].set_title(f'Pairwise Comparisons (N={len(comp_results)})')
    axes[1,0].set_ylabel('Count')
    
    # Plot 5: Ranking visualization (first few rankings)
    rankings_to_show = min(5, len(annotation_data['rankings']['orders']))
    colors = ['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral']
    
    for i in range(rankings_to_show):
        item_set = [int(x) for x in annotation_data['rankings']['item_sets'][i]]  # Convert to int
        order = [int(x) for x in annotation_data['rankings']['orders'][i]]      # Convert to int
        # Create ranking visualization
        y_pos = np.arange(len(item_set)) + i * (len(item_set) + 0.5)  # Offset each ranking
        sorted_items = [item_set[order[j]-1] for j in range(len(order))]
        axes[1,1].barh(y_pos, range(len(sorted_items), 0, -1), 
                      color=colors[i % len(colors)], alpha=0.7, label=f'Ranking {i+1}')
        
        # Add item labels
        for j, item_id in enumerate(sorted_items):
            axes[1,1].text(0.1, y_pos[j], f'Item {item_id}', 
                          va='center', fontsize=8)
    
    axes[1,1].set_title(f'Sample Rankings (showing {rankings_to_show}/{len(annotation_data["rankings"]["orders"])})')
    axes[1,1].set_xlabel('Rank (higher = better)')
    axes[1,1].set_ylabel('Position in ranking')
    axes[1,1].legend()
    
    # Plot 6: Query type distribution
    query_types = ['Ratings', 'Comparisons', 'Rankings']
    query_counts = [len(annotation_data['ratings']['values']),
                   len(annotation_data['comparisons']['results']), 
                   len(annotation_data['rankings']['orders'])]
    axes[1,2].pie(query_counts, labels=query_types, autopct='%1.1f%%')
    axes[1,2].set_title('Distribution of Query Types')
    
    # Plot 7: Annotator activity (ratings)
    annotators_int = np.array(annotation_data['ratings']['annotators'], dtype=int)
    annotator_activity = np.bincount(annotators_int, minlength=params['J']+1)[1:]
    axes[2,0].bar(range(1, params['J']+1), annotator_activity)
    axes[2,0].set_title('Annotator Activity (Ratings)')
    axes[2,0].set_xlabel('Annotator ID')
    axes[2,0].set_ylabel('Number of Ratings')
    
    # Plot 8: Item coverage (how often each item appears in queries)
    all_items = (annotation_data['ratings']['items'] + 
                annotation_data['comparisons']['item_a'] + 
                annotation_data['comparisons']['item_b'] +
                [item for sublist in annotation_data['rankings']['item_sets'] for item in sublist])
    all_items_int = np.array(all_items, dtype=int)
    item_coverage = np.bincount(all_items_int, minlength=params['K']+1)[1:]
    axes[2,1].bar(range(1, params['K']+1), item_coverage)
    axes[2,1].set_title('Item Coverage Across All Queries')
    axes[2,1].set_xlabel('Item ID')
    axes[2,1].set_ylabel('Times Queried')
    
    # Plot 9: Base scores heatmap (first attribute across all annotators)
    first_attr_scores = base_scores[:params['J'], :]  # First attribute, all annotators
    im = axes[2,2].imshow(first_attr_scores, aspect='auto', cmap='RdBu_r')
    axes[2,2].set_title('Base Scores Heatmap (Attribute 1)')
    axes[2,2].set_xlabel('Items')
    axes[2,2].set_ylabel('Annotators')
    plt.colorbar(im, ax=axes[2,2])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "test_output_mixed_annotations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Summary plots saved to: {output_path}")
    
    # Also show if running interactively
    plt.show()

def save_generated_data(params, embeddings, annotator_preferences, base_scores, annotation_data):
    """Save generated data to JSON files for inspection."""
    
    logger.info("Saving generated data to files...")
    
    # Create data directory
    data_dir = Path(__file__).parent / "generated_data"
    data_dir.mkdir(exist_ok=True)
    
    # Save ground truth model components
    ground_truth = {
        'metadata': {
            'generation_params': params,
            'description': 'Ground truth embeddings, preferences, and scores for synthetic ranking data'
        },
        'embeddings': embeddings.tolist(),
        'annotator_preferences': annotator_preferences.tolist(), 
        'base_scores': base_scores.tolist()
    }
    
    ground_truth_path = data_dir / "ground_truth_model.json"
    with open(ground_truth_path, 'w') as f:
        import json
        json.dump(ground_truth, f, indent=2)
    logger.info(f"Ground truth model saved to: {ground_truth_path}")
    
    # Save annotation observations
    observations = {
        'metadata': {
            'generation_params': params,
            'description': 'Mixed annotation observations: ratings, comparisons, and rankings',
            'query_counts': {
                'ratings': len(annotation_data['ratings']['values']),
                'comparisons': len(annotation_data['comparisons']['results']),
                'rankings': len(annotation_data['rankings']['orders'])
            }
        },
        'observations': annotation_data
    }
    
    observations_path = data_dir / "annotation_observations.json" 
    with open(observations_path, 'w') as f:
        json.dump(observations, f, indent=2)
    logger.info(f"Annotation observations saved to: {observations_path}")
    
    # Save human-readable sample data
    sample_data = []
    
    # Add rating samples
    for i in range(min(10, len(annotation_data['ratings']['values']))):
        sample_data.append({
            'type': 'rating',
            'query': f"Annotator {annotation_data['ratings']['annotators'][i]} rates Item {annotation_data['ratings']['items'][i]} on Attribute {annotation_data['ratings']['attributes'][i]}",
            'result': f"Rating: {annotation_data['ratings']['values'][i]}/5"
        })
    
    # Add comparison samples  
    for i in range(min(5, len(annotation_data['comparisons']['results']))):
        winner = 'A' if annotation_data['comparisons']['results'][i] == 1 else 'B'
        sample_data.append({
            'type': 'comparison',
            'query': f"Annotator {annotation_data['comparisons']['annotators'][i]} compares Item {annotation_data['comparisons']['item_a'][i]} vs Item {annotation_data['comparisons']['item_b'][i]} on Attribute {annotation_data['comparisons']['attributes'][i]}",
            'result': f"Winner: Item {winner}"
        })
    
    # Add ranking samples
    for i in range(min(3, len(annotation_data['rankings']['orders']))):
        item_set = [int(x) for x in annotation_data['rankings']['item_sets'][i]]
        order = [int(x) for x in annotation_data['rankings']['orders'][i]]
        ranked_items = [item_set[order[j]-1] for j in range(len(order))]
        sample_data.append({
            'type': 'ranking', 
            'query': f"Annotator {annotation_data['rankings']['annotators'][i]} ranks items {item_set} on Attribute {annotation_data['rankings']['attributes'][i]}",
            'result': f"Ranking: {ranked_items} (best to worst)"
        })
    
    samples_path = data_dir / "sample_annotations.json"
    with open(samples_path, 'w') as f:
        json.dump({'sample_queries': sample_data}, f, indent=2)
    logger.info(f"Sample annotations saved to: {samples_path}")
    
    logger.info(f"✓ All data saved to {data_dir}/")

if __name__ == "__main__":
    logger.info("Testing basic Stan data generation...")
    
    # Ensure models directory exists
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    success = test_basic_data_generation()
    
    if success:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Tests failed!")
        exit(1)