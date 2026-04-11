"""
Graph structure visualization for Bayesian Networks.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_bayesian_network_plot(adjacency_matrix: np.ndarray, 
                                output_dir: str = "attention_analysis") -> None:
    """
    Create a separate visualization of the Bayesian Network structure.
    
    Args:
        adjacency_matrix: True BN adjacency matrix [n_nodes, n_nodes]
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_nodes = adjacency_matrix.shape[0]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(f'N{i}')
    
    # Add edges from adjacency matrix
    for parent in range(n_nodes):
        for child in range(n_nodes):
            if adjacency_matrix[parent, child] == 1:
                G.add_edge(f'N{parent}', f'N{child}')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Use circular layout for clean visualization
    pos = nx.circular_layout(G)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, 
                          arrowsize=20, arrowstyle='->', ax=ax)
    
    ax.set_title(f'Bayesian Network Structure ({n_nodes} nodes)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bayesian_network_structure.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Bayesian Network structure saved to {output_dir}/bayesian_network_structure.png")
    plt.close()