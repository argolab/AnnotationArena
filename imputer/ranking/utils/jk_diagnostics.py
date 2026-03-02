#!/usr/bin/env python3
"""
JK bipartite diagnostics infrastructure.

Computes JK metric matrices, rank-1 approximations, topology summaries, and correlation
analyses to diagnose Marformer performance gaps in JK-only experiments.

Usage:
    PYTHONPATH=. python utils/jk_diagnostics.py \\
        --data-bundle OUTPUT/generated_data/base_run/data_bundle.json \\
        --imputer-predictives OUTPUT/IMPUTER/base_run_marformer \\
        --slice train_missing
    (Loads train_predictives.json or test_predictives.json from the run dir by slice; falls back to predictives.json)
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns

from stan.pipeline.bundle import GroundTruthBundle

# Suppress expected warnings for empty slices (handled gracefully)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


def load_bundle_and_predictives(
    bundle_path: Path,
    predictives_path: Path,
) -> Tuple[GroundTruthBundle, Dict[str, Any]]:
    """Load data bundle and imputer predictives."""
    with open(bundle_path, 'r') as f:
        bundle_dict = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_dict)
    
    with open(predictives_path, 'r') as f:
        predictives = json.load(f)
    
    return bundle, predictives


def compute_jk_matrices(
    bundle: GroundTruthBundle,
    predictives: Dict[str, Any],
    slice_name: str = "test_missing",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute JK metric matrices on specified slice.
    
    Returns:
        (JK_logloss, JK_acc, JK_rmse, JK_count) all shape [J, K]
    """
    # Get dimensions from bundle stats or infer from data
    J = bundle.stats.get("J")
    K = bundle.stats.get("total_items")
    C = bundle.stats.get("C", 5)  # Likert classes
    
    if J is None:
        # Infer from data (bundle uses 1-indexed)
        J = max(r["annotator"] for r in bundle.all_ratings) if bundle.all_ratings else 0
    if K is None:
        K = bundle.stats.get("K_train", 0) + bundle.stats.get("K_test", 0)
        if K == 0:
            # Infer from data
            K = max(r["item"] for r in bundle.all_ratings) if bundle.all_ratings else 0
    
    # Initialize matrices
    JK_logloss = np.full((J, K), np.nan)
    JK_acc = np.full((J, K), np.nan)
    JK_rmse = np.full((J, K), np.nan)
    JK_count = np.zeros((J, K), dtype=int)
    
    # Aggregate logloss, acc, rmse per (j,k) across attributes i
    # Aggregate over attributes i
    
    # Map predictions by (attribute, annotator, item) for lookup
    # Predictives use 0-indexed, bundle uses 1-indexed
    pred_dict = {}
    for pred in predictives.get("predictions", []):
        if pred.get("is_listwise", False):
            continue  # Skip pairwise rankings
        
        # Extract key (predictives are 0-indexed)
        i_pred = pred["attribute"]  # 0-indexed
        j_pred = pred["annotator"]  # 0-indexed
        items = pred["items"]
        if len(items) != 1:
            continue
        k_pred = items[0]  # 0-indexed
        
        # Convert to 1-indexed to match bundle
        i_bundle = i_pred + 1
        j_bundle = j_pred + 1
        k_bundle = k_pred + 1
        
        key = (i_bundle, j_bundle, k_bundle)
        if key not in pred_dict:
            pred_dict[key] = []
        pred_dict[key].append(pred)
    
    # Process ratings from bundle. For each rating we need a matching prediction in pred_dict.
    # pred_dict is built from predictives["predictions"]; run_imputer must save predictives for
    # both train_missing and test_all so that train_missing slice has non-empty heatmaps.
    if slice_name == "test_missing":
        target_ratings = [r for r in bundle.missing_ratings if r["instance"] == "test"]
    elif slice_name == "train_missing":
        target_ratings = [r for r in bundle.missing_ratings if r["instance"] == "train"]
    else:
        raise ValueError(f"Unknown slice: {slice_name}")
    
    for rating in target_ratings:
        i = rating["attribute"]
        j = rating["annotator"]  # 1-indexed
        k = rating["item"]  # 1-indexed
        true_value = rating["value"]  # 1-indexed (1..C)
        
        # Find matching prediction
        key = (i, j, k)
        if key not in pred_dict:
            continue
        
        # Use first matching prediction (should be unique for ratings)
        pred = pred_dict[key][0]
        
        # Extract probabilities and predicted class
        probs = np.array(pred.get("rating_probabilities", []))
        if len(probs) == 0:
            continue
        
        pred_class = pred.get("predicted_rating_class", np.argmax(probs))
        
        # Convert to 1-indexed for comparison
        # Predictives use 0-indexed classes, bundle uses 1-indexed
        pred_class_1idx = pred_class + 1
        true_class_1idx = true_value
        
        # Compute metrics
        # Logloss: negative log probability of true class
        true_class_0idx = true_value - 1
        if 0 <= true_class_0idx < len(probs):
            logloss = -np.log(max(probs[true_class_0idx], 1e-10))
        else:
            logloss = np.nan
        
        # Accuracy: 1 if correct, 0 otherwise
        acc = 1.0 if pred_class_1idx == true_class_1idx else 0.0
        
        # Squared error for RMSE (we'll compute sqrt of mean later)
        squared_error = (pred_class_1idx - true_class_1idx) ** 2
        
        # Convert j, k to 0-indexed for matrix indexing
        j_idx = j - 1
        k_idx = k - 1
        
        # Accumulate (handle multiple entries per j,k across attributes)
        # For RMSE, we accumulate squared errors, then take sqrt of mean
        if np.isnan(JK_logloss[j_idx, k_idx]):
            JK_logloss[j_idx, k_idx] = logloss
            JK_acc[j_idx, k_idx] = acc
            JK_rmse[j_idx, k_idx] = squared_error  # Store squared error
        else:
            # Average across attributes
            n = JK_count[j_idx, k_idx]
            JK_logloss[j_idx, k_idx] = (JK_logloss[j_idx, k_idx] * n + logloss) / (n + 1)
            JK_acc[j_idx, k_idx] = (JK_acc[j_idx, k_idx] * n + acc) / (n + 1)
            JK_rmse[j_idx, k_idx] = (JK_rmse[j_idx, k_idx] * n + squared_error) / (n + 1)
        
        JK_count[j_idx, k_idx] += 1
    
    # Convert squared errors to RMSE (sqrt of mean)
    JK_rmse = np.sqrt(JK_rmse)
    
    return JK_logloss, JK_acc, JK_rmse, JK_count


def compute_jk_connectedness(bundle: GroundTruthBundle, J: int, K: int) -> np.ndarray:
    """
    Compute JK connectedness matrix: for each (j,k), fraction of attributes i for which
    (i,j,k) is observed (observation rate per cell, average over I).
    Shape [J, K], values in [0, 1]. Higher = more observed = more connected.
    """
    I = bundle.stats.get("I", 1)
    if I < 1:
        I = 1
    # Count distinct attributes observed per (j, k); bundle uses 1-indexed j, k
    seen_i_per_jk: Dict[Tuple[int, int], set] = defaultdict(set)
    for r in bundle.observed_ratings:
        j, k = r["annotator"], r["item"]
        i = r["attribute"]
        if 1 <= j <= J and 1 <= k <= K:
            seen_i_per_jk[(j, k)].add(i)
    JK_connectedness = np.zeros((J, K))
    for (j, k), seen_i in seen_i_per_jk.items():
        JK_connectedness[j - 1, k - 1] = len(seen_i) / I
    return JK_connectedness


def get_observed_jk_mask(bundle: GroundTruthBundle, J: int, K: int) -> np.ndarray:
    """
    Get boolean mask of observed (j,k) pairs from train-observed ratings.
    
    Args:
        bundle: Data bundle
        J: Number of annotators
        K: Number of items
    
    Returns:
        Boolean array shape [J, K], True where (j,k) is observed
    """
    mask = np.zeros((J, K), dtype=bool)
    train_observed = [r for r in bundle.observed_ratings if r["instance"] == "train"]
    for r in train_observed:
        j = r["annotator"]  # 1-indexed
        k = r["item"]  # 1-indexed
        if 1 <= j <= J and 1 <= k <= K:
            mask[j - 1, k - 1] = True
    return mask


def fit_rank1_approximation(M: np.ndarray, observed_mask: Optional[np.ndarray] = None, default_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit rank-1 approximation M ≈ u ⊗ v using SVD.
    
    Args:
        M: Matrix shape [J, K] (may contain NaN)
        observed_mask: Boolean mask shape [J, K], True for observed entries (will be set to default_value)
        default_value: Value to use for observed entries (default: 0.0)
    
    Returns:
        (u, v, residual) where u shape [J], v shape [K], residual shape [J, K]
    """
    # Create a copy for modification
    M_for_fitting = M.copy()
    
    # Set observed entries to default value if mask provided
    if observed_mask is not None:
        M_for_fitting[observed_mask] = default_value
    
    # Fill remaining NaN with 0 for SVD (weighted approach would be better but simpler for now)
    M_filled = np.nan_to_num(M_for_fitting, nan=0.0)
    
    # Use SVD to get rank-1 approximation
    # M ≈ σ₁ u₁ v₁ᵀ
    try:
        U, s, Vt = svds(M_filled.astype(float), k=1)
        if len(s) == 0 or s[0] <= 0:
            # Fallback: use zeros if SVD fails
            u = np.zeros(M.shape[0])
            v = np.zeros(M.shape[1])
        else:
            u = U[:, 0] * np.sqrt(s[0])
            v = Vt[0, :] * np.sqrt(s[0])
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        # Fallback: use zeros if SVD fails
        u = np.zeros(M.shape[0])
        v = np.zeros(M.shape[1])
    
    # Optional sign flip: if the sum of values is < 0 for both u and v, flip both by -1.
    # M ≈ u⊗v = (-u)⊗(-v), so this does not change the approximation or residual.
    if np.sum(u) < 0 and np.sum(v) < 0:
        u = -u
        v = -v
    
    # Compute residual (use original M, not M_for_fitting)
    M_approx = np.outer(u, v)
    residual = M - M_approx
    
    return u, v, residual


def compute_topology_summaries(
    bundle: GroundTruthBundle,
    J: int,
    K: int,
) -> Dict[str, Any]:
    """
    Compute topology summaries from train-observed bipartite graph.
    
    Args:
        bundle: Data bundle
        J: Number of annotators (to ensure consistent array sizes)
        K: Number of items (to ensure consistent array sizes)
    
    Returns:
        Dictionary with degree distributions, connectivity metrics, etc.
    """
    # Build bipartite graph from train-observed ratings
    train_observed = [r for r in bundle.observed_ratings if r["instance"] == "train"]
    
    # Track edges: (j, k) pairs
    edges = set()
    for r in train_observed:
        j = r["annotator"]
        k = r["item"]
        edges.add((j, k))
    
    # Compute degrees
    annotator_degree = defaultdict(int)
    item_degree = defaultdict(int)
    for j, k in edges:
        annotator_degree[j] += 1
        item_degree[k] += 1
    
    # Convert to arrays with consistent sizes (pad with 0 for missing annotators/items)
    deg_j = np.array([annotator_degree.get(j, 0) for j in range(1, J + 1)])
    deg_k = np.array([item_degree.get(k, 0) for k in range(1, K + 1)])
    
    # Compute connectivity (simple: count isolates)
    isolates_j = np.sum(deg_j == 0)
    isolates_k = np.sum(deg_k == 0)
    
    # For connected components, use simple union-find on bipartite graph
    # Simplified: just report giant component fraction (approximate)
    non_isolates_j = np.sum(deg_j > 0)
    non_isolates_k = np.sum(deg_k > 0)
    
    return {
        "annotator_degrees": deg_j.tolist(),
        "item_degrees": deg_k.tolist(),
        "annotator_degree_mean": float(np.mean(deg_j)),
        "annotator_degree_std": float(np.std(deg_j)),
        "item_degree_mean": float(np.mean(deg_k)),
        "item_degree_std": float(np.std(deg_k)),
        "num_edges": len(edges),
        "isolates_annotator": int(isolates_j),
        "isolates_item": int(isolates_k),
        "non_isolates_annotator": int(non_isolates_j),
        "non_isolates_item": int(non_isolates_k),
    }


def compute_correlations(
    bundle: GroundTruthBundle,
    JK_logloss: np.ndarray,
    JK_acc: np.ndarray,
    JK_rmse: np.ndarray,
    JK_connectedness: np.ndarray,
    topology: Dict[str, Any],
    J_train: int,
    K_train: int,
) -> Dict[str, Any]:
    """
    Compute correlations between degree and metrics.
    
    Uses only train instance data (annotators 1..J_train, items 1..K_train).
    
    Returns:
        Dictionary with correlation coefficients and aggregated metrics.
    """
    # Use only train instance (first J_train annotators and K_train items)
    deg_j = np.array(topology["annotator_degrees"])[:J_train]
    deg_k = np.array(topology["item_degrees"])[:K_train]
    
    # Slice JK matrices to train instance
    JK_logloss_train = JK_logloss[:J_train, :K_train]
    JK_acc_train = JK_acc[:J_train, :K_train]
    JK_rmse_train = JK_rmse[:J_train, :K_train]
    JK_connectedness_train = JK_connectedness[:J_train, :K_train]
    
    # Aggregate metrics per annotator and per item
    # For annotators: average over items (columns)
    # Use nanmean with where to avoid warnings on empty slices
    with np.errstate(invalid='ignore'):
        avg_logloss_j = np.nanmean(JK_logloss_train, axis=1)
        avg_acc_j = np.nanmean(JK_acc_train, axis=1)
        avg_rmse_j = np.nanmean(JK_rmse_train, axis=1)
        avg_connectedness_j = np.nanmean(JK_connectedness_train, axis=1)
        
        # For items: average over annotators (rows)
        avg_logloss_k = np.nanmean(JK_logloss_train, axis=0)
        avg_acc_k = np.nanmean(JK_acc_train, axis=0)
        avg_rmse_k = np.nanmean(JK_rmse_train, axis=0)
        avg_connectedness_k = np.nanmean(JK_connectedness_train, axis=0)
    
    # Filter out NaN for correlation
    def safe_corr(x, y):
        # Ensure arrays have same length
        if len(x) != len(y):
            return {"pearson": np.nan, "spearman": np.nan, "error": f"Length mismatch: {len(x)} vs {len(y)}"}
        mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(mask) < 2:
            return {"pearson": np.nan, "spearman": np.nan}
        x_clean = x[mask]
        y_clean = y[mask]
        try:
            # Check for constant arrays (correlation undefined)
            if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                return {"pearson": np.nan, "spearman": np.nan, "error": "Constant input"}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
                pearson = stats.pearsonr(x_clean, y_clean)[0]
                spearman = stats.spearmanr(x_clean, y_clean)[0]
            return {"pearson": float(pearson), "spearman": float(spearman)}
        except (ValueError, RuntimeWarning) as e:
            return {"pearson": np.nan, "spearman": np.nan, "error": str(e)}
    
    correlations = {
        "annotator_degree_vs_logloss": safe_corr(deg_j, avg_logloss_j),
        "annotator_degree_vs_acc": safe_corr(deg_j, avg_acc_j),
        "annotator_degree_vs_rmse": safe_corr(deg_j, avg_rmse_j),
        "annotator_degree_vs_connectedness": safe_corr(deg_j, avg_connectedness_j),
        "item_degree_vs_logloss": safe_corr(deg_k, avg_logloss_k),
        "item_degree_vs_acc": safe_corr(deg_k, avg_acc_k),
        "item_degree_vs_rmse": safe_corr(deg_k, avg_rmse_k),
        "item_degree_vs_connectedness": safe_corr(deg_k, avg_connectedness_k),
    }
    
    return {
        "correlations": correlations,
        "annotator_metrics": {
            "avg_logloss": avg_logloss_j.tolist(),
            "avg_acc": avg_acc_j.tolist(),
            "avg_rmse": avg_rmse_j.tolist(),
            "avg_connectedness": avg_connectedness_j.tolist(),
        },
        "item_metrics": {
            "avg_logloss": avg_logloss_k.tolist(),
            "avg_acc": avg_acc_k.tolist(),
            "avg_rmse": avg_rmse_k.tolist(),
            "avg_connectedness": avg_connectedness_k.tolist(),
        },
    }


def create_plots(
    out_dir: Path,
    JK_logloss: np.ndarray,
    JK_acc: np.ndarray,
    JK_rmse: np.ndarray,
    JK_connectedness: np.ndarray,
    JK_count: np.ndarray,
    topology: Dict[str, Any],
    correlations: Dict[str, Any],
    J_train: int,
    K_train: int,
    u_logloss: np.ndarray,
    v_logloss: np.ndarray,
    u_acc: np.ndarray,
    v_acc: np.ndarray,
    u_rmse: np.ndarray,
    v_rmse: np.ndarray,
    u_connectedness: np.ndarray,
    v_connectedness: np.ndarray,
    residual_logloss: np.ndarray,
    residual_acc: np.ndarray,
    residual_rmse: np.ndarray,
    residual_connectedness: np.ndarray,
    observed_mask: np.ndarray,
):
    """Create all diagnostic plots."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    J, K = JK_logloss.shape
    # Topology is already sliced to train-only
    deg_j = np.array(topology["annotator_degrees"])
    deg_k = np.array(topology["item_degrees"])
    
    # Helper to create rank-1 summary plot
    def plot_rank1_summary(M, u, v, residual, metric_name, filename_base):
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main heatmap (no colorbar so marginal u/v axes read clearly)
        ax_main = fig.add_subplot(gs[1:, :2])
        im = ax_main.imshow(M, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_main.set_xlabel('Item (K)', fontsize=10)
        ax_main.set_ylabel('Annotator (J)', fontsize=10)
        ax_main.set_title(f'{metric_name} Matrix', fontsize=12)
        
        # Y-axis: annotator factor u
        ax_u = fig.add_subplot(gs[1:, 2])
        ax_u.plot(u, list(range(len(u))), 'o-', markersize=4)
        ax_u.set_ylabel('Annotator (J)', fontsize=10)
        ax_u.set_xlabel('u factor', fontsize=10)
        ax_u.set_title('Annotator Factor', fontsize=10)
        ax_u.invert_yaxis()
        ax_u.grid(True, alpha=0.3)
        
        # X-axis: item factor v
        ax_v = fig.add_subplot(gs[0, :2])
        ax_v.plot(list(range(len(v))), v, 'o-', markersize=4)
        ax_v.set_xlabel('Item (K)', fontsize=10)
        ax_v.set_ylabel('v factor', fontsize=10)
        ax_v.set_title('Item Factor', fontsize=10)
        ax_v.grid(True, alpha=0.3)
        
        plt.suptitle(f'{metric_name} Rank-1 Approximation', fontsize=14, y=0.98)
        plt.savefig(plots_dir / f'{filename_base}.png', bbox_inches='tight')
        plt.close()
        
        # Residual heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(residual, aspect='auto', cmap='RdBu_r', 
                        interpolation='nearest', vmin=-np.nanmax(np.abs(residual)), 
                        vmax=np.nanmax(np.abs(residual)))
        ax.set_xlabel('Item (K)', fontsize=10)
        ax.set_ylabel('Annotator (J)', fontsize=10)
        ax.set_title(f'{metric_name} Residual (M - u⊗v)', fontsize=12)
        plt.colorbar(im, ax=ax)
        plt.savefig(plots_dir / f'{filename_base}_residual.png', bbox_inches='tight')
        plt.close()
    
    # Create rank-1 plots
    plot_rank1_summary(JK_logloss, u_logloss, v_logloss, residual_logloss, 
                       'Logloss', 'jk_rank1_approx_logloss')
    plot_rank1_summary(JK_acc, u_acc, v_acc, residual_acc,
                       'Accuracy', 'jk_rank1_approx_acc')
    plot_rank1_summary(JK_rmse, u_rmse, v_rmse, residual_rmse,
                       'RMSE', 'jk_rank1_approx_rmse')
    plot_rank1_summary(JK_connectedness, u_connectedness, v_connectedness, residual_connectedness,
                       'Connectedness', 'jk_rank1_approx_connectedness')
    
    # Heatmaps with marginals (similar layout to rank-1 plots)
    def plot_heatmap_with_marginals(M, name, filename, observed_mask):
        """Create heatmap with row/column averages as marginals."""
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # For connectedness: use all entries (both observed and missing)
        # For other metrics: only use missing entries
        if name == 'Connectedness':
            valid_mask = ~np.isnan(M)  # Use all entries
        else:
            valid_mask = ~np.isnan(M) & ~observed_mask  # Only missing entries
        
        # Compute row averages (annotator averages) - only on valid missing entries
        row_avg = np.full(M.shape[0], np.nan)
        for j in range(M.shape[0]):
            valid_cols = valid_mask[j, :]
            if np.any(valid_cols):
                row_avg[j] = np.nanmean(M[j, valid_cols])
        
        # Compute column averages (item averages) - only on valid missing entries
        col_avg = np.full(M.shape[1], np.nan)
        for k in range(M.shape[1]):
            valid_rows = valid_mask[:, k]
            if np.any(valid_rows):
                col_avg[k] = np.nanmean(M[valid_rows, k])
        
        # Main heatmap (no colorbar so marginal axes read clearly)
        ax_main = fig.add_subplot(gs[1:, :2])
        im = ax_main.imshow(M, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_main.set_xlabel('Item (K)', fontsize=10)
        ax_main.set_ylabel('Annotator (J)', fontsize=10)
        ax_main.set_title(f'{name} Matrix', fontsize=12)
        
        # Y-axis: annotator average (row average)
        ax_row = fig.add_subplot(gs[1:, 2])
        # Only plot non-NaN values
        valid_row_indices = ~np.isnan(row_avg)
        if np.any(valid_row_indices):
            ax_row.plot(row_avg[valid_row_indices], np.where(valid_row_indices)[0], 'o-', markersize=4)
        ax_row.set_ylabel('Annotator (J)', fontsize=10)
        ax_row.set_xlabel('Row Average', fontsize=10)
        ax_row.set_title('Annotator Average', fontsize=10)
        ax_row.invert_yaxis()
        ax_row.grid(True, alpha=0.3)
        
        # X-axis: item average (column average)
        ax_col = fig.add_subplot(gs[0, :2])
        # Only plot non-NaN values
        valid_col_indices = ~np.isnan(col_avg)
        if np.any(valid_col_indices):
            ax_col.plot(np.where(valid_col_indices)[0], col_avg[valid_col_indices], 'o-', markersize=4)
        ax_col.set_xlabel('Item (K)', fontsize=10)
        ax_col.set_ylabel('Column Average', fontsize=10)
        ax_col.set_title('Item Average', fontsize=10)
        ax_col.grid(True, alpha=0.3)
        
        plt.suptitle(f'{name} Matrix with Marginals', fontsize=14, y=0.98)
        plt.savefig(plots_dir / f'{filename}.png', bbox_inches='tight')
        plt.close()
    
    for M, name, filename in [
        (JK_logloss, 'Logloss', 'jk_heatmap_logloss'),
        (JK_acc, 'Accuracy', 'jk_heatmap_acc'),
        (JK_rmse, 'RMSE', 'jk_heatmap_rmse'),
        (JK_connectedness, 'Connectedness', 'jk_heatmap_connectedness'),
    ]:
        plot_heatmap_with_marginals(M, name, filename, observed_mask)
    
    # Master marginal plot: all metrics together
    def compute_marginals(M, observed_mask, exclude_observed=True):
        """
        Compute row and column averages for a matrix.
        
        Args:
            M: Matrix to compute marginals for
            observed_mask: Boolean mask of observed entries
            exclude_observed: If True, only use missing entries. If False, use all entries.
        """
        if exclude_observed:
            valid_mask = ~np.isnan(M) & ~observed_mask
        else:
            # For connectedness: use all entries (both observed and missing)
            valid_mask = ~np.isnan(M)
        
        # Row averages (annotator averages)
        row_avg = np.full(M.shape[0], np.nan)
        for j in range(M.shape[0]):
            valid_cols = valid_mask[j, :]
            if np.any(valid_cols):
                row_avg[j] = np.nanmean(M[j, valid_cols])
        
        # Column averages (item averages)
        col_avg = np.full(M.shape[1], np.nan)
        for k in range(M.shape[1]):
            valid_rows = valid_mask[:, k]
            if np.any(valid_rows):
                col_avg[k] = np.nanmean(M[valid_rows, k])
        
        return row_avg, col_avg
    
    # Compute marginals for all metrics
    # For performance metrics: only use missing entries
    row_logloss, col_logloss = compute_marginals(JK_logloss, observed_mask, exclude_observed=True)
    row_acc, col_acc = compute_marginals(JK_acc, observed_mask, exclude_observed=True)
    row_rmse, col_rmse = compute_marginals(JK_rmse, observed_mask, exclude_observed=True)
    # For connectedness: use all entries (both observed and missing)
    row_connectedness, col_connectedness = compute_marginals(JK_connectedness, observed_mask, exclude_observed=False)
    
    # Create master marginal plot
    fig, (ax_j, ax_k) = plt.subplots(1, 2, figsize=(16, 6))
    
    # J-axis (annotator) marginals - plot each metric separately, handling NaN
    j_indices = np.arange(len(row_logloss))
    ax_j.plot(j_indices, row_logloss, 'o-', label='Logloss', markersize=4, linewidth=2, alpha=0.7)
    ax_j.plot(j_indices, row_acc, 'o-', label='Accuracy', markersize=4, linewidth=2, alpha=0.7)
    ax_j.plot(j_indices, row_rmse, 'o-', label='RMSE', markersize=4, linewidth=2, alpha=0.7)
    ax_j.plot(j_indices, row_connectedness, 'o-', label='Connectedness', markersize=4, linewidth=2, alpha=0.7)
    
    ax_j.set_xlabel('Annotator (J)', fontsize=12)
    ax_j.set_ylabel('Row Average', fontsize=12)
    ax_j.set_title('Annotator Marginals (J-axis)', fontsize=14)
    ax_j.legend(loc='best')
    ax_j.grid(True, alpha=0.3)
    
    # K-axis (item) marginals - plot each metric separately, handling NaN
    k_indices = np.arange(len(col_logloss))
    ax_k.plot(k_indices, col_logloss, 'o-', label='Logloss', markersize=4, linewidth=2, alpha=0.7)
    ax_k.plot(k_indices, col_acc, 'o-', label='Accuracy', markersize=4, linewidth=2, alpha=0.7)
    ax_k.plot(k_indices, col_rmse, 'o-', label='RMSE', markersize=4, linewidth=2, alpha=0.7)
    ax_k.plot(k_indices, col_connectedness, 'o-', label='Connectedness', markersize=4, linewidth=2, alpha=0.7)
    
    ax_k.set_xlabel('Item (K)', fontsize=12)
    ax_k.set_ylabel('Column Average', fontsize=12)
    ax_k.set_title('Item Marginals (K-axis)', fontsize=14)
    ax_k.legend(loc='best')
    ax_k.grid(True, alpha=0.3)
    
    plt.suptitle('JK Marginals: All Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / 'jk_marginals_master.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Degree vs metric scatter plots
    annotator_metrics = correlations["annotator_metrics"]
    item_metrics = correlations["item_metrics"]
    
    for entity, deg, metrics_dict, suffix in [
        ("annotator", deg_j, annotator_metrics, "annotator"),
        ("item", deg_k, item_metrics, "item"),
    ]:
        for metric_name, metric_values in [
            ("logloss", metrics_dict["avg_logloss"]),
            ("acc", metrics_dict["avg_acc"]),
            ("rmse", metrics_dict["avg_rmse"]),
            ("connectedness", metrics_dict["avg_connectedness"]),
        ]:
            metric_arr = np.array(metric_values)
            mask = ~np.isnan(metric_arr)
            
            if np.sum(mask) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(deg[mask], metric_arr[mask], alpha=0.6, s=30)
            
            # Fit line (only if we have enough data and x is not constant)
            deg_clean = deg[mask]
            metric_clean = metric_arr[mask]
            
            if len(deg_clean) >= 2 and np.std(deg_clean) > 1e-10:
                try:
                    z = np.polyfit(deg_clean, metric_clean, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(deg_clean.min(), deg_clean.max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                except (np.linalg.LinAlgError, ValueError) as e:
                    # Skip line fitting if it fails (e.g., insufficient data, constant values)
                    pass
            
            # Annotate correlation
            corr_key = f"{entity}_degree_vs_{metric_name}"
            if corr_key in correlations["correlations"]:
                corr = correlations["correlations"][corr_key]
                pearson = corr.get("pearson", np.nan)
                spearman = corr.get("spearman", np.nan)
                if not np.isnan(pearson):
                    ax.text(0.05, 0.95, f'Pearson: {pearson:.3f}\nSpearman: {spearman:.3f}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'{entity.capitalize()} Degree', fontsize=10)
            ax.set_ylabel(f'Average {metric_name.upper()}', fontsize=10)
            ax.set_title(f'{entity.capitalize()} Degree vs {metric_name.upper()}', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'degree_vs_{metric_name}_{suffix}.png', bbox_inches='tight')
            plt.close()
    
    # Binned trend plots
    for entity, deg, metrics_dict, suffix in [
        ("annotator", deg_j, annotator_metrics, "annotator"),
        ("item", deg_k, item_metrics, "item"),
    ]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for idx, (metric_name, metric_values) in enumerate([
            ("logloss", metrics_dict["avg_logloss"]),
            ("acc", metrics_dict["avg_acc"]),
            ("rmse", metrics_dict["avg_rmse"]),
            ("connectedness", metrics_dict["avg_connectedness"]),
        ]):
            metric_arr = np.array(metric_values)
            mask = ~np.isnan(metric_arr)
            
            if np.sum(mask) < 2:
                axes[idx].text(0.5, 0.5, 'Insufficient data', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            # Bin by deciles
            deg_clean = deg[mask]
            metric_clean = metric_arr[mask]
            
            # Check if we have enough variation to bin
            if np.std(deg_clean) < 1e-10:
                axes[idx].text(0.5, 0.5, 'Insufficient variation\nfor binning', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            try:
                deciles = np.percentile(deg_clean, np.linspace(0, 100, 11))
                bin_means = []
                bin_stds = []
                bin_counts = []
                bin_centers = []
                
                for i in range(len(deciles) - 1):
                    bin_mask = (deg_clean >= deciles[i]) & (deg_clean < deciles[i + 1])
                    if i == len(deciles) - 2:  # Include upper bound for last bin
                        bin_mask = (deg_clean >= deciles[i]) & (deg_clean <= deciles[i + 1])
                    
                    if np.sum(bin_mask) > 0:
                        bin_means.append(np.mean(metric_clean[bin_mask]))
                        bin_stds.append(np.std(metric_clean[bin_mask]))
                        bin_counts.append(np.sum(bin_mask))
                        bin_centers.append((deciles[i] + deciles[i + 1]) / 2)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Binning error:\n{str(e)[:50]}', 
                             ha='center', va='center', transform=axes[idx].transAxes,
                             fontsize=8, color='red')
                continue
            
            if len(bin_means) > 0:
                try:
                    axes[idx].errorbar(bin_centers, bin_means, yerr=bin_stds, 
                                     fmt='o-', capsize=5, capthick=2, markersize=6)
                    axes[idx].set_xlabel(f'{entity.capitalize()} Degree (deciles)', fontsize=9)
                    axes[idx].set_ylabel(f'Mean {metric_name.upper()}', fontsize=9)
                    axes[idx].set_title(f'{metric_name.upper()}', fontsize=10)
                    axes[idx].grid(True, alpha=0.3)
                except Exception as e:
                    # Skip plotting if errorbar fails
                    axes[idx].text(0.5, 0.5, f'Plotting error:\n{str(e)[:50]}', 
                                 ha='center', va='center', transform=axes[idx].transAxes,
                                 fontsize=8, color='red')
                    axes[idx].set_title(f'{metric_name.upper()} (error)', fontsize=10)
        
        plt.suptitle(f'{entity.capitalize()} Degree Binned Trends', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / f'degree_binned_vs_metric_{suffix}.png', bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="JK bipartite diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-bundle", type=str, required=True,
                       help="Path to data_bundle.json")
    parser.add_argument("--imputer-predictives", type=str, required=True,
                       help="Path to imputer run dir (with train_predictives.json / test_predictives.json) or legacy predictives.json")
    parser.add_argument("--out-dir", type=str, default=None,
                       help="Output directory (default: jk_diagnostics under imputer run dir)")
    parser.add_argument("--slice", type=str, default="train_missing",
                       choices=["test_missing", "train_missing"],
                       help="Which slice to analyze (default: train_missing)")
    
    args = parser.parse_args()
    
    bundle_path = Path(args.data_bundle)
    p = Path(args.imputer_predictives)
    # Resolve predictives file: dir → train_predictives.json / test_predictives.json by slice; fallback to predictives.json
    if p.is_dir():
        fname = "train_predictives.json" if args.slice == "train_missing" else "test_predictives.json"
        predictives_path = p / fname
        if not predictives_path.exists():
            predictives_path = p / "predictives.json"
        run_dir_for_out = p
    else:
        predictives_path = p
        run_dir_for_out = p.parent
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = run_dir_for_out / "jk_diagnostics"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading bundle from {bundle_path}")
    print(f"Loading predictives from {predictives_path}")
    
    bundle, predictives = load_bundle_and_predictives(bundle_path, predictives_path)
    
    print(f"Computing JK matrices on {args.slice}...")
    JK_logloss, JK_acc, JK_rmse, JK_count = compute_jk_matrices(
        bundle, predictives, args.slice
    )
    
    # Determine train-only dimensions (train annotators and train items only)
    J_actual, K_actual = JK_logloss.shape
    J_train = (2 * J_actual) // 3  # Train annotators: 1..2J/3
    K_train = bundle.stats.get("K_train")
    if K_train is None:
        K_train = K_actual // 2
    
    print(f"Full matrix shape: J={J_actual}, K={K_actual}")
    print(f"Train-only dimensions: J_train={J_train} (annotators 1..{J_train}), K_train={K_train} (items 1..{K_train})")
    
    # Slice matrices to train-only (remove test annotators and test items)
    # This ensures all JK analysis uses only train data
    JK_logloss = JK_logloss[:J_train, :K_train].copy()
    JK_acc = JK_acc[:J_train, :K_train].copy()
    JK_rmse = JK_rmse[:J_train, :K_train].copy()
    JK_count = JK_count[:J_train, :K_train].copy()
    
    print(f"Sliced matrices to train-only: J={J_train}, K={K_train}")
    
    # Get observed (j,k) mask for train-only region
    observed_mask_full = get_observed_jk_mask(bundle, J_actual, K_actual)
    observed_mask = observed_mask_full[:J_train, :K_train].copy()
    num_observed = np.sum(observed_mask)
    print(f"Found {num_observed} observed (j,k) pairs (will set to defaults for rank-1 fitting)")
    
    print(f"Fitting rank-1 approximations...")
    # Set observed entries to defaults: acc=1, logloss=0, rmse=0
    u_logloss, v_logloss, residual_logloss = fit_rank1_approximation(JK_logloss, observed_mask, default_value=0.0)
    u_acc, v_acc, residual_acc = fit_rank1_approximation(JK_acc, observed_mask, default_value=1.0)
    u_rmse, v_rmse, residual_rmse = fit_rank1_approximation(JK_rmse, observed_mask, default_value=0.0)
    
    print(f"Computing JK connectedness (observation rate per (j,k), avg over I)...")
    JK_connectedness_full = compute_jk_connectedness(bundle, J_actual, K_actual)
    # Slice to train-only (remove test annotators and test items)
    JK_connectedness = JK_connectedness_full[:J_train, :K_train].copy()
    
    print(f"Fitting rank-1 for connectedness...")
    # Set observed entries to 0 for rank-1 fitting (default already correct)
    u_connectedness, v_connectedness, residual_connectedness = fit_rank1_approximation(JK_connectedness, observed_mask, default_value=0.0)
    
    print(f"Computing topology summaries...")
    # Compute topology on full bundle, but slice to train-only for analysis
    topology_full = compute_topology_summaries(bundle, J_actual, K_actual)
    # Slice topology to train-only
    topology = {
        "annotator_degrees": topology_full["annotator_degrees"][:J_train],
        "item_degrees": topology_full["item_degrees"][:K_train],
        "annotator_degree_mean": float(np.mean(topology_full["annotator_degrees"][:J_train])),
        "annotator_degree_std": float(np.std(topology_full["annotator_degrees"][:J_train])),
        "item_degree_mean": float(np.mean(topology_full["item_degrees"][:K_train])),
        "item_degree_std": float(np.std(topology_full["item_degrees"][:K_train])),
        "num_edges": topology_full["num_edges"],  # Keep total for reference
        "isolates_annotator": int(np.sum(np.array(topology_full["annotator_degrees"][:J_train]) == 0)),
        "isolates_item": int(np.sum(np.array(topology_full["item_degrees"][:K_train]) == 0)),
        "non_isolates_annotator": int(np.sum(np.array(topology_full["annotator_degrees"][:J_train]) > 0)),
        "non_isolates_item": int(np.sum(np.array(topology_full["item_degrees"][:K_train]) > 0)),
    }
    
    print(f"Computing correlations (train-only: J={J_train}, K={K_train})...")
    correlations = compute_correlations(
        bundle, JK_logloss, JK_acc, JK_rmse, JK_connectedness, topology, J_train, K_train
    )
    
    print(f"Creating plots...")
    create_plots(
        out_dir, JK_logloss, JK_acc, JK_rmse, JK_connectedness, JK_count,
        topology, correlations, J_train, K_train,
        u_logloss, v_logloss, u_acc, v_acc, u_rmse, v_rmse,
        u_connectedness, v_connectedness,
        residual_logloss, residual_acc, residual_rmse, residual_connectedness,
        observed_mask,
    )
    
    # Save matrices as CSV
    print(f"Saving matrices...")
    pd.DataFrame(JK_logloss).to_csv(out_dir / "jk_matrix_logloss.csv", index=False)
    pd.DataFrame(JK_acc).to_csv(out_dir / "jk_matrix_acc.csv", index=False)
    pd.DataFrame(JK_rmse).to_csv(out_dir / "jk_matrix_rmse.csv", index=False)
    pd.DataFrame(JK_connectedness).to_csv(out_dir / "jk_matrix_connectedness.csv", index=False)
    pd.DataFrame(JK_count).to_csv(out_dir / "jk_matrix_count.csv", index=False)
    
    # Save summary JSON
    summary = {
        "slice": args.slice,
        "train_instance": {"J_train": J_train, "K_train": K_train},
        "rating_scale": "1..C (1-indexed)",
        "topology": topology,
        "correlations": correlations,
        "rank1_factors": {
            "logloss": {
                "u": u_logloss.tolist(),
                "v": v_logloss.tolist(),
            },
            "acc": {
                "u": u_acc.tolist(),
                "v": v_acc.tolist(),
            },
            "rmse": {
                "u": u_rmse.tolist(),
                "v": v_rmse.tolist(),
            },
            "connectedness": {
                "u": u_connectedness.tolist(),
                "v": v_connectedness.tolist(),
            },
        },
    }
    
    with open(out_dir / "jk_diagnostics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Diagnostics saved to {out_dir}")
    print(f"  - JSON: jk_diagnostics.json")
    print(f"  - CSV matrices: jk_matrix_*.csv")
    print(f"  - Plots: plots/")


if __name__ == "__main__":
    main()
