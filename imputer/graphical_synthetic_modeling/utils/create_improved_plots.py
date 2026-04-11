#!/usr/bin/env python3
"""
Create improved plots from saved experimental data.

Improvements:
1. Unified 3-subplot scatterplot: Neural vs True, EM vs True, with consistent formatting
2. Enhanced log-loss comparison with emphasized true model baseline
3. Consistent axes and dot sizes for direct comparison
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import argparse

from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np

# Professional publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'figure.dpi': 100
})

def load_results(pickle_path):
    """Load experimental results from pickle file."""
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    return results

from matplotlib.colors import Normalize
import numpy as np

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path

def create_unified_budget_field_plots(
    results,
    output_dir="improved_plots",
    missing_rate=None,
    gridsize=120,           # grid resolution for the field
    sigma=1.2,              # Gaussian blur (in grid cells)
    mincnt=8,               # require at least this many samples per cell
    cmap_budget="inferno_r" # darker = higher budget
):
    """
    Three-panel figure:
      (1) Neural vs True: smooth 2D field where color = mean budget
      (2) Neural vs True: KDE contours per fixed imputer size (no dots)
      (3) EM vs True:     smooth 2D field where color = mean budget
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    neural_true_data, em_true_data = [], []
    budgets = set()

    # ---- collect data (same logic you use) ----
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Unknown')

        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)

            tvals = step_result.get('true_model_log_loss_values', [])
            nvals = step_result.get('neural_log_loss_values', [])
            evals = step_result.get('domain_log_loss_values', [])

            m = min(len(tvals), len(nvals))
            for i in range(m):
                tv, nv = tvals[i], nvals[i]
                if not (np.isnan(tv) or np.isinf(tv) or np.isnan(nv) or np.isinf(nv)):
                    neural_true_data.append((tv, nv, budget, imputer_size))

            m2 = min(len(tvals), len(evals))
            for i in range(m2):
                tv, ev = tvals[i], evals[i]
                if not (np.isnan(tv) or np.isinf(tv) or np.isnan(ev) or np.isinf(ev)):
                    em_true_data.append((tv, ev, budget))

    if not neural_true_data and not em_true_data:
        print("No valid data for budget-field plots.")
        return

    # ---- shared limits ----
    all_true = ([x[0] for x in neural_true_data] + [x[0] for x in em_true_data])
    all_pred = ([x[1] for x in neural_true_data] + [x[1] for x in em_true_data])
    vmin = min(min(all_true), min(all_pred))
    vmax = max(max(all_true), max(all_pred))
    pad  = (vmax - vmin) * 0.05
    xmin, xmax = vmin - pad, vmax + pad
    ymin, ymax = xmin, xmax  # square axes

    bmin, bmax = min(budgets), max(budgets)
    norm = Normalize(vmin=bmin, vmax=bmax)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ---------- helper to compute smooth mean-budget field ----------
    def mean_budget_field(x, y, b):
        # sum of budgets per cell
        sumB, xedges, yedges, _ = binned_statistic_2d(
            x, y, b, statistic="sum", bins=gridsize,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        # count per cell
        cnt, _, _, _ = binned_statistic_2d(
            x, y, None, statistic="count", bins=gridsize,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        # mean where count>=mincnt, else NaN
        with np.errstate(invalid="ignore", divide="ignore"):
            meanB = sumB / cnt
        meanB[cnt < mincnt] = np.nan

        # light smoothing; preserve NaNs (mask, blur, re-mask)
        mask = np.isnan(meanB)
        filled = meanB.copy()
        # fill NaNs with local median-ish to avoid edge bleeding
        fill_value = np.nanmedian(meanB)
        if np.isnan(fill_value):
            fill_value = (bmin + bmax) / 2
        filled[mask] = fill_value
        smoothed = gaussian_filter(filled, sigma=sigma, mode="nearest")
        smoothed[mask] = np.nan

        extent = [xmin, xmax, ymin, ymax]
        return smoothed.T, extent  # transpose so x~columns, y~rows

    # ---------- (1) Neural vs True: smooth mean-budget background ----------
    if neural_true_data:
        t = np.array([x[0] for x in neural_true_data])
        n = np.array([x[1] for x in neural_true_data])
        b = np.array([x[2] for x in neural_true_data])

        field, extent = mean_budget_field(t, n, b)
        im1 = ax1.imshow(
            field, origin="lower", extent=extent, cmap=cmap_budget, norm=norm,
            aspect="auto", interpolation="bilinear"
        )
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label('Mean Budget (Training Samples)', fontsize=10)

    ax1.plot([xmin, xmax], [ymin, ymax], 'k--', alpha=0.7, linewidth=2, label='Perfect Agreement')
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('True Model Log-Loss', fontsize=12)
    ax1.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax1.set_title('Neural vs True (Budget Field — Darker = Higher Budget)', fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ---------- (2) Neural vs True: KDE contours per imputer size (no dots) ----------
    from scipy.stats import gaussian_kde

    def draw_kde(ax, x, y, color, levels=6, lw=1.6):
        if len(x) < 40:
            return
        xx = np.linspace(xmin, xmax, 220)
        yy = np.linspace(ymin, ymax, 220)
        X, Y = np.meshgrid(xx, yy)
        kde = gaussian_kde(np.vstack([x, y]))
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=lw, alpha=0.95)

    if neural_true_data:
        sizes = np.array([x[3] for x in neural_true_data], dtype=object)
        t = np.array([x[0] for x in neural_true_data])
        n = np.array([x[1] for x in neural_true_data])

        palette = {'Tiny': '#fcaeae', 'Small': '#d04a4a', 'Large': '#8c1515'}
        for sz in ['Tiny', 'Small', 'Large']:
            m = sizes == sz
            if m.any():
                draw_kde(ax2, t[m], n[m], palette.get(sz, '#777777'))
                ax2.scatter([], [], color=palette.get(sz, '#777777'), s=50, label=f'{sz} Imputer')

    ax2.plot([xmin, xmax], [ymin, ymax], 'k--', alpha=0.7, linewidth=2, label='Perfect Agreement')
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('True Model Log-Loss', fontsize=12)
    ax2.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax2.set_title('Neural vs True (Imputer Size — KDE Contours)', fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ---------- (3) EM vs True: smooth mean-budget background ----------
    if em_true_data:
        t = np.array([x[0] for x in em_true_data])
        e = np.array([x[1] for x in em_true_data])
        b = np.array([x[2] for x in em_true_data])

        field, extent = mean_budget_field(t, e, b)
        im3 = ax3.imshow(
            field, origin="lower", extent=extent, cmap=cmap_budget, norm=norm,
            aspect="auto", interpolation="bilinear"
        )
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_label('Mean Budget (Training Samples)', fontsize=10)

    ax3.plot([xmin, xmax], [ymin, ymax], 'k--', alpha=0.7, linewidth=2, label='Perfect Agreement')
    ax3.set_xlim(xmin, xmax); ax3.set_ylim(ymin, ymax)
    ax3.set_xlabel('True Model Log-Loss', fontsize=12)
    ax3.set_ylabel('EM Model Log-Loss', fontsize=12)
    ax3.set_title('EM vs True (Budget Field — Darker = Higher Budget)', fontsize=12)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/unified_budget_field{suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Unified budget-field plots saved to {save_path}")


def create_unified_scatterplots(results, output_dir="improved_plots", missing_rate=None):
    """
    Create unified 3-subplot scatterplot: Neural vs True, EM vs True, all on same axes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract all data for consistent axis scaling
    neural_true_data = []
    em_true_data = []
    
    budgets = set()
    imputer_sizes = set()
    
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Unknown')
        
        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)
            imputer_sizes.add(imputer_size)
            
            # Get individual sample values
            true_values = step_result.get('true_model_log_loss_values', [])
            neural_values = step_result.get('neural_log_loss_values', [])
            em_values = step_result.get('domain_log_loss_values', [])
            
            # Collect neural vs true pairs
            min_len_neural = min(len(true_values), len(neural_values))
            for i in range(min_len_neural):
                if not (np.isnan(true_values[i]) or np.isinf(true_values[i]) or 
                       np.isnan(neural_values[i]) or np.isinf(neural_values[i])):
                    neural_true_data.append((true_values[i], neural_values[i], budget, imputer_size))
            
            # Collect EM vs true pairs
            min_len_em = min(len(true_values), len(em_values))
            for i in range(min_len_em):
                if not (np.isnan(true_values[i]) or np.isinf(true_values[i]) or 
                       np.isnan(em_values[i]) or np.isinf(em_values[i])):
                    em_true_data.append((true_values[i], em_values[i], budget))
    
    if not neural_true_data and not em_true_data:
        print("No valid data for scatterplots")
        return
    
    # Get consistent axis limits across all subplots
    all_true = ([x[0] for x in neural_true_data] + [x[0] for x in em_true_data])
    all_pred = ([x[1] for x in neural_true_data] + [x[1] for x in em_true_data])
    
    min_val = min(min(all_true), min(all_pred))
    max_val = max(max(all_true), max(all_pred))
    axis_margin = (max_val - min_val) * 0.05
    axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
    
    # Create 3-subplot figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Consistent scatter parameters
    scatter_size = 20
    scatter_alpha = 0.3
    
    budgets_list = sorted(budgets)
    budget_min, budget_max = min(budgets_list), max(budgets_list)
    
    # Subplot 1: Neural vs True (Budget Progression)
    if neural_true_data:
        true_vals = [x[0] for x in neural_true_data]
        neural_vals = [x[1] for x in neural_true_data]
        budget_vals = [x[2] for x in neural_true_data]
        
        scatter1 = ax1.scatter(true_vals, neural_vals, c=budget_vals, 
                              cmap='viridis', alpha=scatter_alpha, s=scatter_size, 
                              vmin=budget_min, vmax=budget_max)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Budget (Training Samples)', fontsize=10)
    
    # Perfect agreement line (all subplots)
    ax1.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, 
            label='Perfect Agreement', linewidth=2)
    
    ax1.set_xlim(axis_min, axis_max)
    ax1.set_ylim(axis_min, axis_max)
    ax1.set_xlabel('True Model Log-Loss', fontsize=12)
    ax1.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax1.set_title('Neural vs True (Budget Progression)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Neural vs True (Imputer Size)
    if neural_true_data:
        size_color_values = {'Tiny': 0.2, 'Small': 0.5, 'Large': 0.8}
        size_vals = [size_color_values.get(x[3], 0.5) for x in neural_true_data]
        
        scatter2 = ax2.scatter(true_vals, neural_vals, c=size_vals, 
                              cmap='Reds', alpha=scatter_alpha, s=scatter_size, vmin=0, vmax=1)
        
        # Add discrete legend
        for size in sorted(set([x[3] for x in neural_true_data])):
            if size in size_color_values:
                color_val = size_color_values[size]
                color = plt.cm.Reds(color_val)
                ax2.scatter([], [], c=[color], label=f'{size} Imputer', s=50)
    
    ax2.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7,
            label='Perfect Agreement', linewidth=2)
    
    ax2.set_xlim(axis_min, axis_max)
    ax2.set_ylim(axis_min, axis_max)
    ax2.set_xlabel('True Model Log-Loss', fontsize=12)
    ax2.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax2.set_title('Neural vs True (Imputer Size)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: EM vs True (Budget Progression)
    if em_true_data:
        true_vals_em = [x[0] for x in em_true_data]
        em_vals = [x[1] for x in em_true_data]
        budget_vals_em = [x[2] for x in em_true_data]
        
        scatter3 = ax3.scatter(true_vals_em, em_vals, c=budget_vals_em, 
                              cmap='viridis', alpha=scatter_alpha, s=scatter_size,
                              vmin=budget_min, vmax=budget_max)
        
        # Add colorbar
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Budget (Training Samples)', fontsize=10)
    
    ax3.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7,
            label='Perfect Agreement', linewidth=2)
    
    ax3.set_xlim(axis_min, axis_max)
    ax3.set_ylim(axis_min, axis_max)
    ax3.set_xlabel('True Model Log-Loss', fontsize=12)
    ax3.set_ylabel('EM Model Log-Loss', fontsize=12)
    ax3.set_title('EM vs True (Budget Progression)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/unified_scatterplots{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Unified scatterplots saved to {save_path}")

def create_enhanced_log_loss_curves(results, output_dir="improved_plots", missing_rate=None):
    """
    Create enhanced log-loss curves with emphasized true model baseline.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group results by node size
    results_by_nodes = {}
    for key, policy_results in results.items():
        if isinstance(key, tuple) and len(key) >= 1:
            n_nodes = key[0]
            if n_nodes not in results_by_nodes:
                results_by_nodes[n_nodes] = {}
            results_by_nodes[n_nodes][key] = policy_results
    
    # Create plot for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Track true model baseline (should be constant)
        true_baseline_plotted = False
        
        # Color scheme
        colors = {'Tiny': '#ff9999', 'Small': '#cc4444', 'Large': '#990000'}
        
        for key, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Extract data
            costs = [r['budget'] for r in experiment_results]
            neural_log_loss = [r.get('neural_log_loss', float('inf')) for r in experiment_results]
            em_log_loss = [r.get('domain_log_loss', float('inf')) for r in experiment_results]
            true_log_loss = [r.get('true_model_log_loss', float('inf')) for r in experiment_results]
            
            # Plot true model baseline (thick dotted line, only once)
            if not true_baseline_plotted:
                ax.plot(costs, true_log_loss, 'k:', linewidth=4, alpha=1.0,
                       label='True Model + True Params (Baseline)', markersize=0)
                
                # Also plot EM baseline (once)
                ax.plot(costs, em_log_loss, 's-', color='#1f77b4', 
                       linewidth=1, markersize=6, alpha=0.6,
                       label='True Model + EM Params')
                
                true_baseline_plotted = True
            
            # Plot neural imputer
            color = colors.get(imputer_size, '#666666')
            ax.plot(costs, neural_log_loss, '^-', color=color,
                   linewidth=1, markersize=6, alpha=0.6,
                   label=f'Neural Imputer ({imputer_size})')
        
        ax.set_xlabel('Budget (Number of Training Samples)', fontsize=12)
        ax.set_ylabel('Log-Loss', fontsize=12)
        ax.set_title(f'Log-Loss Comparison: {n_nodes} Nodes', fontsize=12)
        
        # Enhance legend - put true baseline first
        handles, labels = ax.get_legend_handles_labels()
        baseline_idx = next(i for i, label in enumerate(labels) if 'Baseline' in label)
        handles = [handles[baseline_idx]] + handles[:baseline_idx] + handles[baseline_idx+1:]
        labels = [labels[baseline_idx]] + labels[:baseline_idx] + labels[baseline_idx+1:]
        
        ax.legend(handles, labels, fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/enhanced_log_loss_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced log-loss curves for {n_nodes} nodes saved to {save_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create improved plots from experimental results')
    parser.add_argument('--pickle-path', required=True, 
                       help='Path to pickle file with experimental results')
    parser.add_argument('--output-dir', default='improved_plots',
                       help='Directory to save improved plots')
    parser.add_argument('--missing-rate', type=float, default=None,
                       help='Missing rate for filename suffix')
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.pickle_path}...")
    results = load_results(args.pickle_path)

    create_unified_budget_field_plots(results, args.output_dir, 0.5)
    
    print(f"All improved plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()