"""
Bootstrap confidence interval utilities for progressive imputation experiments.

Provides robust statistical confidence intervals using bootstrap resampling
from individual sample results rather than graph-level aggregates.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(values: List[float], n_bootstrap: int = 1000,
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval from individual sample results.

    Uses percentile method to compute confidence bounds that are more
    accurate than standard deviation-based error bars.

    Args:
        values: Individual sample results (not graph-level means)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        logger.warning("Insufficient data for bootstrap CI, returning mean ± 0")
        mean_val = np.mean(values) if values else 0.0
        return mean_val, mean_val

    values = np.array(values)
    bootstrap_means = []

    # Bootstrap resampling
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentile-based confidence interval
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    logger.debug(f"Bootstrap CI from {len(values)} values: "
                f"[{lower:.4f}, {upper:.4f}] ({confidence*100:.1f}% CI)")

    return lower, upper


def bootstrap_aggregated_results(graph_results: List[Dict[str, Dict[str, Any]]],
                                n_graphs: int, n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate experimental results with bootstrap confidence intervals.

    Replaces standard deviation error bars with bootstrap confidence intervals
    computed from individual sample results across all graph instances.

    Args:
        graph_results: List of result dicts from each graph instance
        n_graphs: Number of graph instances
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level for intervals

    Returns:
        Dict with aggregated results including bootstrap CIs
    """
    logger.debug(f"Computing bootstrap aggregation from {n_graphs} graph instances")

    if not graph_results:
        return {}

    # Get all policy-imputer combinations from first graph
    all_keys = list(graph_results[0].keys())
    aggregated = {}

    for policy_imputer_key in all_keys:
        logger.debug(f"Bootstrap aggregating results for {policy_imputer_key}")

        # Get progressive results from all graphs for this policy-imputer combination
        all_progressive_results = []
        for graph_result in graph_results:
            if policy_imputer_key in graph_result:
                all_progressive_results.append(graph_result[policy_imputer_key]['results'])

        # DEBUG: Check what fields are actually in the first step
        if all_progressive_results and all_progressive_results[0]:
            first_step = all_progressive_results[0][0]
            domain_keys = [k for k in first_step.keys() if 'domain' in k]
            logger.warning(f"DEBUG: Available domain keys in raw data: {domain_keys}")
            logger.warning(f"DEBUG: domain_1_kl = {first_step.get('domain_1_kl', 'MISSING')}")
            logger.warning(f"DEBUG: domain_kl = {first_step.get('domain_kl', 'MISSING')}")

        if not all_progressive_results:
            continue

        # Get number of budget steps
        n_steps = len(all_progressive_results[0])

        # Aggregate step by step with bootstrap CIs
        aggregated_steps = []
        for step_idx in range(n_steps):
            # Collect individual sample arrays for bootstrap (not graph means)
            neural_log_loss_arrays = []
            domain_log_loss_arrays = []
            domain_1_log_loss_arrays = []  # EM (1 restart)
            true_log_loss_arrays = []
            neural_cross_entropy_arrays = []
            domain_cross_entropy_arrays = []
            domain_1_cross_entropy_arrays = []  # EM (1 restart)
            true_entropy_arrays = []
            neural_kl_arrays = []
            domain_kl_arrays = []
            domain_1_kl_arrays = []  # EM (1 restart)

            # Timing data (per graph)
            neural_times = []
            domain_times = []
            budgets = []
            n_training_samples = []

            for graph_progressive_results in all_progressive_results:
                if step_idx < len(graph_progressive_results):
                    step_result = graph_progressive_results[step_idx]

                    # Collect individual sample arrays from each graph
                    neural_log_loss_arrays.extend(step_result.get('neural_log_loss_values', []))
                    domain_log_loss_arrays.extend(step_result.get('domain_log_loss_values', []))
                    domain_1_log_loss_arrays.extend(step_result.get('domain_1_log_loss_values', []))
                    true_log_loss_arrays.extend(step_result.get('true_model_log_loss_values', []))
                    neural_cross_entropy_arrays.extend(step_result.get('neural_cross_entropy_values', []))
                    domain_cross_entropy_arrays.extend(step_result.get('domain_cross_entropy_values', []))
                    domain_1_cross_entropy_arrays.extend(step_result.get('domain_1_cross_entropy_values', []))
                    true_entropy_arrays.extend(step_result.get('true_entropy_values', []))

                    # KL divergence individual values
                    neural_kl_arrays.extend(step_result.get('neural_kl_distribution', []))

                    # Domain KL values (single values per graph, not individual sample arrays)
                    domain_kl_val = step_result.get('domain_kl', float('inf'))
                    if not np.isinf(domain_kl_val):
                        domain_kl_arrays.append(domain_kl_val)

                    domain_1_kl_val = step_result.get('domain_1_kl', float('inf'))
                    if not np.isinf(domain_1_kl_val):
                        domain_1_kl_arrays.append(domain_1_kl_val)

                    # Graph-level timing data
                    neural_times.append(step_result.get('neural_time', 0.0))
                    domain_times.append(step_result.get('domain_time', 0.0))
                    budgets.append(step_result.get('budget', 0))
                    n_training_samples.append(step_result.get('n_training_samples', 0))

            # Compute bootstrap confidence intervals for individual sample metrics
            neural_kl_mean = np.mean(neural_kl_arrays) if neural_kl_arrays else float('inf')
            neural_kl_lower, neural_kl_upper = bootstrap_confidence_interval(
                neural_kl_arrays, n_bootstrap, confidence) if neural_kl_arrays else (neural_kl_mean, neural_kl_mean)

            domain_kl_mean = np.mean(domain_kl_arrays) if domain_kl_arrays else float('inf')
            domain_kl_lower, domain_kl_upper = bootstrap_confidence_interval(
                domain_kl_arrays, n_bootstrap, confidence) if domain_kl_arrays else (domain_kl_mean, domain_kl_mean)

            neural_log_loss_mean = np.mean(neural_log_loss_arrays) if neural_log_loss_arrays else float('inf')
            neural_log_loss_lower, neural_log_loss_upper = bootstrap_confidence_interval(
                neural_log_loss_arrays, n_bootstrap, confidence) if neural_log_loss_arrays else (neural_log_loss_mean, neural_log_loss_mean)

            domain_log_loss_mean = np.mean(domain_log_loss_arrays) if domain_log_loss_arrays else float('inf')
            domain_log_loss_lower, domain_log_loss_upper = bootstrap_confidence_interval(
                domain_log_loss_arrays, n_bootstrap, confidence) if domain_log_loss_arrays else (domain_log_loss_mean, domain_log_loss_mean)

            # EM (1 restart) metrics
            domain_1_kl_mean = np.mean(domain_1_kl_arrays) if domain_1_kl_arrays else float('inf')
            domain_1_kl_lower, domain_1_kl_upper = bootstrap_confidence_interval(
                domain_1_kl_arrays, n_bootstrap, confidence) if domain_1_kl_arrays else (domain_1_kl_mean, domain_1_kl_mean)

            domain_1_log_loss_mean = np.mean(domain_1_log_loss_arrays) if domain_1_log_loss_arrays else float('inf')
            domain_1_log_loss_lower, domain_1_log_loss_upper = bootstrap_confidence_interval(
                domain_1_log_loss_arrays, n_bootstrap, confidence) if domain_1_log_loss_arrays else (domain_1_log_loss_mean, domain_1_log_loss_mean)

            # Standard aggregation for timing (graph-level metrics)
            neural_time_mean = np.mean(neural_times) if neural_times else 0.0
            domain_time_mean = np.mean(domain_times) if domain_times else 0.0

            aggregated_step = {
                'budget': budgets[0] if budgets else 0,
                'n_training_samples': n_training_samples[0] if n_training_samples else 0,

                # Neural metrics with bootstrap CIs
                'neural_kl': neural_kl_mean,
                'neural_kl_lower': neural_kl_lower,
                'neural_kl_upper': neural_kl_upper,
                'neural_log_loss': neural_log_loss_mean,
                'neural_log_loss_lower': neural_log_loss_lower,
                'neural_log_loss_upper': neural_log_loss_upper,
                'neural_time': neural_time_mean,

                # Domain metrics with bootstrap CIs (EM 5/10 restarts)
                'domain_kl': domain_kl_mean,
                'domain_kl_lower': domain_kl_lower,
                'domain_kl_upper': domain_kl_upper,
                'domain_log_loss': domain_log_loss_mean,
                'domain_log_loss_lower': domain_log_loss_lower,
                'domain_log_loss_upper': domain_log_loss_upper,
                'domain_time': domain_time_mean,

                # Domain 1-restart metrics with bootstrap CIs (EM 1 restart)
                'domain_1_kl': domain_1_kl_mean,
                'domain_1_kl_lower': domain_1_kl_lower,
                'domain_1_kl_upper': domain_1_kl_upper,
                'domain_1_log_loss': domain_1_log_loss_mean,
                'domain_1_log_loss_lower': domain_1_log_loss_lower,
                'domain_1_log_loss_upper': domain_1_log_loss_upper,

                # True model metrics
                'true_model_log_loss': np.mean(true_log_loss_arrays) if true_log_loss_arrays else float('inf'),

                # Raw values for plotting (flattened across graphs)
                'neural_log_loss_values': neural_log_loss_arrays,
                'domain_log_loss_values': domain_log_loss_arrays,
                'domain_1_log_loss_values': domain_1_log_loss_arrays,
                'true_model_log_loss_values': true_log_loss_arrays,
                'neural_cross_entropy_values': neural_cross_entropy_arrays,
                'domain_cross_entropy_values': domain_cross_entropy_arrays,
                'domain_1_cross_entropy_values': domain_1_cross_entropy_arrays,
                'true_entropy_values': true_entropy_arrays,
                'neural_kl_values': neural_kl_arrays,
                'domain_kl_values': domain_kl_arrays,
                'domain_1_kl_values': domain_1_kl_arrays,

                # Backward compatibility (keep old std fields for legacy code)
                'neural_kl_std': np.std(neural_kl_arrays) if len(neural_kl_arrays) > 1 else 0.0,
                'domain_kl_std': np.std(domain_kl_arrays) if len(domain_kl_arrays) > 1 else 0.0,
                'neural_log_loss_std': np.std(neural_log_loss_arrays) if len(neural_log_loss_arrays) > 1 else 0.0,
                'domain_log_loss_std': np.std(domain_log_loss_arrays) if len(domain_log_loss_arrays) > 1 else 0.0,

                # Evaluation counts
                'neural_n_evaluations': len([x for x in neural_kl_arrays if not np.isinf(x)]),
                'domain_n_evaluations': len([x for x in domain_kl_arrays if not np.isinf(x)]),
                'neural_failed_rate': sum(1 for x in neural_kl_arrays if np.isinf(x)) / len(neural_kl_arrays) if neural_kl_arrays else 1.0,
                'domain_failed_rate': sum(1 for x in domain_kl_arrays if np.isinf(x)) / len(domain_kl_arrays) if domain_kl_arrays else 1.0
            }

            aggregated_steps.append(aggregated_step)

        # Get metadata from first graph result
        first_graph_data = graph_results[0][policy_imputer_key]

        # Compute total time statistics
        all_total_times = [graph_results[i][policy_imputer_key].get('total_time', 0.0)
                          for i in range(len(graph_results)) if policy_imputer_key in graph_results[i]]

        # Create aggregated result structure
        aggregated[policy_imputer_key] = {
            'results': aggregated_steps,
            'total_time': np.mean(all_total_times) if all_total_times else 0.0,
            'total_time_std': np.std(all_total_times) if len(all_total_times) > 1 else 0.0,
            'n_graphs': n_graphs,
            'config': first_graph_data.get('config', {}),
            'policy_name': first_graph_data.get('policy_name', ''),
            'imputer_size': first_graph_data.get('imputer_size', ''),
            'policy_info': first_graph_data.get('policy_info', {})
        }

    logger.info(f"Bootstrap aggregated {len(aggregated)} policy-imputer combinations "
               f"across {n_graphs} graphs with {confidence*100:.1f}% CIs")
    return aggregated