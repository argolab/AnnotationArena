import json
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Any
from scipy.special import kl_div
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PredictionEvaluator:
    """Evaluates model predictions against ground truth data."""
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize evaluator.
        
        Args:
            epsilon: Small value to add for numerical stability
        """
        self.epsilon = epsilon
    
    def load_data(self, data_file: str) -> Dict:
        """Load ground truth data from JSON file."""
        with open(data_file, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                data = content['data']
                metadata = content.get('metadata', {})
            else:
                data = content
                metadata = {}
        
        logger.info(f"Loaded ground truth data with {len(data)} examples")
        return {'data': data, 'metadata': metadata}
    
    def load_predictions(self, predictions_file: str) -> Dict[str, List[float]]:
        """Load model predictions from JSON file."""
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        logger.info(f"Loaded predictions for {len(predictions)} (example, position) pairs")
        return predictions
    
    def normalize_distribution(self, dist: np.ndarray) -> np.ndarray:
        """
        Normalize a probability distribution and add epsilon for stability.
        
        Args:
            dist: Probability distribution array
            
        Returns:
            Normalized distribution with epsilon added
        """
        dist = np.array(dist)
        dist = dist + self.epsilon
        return dist / dist.sum()
    
    def compute_kl_divergence(self, true_dist: np.ndarray, pred_dist: np.ndarray) -> float:
        """
        Compute KL divergence: KL(true || pred) = sum(true * log(true / pred))
        
        Args:
            true_dist: True probability distribution
            pred_dist: Predicted probability distribution
            
        Returns:
            KL divergence value
        """
        # Normalize both distributions
        true_norm = self.normalize_distribution(true_dist)
        pred_norm = self.normalize_distribution(pred_dist)
        
        # Compute KL divergence
        kl_div_val = np.sum(true_norm * np.log(true_norm / pred_norm))
        return kl_div_val
    
    def compute_log_loss(self, true_dist: np.ndarray, pred_dist: np.ndarray) -> float:
        """
        Compute log loss as cross-entropy: -sum(true * log(pred))
        
        Args:
            true_dist: True probability distribution
            pred_dist: Predicted probability distribution
            
        Returns:
            Cross-entropy loss value
        """
        # Normalize distributions
        true_norm = self.normalize_distribution(true_dist)
        pred_norm = self.normalize_distribution(pred_dist)
        
        # Compute cross-entropy: -sum(true * log(pred))
        cross_entropy = -np.sum(true_norm * np.log(pred_norm))
        return cross_entropy
    
    def is_probabilistic_distribution(self, dist: List[float]) -> bool:
        """
        Check if a distribution is probabilistic (not one-hot).
        
        Args:
            dist: Distribution to check
            
        Returns:
            True if probabilistic, False if one-hot
        """
        dist_array = np.array(dist)
        # Check if more than one element is positive
        return (dist_array > self.epsilon).sum() > 1
    
    def evaluate_predictions(self, data_content: Dict, predictions: Dict[str, List[float]],
                           only_probabilistic: bool = True) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth data for masked positions only.
        
        Args:
            data_content: Dictionary containing ground truth data
            predictions: Dictionary of predictions with "example_position" keys
            only_probabilistic: If True, only evaluate positions with probabilistic ground truth
            
        Returns:
            Dictionary containing evaluation metrics
        """
        data = data_content['data']
        kl_divergences = []
        log_losses = []
        position_details = []
        
        total_positions = 0
        total_masked_positions = 0
        evaluated_positions = 0
        skipped_onehot = 0
        skipped_missing = 0
        skipped_not_masked = 0
        
        logger.info("Computing evaluation metrics for masked positions only...")
        
        for example_idx in range(len(data)):
            entry = data[example_idx]
            answers = entry['answers']
            inputs = entry['input']
            
            for position_idx in range(len(answers)):
                total_positions += 1
                
                # Check if this position is masked (first element of input is 1)
                if len(inputs[position_idx]) == 0 or inputs[position_idx][0] != 1:
                    skipped_not_masked += 1
                    continue
                
                total_masked_positions += 1
                key = f"{example_idx}_{position_idx}"
                
                # Check if we have prediction for this position
                if key not in predictions:
                    skipped_missing += 1
                    continue
                
                true_dist = answers[position_idx]
                pred_dist = predictions[key]
                
                # Skip if not a valid probability distribution
                if abs(sum(true_dist) - 1.0) > 1e-5:
                    skipped_missing += 1
                    continue
                
                # Skip one-hot distributions if only_probabilistic is True
                if only_probabilistic and not self.is_probabilistic_distribution(true_dist):
                    skipped_onehot += 1
                    continue
                
                # Compute metrics
                try:
                    kl_div = self.compute_kl_divergence(true_dist, pred_dist)
                    log_loss = self.compute_log_loss(true_dist, pred_dist)
                    
                    # Check for valid results
                    if not (np.isnan(kl_div) or np.isinf(kl_div)):
                        kl_divergences.append(kl_div)
                    
                    if not (np.isnan(log_loss) or np.isinf(log_loss)):
                        log_losses.append(log_loss)
                    
                    evaluated_positions += 1
                    
                    # Store detailed information
                    position_details.append({
                        'example': example_idx,
                        'position': position_idx,
                        'kl_divergence': kl_div,
                        'log_loss': log_loss,
                        'true_entropy': -np.sum(self.normalize_distribution(true_dist) * 
                                              np.log(self.normalize_distribution(true_dist))),
                        'pred_entropy': -np.sum(self.normalize_distribution(pred_dist) * 
                                              np.log(self.normalize_distribution(pred_dist)))
                    })
                    
                except Exception as e:
                    logger.warning(f"Error computing metrics for {key}: {e}")
                    continue
        
        # Compute summary statistics
        results = {
            'kl_divergence': {
                'mean': np.mean(kl_divergences) if kl_divergences else 0.0,
                'std': np.std(kl_divergences) if kl_divergences else 0.0,
                'median': np.median(kl_divergences) if kl_divergences else 0.0,
                'min': np.min(kl_divergences) if kl_divergences else 0.0,
                'max': np.max(kl_divergences) if kl_divergences else 0.0,
                'percentile_25': np.percentile(kl_divergences, 25) if kl_divergences else 0.0,
                'percentile_75': np.percentile(kl_divergences, 75) if kl_divergences else 0.0,
                'count': len(kl_divergences)
            },
            'log_loss': {
                'mean': np.mean(log_losses) if log_losses else 0.0,
                'std': np.std(log_losses) if log_losses else 0.0,
                'median': np.median(log_losses) if log_losses else 0.0,
                'min': np.min(log_losses) if log_losses else 0.0,
                'max': np.max(log_losses) if log_losses else 0.0,
                'percentile_25': np.percentile(log_losses, 25) if log_losses else 0.0,
                'percentile_75': np.percentile(log_losses, 75) if log_losses else 0.0,
                'count': len(log_losses)
            },
            'evaluation_summary': {
                'total_positions': total_positions,
                'total_masked_positions': total_masked_positions,
                'evaluated_positions': evaluated_positions,
                'skipped_onehot': skipped_onehot,
                'skipped_missing': skipped_missing,
                'skipped_not_masked': skipped_not_masked,
                'evaluation_rate': evaluated_positions / total_masked_positions if total_masked_positions > 0 else 0.0,
                'masking_rate': total_masked_positions / total_positions if total_positions > 0 else 0.0
            },
            'position_details': position_details[:100],  # Limit to first 100 for file size
            'raw_values': {
                'kl_divergences': kl_divergences,
                'log_losses': log_losses
            }
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any], predictions_file: str, data_file: str):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("PREDICTION EVALUATION RESULTS (MASKED POSITIONS ONLY)")
        print("="*80)
        print(f"Predictions file: {predictions_file}")
        print(f"Data file: {data_file}")
        print()
        
        summary = results['evaluation_summary']
        print(f"Total positions: {summary['total_positions']}")
        print(f"Total masked positions: {summary['total_masked_positions']}")
        print(f"Evaluated masked positions: {summary['evaluated_positions']}")
        print(f"Skipped (one-hot): {summary['skipped_onehot']}")
        print(f"Skipped (missing/invalid): {summary['skipped_missing']}")
        print(f"Skipped (not masked): {summary['skipped_not_masked']}")
        print(f"Masking rate: {summary['masking_rate']:.2%}")
        print(f"Evaluation rate (of masked): {summary['evaluation_rate']:.2%}")
        print()
        
        # KL Divergence results
        kl_results = results['kl_divergence']
        print("KL DIVERGENCE RESULTS:")
        print(f"  Mean: {kl_results['mean']:.6f}")
        print(f"  Std:  {kl_results['std']:.6f}")
        print(f"  Median: {kl_results['median']:.6f}")
        print(f"  Min: {kl_results['min']:.6f}")
        print(f"  Max: {kl_results['max']:.6f}")
        print(f"  25th percentile: {kl_results['percentile_25']:.6f}")
        print(f"  75th percentile: {kl_results['percentile_75']:.6f}")
        print(f"  Count: {kl_results['count']}")
        print()
        
        # Log Loss results
        ll_results = results['log_loss']
        print("LOG LOSS RESULTS:")
        print(f"  Mean: {ll_results['mean']:.6f}")
        print(f"  Std:  {ll_results['std']:.6f}")
        print(f"  Median: {ll_results['median']:.6f}")
        print(f"  Min: {ll_results['min']:.6f}")
        print(f"  Max: {ll_results['max']:.6f}")
        print(f"  25th percentile: {ll_results['percentile_25']:.6f}")
        print(f"  75th percentile: {ll_results['percentile_75']:.6f}")
        print(f"  Count: {ll_results['count']}")
        print("="*80)


def evaluate_predictions(data_file: str, predictions_file: str, 
                        only_probabilistic: bool = True,
                        output_file: str = None) -> Dict[str, Any]:
    """
    Main evaluation function.
    
    Args:
        data_file: Path to ground truth data JSON file
        predictions_file: Path to predictions JSON file
        only_probabilistic: If True, only evaluate probabilistic ground truth
        output_file: Optional path to save detailed results
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = PredictionEvaluator()
    
    # Load data and predictions
    data_content = evaluator.load_data(data_file)
    predictions = evaluator.load_predictions(predictions_file)
    
    # Evaluate predictions
    results = evaluator.evaluate_predictions(
        data_content, predictions, only_probabilistic=only_probabilistic
    )
    
    # Print summary
    evaluator.print_summary(results, predictions_file, data_file)
    
    # Save detailed results if requested
    if output_file:
        # Remove raw values for file size if there are many
        results_to_save = results.copy()
        if len(results['raw_values']['kl_divergences']) > 10000:
            results_to_save['raw_values'] = {
                'note': 'Raw values omitted due to size. Use --keep_raw to include them.'
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logger.info(f"Detailed results saved to: {output_file}")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate model predictions against ground truth')
    parser.add_argument('--data_file', required=True, help='Path to ground truth data JSON file')
    parser.add_argument('--predictions_file', required=True, help='Path to predictions JSON file')
    parser.add_argument('--output_file', help='Path to save detailed evaluation results')
    parser.add_argument('--include_onehot', action='store_true', 
                       help='Include one-hot ground truth in evaluation (default: probabilistic only)')
    parser.add_argument('--epsilon', type=float, default=1e-10,
                       help='Epsilon value for numerical stability (default: 1e-10)')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if not args.output_file:
        pred_base = os.path.splitext(os.path.basename(args.predictions_file))[0]
        data_base = os.path.splitext(os.path.basename(args.data_file))[0]
        args.output_file = f"evaluation_{pred_base}_{data_base}.json"
    
    # Run evaluation
    results = evaluate_predictions(
        data_file=args.data_file,
        predictions_file=args.predictions_file,
        only_probabilistic=not args.include_onehot,
        output_file=args.output_file
    )
    
    # Print final summary for easy parsing
    print(f"\nFINAL METRICS:")
    print(f"KL Divergence: {results['kl_divergence']['mean']:.6f} ± {results['kl_divergence']['std']:.6f}")
    print(f"Log Loss: {results['log_loss']['mean']:.6f} ± {results['log_loss']['std']:.6f}")
    print(f"Evaluated: {results['evaluation_summary']['evaluated_positions']} positions")


if __name__ == "__main__":
    import os
    main()