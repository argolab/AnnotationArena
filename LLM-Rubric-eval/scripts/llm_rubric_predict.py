from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats 

import torch
import sys
sys.path.insert(0, "/home/stone/AnnotationArena/LLM-Rubric")
from llm_rubric import pd_utils
from llm_rubric.model.torch_impl import (
    Hyperparameters, PersonalizedCalibrationNetwork
)


def load_json_data_to_dataframe(json_path: Path) -> pd.DataFrame:
    """
    Load JSON data and convert to dataframe format.

    The JSON contains an 'all_ratings' list of records with fields:
      attribute (1-9), annotator (1-24 human, 25 LLM), item, value, rating_dist

    Returns one row per (item, human_annotator) pair, with LLM (annotator 25)
    ratings as input features and human ratings as output labels.
    In test mode: LLM questions are known, human questions are unknown.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ratings = data['all_ratings']

    from collections import defaultdict
    llm_ratings = defaultdict(dict)        # item -> {attribute -> rating_dist}
    human_ratings = defaultdict(lambda: defaultdict(dict))  # item -> annotator -> {attribute -> rating_dist}

    for r in ratings:
        if r['instance'] != 'test':
            continue
        item = r['item']
        attr = r['attribute'] - 1  # convert 1-9 to 0-indexed 0-8
        if r['annotator'] == 25:
            llm_ratings[item][attr] = r['rating_dist']
        else:
            human_ratings[item][r['annotator']][attr] = r['rating_dist']

    rows = []
    for item, annotators_dict in human_ratings.items():
        text_id = f"text_{item}"
        llm_data = llm_ratings.get(item, {})

        for annotator, questions_dict in annotators_dict.items():
            row = {
                'text_id': text_id,
                'annotator': annotator,
            }

            # Add LLM probability distributions as input features (Q0-Q8, 4 probs each)
            for q in range(9):
                if q in llm_data:
                    for ans_idx, prob in enumerate(llm_data[q], start=1):
                        row[f'Q{q}_{ans_idx}_prob'] = prob
                else:
                    for ans_idx in range(1, 5):
                        row[f'Q{q}_{ans_idx}_prob'] = 0.0

            # Add human answers as output labels (Q0-Q8)
            # In test mode: human questions are unknown (has_unknown=True)
            for q in range(9):
                if q in questions_dict:
                    answer_probs = questions_dict[q]
                    label = int(np.argmax(answer_probs) + 1)
                    row[f'Q{q}'] = label
                else:
                    row[f'Q{q}'] = -1

            row['has_unknown'] = True  # human labels are always unknown in test
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def evaluate_models(
    test_json_path: str,
    model_dir: str,
    judge_map_path: str,
    output_predictions_path: str = None,
    num_questions: int = 9,
    num_answers: int = 4,
    layer1_size: int = None,
    layer2_size: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    pretraining_epochs: int = None,
    finetuning_epochs: int = None,
): 
    # Load test data from JSON
    print(f"Loading test data from {test_json_path}")
    df = load_json_data_to_dataframe(Path(test_json_path))
    
    # For testing, we only use rows where we need to predict (has unknown labels)
    df = df[df['has_unknown'] == True].copy()
    
    print(f"Loaded {len(df)} test samples")
    print(f"Unique texts: {df['text_id'].nunique()}")
    print(f"Unique annotators: {df['annotator'].nunique()}")
    
    # Create annotator name column
    df['annotator_name'] = df['annotator'].apply(lambda x: f'annotator_{x}')
    
    # Load judge mapping
    with open(judge_map_path, 'r') as fh:
        judge_id_map = json.load(fh)
    
    pd_utils.add_judge_ids(df, 'annotator_name', 'judge_id', judge_id_map=judge_id_map)
    num_judges = len(judge_id_map)
    print(f"Number of judges: {num_judges}")
    
    # Define input and output criteria
    input_criteria = [f'Q{i}' for i in range(num_questions)]
    output_criteria = [f'Q{i}' for i in range(num_questions)]
    
    input_size = len(input_criteria) * num_answers
    output_size = len(output_criteria)
    
    # Set up hyperparameters
    hp_args = {}
    hp_args["all_data_size"] = len(df)
    
    if layer1_size is not None:
        hp_args["layer1_size"] = layer1_size
    if layer2_size is not None:
        hp_args["layer2_size"] = layer2_size
    if batch_size is not None:
        hp_args["batch_size"] = batch_size
    if learning_rate is not None:
        hp_args["learning_rate"] = learning_rate
    if pretraining_epochs is not None:
        hp_args["pretraining_epochs"] = pretraining_epochs
    if finetuning_epochs is not None:
        hp_args["finetuning_epochs"] = finetuning_epochs

    hp = Hyperparameters(
        input_size=input_size,
        output_size=output_size,
        num_judges=num_judges,
        **hp_args,
    )
    
    # Create dataset
    ds_test = pd_utils.make_dataset(df, input_criteria, 'judge_id', output_criteria)
    
    # Collect all predictions and ground truths across all questions
    all_predictions_flat = []
    all_ground_truth_flat = []
    all_losses = []
    
    # Evaluate each question-specific model
    results = {}
    all_predictions = []
    
    model_dir_path = Path(model_dir)
    
    for q_idx in range(num_questions):
        print(f"\n{'='*60}")
        print(f"Evaluating model for Q{q_idx}")
        print(f"{'='*60}")
        
        # Load question-specific model
        model_path = model_dir_path / f"model_Q{q_idx}.pt"
        if not model_path.exists():
            print(f"Warning: Model for Q{q_idx} not found at {model_path}")
            continue
        
        model = PersonalizedCalibrationNetwork(hp)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Predict for this question
        with torch.no_grad():
            test_loss = model.loss(ds_test.X, ds_test.A, ds_test.Y, I=[q_idx]).detach().numpy()
            yhat = model.decode(ds_test.X, ds_test.A, I=[q_idx])[:, 0].detach().numpy()
            y = ds_test.Y[:, q_idx].detach().numpy()
        
        # Calculate metrics for this question
        rmse = metrics.root_mean_squared_error(y, yhat)
        p_corr, _ = stats.pearsonr(y, yhat)
        s_corr, _ = stats.spearmanr(y, yhat)
        t_corr, _ = stats.kendalltau(y, yhat)
        
        print(f"Test Loss for Q{q_idx}: {test_loss}")
        print(f"Test RMSE for Q{q_idx}: {rmse}")
        print(f"Test Pearson for Q{q_idx}: {p_corr}")
        print(f"Test Spearman for Q{q_idx}: {s_corr}")
        print(f"Test Kendall for Q{q_idx}: {t_corr}")
        
        # Store results
        results[f'Q{q_idx}'] = {
            'loss': float(test_loss),
            'rmse': float(rmse),
            'pearson': float(p_corr),
            'spearman': float(s_corr),
            'kendall': float(t_corr),
        }
        
        # Store predictions
        all_predictions.append({
            'question': q_idx,
            'predictions': yhat.tolist(),
            'ground_truth': y.tolist(),
        })
        
        # Collect for overall metrics
        all_predictions_flat.extend(yhat.tolist())
        all_ground_truth_flat.extend(y.tolist())
        all_losses.append(test_loss)
    
    # Print summary per question
    print(f"\n{'='*60}")
    print("Summary of Results (Per Question)")
    print(f"{'='*60}")
    for q_idx in range(num_questions):
        if f'Q{q_idx}' in results:
            r = results[f'Q{q_idx}']
            print(f"Q{q_idx}: Pearson={r['pearson']:.4f}, Spearman={r['spearman']:.4f}, RMSE={r['rmse']:.4f}, Loss={r['loss']:.4f}")
    
    # Calculate overall metrics across all individual evaluations
    print(f"\n{'='*60}")
    print("Overall Metrics (All Individual Evaluations Combined)")
    print(f"{'='*60}")
    
    all_predictions_flat = np.array(all_predictions_flat)
    all_ground_truth_flat = np.array(all_ground_truth_flat)
    
    overall_rmse = metrics.root_mean_squared_error(all_ground_truth_flat, all_predictions_flat)
    overall_pearson, _ = stats.pearsonr(all_ground_truth_flat, all_predictions_flat)
    overall_spearman, _ = stats.spearmanr(all_ground_truth_flat, all_predictions_flat)
    overall_kendall, _ = stats.kendalltau(all_ground_truth_flat, all_predictions_flat)
    overall_loss = np.mean(all_losses)  # Average of per-question losses
    
    print(f"Total evaluations: {len(all_predictions_flat)}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall Pearson: {overall_pearson:.4f}")
    print(f"Overall Spearman: {overall_spearman:.4f}")
    print(f"Overall Kendall: {overall_kendall:.4f}")
    print(f"Overall Loss (avg): {overall_loss:.4f}")
    
    overall_results = {
        'total_evaluations': len(all_predictions_flat),
        'rmse': float(overall_rmse),
        'pearson': float(overall_pearson),
        'spearman': float(overall_spearman),
        'kendall': float(overall_kendall),
        'loss': float(overall_loss),
    }
    
    # Save predictions if output path provided
    if output_predictions_path:
        output_data = {
            'per_question_results': results,
            'overall_results': overall_results,
            'predictions': all_predictions,
        }
        with open(output_predictions_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nPredictions saved to {output_predictions_path}")
    
    return results, overall_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate personalized calibration models')
    
    # Required arguments
    parser.add_argument('--test-json', type=str, required=True,
                        help='Path to test JSON file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--judge-map', type=str, required=True,
                        help='Path to judge ID mapping JSON')
    
    # Optional arguments
    parser.add_argument('--output-predictions', type=str, default=None,
                        help='Path to save predictions JSON (optional)')
    
    # Model hyperparameters (must match training)
    parser.add_argument('--layer1-size', type=int, default=100,
                        help='Size of first hidden layer (default: 100)')
    parser.add_argument('--layer2-size', type=int, default=100,
                        help='Size of second hidden layer (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--pretraining-epochs', type=int, default=50,
                        help='Number of pretraining epochs (default: 50)')
    parser.add_argument('--finetuning-epochs', type=int, default=50,
                        help='Number of finetuning epochs (default: 50)')
    
    # Other parameters
    parser.add_argument('--num-questions', type=int, default=9,
                        help='Number of questions (default: 9)')
    parser.add_argument('--num-answers', type=int, default=4,
                        help='Number of answer choices per question (default: 4)')
    
    args = parser.parse_args()
    
    evaluate_models(
        test_json_path=args.test_json,
        model_dir=args.model_dir,
        judge_map_path=args.judge_map,
        output_predictions_path=args.output_predictions,
        num_questions=args.num_questions,
        num_answers=args.num_answers,
        layer1_size=args.layer1_size,
        layer2_size=args.layer2_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretraining_epochs=args.pretraining_epochs,
        finetuning_epochs=args.finetuning_epochs,
    )
