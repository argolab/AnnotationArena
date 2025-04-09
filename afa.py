import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import sys
sys.path.insert(0, "..")
from joint_dataset import JointDataset
from Imputer import Imputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from collections import defaultdict
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

def myopic_policy_imputer(imputer, dataset, target_question, policy_question):

    all_predictions = {}
    all_true_scores = {}
    
    with torch.no_grad():
        device = next(imputer.parameters()).device
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Convert question names to indices if needed
        target_idx = imputer.var_to_index.get(target_question, target_question)
        policy_idx = imputer.var_to_index.get(policy_question, policy_question)
        
        all_question_vars = [q for q in list(imputer.question_num_to_question_name.values()) 
                             if q != target_question]
        max_questions = len(all_question_vars)
        print(f"Max question: {max_questions}")
        
        for i in range(max_questions + 1):  # +1 because we include 0 questions
            all_predictions[i] = []
            all_true_scores[i] = []
        
        # Process each instance separately
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing instances")):
            inputs = batch[1].to(device)
            labels = batch[2].to(device)
            annotators = batch[3].to(device)
            questions = batch[4].to(device)
            
            # Find target question position
            target_pos = None
            for j in range(inputs.shape[1]):
                if questions[0, j].item() == target_idx:
                    target_pos = j
                    break
            
            # Skip this instance if target not found or already observed
            if target_pos is None or inputs[0, target_pos, 0] == 0:
                continue
            
            # Get the target variable info
            target_var_name = imputer.question_num_to_question_name[target_idx]
            target_var = imputer.variables[target_var_name]
            var_dim = target_var.param_dim()
            
            possible_labels = torch.arange(1, var_dim + 1, dtype=torch.float, device=device)
            true_score = torch.sum(labels[0, target_pos] * possible_labels).item()
            
            # Make copy of inputs to modify for this instance
            instance_inputs = inputs.clone()
            
            # Add questions one by one
            added_questions = set()
            
            # Calculate and store prediction with 0 questions
            outputs_0 = imputer(instance_inputs, questions)
            pred_features_0 = target_var.to_features(outputs_0[0, target_pos, -var_dim:])
            possible_labels = torch.arange(1, var_dim + 1, dtype=torch.float, device=device)
            pred_score_0 = torch.sum(pred_features_0 * possible_labels).item()
            
            all_predictions[0].append(pred_score_0)
            all_true_scores[0].append(true_score)
            
            # Get unknown questions for this instance (excluding target)
            instance_questions = set()
            for j in range(inputs.shape[1]):
                q_idx = questions[0, j].item()
                if q_idx != target_idx:
                    q_name = imputer.question_num_to_question_name[q_idx]
                    instance_questions.add(q_name)
            
            # Add questions one by one until all are added
            question_count = 0
            while len(added_questions) < len(instance_questions):
                question_count += 1
                
                # Find the best question to add based on VOI for this instance
                max_voi = float("-inf")
                best_question = None
                
                for question in instance_questions:
                    if question in added_questions:
                        continue
                    
                    # Compute VOI for this question with current observations
                    voi = imputer.compute_voi(
                        policy_question, 
                        question,
                        instance_inputs,
                        labels,
                        questions,
                        device,
                        added_variables=added_questions
                    )
                    
                    if voi > max_voi:
                        best_question = question
                        max_voi = voi
                
                # Add the best question
                if best_question:
                    print(f"Add {best_question}")
                    added_questions.add(best_question)
                    best_question_idx = imputer.var_to_index.get(best_question)
                    
                    # Update inputs to mark question as observed
                    for j in range(instance_inputs.shape[1]):
                        if questions[0, j].item() == best_question_idx:
                            instance_inputs[0, j, 0] = 0  # Mark as observed
                            instance_inputs[0, j, 1:] = labels[0, j]  # Set to true value
                    
                    # Make prediction with current set of questions
                    outputs = imputer(instance_inputs, questions)
                    pred_features = target_var.to_features(outputs[0, target_pos, -var_dim:])
                    pred_score = torch.sum(pred_features * possible_labels).item()
                    
                    # Store prediction for this question count
                    all_predictions[question_count].append(pred_score)
                    all_true_scores[question_count].append(true_score)
                else:
                    break  # No more questions to add
    
    # Calculate metrics for all question counts
    all_metrics = {}
    for question_count, preds in all_predictions.items():
        if not preds:  # Skip if no predictions
            continue
            
        true_scores = all_true_scores[question_count]
        metrics = {
            'rmse': np.sqrt(np.mean((np.array(preds) - np.array(true_scores)) ** 2)),
            'pearson': pearsonr(preds, true_scores)[0] if len(set(preds)) > 1 else 0.0,
            'spearman': spearmanr(preds, true_scores)[0] if len(set(preds)) > 1 else 0.0,
            'kendall': kendalltau(preds, true_scores)[0] if len(set(preds)) > 1 else 0.0
        }
        
        all_metrics[question_count] = metrics
        print(f"Question count {question_count}, metrics: {metrics}")
    
    return all_predictions, all_true_scores, all_metrics

model_path = "imputer_model_20.pth"
print(f"Loading imputer model from {model_path}")
imputer = Imputer(
        total_embedding_dimension=30,
        num_heads=6,
        num_layers=5,
        ff_dim=1024,
        dropout=0.1
    )
imputer.load_state_dict(torch.load(model_path))
imputer.eval()
dataset_path = "gaussian_0.json"
# Load the dataset
print(f"Loading dataset from {dataset_path}")
dataset = JointDataset(dataset_path)
print(f"Dataset loaded with {len(dataset)} instances")
    
for target_question in range(1, 6):
    target_question = f"Q{target_question}"
    policy_question = target_question
    print(f"Running myopic policy imputer with target={target_question}, policy={policy_question}")
    all_predictions, all_true_scores, all_metrics = myopic_policy_imputer(
        imputer=imputer,
        dataset=dataset,
        target_question=target_question,
        policy_question=policy_question
    )
    
    # Save raw results
    results = {
        "predictions": {str(k): v for k, v in all_predictions.items()},
        "true_scores": {str(k): v for k, v in all_true_scores.items()},
        "metrics": {str(k): v for k, v in all_metrics.items()}
    }
    
    
    # Create plots
    question_counts = sorted([int(k) for k in all_metrics.keys()])
    
    # Plot RMSE vs question count
    plt.figure(figsize=(10, 6))
    rmse_values = [all_metrics[k]['rmse'] for k in question_counts]
    plt.plot(question_counts, rmse_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Number of Questions')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Number of Questions for Target: {target_question}')
    plt.grid(True)
    plt.savefig("rmse_vs_questions.png")
    plt.show()
    
    # Plot correlation metrics vs question count
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['pearson', 'spearman', 'kendall']
    for metric in metrics_to_plot:
        values = [all_metrics[k][metric] for k in question_counts]
        plt.plot(question_counts, values, marker='o', linestyle='-', linewidth=2, label=f'{metric.capitalize()} Correlation')
    
    plt.xlabel('Number of Questions')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Correlation Metrics vs Number of Questions for Target: {target_question}')
    plt.legend()
    plt.grid(True)
    plt.savefig("correlation_vs_questions.png")
    plt.show()
    
    