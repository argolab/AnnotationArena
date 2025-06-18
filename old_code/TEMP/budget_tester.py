import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from tqdm.auto import tqdm
import os
from imputer_embedding import ImputerEmbedding
from utils_prabhav import AnnotationDataset
from selection_fixed import SelectionFactory

class OverallRMSETester:
    def __init__(self, voi_model_path, test_dataset_path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('overall_rmse_test.log', mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.voi_model = self._load_model(voi_model_path)
        
        with open(test_dataset_path, 'r') as f:
            self.original_data = json.load(f)
        
        self.save_dir = os.path.dirname(test_dataset_path)
        print(f"Loaded VOI model and {len(self.original_data)} test examples")
    
    def _load_model(self, model_path):
        model = ImputerEmbedding(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    def create_fully_masked_entry(self, example_idx):
        """Create a fully masked version - model knows nothing except text embeddings."""
        entry = deepcopy(self.original_data[example_idx])
        
        # Mask ALL positions - model starts knowing nothing
        for pos in range(len(entry['input'])):
            entry['input'][pos][0] = 1  # Mask bit = 1
            entry['input'][pos][1:] = [0.0] * 5  # Zero out annotation
            entry['known_questions'][pos] = 0  # Mark as unknown
        
        return entry
    
    def unmask_position(self, entry, example_idx, position):
        """Unmask a position using ground truth."""
        original_entry = self.original_data[example_idx]
        
        # Set mask bit to 0 (observed)
        entry['input'][position][0] = 0
        # Restore the answer from ground truth
        entry['input'][position][1:] = original_entry['answers'][position]
        # Mark as known
        entry['known_questions'][position] = 1
    
    def get_overall_rmse(self, entry, example_idx):
        """Get RMSE for ALL masked positions."""
        original_entry = self.original_data[example_idx]
        
        positions = []
        for i in range(len(entry['input'])):
            positions.append(i)
        
        # Create single-example dataset
        temp_dataset = AnnotationDataset([entry])
        known_questions, inputs, answers, annotators, questions, embeddings = temp_dataset[0]
        
        with torch.no_grad():
            outputs = self.voi_model(inputs.unsqueeze(0), annotators.unsqueeze(0), 
                                   questions.unsqueeze(0), embeddings.unsqueeze(0) if embeddings is not None else None)
            
            squared_errors = []
            
            for pos in positions:
                pred_probs = torch.softmax(outputs[0, pos, :], dim=0)
                pred_class = torch.argmax(pred_probs).item() + 1
                
                # Get true class from original ground truth
                true_class = torch.argmax(torch.tensor(original_entry['answers'][pos])).item() + 1
                
                squared_error = (pred_class - true_class) ** 2
                squared_errors.append(squared_error)
            
            rmse = np.sqrt(np.mean(squared_errors))
            return rmse, len(positions)
    
    def get_costs(self, example_idx):
        """Get annotation costs for each position."""
        entry = self.original_data[example_idx]
        costs = {}
        
        for pos, (question, annotator) in enumerate(zip(entry['questions'], entry['annotators'])):
            if annotator >= 0:  # Human annotation
                costs[pos] = 1.0
            else:  # LLM annotation
                costs[pos] = 0.2
        
        return costs
    
    def get_selectable_positions(self, entry):
        """Get ALL positions that are still masked."""
        selectable = []
        
        for pos in range(len(entry['input'])):
            if entry['input'][pos][0] == 1:  # Currently masked
                selectable.append(pos)
        
        return selectable
    
    def test_overall_rmse_reduction(self, example_idx, budget):
        """Test overall RMSE reduction with budget constraint."""
        costs = self.get_costs(example_idx)
        
        # Start with fully masked entry
        entry = self.create_fully_masked_entry(example_idx)
        
        results = []
        total_cost = 0.0
        query_log = []  # Track what was queried
        
        # Initial RMSE
        rmse, num_masked = self.get_overall_rmse(entry, example_idx)
        results.append({
            'cost': total_cost,
            'rmse': rmse,
            'num_masked': num_masked,
            'within_budget': True
        })
        
        while rmse > 0:
            # Get selectable positions
            selectable = self.get_selectable_positions(entry)
            
            if not selectable:
                break
            
            # Use VOI to select best position
            try:
                temp_dataset = AnnotationDataset([entry])
                
                feature_selector = SelectionFactory.create_feature_strategy('voi', self.voi_model, 'cpu')
                voi_results = feature_selector.select_features(
                    0, temp_dataset,
                    num_to_select=len(selectable),
                    loss_type='l2',
                    target_questions=[0]  # All questions as targets
                )
                
                if not voi_results:
                    break
                
                # Find best affordable choice
                best_pos = None
                best_ratio = -float('inf')
                
                for pos, benefit, _, _ in voi_results:
                    if pos in selectable:
                        cost = costs.get(pos, 1.0)
                        if total_cost + cost <= budget:
                            ratio = benefit / cost
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_pos = pos
                
                if best_pos is None:
                    break
                
                # Unmask the selected position
                cost = costs.get(best_pos, 1.0)
                total_cost += cost
                
                # Log the query
                question = entry['questions'][best_pos]
                annotator = entry['annotators'][best_pos]
                query_type = 'Human' if annotator >= 0 else 'LLM'
                
                query_log.append({
                    'position': best_pos,
                    'question': question,
                    'annotator_type': query_type,
                    'cost': cost
                })
                
                self.unmask_position(entry, example_idx, best_pos)
                rmse, num_masked = self.get_overall_rmse(entry, example_idx)
                
                results.append({
                    'cost': total_cost,
                    'rmse': rmse,
                    'num_masked': num_masked,
                    'within_budget': total_cost <= budget
                })
                
            except Exception as e:
                print(f"Example {example_idx} error: {e}")
                break
        
        return results, query_log
    
    def plot_rmse_progression(self, all_results, budget):
        """Plot RMSE progression with budget analysis."""
        
        # Extract progressions
        all_progressions = []
        within_budget_progressions = []
        exceeded_budget_progressions = []
        
        for result in all_results:
            if result['results']:
                rmse_progression = [step['rmse'] for step in result['results']]
                
                all_progressions.append(rmse_progression)
                
                if result['within_budget']:
                    within_budget_progressions.append(rmse_progression)
                else:
                    exceeded_budget_progressions.append(rmse_progression)
        
        # Calculate max steps
        max_steps = max(len(p) for p in all_progressions) if all_progressions else 0
        
        # Calculate statistics
        def calc_stats(progressions, max_steps):
            means, stds, counts = [], [], []
            for step in range(max_steps):
                values_at_step = [p[step] for p in progressions if step < len(p)]
                if values_at_step:
                    means.append(np.mean(values_at_step))
                    stds.append(np.std(values_at_step))
                    counts.append(len(values_at_step))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    counts.append(0)
            return means, stds, counts
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual progressions (faded)
        for progression in within_budget_progressions:
            steps = list(range(len(progression)))
            plt.plot(steps, progression, color='green', alpha=0.2, linewidth=0.5)
        
        for progression in exceeded_budget_progressions:
            steps = list(range(len(progression)))
            plt.plot(steps, progression, color='red', alpha=0.2, linewidth=0.5)
        
        # Plot mean progressions
        if within_budget_progressions:
            wb_means, wb_stds, wb_counts = calc_stats(within_budget_progressions, max_steps)
            valid_steps = [i for i, m in enumerate(wb_means) if not np.isnan(m)]
            if valid_steps:
                plt.errorbar([i for i in valid_steps], [wb_means[i] for i in valid_steps], 
                            yerr=[wb_stds[i] for i in valid_steps],
                            color='green', linewidth=2, marker='o', markersize=4,
                            capsize=3, label=f'Within Budget ({len(within_budget_progressions)} examples)')
        
        if exceeded_budget_progressions:
            ex_means, ex_stds, ex_counts = calc_stats(exceeded_budget_progressions, max_steps)
            valid_steps = [i for i, m in enumerate(ex_means) if not np.isnan(m)]
            if valid_steps:
                plt.errorbar([i for i in valid_steps], [ex_means[i] for i in valid_steps], 
                            yerr=[ex_stds[i] for i in valid_steps],
                            color='red', linewidth=2, marker='s', markersize=4,
                            capsize=3, label=f'Exceeded Budget ({len(exceeded_budget_progressions)} examples)')
        
        plt.xlabel('Number of Queries')
        plt.ylabel('Overall RMSE')
        plt.title('RMSE Reduction Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add budget info
        within_count = len(within_budget_progressions)
        total_count = len(all_progressions)
        plt.text(0.02, 0.98, f'Budget: {budget}\nWithin Budget: {within_count}/{total_count} ({within_count/total_count*100:.1f}%)', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_query_analysis(self, all_query_logs):
        """Plot analysis of what was queried."""
        
        # Extract query information
        query_types = [q['annotator_type'] for q in all_query_logs]
        questions = [q['question'] for q in all_query_logs]
        costs = [q['cost'] for q in all_query_logs]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. LLM vs Human queries
        ax1 = axes[0]
        type_counts = {}
        for qtype in query_types:
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = ['lightcoral' if t == 'Human' else 'lightblue' for t in types]
        
        bars = ax1.bar(types, counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Number of Queries')
        ax1.set_title('LLM vs Human Queries')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Questions requested
        ax2 = axes[1]
        question_counts = {}
        for q in questions:
            question_counts[f'Q{q}'] = question_counts.get(f'Q{q}', 0) + 1
        
        q_labels = sorted(question_counts.keys())
        q_counts = [question_counts[q] for q in q_labels]
        
        bars = ax2.bar(q_labels, q_counts, color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Number of Queries')
        ax2.set_title('Question Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for bar, count in zip(bars, q_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cost distribution
        ax3 = axes[2]
        ax3.hist(costs, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax3.set_xlabel('Query Cost')
        ax3.set_ylabel('Number of Queries')
        ax3.set_title('Cost Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        avg_cost = np.mean(costs)
        ax3.text(0.95, 0.95, f'Avg Cost: {avg_cost:.2f}\nTotal Queries: {len(costs)}', 
                 transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def run_overall_rmse_analysis(self, num_examples=100, budget=10.0):
        """Run overall RMSE reduction analysis."""
        print(f"Starting overall RMSE analysis: {num_examples} examples, budget {budget}")
        
        all_results = []
        all_query_logs = []
        
        for example_idx in tqdm(range(num_examples), desc="Testing examples"):
            results, query_log = self.test_overall_rmse_reduction(example_idx, budget)
            
            example_result = {
                'example_idx': example_idx,
                'results': results,
                'query_log': query_log,
                'final_rmse': results[-1]['rmse'] if results else float('inf'),
                'within_budget': results[-1]['within_budget'] if results else False,
                'final_cost': results[-1]['cost'] if results else 0.0,
                'num_queries': len(query_log)
            }
            all_results.append(example_result)
            all_query_logs.extend(query_log)
        
        # Generate plots
        self.plot_rmse_progression(all_results, budget)
        self.plot_query_analysis(all_query_logs)
        
        return all_results, all_query_logs
