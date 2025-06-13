"""
Selection strategies for active learning and feature acquisition.
Fixed version that never uses ground truth labels for unobserved positions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class ExampleSelectionStrategy(ABC):
    """Base class for example selection strategies."""
    
    def __init__(self, name, model, device=None):
        self.name = name
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def select_examples(self, dataset, num_to_select=1, **kwargs):
        """Select examples from the dataset."""
        pass


class FeatureSelectionStrategy(ABC):
    """Base class for feature selection strategies."""
    
    def __init__(self, name, model, device=None):
        self.name = name
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def select_features(self, example_idx, dataset, num_to_select=1, **kwargs):
        """Select features from an example."""
        pass


class RandomExampleSelectionStrategy(ExampleSelectionStrategy):
    """Random example selection strategy."""
    
    def __init__(self, model, device=None):
        super().__init__("random", model, device)
    
    def select_examples(self, dataset, num_to_select=1, **kwargs):
        """Randomly select examples."""
        available_indices = list(range(len(dataset)))
        selected = random.sample(available_indices, min(num_to_select, len(available_indices)))
        return selected, [0.0] * len(selected)


class RandomFeatureSelectionStrategy(FeatureSelectionStrategy):
    """Random feature selection strategy."""
    
    def __init__(self, model, device=None):
        super().__init__("random", model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, **kwargs):
        """Randomly select features."""
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        num_to_select = min(num_to_select, len(masked_positions))
        selected_positions = random.sample(masked_positions, num_to_select)
        
        return [(pos, 0.0, 1.0, 0.0) for pos in selected_positions]


class EntropyExampleSelectionStrategy(ExampleSelectionStrategy):
    """Entropy-based example selection strategy."""
    
    def __init__(self, model, device=None):
        super().__init__("entropy", model, device)
    
    def select_examples(self, dataset, num_to_select=1, **kwargs):
        """Select examples with highest prediction uncertainty."""
        uncertainties = []
        
        for idx in range(len(dataset)):
            known_questions, inputs, answers, annotators, questions, embeddings = dataset[idx]
            inputs = inputs.unsqueeze(0).to(self.device)
            annotators = annotators.unsqueeze(0).to(self.device)
            questions = questions.unsqueeze(0).to(self.device)
            if embeddings is not None:
                embeddings = embeddings.unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs, annotators, questions, embeddings)
                
                total_entropy = 0.0
                masked_count = 0
                
                for pos in range(outputs.shape[1]):
                    if inputs[0, pos, 0] == 1:  # Masked position
                        probs = F.softmax(outputs[0, pos], dim=0)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                        total_entropy += entropy.item()
                        masked_count += 1
                
                avg_entropy = total_entropy / max(masked_count, 1)
                uncertainties.append((idx, avg_entropy))
        
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in uncertainties[:num_to_select]]
        selected_scores = [score for _, score in uncertainties[:num_to_select]]
        
        return selected_indices, selected_scores


class EntropyFeatureSelectionStrategy(FeatureSelectionStrategy):
    """Entropy-based feature selection strategy."""
    
    def __init__(self, model, device=None):
        super().__init__("entropy", model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, **kwargs):
        """Select features with highest prediction uncertainty."""
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, annotators, questions, embeddings)
            
            position_entropies = []
            for pos in masked_positions:
                probs = F.softmax(outputs[0, pos], dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                position_entropies.append((pos, entropy, 1.0, entropy))
        
        position_entropies.sort(key=lambda x: x[1], reverse=True)
        return position_entropies[:num_to_select]


class BADGESelectionStrategy(ExampleSelectionStrategy):
    """BADGE (Batch Active learning by Diverse Gradient Embeddings) selection strategy."""
    
    def __init__(self, model, device=None):
        super().__init__("badge", model, device)
    
    def select_examples(self, dataset, num_to_select=1, **kwargs):
        """Select diverse examples using gradient embeddings."""
        return random.sample(list(range(len(dataset))), min(num_to_select, len(dataset))), [1.0] * num_to_select


class VOICalculator:
    """Calculator for Value of Information (VOI)."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_loss(self, pred, loss_type="cross_entropy"):
        """Compute loss for prediction."""
        if loss_type == "cross_entropy" or loss_type == "nll":
            probs = F.softmax(pred, dim=-1)
            return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            
        elif loss_type == "l2":
            probs = F.softmax(pred, dim=-1)
            scores = torch.arange(1, 6, device=self.device).float()
            mean = torch.sum(probs * scores, dim=-1)
            variance = torch.sum(probs * (scores - mean.unsqueeze(-1)) ** 2, dim=-1).mean().item()
            return variance 
            
        elif loss_type == "0-1":
            probs = F.softmax(pred, dim=-1)
            max_prob = probs.max(dim=-1)[0].mean().item()
            return 1 - max_prob 
    
    def compute_voi(self, model, inputs, annotators, questions, embeddings, known_questions, candidate_idx, 
                   target_indices, loss_type="cross_entropy", cost=1.0):
        """Compute VOI using batch processing approach."""
        model.eval()
        batch_size = inputs.shape[0]
        
        with torch.no_grad():
            outputs = model(inputs, annotators, questions, embeddings)
            
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                target_preds = outputs[:, target_idx, :]
            else:
                target_preds = torch.cat([outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            loss_initial = self.compute_loss(target_preds, loss_type)
            
            candidate_pred = outputs[:, candidate_idx, :]
            candidate_probs = F.softmax(candidate_pred, dim=-1)
            num_classes = candidate_probs.shape[-1]
            
            expanded_inputs = inputs.repeat(num_classes, 1, 1)
            expanded_annotators = annotators.repeat(num_classes, 1)
            expanded_questions = questions.repeat(num_classes, 1)
            if embeddings is not None:
                expanded_embeddings = embeddings.repeat(num_classes, 1, 1)
            else:
                expanded_embeddings = None
            
            for i in range(num_classes):
                class_batch_start = i * batch_size
                class_batch_end = (i + 1) * batch_size
                
                one_hot = F.one_hot(torch.tensor(i), num_classes=num_classes).float().to(self.device)
                expanded_inputs[class_batch_start:class_batch_end, candidate_idx, 1:] = one_hot
                expanded_inputs[class_batch_start:class_batch_end, candidate_idx, 0] = 0
            
            expanded_outputs = model(expanded_inputs, expanded_annotators, expanded_questions, expanded_embeddings)
            
            all_losses = []
            
            for i in range(num_classes):
                class_batch_start = i * batch_size
                class_batch_end = (i + 1) * batch_size
                
                if isinstance(target_indices, list) and len(target_indices) == 1:
                    target_idx = target_indices[0]
                    class_target_preds = expanded_outputs[class_batch_start:class_batch_end, target_idx, :]
                else:
                    class_target_preds = torch.cat([
                        expanded_outputs[class_batch_start:class_batch_end, idx, :].unsqueeze(1) 
                        for idx in target_indices
                    ], dim=1)
                
                class_loss = self.compute_loss(class_target_preds, loss_type)
                all_losses.append(class_loss)
            
            expected_posterior_loss = 0.0
            for i in range(num_classes):
                expected_posterior_loss += candidate_probs[0, i].item() * all_losses[i]
            
            voi = loss_initial - expected_posterior_loss
            voi_cost_ratio = voi / max(cost, 1e-10)
            
            return voi, voi_cost_ratio, expected_posterior_loss


class FastVOICalculator(VOICalculator):
    """Fast VOI calculator that uses gradient-based approximation for efficiency."""
    
    def __init__(self, model, device=None, loss_type="cross_entropy"):
        super().__init__(model, device)
        self.loss_type = loss_type
    
    def compute_fast_voi(self, model, inputs, annotators, questions, known_questions, embeddings, candidate_idx, target_indices, loss_type=None, num_samples=3, cost=1.0):
        """Compute VOI using gradient-based approximation."""
        model.eval()
        loss_type = loss_type or self.loss_type
        batch_size = inputs.shape[0]
        input_dim = inputs.shape[2]
        
        with torch.enable_grad():
            inputs_grad = inputs.clone().requires_grad_(True)
            
            outputs = model(inputs_grad, annotators, questions, embeddings)
            
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                target_preds = outputs[:, target_idx, :]
            else:
                target_preds = torch.cat([outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            loss_initial = self.compute_loss(target_preds, loss_type)
            
            candidate_pred = outputs[:, candidate_idx, :]
            candidate_probs = F.softmax(candidate_pred, dim=-1)
            num_classes = candidate_probs.shape[-1]
            
            expanded_inputs = inputs.repeat(num_classes, 1, 1)
            expanded_annotators = annotators.repeat(num_classes, 1)
            expanded_questions = questions.repeat(num_classes, 1)
            if embeddings is not None:
                expanded_embeddings = embeddings.repeat(num_classes, 1, 1)
            else:
                expanded_embeddings = None
            
            for i in range(num_classes):
                class_batch_start = i * batch_size
                class_batch_end = (i + 1) * batch_size
                
                one_hot = F.one_hot(torch.tensor(i), num_classes=num_classes).float().to(self.device)
                expanded_inputs[class_batch_start:class_batch_end, candidate_idx, 1:] = one_hot
                expanded_inputs[class_batch_start:class_batch_end, candidate_idx, 0] = 0
            
            expanded_outputs = model(expanded_inputs, expanded_annotators, expanded_questions, expanded_embeddings)
            
            all_losses = []
            
            for i in range(num_classes):
                class_batch_start = i * batch_size
                class_batch_end = (i + 1) * batch_size
                
                if isinstance(target_indices, list) and len(target_indices) == 1:
                    target_idx = target_indices[0]
                    class_target_preds = expanded_outputs[class_batch_start:class_batch_end, target_idx, :]
                else:
                    class_target_preds = torch.cat([
                        expanded_outputs[class_batch_start:class_batch_end, idx, :].unsqueeze(1) 
                        for idx in target_indices
                    ], dim=1)
                
                class_loss = self.compute_loss(class_target_preds, loss_type)
                all_losses.append(class_loss)
            
            expected_posterior_loss = 0.0
            for i in range(num_classes):
                expected_posterior_loss += candidate_probs[0, i].item() * all_losses[i]
            
            voi = loss_initial - expected_posterior_loss
            voi_cost_ratio = voi / max(cost, 1e-10)
            
            return voi, voi_cost_ratio, expected_posterior_loss, 0


class ArgmaxVOICalculator(VOICalculator):
    """VOI calculator that only considers the argmax value."""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def compute_argmax_voi(self, model, inputs, annotators, questions, known_questions, embeddings, candidate_idx, target_indices, loss_type="cross_entropy", cost=1.0):
        """Compute VOI using only the argmax value instead of expectation."""
        model.eval()

        with torch.no_grad():
            outputs = model(inputs, annotators, questions, embeddings)
            
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                target_preds = outputs[:, target_idx, :]
            else:
                target_preds = torch.cat([outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            loss_initial = self.compute_loss(target_preds, loss_type)
            
            candidate_pred = outputs[:, candidate_idx, :]
            candidate_probs = F.softmax(candidate_pred, dim=-1)
            
            most_likely_class = torch.argmax(candidate_probs, dim=1)[0].item()
            
            batch_size = inputs.shape[0]
            num_classes = candidate_probs.shape[-1]
            
            input_with_answer = inputs.clone()
            one_hot = F.one_hot(torch.tensor(most_likely_class), num_classes=num_classes).float().to(self.device)
            input_with_answer[:, candidate_idx, 1:] = one_hot
            input_with_answer[:, candidate_idx, 0] = 0
          
            new_outputs = model(input_with_answer, annotators, questions, embeddings)
            
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                new_target_preds = new_outputs[:, target_idx, :]
            else:
                new_target_preds = torch.cat([new_outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            posterior_loss = self.compute_loss(new_target_preds, loss_type)
            
            voi = loss_initial - posterior_loss
            voi_cost_ratio = voi / max(cost, 1e-10)
            
            return voi, voi_cost_ratio, posterior_loss


class VOISelectionStrategy(FeatureSelectionStrategy):
    """VOI-based feature selection strategy for Active Feature Acquisition."""
    
    def __init__(self, model, device=None):
        super().__init__("voi", model, device)
        self.voi_calculator = VOICalculator(model, device)

    def select_features(self, example_idx, dataset, num_to_select=1, target_questions=None, 
                   loss_type="cross_entropy", costs=None, **kwargs):
        """Select features using VOI within a given example."""
        if target_questions is None:
            target_questions = [0]
        
        if isinstance(target_questions[0], str):
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            target_questions = [question_list.index(q) for q in target_questions if q in question_list]
        
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        answers = answers.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        known_questions = known_questions.unsqueeze(0).to(self.device)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        
        target_indices = []
        for q_idx in target_questions:
            for i in range(questions.shape[1]):
                if questions[0, i].item() == q_idx and annotators[0, i].item() >= 0:
                    target_indices.append(i)
        
        if not target_indices:
            return []
        
        position_vois = []
        for position in masked_positions:
            cost = 1.0
            if costs and position in costs:
                cost = costs[position]
                
            voi, voi_cost_ratio, posterior_loss = self.voi_calculator.compute_voi(
                self.model, inputs, annotators, questions, embeddings, known_questions,
                position, target_indices, loss_type, cost=cost
            )
            
            position_vois.append((position, voi, cost, voi_cost_ratio))
        
        position_vois.sort(key=lambda x: x[3], reverse=True)
        
        return position_vois[:num_to_select]


class FastVOISelectionStrategy(FeatureSelectionStrategy):
    """Fast VOI-based feature selection strategy for Active Feature Acquisition."""
    
    def __init__(self, model, device=None, loss_type="cross_entropy"):
        super().__init__("fast_voi", model, device)
        self.voi_calculator = FastVOICalculator(model, device, loss_type)
        self.loss_type = loss_type
    
    def select_features(self, example_idx, dataset, num_to_select=1, target_questions=None, 
                       loss_type=None, num_samples=3, costs=None, **kwargs):
        """Select features using gradient-based VOI approximation."""
        loss_type = loss_type or self.loss_type
        
        if target_questions is None:
            target_questions = [0]
        
        if isinstance(target_questions[0], str):
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            target_questions = [question_list.index(q) for q in target_questions if q in question_list]
        
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        answers = answers.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        known_questions = known_questions.unsqueeze(0).to(self.device)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        
        target_indices = []
        for q_idx in target_questions:
            for i in range(questions.shape[1]):
                if questions[0, i].item() == q_idx and annotators[0, i].item() >= 0:
                    target_indices.append(i)
        
        if not target_indices:
            return []
        
        position_vois = []
        for position in masked_positions:
            cost = 1.0
            if costs and position in costs:
                cost = costs[position]
                
            voi, voi_cost_ratio, posterior_loss, most_informative_class = self.voi_calculator.compute_fast_voi(
                self.model, inputs, annotators, questions, known_questions, embeddings,
                position, target_indices, loss_type, num_samples, cost=cost
            )
            
            position_vois.append((position, voi, cost, voi_cost_ratio, most_informative_class))
        
        position_vois.sort(key=lambda x: x[3], reverse=True)
        
        return position_vois[:num_to_select]


class ArgmaxVOISelectionStrategy(FeatureSelectionStrategy):
    """VOI-based feature selection strategy that only considers the argmax value."""
    
    def __init__(self, model, device=None):
        super().__init__("voi_argmax", model, device)
        self.voi_calculator = ArgmaxVOICalculator(model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, target_questions=None, 
                       loss_type="cross_entropy", costs=None, **kwargs):
        """Select features using ArgmaxVOI within a given example."""
        if target_questions is None:
            target_questions = [0]
        
        if isinstance(target_questions[0], str):
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            target_questions = [question_list.index(q) for q in target_questions if q in question_list]
        
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        answers = answers.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        known_questions = known_questions.unsqueeze(0).to(self.device)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        
        target_indices = []
        for q_idx in target_questions:
            for i in range(questions.shape[1]):
                if questions[0, i].item() == q_idx and annotators[0, i].item() >= 0:
                    target_indices.append(i)
        
        if not target_indices:
            return []
        
        position_vois = []
        for position in masked_positions:
            cost = 1.0
            if costs and position in costs:
                cost = costs[position]
                
            voi, voi_cost_ratio, posterior_loss = self.voi_calculator.compute_argmax_voi(
                self.model, inputs, annotators, questions, known_questions, embeddings,
                position, target_indices, loss_type, cost=cost
            )
            
            position_vois.append((position, voi, cost, voi_cost_ratio))
        
        position_vois.sort(key=lambda x: x[3], reverse=True)
        
        return position_vois[:num_to_select]


class GradientSelector:
    """Helper class for gradient-based selection. FIXED to never use ground truth for unobserved positions."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def normalize_gradient(self, grad_dict):
        """Normalize gradients by their total L2 norm."""
        total_norm_squared = 0.0
        for name, grad in grad_dict.items():
            total_norm_squared += torch.sum(grad ** 2).item()
        
        if total_norm_squared <= 1e-10:
            return grad_dict
        
        total_norm = math.sqrt(total_norm_squared)
        normalized_grad_dict = {}
        
        for name, grad in grad_dict.items():
            normalized_grad_dict[name] = grad / total_norm
        
        return normalized_grad_dict
    
    def compute_grad_dot_product(self, grad_dict1, grad_dict2):
        """Compute dot product between two gradient dictionaries."""
        dot_product = 0.0
        
        for name in grad_dict1:
            if name in grad_dict2:
                dot_product += torch.sum(-grad_dict1[name] * grad_dict2[name]).item()
        
        return dot_product
    
    def compute_sample_gradient(self, model, inputs, labels, annotators, questions, embeddings):
        """Compute gradient for a single example using autoregressive sampling. FIXED VERSION."""
        model.train()
        grad_dict = {}
        
        masked_positions = []
        for j in range(inputs.shape[1]):
            if inputs[0, j, 0] == 1:
                masked_positions.append(j)
        
        if not masked_positions:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad_dict[name] = torch.zeros_like(param)
            return grad_dict
        
        temp_inputs = inputs.clone()
        temp_labels = torch.zeros_like(labels)
        
        for pos in range(inputs.shape[1]):
            if inputs[0, pos, 0] == 0:
                temp_labels[0, pos] = labels[0, pos]
            else:
                with torch.no_grad():
                    current_outputs = model(temp_inputs, annotators, questions, embeddings)
                    var_outputs = current_outputs[0, pos]
                    var_probs = F.softmax(var_outputs, dim=0)
                
                sampled_class = torch.multinomial(var_probs, 1).item()
                
                one_hot = torch.zeros(model.max_choices, device=self.device)
                one_hot[sampled_class] = 1.0
                
                temp_inputs[0, pos, 0] = 0
                temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                temp_labels[0, pos] = one_hot
        
        model.zero_grad()
        
        outputs = model(temp_inputs, annotators, questions, embeddings)
        loss = model.compute_total_loss(
            outputs, temp_labels, temp_inputs, questions, embeddings,
            full_supervision=True
        )
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] = param.grad.detach().clone()
        
        model.zero_grad()
        
        return grad_dict
    
    def compute_example_gradients(self, model, inputs, labels, annotators, questions, embeddings, num_samples=5):
        """Compute gradients for a single example with multiple samples."""
        grad_dict = {}
        
        for _ in range(num_samples):
            sample_grad_dict = self.compute_sample_gradient(
                model, inputs, labels, annotators, questions, embeddings
            )
            
            for name, grad in sample_grad_dict.items():
                if name not in grad_dict:
                    grad_dict[name] = grad
                else:
                    grad_dict[name] += grad
        
        if num_samples > 0:
            for name in grad_dict:
                grad_dict[name] /= num_samples
        
        return grad_dict
    
    def compute_validation_gradient_sampled(self, model, val_dataloader, num_samples=5):
        """Compute validation gradients using sampling approach. FIXED VERSION."""
        model.train()
        grad_samples = []
        
        for _ in tqdm(range(num_samples), desc="Computing validation gradients"):
            temp_grad_dict = {}
            sample_count = 0
            
            for batch in val_dataloader:
                known_questions, inputs, labels, annotators, questions, embeddings = batch
                inputs, labels, annotators, questions = (
                    inputs.to(self.device), labels.to(self.device), 
                    annotators.to(self.device), questions.to(self.device)
                )

                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                
                batch_size = inputs.shape[0]
                
                for i in range(batch_size):
                    temp_inputs = inputs[i:i+1].clone()
                    temp_labels = torch.zeros_like(labels[i:i+1])
                    
                    for pos in range(temp_inputs.shape[1]):
                        if temp_inputs[0, pos, 0] == 0:
                            temp_labels[0, pos] = labels[i, pos]
                        else:
                            with torch.no_grad():
                                current_outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                                      embeddings[i:i+1] if embeddings is not None else None)
                                var_outputs = current_outputs[0, pos]
                                var_probs = F.softmax(var_outputs, dim=0)
                            
                            sampled_class = torch.multinomial(var_probs, 1).item()
                            
                            one_hot = torch.zeros(model.max_choices, device=self.device)
                            one_hot[sampled_class] = 1.0
                            
                            temp_inputs[0, pos, 0] = 0
                            temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                            temp_labels[0, pos] = one_hot
                    
                    model.zero_grad()
                    outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                  embeddings[i:i+1] if embeddings is not None else None)
                    batch_loss = model.compute_total_loss(
                        outputs, temp_labels, temp_inputs, questions[i:i+1], 
                        embeddings[i:i+1] if embeddings is not None else None,
                        full_supervision=True
                    )
                    
                    if batch_loss > 0:
                        batch_loss.backward()
                        sample_count += 1
                        
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                if name not in temp_grad_dict:
                                    temp_grad_dict[name] = param.grad.detach().clone()
                                else:
                                    temp_grad_dict[name] += param.grad.detach().clone()
            
            if sample_count > 0:
                for name in temp_grad_dict:
                    temp_grad_dict[name] /= sample_count
                
                normalized_grad_dict = self.normalize_gradient(temp_grad_dict)
                grad_samples.append(normalized_grad_dict)
        
        return grad_samples
    
    def compute_all_variable_gradients_efficient(self, model, inputs, labels, annotators, questions, embeddings, masked_positions, num_samples=5):
        """Efficiently compute gradients for all masked positions."""
        all_position_grads = {pos: {} for pos in masked_positions}
        
        for sample_idx in range(num_samples):
            model.train()
            temp_inputs = inputs.clone()
            temp_labels = torch.zeros_like(labels)
            
            for pos in range(temp_inputs.shape[1]):
                if temp_inputs[0, pos, 0] == 0:
                    temp_labels[0, pos] = labels[0, pos]
                else:
                    with torch.no_grad():
                        current_outputs = model(temp_inputs, annotators, questions, embeddings)
                        var_outputs = current_outputs[0, pos]
                        var_probs = F.softmax(var_outputs, dim=0)
                    
                    sampled_class = torch.multinomial(var_probs, 1).item()
                    
                    one_hot = torch.zeros(model.max_choices, device=self.device)
                    one_hot[sampled_class] = 1.0
                    
                    temp_inputs[0, pos, 0] = 0
                    temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                    temp_labels[0, pos] = one_hot
            
            for target_pos in masked_positions:
                model.zero_grad()
                outputs = model(temp_inputs, annotators, questions, embeddings)
                
                position_output = outputs[0, target_pos]
                position_label = temp_labels[0, target_pos]
                loss = F.cross_entropy(position_output.unsqueeze(0), 
                                      torch.argmax(position_label).unsqueeze(0))
                
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if name not in all_position_grads[target_pos]:
                            all_position_grads[target_pos][name] = param.grad.detach().clone()
                        else:
                            all_position_grads[target_pos][name] += param.grad.detach().clone()
                
                model.zero_grad()
        
        if num_samples > 0:
            for pos in masked_positions:
                for name in all_position_grads[pos]:
                    all_position_grads[pos][name] /= num_samples
        
        return all_position_grads


class GradientTopOnlySelector:
    """Helper class for gradient-based selection (top layer only). FIXED VERSION."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_top_layer_param(self, param_name):
        """Helper method to identify if a parameter belongs to the top layer."""
        top_layer_identifiers = ['encoder.layers.5.out']
        return any(identifier in param_name.lower() for identifier in top_layer_identifiers)
    
    def normalize_gradient(self, grad_dict):
        """Normalize gradients by their total L2 norm."""
        total_norm_squared = 0.0
        for name, grad in grad_dict.items():
            total_norm_squared += torch.sum(grad ** 2).item()
        
        if total_norm_squared <= 1e-10:
            return grad_dict
        
        total_norm = math.sqrt(total_norm_squared)
        normalized_grad_dict = {}
        
        for name, grad in grad_dict.items():
            normalized_grad_dict[name] = grad / total_norm
        
        return normalized_grad_dict
    
    def compute_grad_dot_product(self, grad_dict1, grad_dict2):
        """Compute dot product between two gradient dictionaries."""
        dot_product = 0.0
        
        for name in grad_dict1:
            if name in grad_dict2:
                dot_product += torch.sum(-grad_dict1[name] * grad_dict2[name]).item()
        
        return dot_product
    
    def compute_sample_gradient(self, model, inputs, labels, annotators, questions, embeddings):
        """Compute gradient for a single example using autoregressive sampling (top layer only). FIXED VERSION."""
        model.train()
        grad_dict = {}
        
        masked_positions = []
        for j in range(inputs.shape[1]):
            if inputs[0, j, 0] == 1:
                masked_positions.append(j)
        
        if not masked_positions:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad_dict[name] = torch.zeros_like(param)
            return grad_dict
        
        temp_inputs = inputs.clone()
        temp_labels = torch.zeros_like(labels)
        
        for pos in range(inputs.shape[1]):
            if inputs[0, pos, 0] == 0:
                temp_labels[0, pos] = labels[0, pos]
            else:
                with torch.no_grad():
                    current_outputs = model(temp_inputs, annotators, questions, embeddings)
                    var_outputs = current_outputs[0, pos]
                    var_probs = F.softmax(var_outputs, dim=0)
                
                sampled_class = torch.multinomial(var_probs, 1).item()
                
                one_hot = torch.zeros(model.max_choices, device=self.device)
                one_hot[sampled_class] = 1.0
                
                temp_inputs[0, pos, 0] = 0
                temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                temp_labels[0, pos] = one_hot
        
        non_top_params = []
        for name, param in model.named_parameters():
            if not self._is_top_layer_param(name):
                if param.requires_grad:
                    param.requires_grad = False
                    non_top_params.append(param)

        model.zero_grad()
        outputs = model(temp_inputs, annotators, questions, embeddings)
        loss = model.compute_total_loss(
            outputs, temp_labels, temp_inputs, questions, embeddings,
            full_supervision=True
        )
        loss.backward()

        for param in non_top_params:
            param.requires_grad = True
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and self._is_top_layer_param(name):
                grad_dict[name] = param.grad.detach().clone()
        
        model.zero_grad()
        
        return grad_dict
    
    def compute_example_gradients(self, model, inputs, labels, annotators, questions, embeddings, num_samples=5):
        """Compute gradients for a single example with multiple samples."""
        grad_dict = {}
        
        for _ in range(num_samples):
            sample_grad_dict = self.compute_sample_gradient(
                model, inputs, labels, annotators, questions, embeddings
            )
            
            for name, grad in sample_grad_dict.items():
                if name not in grad_dict:
                    grad_dict[name] = grad
                else:
                    grad_dict[name] += grad
        
        if num_samples > 0:
            for name in grad_dict:
                grad_dict[name] /= num_samples
        
        return grad_dict
    
    def compute_validation_gradient_sampled(self, model, val_dataloader, num_samples=5):
        """Compute validation gradients using sampling approach (top layer only). FIXED VERSION."""
        model.train()
        grad_samples = []
        
        for _ in tqdm(range(num_samples), desc="Computing validation gradients"):
            temp_grad_dict = {}
            sample_count = 0
            
            for batch in val_dataloader:
                known_questions, inputs, labels, annotators, questions, embeddings = batch
                inputs, labels, annotators, questions = (
                    inputs.to(self.device), labels.to(self.device), 
                    annotators.to(self.device), questions.to(self.device)
                )

                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                
                batch_size = inputs.shape[0]
                
                for i in range(batch_size):
                    temp_inputs = inputs[i:i+1].clone()
                    temp_labels = torch.zeros_like(labels[i:i+1])
                    
                    for pos in range(temp_inputs.shape[1]):
                        if temp_inputs[0, pos, 0] == 0:
                            temp_labels[0, pos] = labels[i, pos]
                        else:
                            with torch.no_grad():
                                current_outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                                      embeddings[i:i+1] if embeddings is not None else None)
                                var_outputs = current_outputs[0, pos]
                                var_probs = F.softmax(var_outputs, dim=0)
                            
                            sampled_class = torch.multinomial(var_probs, 1).item()
                            
                            one_hot = torch.zeros(model.max_choices, device=self.device)
                            one_hot[sampled_class] = 1.0
                            
                            temp_inputs[0, pos, 0] = 0
                            temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                            temp_labels[0, pos] = one_hot
                    
                    non_top_params = []
                    for name, param in model.named_parameters():
                        if not self._is_top_layer_param(name):
                            if param.requires_grad:
                                param.requires_grad = False
                                non_top_params.append(param)

                    model.zero_grad()
                    outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                  embeddings[i:i+1] if embeddings is not None else None)
                    batch_loss = model.compute_total_loss(
                        outputs, temp_labels, temp_inputs, questions[i:i+1], 
                        embeddings[i:i+1] if embeddings is not None else None,
                        full_supervision=True
                    )
                    
                    if batch_loss > 0:
                        batch_loss.backward()
                        sample_count += 1
                        
                        for name, param in model.named_parameters():
                            if param.grad is not None and self._is_top_layer_param(name):
                                if name not in temp_grad_dict:
                                    temp_grad_dict[name] = param.grad.detach().clone()
                                else:
                                    temp_grad_dict[name] += param.grad.detach().clone()
                    for param in non_top_params:
                        param.requires_grad = True
            
            if sample_count > 0:
                for name in temp_grad_dict:
                    temp_grad_dict[name] /= sample_count
                
                normalized_grad_dict = self.normalize_gradient(temp_grad_dict)
                grad_samples.append(normalized_grad_dict)
        
        return grad_samples


class VariableGradientTopOnlySelector:
    """Variable-level gradient selector that only considers top layer parameters. FIXED VERSION."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_top_layer_param(self, param_name):
        """Helper method to identify if a parameter belongs to the top layer."""
        top_layer_identifiers = ['encoder.layers.5.out']
        return any(identifier in param_name.lower() for identifier in top_layer_identifiers)
    
    def normalize_gradient(self, grad_dict):
        """Normalize gradients by their total L2 norm."""
        total_norm_squared = 0.0
        for name, grad in grad_dict.items():
            total_norm_squared += torch.sum(grad ** 2).item()
        
        if total_norm_squared <= 1e-10:
            return grad_dict
        
        total_norm = math.sqrt(total_norm_squared)
        normalized_grad_dict = {}
        
        for name, grad in grad_dict.items():
            normalized_grad_dict[name] = grad / total_norm
        
        return normalized_grad_dict
    
    def compute_grad_dot_product(self, grad_dict1, grad_dict2):
        """Compute dot product between two gradient dictionaries."""
        dot_product = 0.0
        
        for name in grad_dict1:
            if name in grad_dict2:
                dot_product += torch.sum(-grad_dict1[name] * grad_dict2[name]).item()
        
        return dot_product
    
    def compute_validation_gradient_sampled(self, model, val_dataloader, num_samples=5):
        """Compute validation gradients using sampling approach (top layer only). FIXED VERSION."""
        model.train()
        grad_samples = []
        
        for _ in tqdm(range(num_samples), desc="Computing validation gradients"):
            temp_grad_dict = {}
            sample_count = 0
            
            for batch in val_dataloader:
                known_questions, inputs, labels, annotators, questions, embeddings = batch
                inputs, labels, annotators, questions = (
                    inputs.to(self.device), labels.to(self.device), 
                    annotators.to(self.device), questions.to(self.device)
                )

                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                
                batch_size = inputs.shape[0]
                
                for i in range(batch_size):
                    temp_inputs = inputs[i:i+1].clone()
                    temp_labels = torch.zeros_like(labels[i:i+1])
                    
                    for pos in range(temp_inputs.shape[1]):
                        if temp_inputs[0, pos, 0] == 0:
                            temp_labels[0, pos] = labels[i, pos]
                        else:
                            with torch.no_grad():
                                current_outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                                      embeddings[i:i+1] if embeddings is not None else None)
                                var_outputs = current_outputs[0, pos]
                                var_probs = F.softmax(var_outputs, dim=0)
                            
                            sampled_class = torch.multinomial(var_probs, 1).item()
                            
                            one_hot = torch.zeros(model.max_choices, device=self.device)
                            one_hot[sampled_class] = 1.0
                            
                            temp_inputs[0, pos, 0] = 0
                            temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                            temp_labels[0, pos] = one_hot
                    
                    non_top_params = []
                    for name, param in model.named_parameters():
                        if not self._is_top_layer_param(name):
                            if param.requires_grad:
                                param.requires_grad = False
                                non_top_params.append(param)

                    model.zero_grad()
                    outputs = model(temp_inputs, annotators[i:i+1], questions[i:i+1], 
                                  embeddings[i:i+1] if embeddings is not None else None)
                    batch_loss = model.compute_total_loss(
                        outputs, temp_labels, temp_inputs, questions[i:i+1], 
                        embeddings[i:i+1] if embeddings is not None else None,
                        full_supervision=True
                    )
                    
                    if batch_loss > 0:
                        batch_loss.backward()
                        sample_count += 1
                        
                        for name, param in model.named_parameters():
                            if param.grad is not None and self._is_top_layer_param(name):
                                if name not in temp_grad_dict:
                                    temp_grad_dict[name] = param.grad.detach().clone()
                                else:
                                    temp_grad_dict[name] += param.grad.detach().clone()
                    for param in non_top_params:
                        param.requires_grad = True
            
            if sample_count > 0:
                for name in temp_grad_dict:
                    temp_grad_dict[name] /= sample_count
                
                normalized_grad_dict = self.normalize_gradient(temp_grad_dict)
                grad_samples.append(normalized_grad_dict)
        
        return grad_samples
    
    def compute_all_variable_gradients_efficient(self, model, inputs, labels, annotators, questions, embeddings, masked_positions, num_samples=5):
        """Efficiently compute gradients for all masked positions (top layer only). FIXED VERSION."""
        all_position_grads = {pos: {} for pos in masked_positions}
        
        for sample_idx in range(num_samples):
            model.train()
            temp_inputs = inputs.clone()
            temp_labels = torch.zeros_like(labels)
            
            for pos in range(temp_inputs.shape[1]):
                if temp_inputs[0, pos, 0] == 0:
                    temp_labels[0, pos] = labels[0, pos]
                else:
                    with torch.no_grad():
                        current_outputs = model(temp_inputs, annotators, questions, embeddings)
                        var_outputs = current_outputs[0, pos]
                        var_probs = F.softmax(var_outputs, dim=0)
                    
                    sampled_class = torch.multinomial(var_probs, 1).item()
                    
                    one_hot = torch.zeros(model.max_choices, device=self.device)
                    one_hot[sampled_class] = 1.0
                    
                    temp_inputs[0, pos, 0] = 0
                    temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
                    temp_labels[0, pos] = one_hot
            
            non_top_params = []
            for name, param in model.named_parameters():
                if not self._is_top_layer_param(name):
                    if param.requires_grad:
                        param.requires_grad = False
                        non_top_params.append(param)
            
            for target_pos in masked_positions:
                model.zero_grad()
                outputs = model(temp_inputs, annotators, questions, embeddings)
                
                position_output = outputs[0, target_pos]
                position_label = temp_labels[0, target_pos]
                loss = F.cross_entropy(position_output.unsqueeze(0), 
                                      torch.argmax(position_label).unsqueeze(0))
                
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None and self._is_top_layer_param(name):
                        if name not in all_position_grads[target_pos]:
                            all_position_grads[target_pos][name] = param.grad.detach().clone()
                        else:
                            all_position_grads[target_pos][name] += param.grad.detach().clone()
                
                model.zero_grad()
            
            for param in non_top_params:
                param.requires_grad = True
        
        if num_samples > 0:
            for pos in masked_positions:
                for name in all_position_grads[pos]:
                    all_position_grads[pos][name] /= num_samples
        
        return all_position_grads


class GradientSelectionStrategy(ExampleSelectionStrategy):
    """Gradient-based example selection strategy for Active Learning."""
    
    def __init__(self, model, device=None, gradient_top_only=False):
        super().__init__("gradient", model, device)
        if gradient_top_only:
            self.selector = GradientTopOnlySelector(model, device)
        else:
            self.selector = GradientSelector(model, device)
    
    def select_examples(self, dataset, num_to_select=1, val_dataset=None, 
                        num_samples=5, batch_size=32, costs=None, **kwargs):
        """Select examples using gradient alignment."""
        if val_dataset is None:
            raise ValueError("Validation dataset is required for gradient selection")
        
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        validation_grad_samples = self.selector.compute_validation_gradient_sampled(
            self.model, val_dataloader, num_samples=num_samples
        )
        
        all_scores = []
        all_indices = []
        all_costs = []
        all_bc_ratios = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradient alignment")):
            known_questions, inputs, labels, annotators, questions, embeddings = batch
            inputs, labels, annotators, questions = (
                inputs.to(self.device), labels.to(self.device), 
                annotators.to(self.device), questions.to(self.device)
            )
            if embeddings is not None:
                embeddings = embeddings.to(self.device)
            
            batch_size_actual = inputs.shape[0]
            
            for i in range(batch_size_actual):
                global_example_idx = batch_idx * batch_size + i
                
                masked_positions = []
                for j in range(inputs.shape[1]):
                    if inputs[i, j, 0] == 1:
                        masked_positions.append(j)
                
                if not masked_positions:
                    continue
                
                example_input = inputs[i:i+1]
                example_labels = labels[i:i+1]
                example_annotator = annotators[i:i+1]
                example_question = questions[i:i+1]
                example_embedding = embeddings[i:i+1] if embeddings is not None else None
                
                example_grad_dict = self.selector.compute_example_gradients(
                    self.model, 
                    example_input, example_labels, 
                    example_annotator, example_question, example_embedding,
                    num_samples=num_samples
                )
                
                example_grad_dict = self.selector.normalize_gradient(example_grad_dict)
                
                alignment_scores = []
                for val_grad in validation_grad_samples:
                    alignment = self.selector.compute_grad_dot_product(val_grad, example_grad_dict)
                    alignment_scores.append(alignment)
                
                cost = 1.0
                if costs and global_example_idx in costs:
                    cost = costs[global_example_idx]
                
                avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
                benefit_cost_ratio = avg_alignment / max(cost, 1e-10)
                
                all_scores.append(avg_alignment)
                all_indices.append(global_example_idx)
                all_costs.append(cost)
                all_bc_ratios.append(benefit_cost_ratio)
        
        if all_scores:
            sorted_data = sorted(zip(all_indices, all_scores, all_costs, all_bc_ratios), 
                                key=lambda x: x[3], reverse=True)
            sorted_indices = [idx for idx, _, _, _ in sorted_data]
            sorted_scores = [score for _, score, _, _ in sorted_data]
        else:
            sorted_indices = []
            sorted_scores = []
        
        return sorted_indices[:num_to_select], sorted_scores[:num_to_select]


class VariableGradientSelectionStrategy(ExampleSelectionStrategy):
    """Variable-level gradient-based example selection strategy for Active Learning."""
    
    def __init__(self, model, device=None):
        super().__init__("variable_gradient", model, device)
        self.selector = VariableGradientTopOnlySelector(model, device)
    
    def select_examples(self, dataset, num_to_select=300, val_dataset=None, 
                        num_samples=5, batch_size=32, costs=None, **kwargs):
        """Select variables (masked positions) using gradient alignment."""
        if val_dataset is None:
            raise ValueError("Validation dataset is required for gradient selection")
        
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        validation_grad_samples = self.selector.compute_validation_gradient_sampled(
            self.model, val_dataloader, num_samples=num_samples
        )
        
        all_scores = []
        all_variable_positions = []
        all_costs = []
        all_bc_ratios = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing variable gradient alignment")):
            known_questions, inputs, labels, annotators, questions, embeddings = batch
            inputs, labels, annotators, questions = (
                inputs.to(self.device), labels.to(self.device), 
                annotators.to(self.device), questions.to(self.device)
            )
            if embeddings is not None:
                embeddings = embeddings.to(self.device)
            
            batch_size_actual = inputs.shape[0]
            
            for i in range(batch_size_actual):
                global_example_idx = batch_idx * batch_size + i
                
                masked_positions = []
                for j in range(inputs.shape[1]):
                    if inputs[i, j, 0] == 1:
                        masked_positions.append(j)
                
                if not masked_positions:
                    continue
                
                example_input = inputs[i:i+1]
                example_labels = labels[i:i+1]
                example_annotator = annotators[i:i+1]
                example_question = questions[i:i+1]
                example_embedding = embeddings[i:i+1] if embeddings is not None else None
                
                all_variable_grads = self.selector.compute_all_variable_gradients_efficient(
                    self.model, 
                    example_input, example_labels, 
                    example_annotator, example_question, example_embedding,
                    masked_positions, num_samples=num_samples
                )
                
                for pos in masked_positions:
                    if pos not in all_variable_grads:
                        continue
                        
                    variable_grad_dict = all_variable_grads[pos]
                    variable_grad_dict = self.selector.normalize_gradient(variable_grad_dict)
                    
                    alignment_scores = []
                    for val_grad in validation_grad_samples:
                        alignment = self.selector.compute_grad_dot_product(val_grad, variable_grad_dict)
                        alignment_scores.append(alignment)
                    
                    cost = 1.0
                    variable_key = (global_example_idx, pos)
                    if costs and variable_key in costs:
                        cost = costs[variable_key]
                    
                    avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
                    benefit_cost_ratio = avg_alignment / max(cost, 1e-10)
                    
                    all_scores.append(avg_alignment)
                    all_variable_positions.append(variable_key)
                    all_costs.append(cost)
                    all_bc_ratios.append(benefit_cost_ratio)
        
        if all_scores:
            sorted_data = sorted(zip(all_variable_positions, all_scores, all_costs, all_bc_ratios), 
                                key=lambda x: x[3], reverse=True)
            sorted_positions = [pos for pos, _, _, _ in sorted_data]
            sorted_scores = [score for _, score, _, _ in sorted_data]
        else:
            sorted_positions = []
            sorted_scores = []
        
        return sorted_positions[:num_to_select], sorted_scores[:num_to_select]


class CombinedSelectionStrategy:
    """Combined strategy that uses both example and feature selection."""
    
    def __init__(self, example_strategy, feature_strategy):
        self.example_strategy = example_strategy
        self.feature_strategy = feature_strategy
    
    def select_examples(self, dataset, num_to_select=1, **kwargs):
        """Select examples using the example strategy."""
        return self.example_strategy.select_examples(dataset, num_to_select, **kwargs)
    
    def select_features(self, example_idx, dataset, num_to_select=1, **kwargs):
        """Select features using the feature strategy."""
        return self.feature_strategy.select_features(example_idx, dataset, num_to_select, **kwargs)


class SelectionFactory:
    """Factory for creating selection strategies."""
    
    @staticmethod
    def create_example_strategy(strategy_name, model, device=None, gradient_top_only=False):
        """Create example selection strategy."""
        if strategy_name == "random":
            return RandomExampleSelectionStrategy(model, device)
        elif strategy_name == "gradient":
            return GradientSelectionStrategy(model, device, gradient_top_only=gradient_top_only)
        elif strategy_name == "entropy":
            return EntropyExampleSelectionStrategy(model, device)
        elif strategy_name == "badge":
            return BADGESelectionStrategy(model, device)
        elif strategy_name == "combine":
            return VariableGradientSelectionStrategy(model, device)
        else:
            raise ValueError(f"Unknown example selection strategy: {strategy_name}")
    
    @staticmethod
    def create_feature_strategy(strategy_name, model, device=None):
        """Create feature selection strategy."""
        if strategy_name == "random":
            return RandomFeatureSelectionStrategy(model, device)
        elif strategy_name == "voi":
            return VOISelectionStrategy(model, device)
        elif strategy_name == "fast_voi":
            return FastVOISelectionStrategy(model, device)
        elif strategy_name == "voi_argmax":
            return ArgmaxVOISelectionStrategy(model, device)
        elif strategy_name == "sequential":
            return RandomFeatureSelectionStrategy(model, device)
        elif strategy_name == "entropy":
            return EntropyFeatureSelectionStrategy(model, device)
        else:
            raise ValueError(f"Unknown feature selection strategy: {strategy_name}")
    
    @staticmethod
    def create_combined_strategy(example_strategy_name, feature_strategy_name, model, device=None):
        """Create combined selection strategy."""
        example_strategy = SelectionFactory.create_example_strategy(example_strategy_name, model, device)
        feature_strategy = SelectionFactory.create_feature_strategy(feature_strategy_name, model, device)
        return CombinedSelectionStrategy(example_strategy, feature_strategy)