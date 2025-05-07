import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math
import random
import copy
from torch.utils.data import DataLoader

class SelectionStrategy:
    """
    Base class for selection strategies.
    """
    
    def __init__(self, name, model, device=None):
        """
        Initialize selection strategy.
        
        Args:
            name: Name of the strategy
            model: Model to use for predictions
            device: Device to use for computations
        """
        self.name = name
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExampleSelectionStrategy(SelectionStrategy):
    """
    Base class for Active Learning strategies that select which examples to annotate.
    """
    
    def select_examples(self, dataset, num_to_select=1, costs=None, **kwargs):
        """
        Select examples for annotation.
        
        Args:
            dataset: Dataset to select from
            num_to_select: Number of examples to select
            costs: Dictionary mapping example indices to their annotation costs
            **kwargs: Additional arguments specific to the strategy
            
        Returns:
            list: Indices of selected examples
            list: Corresponding scores for selected examples
        """
        raise NotImplementedError("Subclasses must implement select_examples method")


class FeatureSelectionStrategy(SelectionStrategy):
    """
    Base class for Active Feature Acquisition strategies that select which 
    features (positions) to annotate within an example.
    """
    
    def select_features(self, example_idx, dataset, num_to_select=1, costs=None, **kwargs):
        """
        Select features (positions) to annotate within a given example.
        
        Args:
            example_idx: Index of the example to select features from
            dataset: Dataset containing the example
            num_to_select: Number of features to select
            costs: Dictionary mapping positions to their annotation costs
            **kwargs: Additional arguments specific to the strategy
            
        Returns:
            list: Tuples of (position_idx, benefit, cost, benefit/cost_ratio) for selected positions
        """
        raise NotImplementedError("Subclasses must implement select_features method")
    
    def select_batch_features(self, example_indices, dataset, num_to_select=1, costs=None, **kwargs):
        """
        Select features for multiple examples.
        
        Args:
            example_indices: Indices of examples to select features from
            dataset: Dataset containing the examples
            num_to_select: Number of features to select per example
            costs: Dictionary mapping (example_idx, position_idx) to costs
            **kwargs: Additional arguments specific to the strategy
            
        Returns:
            dict: Mapping from example indices to selected position indices with scores
        """
        selections = {}
        for idx in example_indices:
            example_costs = None
            if costs and idx in costs:
                example_costs = costs[idx]
                
            selected_positions = self.select_features(
                idx, dataset, num_to_select, costs=example_costs, **kwargs
            )
            selections[idx] = selected_positions
        return selections


class RandomExampleSelectionStrategy(ExampleSelectionStrategy):
    """
    Random example selection strategy for Active Learning.
    
    This strategy randomly selects examples without considering model predictions.
    It establishes a baseline for more sophisticated strategies.
    """
    
    def __init__(self, model, device=None):
        """Initialize random example selection strategy."""
        super().__init__("random_example", model, device)
    
    def select_examples(self, dataset, num_to_select=1, costs=None, **kwargs):
        """
        Randomly select examples for annotation.
        
        Args:
            dataset: Dataset to select from
            num_to_select: Number of examples to select
            costs: Dictionary mapping example indices to their annotation costs
            **kwargs: Additional arguments
            
        Returns:
            list: Indices of selected examples
            list: Scores (set to 1.0) for selected examples
        """
        # Get all valid examples (with at least one masked position)
        valid_indices = []
        
        for idx in range(len(dataset)):
            masked_positions = dataset.get_masked_positions(idx)
            if masked_positions:
                valid_indices.append(idx)
        
        # Select random indices
        if len(valid_indices) <= num_to_select:
            selected_indices = valid_indices
        else:
            selected_indices = random.sample(valid_indices, num_to_select)
        
        # Assign uniform scores of 1.0 (no real scoring for random)
        scores = [1.0] * len(selected_indices)
        
        return selected_indices, scores


class RandomFeatureSelectionStrategy(FeatureSelectionStrategy):
    """
    Random feature selection strategy for Active Feature Acquisition.
    
    This strategy randomly selects features within an example without
    considering model predictions. It establishes a baseline for more
    sophisticated feature selection strategies.
    """
    
    def __init__(self, model, device=None):
        """Initialize random feature selection strategy."""
        super().__init__("random_feature", model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, costs=None, **kwargs):
        """
        Randomly select features (positions) to annotate within a given example.
        
        Args:
            example_idx: Index of the example to select features from
            dataset: Dataset containing the example
            num_to_select: Number of features to select
            costs: Dictionary mapping positions to their annotation costs
            **kwargs: Additional arguments
            
        Returns:
            list: Tuples of (position_idx, benefit, cost, benefit/cost_ratio) for selected positions
        """
        # Get all masked positions for this example
        masked_positions = dataset.get_masked_positions(example_idx)
        
        # Select random positions
        if len(masked_positions) <= num_to_select:
            selected_positions = masked_positions
        else:
            selected_positions = random.sample(masked_positions, num_to_select)
        
        # Construct result with benefit/cost information
        result = []
        for pos in selected_positions:
            # Default cost is 1.0 if not specified
            cost = 1.0
            if costs and pos in costs:
                cost = costs[pos]
                
            # For random selection, benefit equals cost (benefit/cost ratio = 1.0)
            benefit = cost
            ratio = 1.0
            
            result.append((pos, benefit, cost, ratio))
        
        return result


class VOICalculator:
    """
    Value of Information (VOI) calculator.
    
    Computes the expected reduction in loss (benefit) from observing a variable,
    considering the cost of observation for true benefit/cost analysis.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize VOI calculator.
        
        Args:
            model: Model to use for predictions
            device: Device to use for computations
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_loss(self, pred, loss_type="cross_entropy"):
        """
        Compute loss for prediction.
        
        Args:
            pred: Prediction logits
            loss_type: Type of loss to compute ("cross_entropy", "l2", or "0-1")
            
        Returns:
            float: Loss value
        """
        if loss_type == "cross_entropy" or loss_type == "nll":
            # Entropy of the distribution (uncertainty)
            probs = F.softmax(pred, dim=-1)
            return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            
        elif loss_type == "l2":
            # Variance of the predicted distribution
            probs = F.softmax(pred, dim=-1)
            scores = torch.arange(1, 6, device=self.device).float()
            mean = torch.sum(probs * scores, dim=-1)
            variance = torch.sum(probs * (scores - mean.unsqueeze(-1)) ** 2, dim=-1).mean().item()
            return variance 
            
        elif loss_type == "0-1":
            # 1 - maximum probability (uncertainty in classification)
            probs = F.softmax(pred, dim=-1)
            max_prob = probs.max(dim=-1)[0].mean().item()
            return 1 - max_prob 
    
    def compute_voi(self, model, inputs, annotators, questions, known_questions, candidate_idx, 
                   target_indices, loss_type="cross_entropy", cost=1.0):
        """
        Compute VOI using batch processing approach.
        
        Args:
            model: Model to use for predictions
            inputs: Input tensor [batch_size, sequence_length, input_dim]
            annotators: Annotator indices [batch_size, sequence_length]
            questions: Question indices [batch_size, sequence_length]
            known_questions: Mask indicating known questions [batch_size, sequence_length]
            candidate_idx: Index of candidate annotation to evaluate
            target_indices: Target indices to compute loss on
            loss_type: Type of loss to compute
            cost: Cost of annotating this position
            
        Returns:
            tuple: (voi_value, voi/cost_ratio, expected_posterior_loss)
        """
        model.eval()

        with torch.no_grad():
            # Get initial outputs and compute initial loss
            outputs = model(inputs, annotators, questions)
            
            # Extract predictions for target indices
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                target_preds = outputs[:, target_idx, :]
            else:
                # Handle multiple target indices - concatenate predictions
                target_preds = torch.cat([outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            # Compute initial loss
            loss_initial = self.compute_loss(target_preds, loss_type)
            
            # Get prediction for candidate
            candidate_pred = outputs[:, candidate_idx, :]
            candidate_probs = F.softmax(candidate_pred, dim=-1)
            
            batch_size = inputs.shape[0]
            num_classes = candidate_probs.shape[-1]  # Usually 5 for our case
            
            # Process all possible answers in one batch
            expanded_inputs = []
            expanded_annotators = []
            expanded_questions = []
            
            for i in range(num_classes):
                # Create a copy of inputs with candidate set to class i
                input_with_answer = inputs.clone()
                one_hot = F.one_hot(torch.tensor(i), num_classes=num_classes).float().to(self.device)
                input_with_answer[:, candidate_idx, 1:] = one_hot
                input_with_answer[:, candidate_idx, 0] = 0  # Mark as observed
                
                expanded_inputs.append(input_with_answer)
                expanded_annotators.append(annotators.clone())
                expanded_questions.append(questions.clone())
            
            expanded_inputs = torch.cat(expanded_inputs, dim=0)
            expanded_annotators = torch.cat(expanded_annotators, dim=0)
            expanded_questions = torch.cat(expanded_questions, dim=0)
            
            # Get predictions for all possible answers
            expanded_outputs = model(expanded_inputs, expanded_annotators, expanded_questions)
            
            # Extract target predictions for each possible answer
            all_losses = []
            
            for i in range(num_classes):
                class_batch_start = i * batch_size
                class_batch_end = (i + 1) * batch_size
                
                if isinstance(target_indices, list) and len(target_indices) == 1:
                    target_idx = target_indices[0]
                    class_target_preds = expanded_outputs[class_batch_start:class_batch_end, target_idx, :]
                else:
                    # Handle multiple target indices
                    class_target_preds = torch.cat([
                        expanded_outputs[class_batch_start:class_batch_end, idx, :].unsqueeze(1) 
                        for idx in target_indices
                    ], dim=1)
                
                # Compute loss for this class assignment
                class_loss = self.compute_loss(class_target_preds, loss_type)
                all_losses.append(class_loss)
            
            # Weight losses by candidate distribution
            expected_posterior_loss = 0.0
            for i in range(num_classes):
                expected_posterior_loss += candidate_probs[0, i].item() * all_losses[i]
            
            # VOI is the expected reduction in loss
            voi = loss_initial - expected_posterior_loss
            
            # Compute benefit/cost ratio
            voi_cost_ratio = voi / max(cost, 1e-10)
            
            return voi, voi_cost_ratio, expected_posterior_loss


class FastVOICalculator(VOICalculator):
    """
    Fast VOI calculator that optimizes the VOI computation.
    
    Approximates VOI by sampling a subset of possible outcomes,
    making it more computationally efficient for large problems.
    """
    
    def __init__(self, model, device=None):
        """Initialize Fast VOI calculator."""
        super().__init__(model, device)
    
    def compute_fast_voi(self, model, inputs, annotators, questions, known_questions, 
                         candidate_idx, target_indices, loss_type="cross_entropy", 
                         num_samples=3, cost=1.0):
        """
        Compute VOI using fast approximation by sampling a subset of possible answers.
        
        Args:
            model: Model to use for predictions
            inputs: Input tensor [batch_size, sequence_length, input_dim]
            annotators: Annotator indices [batch_size, sequence_length]
            questions: Question indices [batch_size, sequence_length]
            known_questions: Mask indicating known questions [batch_size, sequence_length]
            candidate_idx: Index of candidate annotation to evaluate
            target_indices: Target indices to compute loss on
            loss_type: Type of loss to compute
            num_samples: Number of samples to use for approximation
            cost: Cost of annotating this position
            
        Returns:
            tuple: (voi_value, voi/cost_ratio, expected_posterior_loss, most_informative_class)
        """
        model.eval()

        with torch.no_grad():
            # Get initial outputs and compute initial loss
            outputs = model(inputs, annotators, questions)
            
            # Extract predictions for target indices
            if isinstance(target_indices, list) and len(target_indices) == 1:
                target_idx = target_indices[0]
                target_preds = outputs[:, target_idx, :]
            else:
                # Handle multiple target indices
                target_preds = torch.cat([outputs[:, idx, :].unsqueeze(1) for idx in target_indices], dim=1)
            
            # Compute initial loss
            loss_initial = self.compute_loss(target_preds, loss_type)
            
            # Get prediction for candidate
            candidate_pred = outputs[:, candidate_idx, :]
            candidate_probs = F.softmax(candidate_pred, dim=-1)
            
            batch_size = inputs.shape[0]
            num_classes = candidate_probs.shape[-1]  # Usually 5 for our case
            
            # Sample a subset of classes based on probabilities
            if num_samples < num_classes:
                # Sample based on probabilities
                sampled_classes = torch.multinomial(
                    candidate_probs[0], num_samples, replacement=False
                ).tolist()
            else:
                # Use all classes
                sampled_classes = list(range(num_classes))
            
            # Process sampled classes
            all_losses = {}
            
            for class_idx in sampled_classes:
                # Create a copy of inputs with candidate set to class_idx
                input_with_answer = inputs.clone()
                one_hot = F.one_hot(torch.tensor(class_idx), num_classes=num_classes).float().to(self.device)
                input_with_answer[:, candidate_idx, 1:] = one_hot
                input_with_answer[:, candidate_idx, 0] = 0  # Mark as observed
                
                # Get predictions
                class_outputs = model(input_with_answer, annotators, questions)
                
                # Extract predictions for target indices
                if isinstance(target_indices, list) and len(target_indices) == 1:
                    target_idx = target_indices[0]
                    class_target_preds = class_outputs[:, target_idx, :]
                else:
                    # Handle multiple target indices
                    class_target_preds = torch.cat([
                        class_outputs[:, idx, :].unsqueeze(1) for idx in target_indices
                    ], dim=1)
                
                # Compute loss for this class assignment
                class_loss = self.compute_loss(class_target_preds, loss_type)
                all_losses[class_idx] = class_loss
            
            # Compute expected posterior loss (weighted average of per-class losses)
            expected_posterior_loss = 0.0
            sampled_prob_sum = 0.0
            
            for class_idx, class_loss in all_losses.items():
                class_prob = candidate_probs[0, class_idx].item()
                expected_posterior_loss += class_prob * class_loss
                sampled_prob_sum += class_prob
            
            # Normalize by total probability mass of sampled classes
            if sampled_prob_sum > 0:
                expected_posterior_loss /= sampled_prob_sum
            
            # VOI is the expected reduction in loss
            voi = loss_initial - expected_posterior_loss
            
            # Compute benefit/cost ratio
            voi_cost_ratio = voi / max(cost, 1e-10)
            
            # Find most informative class (highest individual VOI)
            class_vois = {}
            for class_idx, class_loss in all_losses.items():
                class_vois[class_idx] = loss_initial - class_loss
            
            if class_vois:
                most_informative_class = max(class_vois, key=class_vois.get)
            else:
                most_informative_class = 0
            
            return voi, voi_cost_ratio, expected_posterior_loss, most_informative_class


class VOISelectionStrategy(FeatureSelectionStrategy):
    """
    VOI-based feature selection strategy for Active Feature Acquisition.
    
    Selects features that provide the highest value of information (expected
    reduction in loss) per unit cost, making annotation more cost-effective.
    """
    
    def __init__(self, model, device=None):
        """Initialize VOI selection strategy."""
        super().__init__("voi", model, device)
        self.voi_calculator = VOICalculator(model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, target_questions=None, 
                       loss_type="cross_entropy", costs=None, **kwargs):
        """
        Select features (positions) using VOI within a given example.
        
        Args:
            example_idx: Index of the example to select features from
            dataset: Dataset containing the example
            num_to_select: Number of features to select
            target_questions: Target questions to compute VOI for
            loss_type: Type of loss to compute
            costs: Dictionary mapping positions to their annotation costs
            **kwargs: Additional arguments
            
        Returns:
            list: Tuples of (position_idx, benefit, cost, benefit/cost_ratio) for selected positions
        """
        if target_questions is None:
            # Default to first question (Q0)
            target_questions = [0]
        
        # Convert target questions to indices if needed
        if isinstance(target_questions[0], str):
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            target_questions = [question_list.index(q) for q in target_questions if q in question_list]
        
        # Get masked positions
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        # Get data
        known_questions, inputs, answers, annotators, questions = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        answers = answers.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        known_questions = known_questions.unsqueeze(0).to(self.device)
        
        # Find target indices (positions that have the target questions)
        target_indices = []
        for q_idx in target_questions:
            for i in range(questions.shape[1]):
                if questions[0, i].item() == q_idx and annotators[0, i].item() >= 0:  # Human annotation
                    target_indices.append(i)
        
        if not target_indices:
            return []
        
        # Calculate VOI for each masked position
        position_vois = []
        for position in masked_positions:
            # Get cost for this position
            cost = 1.0  # Default cost
            if costs and position in costs:
                cost = costs[position]
                
            # Compute VOI
            voi, voi_cost_ratio, posterior_loss = self.voi_calculator.compute_voi(
                self.model, inputs, annotators, questions, known_questions,
                position, target_indices, loss_type, cost=cost
            )
            
            position_vois.append((position, voi, cost, voi_cost_ratio))
        
        # Sort by benefit/cost ratio (highest first)
        position_vois.sort(key=lambda x: x[3], reverse=True)
        
        # Return top selections
        return position_vois[:num_to_select]


class FastVOISelectionStrategy(FeatureSelectionStrategy):
    """
    Fast VOI-based feature selection strategy for Active Feature Acquisition.
    
    Uses sampling to approximate VOI calculations, making it more
    computationally efficient while still providing good selections.
    """
    
    def __init__(self, model, device=None):
        """Initialize Fast VOI selection strategy."""
        super().__init__("fast_voi", model, device)
        self.voi_calculator = FastVOICalculator(model, device)
    
    def select_features(self, example_idx, dataset, num_to_select=1, target_questions=None, 
                       loss_type="cross_entropy", num_samples=3, costs=None, **kwargs):
        """
        Select features (positions) using Fast VOI approximation within a given example.
        
        Args:
            example_idx: Index of the example to select features from
            dataset: Dataset containing the example
            num_to_select: Number of features to select
            target_questions: Target questions to compute VOI for
            loss_type: Type of loss to compute
            num_samples: Number of samples to use for approximation
            costs: Dictionary mapping positions to their annotation costs
            **kwargs: Additional arguments
            
        Returns:
            list: Tuples of (position_idx, benefit, cost, benefit/cost_ratio, class_idx) for selected positions
        """
        if target_questions is None:
            # Default to first question (Q0)
            target_questions = [0]
        
        # Convert target questions to indices if needed
        if isinstance(target_questions[0], str):
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            target_questions = [question_list.index(q) for q in target_questions if q in question_list]
        
        # Get masked positions
        masked_positions = dataset.get_masked_positions(example_idx)
        if not masked_positions:
            return []
        
        # Get data
        known_questions, inputs, answers, annotators, questions = dataset[example_idx]
        inputs = inputs.unsqueeze(0).to(self.device)
        answers = answers.unsqueeze(0).to(self.device)
        annotators = annotators.unsqueeze(0).to(self.device)
        questions = questions.unsqueeze(0).to(self.device)
        known_questions = known_questions.unsqueeze(0).to(self.device)
        
        # Find target indices (positions that have the target questions)
        target_indices = []
        for q_idx in target_questions:
            for i in range(questions.shape[1]):
                if questions[0, i].item() == q_idx and annotators[0, i].item() >= 0:  # Human annotation
                    target_indices.append(i)
        
        if not target_indices:
            return []
        
        # Calculate Fast VOI for each masked position
        position_vois = []
        for position in masked_positions:
            # Get cost for this position
            cost = 1.0  # Default cost
            if costs and position in costs:
                cost = costs[position]
                
            # Compute Fast VOI
            voi, voi_cost_ratio, posterior_loss, most_informative_class = self.voi_calculator.compute_fast_voi(
                self.model, inputs, annotators, questions, known_questions,
                position, target_indices, loss_type, num_samples, cost=cost
            )
            
            position_vois.append((position, voi, cost, voi_cost_ratio, most_informative_class))
        
        # Sort by benefit/cost ratio (highest first)
        position_vois.sort(key=lambda x: x[3], reverse=True)
        
        # Return top selections
        return position_vois[:num_to_select]


class GradientSelector:
    """
    Helper class for gradient-based selection.
    
    Computes and compares gradients for active learning,
    selecting examples that would provide the most training benefit.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize gradient selector.
        
        Args:
            model: Model to use for predictions
            device: Device to use for computations
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def normalize_gradient(self, grad_dict):
        """
        Normalize gradients by their total L2 norm.
        
        Args:
            grad_dict: Dictionary of gradients
            
        Returns:
            dict: Normalized gradients
        """
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
        """
        Compute dot product between two gradient dictionaries.
        
        Args:
            grad_dict1: First gradient dictionary
            grad_dict2: Second gradient dictionary
            
        Returns:
            float: Dot product
        """
        dot_product = 0.0
        
        for name in grad_dict1:
            if name in grad_dict2:
                dot_product += torch.sum(-grad_dict1[name] * grad_dict2[name]).item()
        
        return dot_product
    
    def compute_sample_gradient(self, model, inputs, labels, annotators, questions):
        """
        Compute gradient for a single example using autoregressive sampling.
        
        Args:
            model: Model to use for predictions
            inputs: Input tensor
            labels: Label tensor
            annotators: Annotator indices
            questions: Question indices
            
        Returns:
            dict: Gradient dictionary
        """
        model.train()
        grad_dict = {}
        
        # Identify masked positions
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
        temp_labels = labels.clone()
        
        for pos in masked_positions:
            with torch.no_grad():
                current_outputs = model(temp_inputs, annotators, questions)
                var_outputs = current_outputs[0, pos]
                var_probs = F.softmax(var_outputs, dim=0)
            
            sampled_class = torch.multinomial(var_probs, 1).item()
            
            one_hot = torch.zeros(model.max_choices, device=self.device)
            one_hot[sampled_class] = 1.0
            
            temp_inputs[0, pos, 0] = 0
            temp_inputs[0, pos, 1:1+model.max_choices] = one_hot
            
            temp_labels[0, pos] = one_hot
        
        # Compute loss with full supervision
        model.zero_grad()
        
        outputs = model(temp_inputs, annotators, questions)
        loss = model.compute_total_loss(
            outputs, temp_labels, temp_inputs, questions,
            full_supervision=True
        )
        
        # Compute gradients
        loss.backward()
        
        # Collect gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] = param.grad.detach().clone()
        
        model.zero_grad()
        
        return grad_dict
    
    def compute_example_gradients(self, model, inputs, labels, annotators, questions, num_samples=5):
        """
        Compute gradients for a single example with multiple samples.
        
        Args:
            model: Model to use for predictions
            inputs: Input tensor
            labels: Label tensor
            annotators: Annotator indices
            questions: Question indices
            num_samples: Number of samples to compute
            
        Returns:
            dict: Gradient dictionary
        """
        grad_dict = {}
        
        for _ in range(num_samples):
            sample_grad_dict = self.compute_sample_gradient(
                model, inputs, labels, annotators, questions
            )
            
            # Accumulate gradients
            for name, grad in sample_grad_dict.items():
                if name not in grad_dict:
                    grad_dict[name] = grad
                else:
                    grad_dict[name] += grad
        
        # Average over samples
        if num_samples > 0:
            for name in grad_dict:
                grad_dict[name] /= num_samples
        
        return grad_dict
    
    def compute_validation_gradient_sampled(self, model, val_dataloader, num_samples=5):
        """
        Compute validation gradients using sampling approach.
        
        Args:
            model: Model to use for predictions
            val_dataloader: Validation dataloader
            num_samples: Number of samples to compute
            
        Returns:
            list: List of gradient dictionaries
        """
        model.train()
        grad_samples = []
        
        for _ in tqdm(range(num_samples), desc="Computing validation gradients"):
            temp_grad_dict = {}
            sample_count = 0
            
            for batch in val_dataloader:
                known_questions, inputs, labels, annotators, questions = batch
                inputs, labels, annotators, questions = (
                    inputs.to(self.device), labels.to(self.device), 
                    annotators.to(self.device), questions.to(self.device)
                )
                
                batch_size = inputs.shape[0]
                
                temp_inputs = inputs.clone()
                
                for i in range(batch_size):
                    masked_positions = []
                    for j in range(inputs.shape[1]):
                        if temp_inputs[i, j, 0] == 1:
                            masked_positions.append(j)
                    
                    # Sample values for masked positions
                    for pos in masked_positions:
                        with torch.no_grad():
                            current_outputs = model(temp_inputs, annotators, questions)
                            var_outputs = current_outputs[i, pos]
                            var_probs = F.softmax(var_outputs, dim=0)
                        
                        # Sample a class
                        sampled_class = torch.multinomial(var_probs, 1).item()
                        
                        # Create one-hot encoding
                        one_hot = torch.zeros(model.max_choices, device=self.device)
                        one_hot[sampled_class] = 1.0
                        
                        # Update input
                        temp_inputs[i, pos, 0] = 0
                        temp_inputs[i, pos, 1:1+model.max_choices] = one_hot
                
                # Compute loss with full supervision
                model.zero_grad()
                
                outputs = model(temp_inputs, annotators, questions)
                batch_loss = model.compute_total_loss(
                    outputs, labels, temp_inputs, questions, 
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


class GradientSelectionStrategy(ExampleSelectionStrategy):
    """
    Gradient-based example selection strategy for Active Learning.
    
    Selects examples that have gradient directions most aligned with
    the validation loss gradient, indicating they'd be most helpful
    for improving model performance on the validation set.
    """
    
    def __init__(self, model, device=None):
        """Initialize gradient selection strategy."""
        super().__init__("gradient", model, device)
        self.selector = GradientSelector(model, device)
    
    def select_examples(self, dataset, num_to_select=1, val_dataset=None, 
                        num_samples=5, batch_size=8, costs=None, **kwargs):
        """
        Select examples using gradient alignment.
        
        Args:
            dataset: Dataset to select from
            num_to_select: Number of examples to select
            val_dataset: Validation dataset
            num_samples: Number of samples to compute
            batch_size: Batch size for dataloaders
            costs: Dictionary mapping example indices to their annotation costs
            **kwargs: Additional arguments
            
        Returns:
            tuple: (Selected indices, Alignment scores)
        """
        if val_dataset is None:
            raise ValueError("Validation dataset is required for gradient selection")
        
        # Create validation dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Compute validation gradients
        validation_grad_samples = self.selector.compute_validation_gradient_sampled(
            self.model, val_dataloader, num_samples=num_samples
        )
        
        # Calculate gradient alignment for each example
        all_scores = []
        all_indices = []
        all_costs = []
        all_bc_ratios = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradient alignment")):
            known_questions, inputs, labels, annotators, questions = batch
            inputs, labels, annotators, questions = (
                inputs.to(self.device), labels.to(self.device), 
                annotators.to(self.device), questions.to(self.device)
            )
            
            for i in range(inputs.shape[0]):
                # Skip examples with no masked positions
                if torch.all(inputs[i, :, 0] == 0).item():
                    continue
                    
                example_input = inputs[i:i+1]
                example_labels = labels[i:i+1]
                example_annotator = annotators[i:i+1]
                example_question = questions[i:i+1]
                
                example_grad_dict = self.selector.compute_example_gradients(
                    self.model, 
                    example_input, example_labels, 
                    example_annotator, example_question, 
                    num_samples=num_samples
                )
                
                if not example_grad_dict:
                    continue
                
                example_grad_dict = self.selector.normalize_gradient(example_grad_dict)
                
                alignment_scores = []
                for val_grad in validation_grad_samples:
                    alignment = self.selector.compute_grad_dot_product(val_grad, example_grad_dict)
                    alignment_scores.append(alignment)
                
                global_idx = batch_idx * dataloader.batch_size + i
                
                # Get cost for this example
                cost = 1.0  # Default cost
                if costs and global_idx in costs:
                    cost = costs[global_idx]
                
                avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
                benefit_cost_ratio = avg_alignment / max(cost, 1e-10)
                
                all_scores.append(avg_alignment)
                all_indices.append(global_idx)
                all_costs.append(cost)
                all_bc_ratios.append(benefit_cost_ratio)
        
        if all_scores:
            # Sort by benefit/cost ratio or alignment score
            if kwargs.get('use_benefit_cost_ratio', True):
                sorted_data = sorted(zip(all_indices, all_scores, all_costs, all_bc_ratios), 
                                    key=lambda x: x[3], reverse=True)
                sorted_indices = [idx for idx, _, _, _ in sorted_data]
                sorted_scores = [score for _, score, _, _ in sorted_data]
            else:
                sorted_data = sorted(zip(all_indices, all_scores, all_costs, all_bc_ratios), 
                                    key=lambda x: x[1], reverse=True)
                sorted_indices = [idx for idx, _, _, _ in sorted_data]
                sorted_scores = [score for _, score, _, _ in sorted_data]
            
            selected_indices = sorted_indices[:num_to_select]
            selected_scores = sorted_scores[:num_to_select]
        else:
            selected_indices = []
            selected_scores = []
        
        return selected_indices, selected_scores


class CombinedSelectionStrategy:
    """
    Combines example selection and feature selection strategies.
    
    This allows for a two-stage selection process: first selecting
    the most promising examples, then identifying the most valuable
    features within those examples.
    """
    
    def __init__(self, example_strategy, feature_strategy):
        """
        Initialize combined selection strategy.
        
        Args:
            example_strategy: Strategy for selecting examples
            feature_strategy: Strategy for selecting features within examples
        """
        self.example_strategy = example_strategy
        self.feature_strategy = feature_strategy
        self.name = f"{example_strategy.name}+{feature_strategy.name}"
    
    def select(self, dataset, num_examples=1, num_features=1, example_costs=None, feature_costs=None, **kwargs):
        """
        Select examples and then features within those examples.
        
        Args:
            dataset: Dataset to select from
            num_examples: Number of examples to select
            num_features: Number of features to select per example
            example_costs: Dictionary mapping example indices to their costs
            feature_costs: Dictionary mapping (example_idx, position_idx) to costs
            **kwargs: Additional arguments
            
        Returns:
            list: Tuples of (example_idx, position_idx, benefit, cost, benefit/cost_ratio) for selected features
        """
        # First select examples
        example_result = self.example_strategy.select_examples(
            dataset, num_examples, costs=example_costs, **kwargs
        )
        
        if isinstance(example_result, tuple):
            # Handle case where example strategy returns (indices, scores)
            example_indices, example_scores = example_result
        else:
            example_indices = example_result
            example_scores = [1.0] * len(example_indices)  # Default scores
        
        # Then select features within each example
        selections = []
        for i, example_idx in enumerate(example_indices):
            # Get costs for positions in this example
            if feature_costs and example_idx in feature_costs:
                pos_costs = feature_costs[example_idx]
            else:
                pos_costs = None
                
            # Include example score as additional argument
            kwargs['example_score'] = example_scores[i] if i < len(example_scores) else 1.0
            
            feature_selections = self.feature_strategy.select_features(
                example_idx, dataset, num_features, costs=pos_costs, **kwargs
            )
            
            # Process selections based on different strategy return formats
            for selection in feature_selections:
                if isinstance(selection, tuple):
                    if len(selection) >= 4:  # (position, benefit, cost, ratio, [class])
                        position, benefit, cost, ratio = selection[:4]
                        extra_data = selection[4:] if len(selection) > 4 else []
                        selections.append((example_idx, position, benefit, cost, ratio) + tuple(extra_data))
                    elif len(selection) == 2:  # (position, score)
                        position, score = selection
                        cost = 1.0  # Default cost
                        if pos_costs and position in pos_costs:
                            cost = pos_costs[position]
                        ratio = score / max(cost, 1e-10)
                        selections.append((example_idx, position, score, cost, ratio))
                else:  # Just position
                    position = selection
                    cost = 1.0  # Default cost
                    if pos_costs and position in pos_costs:
                        cost = pos_costs[position]
                    selections.append((example_idx, position, cost, cost, 1.0))
        
        # Sort by benefit/cost ratio (highest first)
        selections.sort(key=lambda x: x[4], reverse=True)
        
        return selections


class SelectionFactory:
    """
    Factory for creating selection strategies.
    
    Provides a centralized way to instantiate different
    selection strategies with consistent configuration.
    """
    
    @staticmethod
    def create_example_strategy(strategy_name, model, device=None):
        """
        Create example selection strategy.
        
        Args:
            strategy_name: Name of the strategy
            model: Model to use for predictions
            device: Device to use for computations
            
        Returns:
            ExampleSelectionStrategy: Example selection strategy
        """
        if strategy_name == "random":
            return RandomExampleSelectionStrategy(model, device)
        elif strategy_name == "gradient":
            return GradientSelectionStrategy(model, device)
        else:
            raise ValueError(f"Unknown example selection strategy: {strategy_name}")
    
    @staticmethod
    def create_feature_strategy(strategy_name, model, device=None):
        """
        Create feature selection strategy.
        
        Args:
            strategy_name: Name of the strategy
            model: Model to use for predictions
            device: Device to use for computations
            
        Returns:
            FeatureSelectionStrategy: Feature selection strategy
        """
        if strategy_name == "random":
            return RandomFeatureSelectionStrategy(model, device)
        elif strategy_name == "voi":
            return VOISelectionStrategy(model, device)
        elif strategy_name == "fast_voi":
            return FastVOISelectionStrategy(model, device)
        elif strategy_name == "sequential":
            # Create a simple strategy that selects positions sequentially
            return RandomFeatureSelectionStrategy(model, device)  # Reuse random but override select_features
        else:
            raise ValueError(f"Unknown feature selection strategy: {strategy_name}")
    
    @staticmethod
    def create_combined_strategy(example_strategy_name, feature_strategy_name, model, device=None):
        """
        Create combined selection strategy.
        
        Args:
            example_strategy_name: Name of the example selection strategy
            feature_strategy_name: Name of the feature selection strategy
            model: Model to use for predictions
            device: Device to use for computations
            
        Returns:
            CombinedSelectionStrategy: Combined selection strategy
        """
        example_strategy = SelectionFactory.create_example_strategy(example_strategy_name, model, device)
        feature_strategy = SelectionFactory.create_feature_strategy(feature_strategy_name, model, device)
        return CombinedSelectionStrategy(example_strategy, feature_strategy)