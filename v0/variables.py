import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod

class Variable(ABC):
    
    def __init__(self, name: str, domain: Any):
        self.name = name
        self.domain = domain
        self.is_query = False
        self.loss_weight = 1.0
        self.loss_fn = None
        self.accept_parameter = True
    
    @abstractmethod
    def param_dim(self) -> int:
        """Return the dimensionality of the parametric representation."""
        pass
    
    @abstractmethod
    def to_features(self, params: torch.Tensor) -> torch.Tensor:
        """Transform parametric representation to featural representation."""
        pass
    
    @abstractmethod
    def get_mask_value(self) -> torch.Tensor:
        """Return the parametric representation for a masked value (typically uniform distribution)."""
        pass
    
    @abstractmethod
    def get_observed_value(self, value: Any) -> torch.Tensor:
        """Return the parametric representation for an observed value (typically point distribution)."""
        pass
    
    @abstractmethod
    def compute_loss(self, pred_params: torch.Tensor, target: Any) -> torch.Tensor:
        """Compute loss between predicted distribution and target value."""
        pass
    
    @abstractmethod
    def sample(self, params: torch.Tensor, sample_size: int = 1) -> torch.Tensor:
        """Sample from the distribution with given parameters."""
        pass
    
    @abstractmethod
    def decode(self, params: torch.Tensor) -> Any:
        """Decode an optimal value from a distribution parameter vector."""
        pass
    
    def set_as_query(self, is_query: bool = True, weight: float = 1.0):
        """Mark this variable as a query variable for VOI computation."""
        self.is_query = is_query
        self.loss_weight = weight
    
    def set_loss_function(self, loss_fn: Callable):
        """Set a custom loss function for this variable."""
        self.loss_fn = loss_fn


class CategoricalVariable(Variable):
    """Variable representing categorical values."""
    
    def __init__(self, name: str, categories: List[Any]):
        super().__init__(name, categories)
        self.num_categories = len(categories)
        # Map from category values to indices
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    def param_dim(self) -> int:
        """Logit vector dimensionality."""
        return self.num_categories
    
    def to_features(self, params: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities using softmax."""
        return F.softmax(params, dim=-1)
    
    def get_mask_value(self) -> torch.Tensor:
        """Return uniform distribution logits (all zeros)."""
        return torch.zeros(self.num_categories)
    
    def get_observed_value(self, value: Any) -> torch.Tensor:
        """Return one-hot-like logits with high value at the observed category."""
        idx = self.cat_to_idx[value]
        logits = torch.ones(self.num_categories) * -10.0  # Very negative for all categories
        logits[idx] = 10.0  # Very positive for the observed category
        return logits
    
    def compute_loss(self, pred_params: torch.Tensor, target: Any) -> torch.Tensor:
        """Compute cross-entropy loss."""
        if self.loss_fn is not None:
            return self.loss_fn(pred_params, target)
        
        target_idx = self.cat_to_idx[target]
        target_tensor = torch.tensor(target_idx, device=pred_params.device)
        return F.cross_entropy(pred_params, target_tensor)
    
    def sample(self, params: torch.Tensor, sample_size: int = 1) -> torch.Tensor:
        """Sample categories from the logit distribution."""
        probs = F.softmax(params, dim=-1)
        samples = torch.multinomial(probs, sample_size, replacement=True)
        return samples
    
    def decode(self, params: torch.Tensor) -> Any:
        """Return the most likely category."""
        idx = torch.argmax(params, dim=-1).item()
        return self.domain[idx]


class CategoricalNumericVariable(CategoricalVariable):
    """Categorical variable with numeric values that can use L2 loss or KL divergence."""
    
    def __init__(self, name: str, numeric_categories: List[float]):
        super().__init__(name, numeric_categories)
        self.numeric_values = torch.tensor(numeric_categories, dtype=torch.float32)
        self.loss_fn = None
    
    def set_loss_function(self, loss_fn):
        """Set a custom loss function for this variable."""
        self.loss_fn = loss_fn
    
    def compute_loss(self, pred_params: torch.Tensor, target: torch.Tensor, parametric: bool = False) -> torch.Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(pred_params, target)
        
        # If we're using distribution-based loss (KL divergence)
        if parametric:
            # Apply log_softmax to pred_params for KL divergence
            log_probs = F.log_softmax(pred_params, dim=-1)
            
            # Calculate KL divergence
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            return kl_criterion(log_probs.unsqueeze(0) if log_probs.dim() == 1 else log_probs, 
                               target.unsqueeze(0) if target.dim() == 1 else target)
        
        # Default to L2/MSE loss for scalar targets
        probs = F.softmax(pred_params, dim=-1)
        expected_value = torch.sum(probs * self.numeric_values.to(pred_params.device))
        
        if isinstance(target, torch.Tensor) and target.dim() > 0 and target.size(0) > 1:
            # If target is a distribution, calculate expected value
            target_value = torch.sum(target * self.numeric_values.to(target.device))
        else:
            # If target is already a scalar
            target_value = target
            
        return F.mse_loss(expected_value, target_value)
    
    def decode(self, params: torch.Tensor) -> float:
        """Return the expected value as the optimal prediction."""
        probs = F.softmax(params, dim=-1)
        expected_value = torch.sum(probs * self.numeric_values.to(params.device))
        return expected_value.item()
    
    def to_features(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probability distribution."""
        return F.softmax(logits, dim=-1)


class NormalVariable(Variable):
    """Variable representing normally distributed real values."""
    
    def __init__(self, name: str, domain: Optional[Tuple[float, float]] = None):
        """
        Initialize a normal variable.
        
        Args:
            name: Variable name
            domain: Optional bounds for the variable (min, max)
        """
        super().__init__(name, domain)
    
    def param_dim(self) -> int:
        """Return dimensionality of (mean, log_variance)."""
        return 2
    
    def to_features(self, params: torch.Tensor) -> torch.Tensor:
        """Convert (mean, log_variance) to (mean, std)."""
        mean, log_var = params[..., 0], params[..., 1]
        std = torch.exp(0.5 * log_var)
        return torch.stack([mean, std], dim=-1)
    
    def get_mask_value(self) -> torch.Tensor:
        """Return parameters for a wide normal distribution."""
        return torch.tensor([0.0, np.log(1000.0)])
    
    def get_observed_value(self, value: float) -> torch.Tensor:
        """Return parameters for a narrow normal distribution centered at the observed value."""
        return torch.tensor([float(value), np.log(0.001)])
    
    def compute_loss(self, pred_params: torch.Tensor, target: float) -> torch.Tensor:
        """Compute negative log likelihood of target under normal distribution."""
        if self.loss_fn is not None:
            return self.loss_fn(pred_params, target)
        
        mean, log_var = pred_params[..., 0], pred_params[..., 1]
        target_tensor = torch.tensor(target, dtype=torch.float32, device=pred_params.device)
        
        # Negative log likelihood of normal distribution
        return 0.5 * (log_var + (target_tensor - mean)**2 / torch.exp(log_var))
    
    def sample(self, params: torch.Tensor, sample_size: int = 1) -> torch.Tensor:
        """Sample from normal distribution with given parameters."""
        mean, log_var = params[..., 0], params[..., 1]
        std = torch.exp(0.5 * log_var)
        
        # Generate normal samples
        eps = torch.randn(sample_size, device=params.device)
        samples = mean + eps * std
        
        # Apply domain constraints if specified
        if self.domain is not None:
            min_val, max_val = self.domain
            samples = torch.clamp(samples, min_val, max_val)
        
        return samples
    
    def decode(self, params: torch.Tensor) -> float:
        """For normal distribution, the mean minimizes expected L2 loss."""
        mean = params[..., 0].item()
        return mean


class DirichletVariable(Variable):
    """Variable representing Dirichlet distributed values (distribution over categories)."""
    
    def __init__(self, name: str, categories: List[Any]):
        """
        Initialize a Dirichlet variable.
        
        Args:
            name: Variable name
            categories: List of possible categories the distribution is over
        """
        super().__init__(name, categories)
        self.num_categories = len(categories)
    
    def param_dim(self) -> int:
        """Return dimensionality of concentration parameters."""
        return self.num_categories
    
    def to_features(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert concentration parameters to (sum, normalized_concentrations).
        
        This gives both the strength and the expected probabilities.
        """
        # Ensure positive concentration parameters
        alpha = torch.exp(params)
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / alpha_sum
        
        # Concatenate sum and probabilities
        return torch.cat([torch.log(alpha_sum), probs], dim=-1)
    
    def get_mask_value(self) -> torch.Tensor:
        """Return parameters for a uniform Dirichlet (all concentrations = 1)."""
        return torch.zeros(self.num_categories)  # log(1) = 0
    
    def get_observed_value(self, value: List[float]) -> torch.Tensor:
        """
        Return parameters for a concentrated Dirichlet.
        
        Args:
            value: List of probabilities summing to 1
        """
        # Convert to high concentration parameters
        scale_factor = 1000.0
        concentrations = torch.tensor([v * scale_factor for v in value])
        return torch.log(concentrations)
    
    def compute_loss(self, pred_params: torch.Tensor, target: List[float]) -> torch.Tensor:
        """Compute KL divergence between predicted Dirichlet and target distribution."""
        if self.loss_fn is not None:
            return self.loss_fn(pred_params, target)
        
        # Convert log-concentrations to concentrations
        alpha_pred = torch.exp(pred_params)
        
        # Create target tensor
        target_tensor = torch.tensor(target, dtype=torch.float32, device=pred_params.device)
        
        # Use MSE between expected probabilities as a simple loss function
        alpha_sum = torch.sum(alpha_pred)
        pred_probs = alpha_pred / alpha_sum
        
        return F.mse_loss(pred_probs, target_tensor)
    
    def sample(self, params: torch.Tensor, sample_size: int = 1) -> torch.Tensor:
        """Sample from Dirichlet distribution with given parameters."""
        alpha = torch.exp(params)
        
        # Generate gamma samples and normalize
        # This is a simple way to sample from Dirichlet
        gamma_samples = torch.zeros((sample_size, self.num_categories), device=params.device)
        for k in range(self.num_categories):
            gamma_samples[:, k] = torch.distributions.Gamma(
                concentration=alpha[k], rate=1.0
            ).sample((sample_size,))
        
        # Normalize to get Dirichlet samples
        samples = gamma_samples / torch.sum(gamma_samples, dim=1, keepdim=True)
        return samples
    
    def decode(self, params: torch.Tensor) -> List[float]:
        """Return the expected probabilities from the Dirichlet distribution."""
        alpha = torch.exp(params)
        alpha_sum = torch.sum(alpha)
        probs = alpha / alpha_sum
        return probs.tolist()


class OrdinalVariable(Variable):
    """Variable representing ordinal values (ordered categories)."""
    
    def __init__(self, name: str, num_bins: int, init_bin_boundaries: Optional[List[float]] = None):
        """
        Initialize an ordinal variable.
        
        Args:
            name: Variable name
            num_bins: Number of ordered categories
            init_bin_boundaries: Initial bin boundaries (learnable parameters)
        """
        super().__init__(name, list(range(num_bins)))
        self.num_bins = num_bins
        
        # Learnable bin boundaries (num_bins - 1 boundaries)
        if init_bin_boundaries is None:
            # Default to evenly spaced boundaries between -3 and 3
            init_bin_boundaries = torch.linspace(-3, 3, num_bins - 1)
        else:
            init_bin_boundaries = torch.tensor(init_bin_boundaries, dtype=torch.float32)
        
        self.register_buffer('bin_boundaries', init_bin_boundaries)
    
    def register_buffer(self, name, tensor):
        """Store a tensor buffer that should be registered with the Imputer."""
        setattr(self, name, tensor)
    
    def param_dim(self) -> int:
        """Return dimensionality of (mean, log_variance)."""
        return 2
    
    def to_features(self, params: torch.Tensor) -> torch.Tensor:
        """Convert (mean, log_variance) to probabilities for each bin."""
        mean, log_var = params[..., 0], params[..., 1]
        std = torch.exp(0.5 * log_var)
        
        # Calculate probability of each bin under the normal distribution
        bin_probs = torch.zeros(self.num_bins, device=params.device)
        
        # First bin: from -inf to first boundary
        bin_probs[0] = torch.distributions.Normal(mean, std).cdf(self.bin_boundaries[0])
        
        # Middle bins: between consecutive boundaries
        for i in range(1, self.num_bins - 1):
            cdf_high = torch.distributions.Normal(mean, std).cdf(self.bin_boundaries[i])
            cdf_low = torch.distributions.Normal(mean, std).cdf(self.bin_boundaries[i-1])
            bin_probs[i] = cdf_high - cdf_low
        
        # Last bin: from last boundary to inf
        bin_probs[-1] = 1.0 - torch.distributions.Normal(mean, std).cdf(self.bin_boundaries[-1])
        
        return bin_probs
    
    def get_mask_value(self) -> torch.Tensor:
        """Return parameters for a wide normal distribution."""
        return torch.tensor([0.0, np.log(100.0)])
    
    def get_observed_value(self, value: int) -> torch.Tensor:
        """Return parameters for a distribution concentrated on the observed bin."""
        # For an observed ordinal value, we can use the bin center or boundary
        if value == 0:
            # First bin
            mean = self.bin_boundaries[0] - 1.0
        elif value == self.num_bins - 1:
            # Last bin
            mean = self.bin_boundaries[-1] + 1.0
        else:
            # Middle bin - use midpoint between boundaries
            mean = (self.bin_boundaries[value-1] + self.bin_boundaries[value]) / 2
        
        # Small variance to concentrate the distribution
        log_var = np.log(0.01)
        
        return torch.tensor([mean, log_var])
    
    def compute_loss(self, pred_params: torch.Tensor, target: int) -> torch.Tensor:
        """Compute negative log likelihood of target bin."""
        if self.loss_fn is not None:
            return self.loss_fn(pred_params, target)
        
        # Convert parameters to bin probabilities
        bin_probs = self.to_features(pred_params)
        
        # Avoid numerical issues
        eps = 1e-10
        bin_probs = torch.clamp(bin_probs, min=eps, max=1.0-eps)
        
        # Negative log likelihood of the target bin
        return -torch.log(bin_probs[target])
    
    def sample(self, params: torch.Tensor, sample_size: int = 1) -> torch.Tensor:
        """Sample from the ordinal distribution."""
        bin_probs = self.to_features(params)
        samples = torch.multinomial(bin_probs, sample_size, replacement=True)
        return samples
    
    def decode(self, params: torch.Tensor) -> int:
        """Return the most likely bin."""
        bin_probs = self.to_features(params)
        return torch.argmax(bin_probs).item()


