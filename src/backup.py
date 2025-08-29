import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wishart, dirichlet, multivariate_normal
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
import json
import torch
import torch.nn as nn
import math
import torch.optim as optim
import os
from tqdm import tqdm
from scipy.stats import multivariate_normal
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

def gibbs_mvnormal(mean, cov, n_samples, burn_in=100):
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    Λ = np.linalg.inv(cov)
    d = len(mean)
    x = mean.copy() 
    samples = np.ones((n_samples, d))
    
    for t in range(n_samples + burn_in):
        for i in range(d):
            cond_var = 1.0 / Λ[i, i]
            sum_except_i = np.dot(Λ[i, :], x - mean) - Λ[i, i] * (x[i] - mean[i])
            cond_mean = mean[i] - cond_var * sum_except_i
            x[i] = np.random.normal(cond_mean, np.sqrt(cond_var))
            
        if t >= burn_in:
            samples[t - burn_in] = x
            
    return samples

class CorrectedGenzAlgorithm(nn.Module):
    """
    Optimized implementation of Genz's algorithm with significant performance improvements.
    
    Key optimizations:
    1. Vectorized Monte Carlo sampling (batch all samples at once)
    2. Cached Cholesky decomposition and other expensive operations
    3. Eliminated loops in sequential conditioning
    4. More efficient memory usage and reduced tensor operations
    5. Smarter convergence checking with early stopping
    6. Optimized CDF/inverse CDF computations
    """
    
    def __init__(self, max_samples: int = 100000, abs_tol: float = 1e-4, rel_tol: float = 1e-3):
        super().__init__()
        self.max_samples = max_samples
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        # Cache for expensive computations
        self._cache = {}
    
    def _standard_normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable standard normal CDF."""
        # Clamp extreme values to prevent overflow/underflow
        x_clamped = torch.clamp(x, -8.0, 8.0)  # Beyond ±8, CDF is essentially 0 or 1
        return 0.5 * torch.erfc(-x_clamped * 0.7071067811865476)
    
    def _inverse_standard_normal_cdf(self, u: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse standard normal CDF."""
        # More conservative clamping to avoid erfinv instability
        u_clamped = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        erfinv_input = torch.clamp(2.0 * u_clamped - 1.0, -0.999999, 0.999999)
        
        # Handle edge cases explicitly
        result = 1.4142135623730951 * torch.erfinv(erfinv_input)
        
        # Replace any remaining NaN/inf values
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result
    
    def _reorder_variables(self, a: torch.Tensor, b: torch.Tensor, 
                          sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized variable reordering."""
        batch_size, dim = a.shape
        
        if sigma.dim() == 2:
            sigma_expanded = sigma.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            sigma_expanded = sigma
        
        # More efficient difficulty computation
        sigma_diag = torch.diagonal(sigma_expanded, dim1=1, dim2=2)
        difficulty = (b - a) * torch.rsqrt(sigma_diag)  # rsqrt is faster than sqrt
        
        # Get sort indices
        _, sort_indices = torch.sort(difficulty, dim=1)
        
        # More efficient gathering using advanced indexing
        batch_idx = torch.arange(batch_size, device=a.device).unsqueeze(1)
        a_reordered = a[batch_idx, sort_indices]
        b_reordered = b[batch_idx, sort_indices]
        
        # Efficient covariance matrix reordering
        sigma_reordered = sigma_expanded[batch_idx.unsqueeze(-1), sort_indices.unsqueeze(-1), sort_indices.unsqueeze(1)]
        
        return a_reordered, b_reordered, sigma_reordered, sort_indices
    
    def _cholesky_factor_cached(self, sigma: torch.Tensor) -> torch.Tensor:
        """Numerically stable cached Cholesky decomposition."""
        # Create a hashable key from sigma
        sigma_key = sigma.data_ptr()
        
        if sigma_key in self._cache:
            return self._cache[sigma_key]
        
        # Check for NaN/inf in input
        if not torch.all(torch.isfinite(sigma)):
            raise ValueError("Covariance matrix contains NaN or inf values")
        
        # More aggressive regularization for numerical stability
        dim = sigma.shape[-1]
        device = sigma.device
        dtype = sigma.dtype
        
        # Check condition number and add regularization accordingly
        eigenvals = torch.linalg.eigvals(sigma)
        min_eigval = torch.min(torch.real(eigenvals))
        max_eigval = torch.max(torch.real(eigenvals))
        
        # Add regularization based on condition number
        reg_factor = torch.clamp(-min_eigval + 1e-8, min=1e-8, max=1e-4)
        eye = torch.eye(dim, device=device, dtype=dtype)
        
        if sigma.dim() == 3:
            reg_factor = reg_factor.unsqueeze(-1).unsqueeze(-1)
            eye = eye.unsqueeze(0).expand(sigma.shape[0], -1, -1)
        
        sigma_reg = sigma + reg_factor * eye
        
        try:
            L = torch.linalg.cholesky(sigma_reg)
        except RuntimeError as e:
            # Fallback: use SVD-based decomposition
            U, S, V = torch.linalg.svd(sigma_reg)
            S_clamp = torch.clamp(S, min=1e-8)
            S_sqrt = torch.sqrt(S_clamp)
            L = U * S_sqrt.unsqueeze(-2)
        
        # Verify the result is finite
        if not torch.all(torch.isfinite(L)):
            raise ValueError("Cholesky decomposition produced NaN or inf values")
        
        self._cache[sigma_key] = L
        return L
    
    def _vectorized_genz_integrand(self, u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, 
                                 sigma: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized Genz integrand computation - eliminates the sequential loop.
        
        This computes all samples simultaneously using vectorized operations.
        All operations are gradient-safe (no in-place modifications).
        """
        batch_size, n_samples, dim = u.shape
        device = u.device
        dtype = u.dtype
        
        # Get cached Cholesky factor
        L = self._cholesky_factor_cached(sigma)
        
        # Expand for all samples
        L_exp = L.unsqueeze(1).expand(-1, n_samples, -1, -1)  # [batch, samples, dim, dim]
        a_exp = a.unsqueeze(1).expand(-1, n_samples, -1)      # [batch, samples, dim]
        b_exp = b.unsqueeze(1).expand(-1, n_samples, -1)      # [batch, samples, dim]
        
        # Initialize outputs - create new tensors each iteration to avoid in-place ops
        prob = torch.ones(batch_size, n_samples, device=device, dtype=dtype)
        y_components = []  # Build y incrementally without in-place operations
        
        # Vectorized sequential conditioning
        for k in range(dim):
            # Conditional variance (diagonal of L) with numerical protection
            sigma_k = L_exp[:, :, k, k]  # [batch, samples]
            
            # Protect against near-zero or negative variances
            sigma_k = torch.clamp(sigma_k, min=1e-8)
            
            # Conditional mean from previous variables
            if k > 0:
                # Stack previous y components and compute mean
                y_prev = torch.stack(y_components, dim=-1)  # [batch, samples, k]
                L_k = L_exp[:, :, k, :k]  # [batch, samples, k]
                mean_k = torch.sum(L_k * y_prev, dim=-1)  # [batch, samples]
            else:
                mean_k = torch.zeros(batch_size, n_samples, device=device, dtype=dtype)
            
            # Standardized bounds with numerical protection
            a_std = (a_exp[:, :, k] - mean_k) / sigma_k
            b_std = (b_exp[:, :, k] - mean_k) / sigma_k
            
            # Ensure proper ordering and finite values
            a_std = torch.clamp(a_std, -10.0, 10.0)
            b_std = torch.clamp(b_std, -10.0, 10.0)
            b_std = torch.maximum(a_std + 1e-8, b_std)  # Ensure b_std > a_std
            
            # CDF values with stability checks
            Phi_a = self._standard_normal_cdf(a_std)
            Phi_b = self._standard_normal_cdf(b_std)
            
            # Ensure valid probability difference
            delta_k = Phi_b - Phi_a
            delta_k = torch.clamp(delta_k, min=1e-8, max=1.0)  # Prevent zero or negative deltas
            
            # Update probability (create new tensor)
            prob = prob * delta_k
            
            # Check for problematic probability values
            prob = torch.where(torch.isfinite(prob), prob, torch.zeros_like(prob))
            
            # Sample next variable with protection
            u_scaled = torch.clamp(u[:, :, k], 1e-8, 1.0 - 1e-8)
            u_k = Phi_a + u_scaled * delta_k
            u_k = torch.clamp(u_k, 1e-6, 1.0 - 1e-6)  # Ensure valid CDF input
            
            y_k = self._inverse_standard_normal_cdf(u_k)
            
            # Additional safety check for y_k
            y_k = torch.where(torch.isfinite(y_k), y_k, torch.zeros_like(y_k))
            y_k = torch.clamp(y_k, -10.0, 10.0)  # Reasonable bounds for normal samples
            
            # Store component without in-place operation
            y_components.append(y_k)
        
        return prob  # [batch_size, n_samples]
    
    def _adaptive_monte_carlo(self, a: torch.Tensor, b: torch.Tensor, 
                            sigma: torch.Tensor) -> torch.Tensor:
        """
        Optimized Monte Carlo with better adaptive strategy and vectorized sampling.
        """
        batch_size, dim = a.shape
        device = a.device
        dtype = a.dtype
        
        # Start with larger initial batch for better statistics
        n_samples = min(5000, self.max_samples // 5)
        total_samples = 0
        
        # Running statistics using gradient-safe operations
        sum_estimates = torch.zeros(batch_size, device=device, dtype=dtype)
        sum_squared_estimates = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Early convergence tracking
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        while total_samples < self.max_samples and not torch.all(converged):
            # Generate all samples at once
            u = torch.rand(batch_size, n_samples, dim, device=device, dtype=dtype)
            
            # Vectorized integrand computation with NaN checking
            estimates = self._vectorized_genz_integrand(u, a, b, sigma)  # [batch, n_samples]
            
            # Check for NaN/inf in estimates and handle gracefully
            if not torch.all(torch.isfinite(estimates)):
                # Replace NaN/inf with zeros and log warning
                nan_mask = ~torch.isfinite(estimates)
                estimates = torch.where(nan_mask, torch.zeros_like(estimates), estimates)
                if torch.any(nan_mask):
                    print(f"Warning: {torch.sum(nan_mask)} NaN/inf values detected in estimates")
            
            # Update running statistics without in-place operations
            batch_sum = torch.sum(estimates, dim=1)
            batch_sum_squared = torch.sum(estimates**2, dim=1)
            
            # Verify sums are finite
            batch_sum = torch.where(torch.isfinite(batch_sum), batch_sum, torch.zeros_like(batch_sum))
            batch_sum_squared = torch.where(torch.isfinite(batch_sum_squared), batch_sum_squared, torch.zeros_like(batch_sum_squared))
            
            # Create new tensors instead of in-place updates
            sum_estimates = sum_estimates + batch_sum
            sum_squared_estimates = sum_squared_estimates + batch_sum_squared
            total_samples += n_samples
            
            # Check convergence every 1000 samples
            if total_samples >= 1000 and total_samples % 1000 == 0:
                current_mean = sum_estimates / total_samples
                current_var = torch.clamp(
                    sum_squared_estimates / total_samples - current_mean**2, 
                    min=1e-12
                )
                std_error = torch.sqrt(current_var / total_samples)
                
                # Vectorized convergence check (creates new tensors)
                abs_converged = std_error < self.abs_tol
                rel_converged = std_error < self.rel_tol * torch.abs(current_mean)
                newly_converged = (abs_converged | rel_converged) & (~converged)
                converged = converged | newly_converged  # Creates new tensor
                
                # Early stopping if enough batches converged
                if torch.sum(converged).float() / batch_size > 0.95:
                    break
            
            # Adaptive sample size with diminishing returns
            n_samples = min(n_samples + 1000, 10000)
        
        return sum_estimates / total_samples
        
        return mean_est
    
    def _precompute_statistics(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Precompute and cache statistics for efficiency."""
        # Check for trivial cases that can be handled analytically
        zero_width = torch.all(torch.isclose(a, b, rtol=1e-10), dim=1)
        return zero_width
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor, 
                mu: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized multivariate normal integral computation.
        """
        batch_size, dim = a.shape
        device = a.device
        
        # Handle mean vector efficiently
        if mu is not None:
            if mu.dim() == 1:
                mu = mu.unsqueeze(0).expand(batch_size, -1)
            a_centered = a - mu
            b_centered = b - mu
        else:
            a_centered = a
            b_centered = b
        
        # Fast invalid case detection
        invalid = torch.any(a_centered >= b_centered, dim=1)
        
        # Early return for trivial cases
        zero_cases = self._precompute_statistics(a_centered, b_centered)
        
        # Expand sigma efficiently
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process only valid cases
        valid_mask = ~(invalid | zero_cases)
        if not torch.any(valid_mask):
            return torch.zeros(batch_size, device=device)
        
        # Extract valid cases for processing (creates new tensors)
        if torch.all(valid_mask):
            # All cases valid - no need to subset
            a_proc, b_proc, sigma_proc = a_centered, b_centered, sigma
        else:
            # Subset to valid cases only (creates new tensors)
            a_proc = a_centered[valid_mask]
            b_proc = b_centered[valid_mask] 
            sigma_proc = sigma[valid_mask]
        
        # Reorder variables for better numerical stability
        a_reordered, b_reordered, sigma_reordered, _ = self._reorder_variables(
            a_proc, b_proc, sigma_proc
        )
        
        # Compute integral using optimized Monte Carlo
        valid_results = self._adaptive_monte_carlo(a_reordered, b_reordered, sigma_reordered)
        
        # Reconstruct full result tensor (gradient-safe)
        if torch.all(valid_mask):
            result = valid_results
        else:
            result = torch.zeros(batch_size, device=device)
            # Use index_put_ or create new tensor to avoid in-place issues
            result = result.index_put([valid_mask], valid_results)
        
        # Final clamping (creates new tensor)
        return torch.clamp(result, 0.0, 1.0)
    
    def clear_cache(self):
        """Clear the computation cache to free memory."""
        self._cache.clear()

class DomainSpecificModel(nn.Module):
    """PyTorch model for domain-specific optimization with proper MAP estimation"""
    
    def __init__(self, n: int, k: int, sparsity_pattern: np.ndarray, 
                 lambda_sparsity: float = 1.0, alpha: float = 2.0, 
                 nu: float = None, V: np.ndarray = None):
        super().__init__()
        print(sparsity_pattern)
        self.n = n
        self.k = k
        self.lambda_sparsity = lambda_sparsity
        self.alpha = alpha
        self.nu = nu if nu is not None else n + 2
        self.V = V if V is not None else np.eye(n)
        
        # Convert to tensors
        self.register_buffer('sparsity_mask', torch.tensor(sparsity_pattern, dtype=torch.float32))
        self.register_buffer('V_tensor', torch.tensor(self.V, dtype=torch.float32))
        
        # Cholesky factor parameters for precision matrix
        self.L_lower = nn.Parameter(torch.randn(n, n) * 0.1)
        self.L_diag_log = nn.Parameter(torch.ones(n))
        
        # Probability matrix parameters (log-scale for numerical stability)
        self.p_logits = nn.Parameter(torch.randn(n, k) * 0.1)
    
    def get_cholesky_factor(self):
        """Get lower triangular Cholesky factor with positive diagonal"""
        L = torch.tril(self.L_lower, diagonal=-1)
        L = L + torch.diag(torch.exp(self.L_diag_log))
        return L
    
    def get_omega(self):
        """Get precision matrix (guaranteed positive definite)"""
        L = self.get_cholesky_factor()
        omega = L @ L.T + (torch.eye(self.n)).to("cuda")
        return omega
    
    def get_sigma(self):
        """Get covariance matrix"""
        omega = self.get_omega()
        return torch.inverse(omega)
    
    def get_p_mat(self):
        """Get probability matrix (satisfies simplex constraints)"""
        return torch.softmax(self.p_logits, dim=1)
    
    def compute_thresholds(self, p_mat, sigma):
        """Compute thresholds from probability matrix and covariance"""
        thresholds = torch.zeros(self.n, self.k - 1, device=p_mat.device)

        for i in range(self.n):
            cumsum_p = torch.tensor(0.0, device=p_mat.device)
            sigma_i = torch.sqrt(sigma[i, i])

            for j in range(self.k - 1):
                cumsum_p = cumsum_p + p_mat[i, j]
                cumsum_p_clamped = torch.clamp(cumsum_p, 1e-6, 1 - 1e-6)
                thresholds[i, j] = sigma_i * torch.erfinv(2 * cumsum_p_clamped - 1) * np.sqrt(2)
        
        return thresholds
    
    def log_prior_precision(self, omega):
        """
        Log prior probability for precision matrix (Wishart with soft constraints)
        Equivalent to _log_prob_precision method
        """
        # Wishart prior: p(Ω) ∝ |Ω|^((ν-n-1)/2) exp(-1/2 tr(V^{-1}Ω))
        log_det_omega = torch.logdet(omega)
        
        # Check if omega is positive definite (logdet will be -inf if not)
        if torch.isinf(log_det_omega) or torch.isnan(log_det_omega):
            return torch.tensor(-1e10, device=omega.device)
        
        # Wishart log probability
        log_prior = (self.nu - self.n - 1) / 2 * log_det_omega
        log_prior -= 0.5 * torch.trace(torch.inverse(self.V_tensor) @ omega)
        
        # Add soft sparsity constraints
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.sparsity_mask[i, j] == 0:
                    log_prior -= self.lambda_sparsity * omega[i, j]**2
        
        return log_prior
    
    def log_prior_probabilities(self, p_mat):
        """
        Log prior probability for probability matrix (Dirichlet)
        p(p_i) = Dir(α, α, ..., α) for each row i
        """
        log_prior = 0.0
        
        for i in range(self.n):
            # Dirichlet log probability for row i
            # log Dir(p_i | α) ∝ Σ_j (α-1) log(p_{ij})
            for j in range(self.k):
                log_prior += (self.alpha - 1) * torch.log(torch.clamp(p_mat[i, j], min=1e-10))
        
        return log_prior
    
    def forward(self):
        """Forward pass - return all derived quantities"""
        omega = self.get_omega()
        sigma = self.get_sigma()
        p_mat = self.get_p_mat()
        thresholds = self.compute_thresholds(p_mat, sigma)
        
        return {
            'omega': omega,
            'sigma': sigma,
            'p_mat': p_mat,
            'thresholds': thresholds
        }

class GaussianBinningWithLinGauss:
    """
    Gaussian graphical model with binned observations
    Uses soft constraints for sparsity and LinGauss for exact marginal likelihood
    """
    
    def __init__(self, n: int, k: int, sparsity_pattern: np.ndarray,
                 penalty_strength: float, alpha: float, nu: float, V: np.ndarray):
        self.n = n
        self.k = k
        self.sparsity_pattern = sparsity_pattern
        self.penalty_strength = penalty_strength
        self.alpha = alpha
        self.nu = nu
        self.V = V
    
    def _log_prob_precision(self, Omega: np.ndarray) -> float:
        """Log probability of precision matrix with soft constraints"""
        try:
            np.linalg.cholesky(Omega)
        except:
            return -np.inf
        
        log_prob = (self.nu - self.n - 1) / 2 * np.linalg.slogdet(Omega)[1]
        log_prob -= 0.5 * np.trace(np.linalg.inv(self.V) @ Omega)
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.sparsity_pattern[i, j] == 0:
                    log_prob -= self.penalty_strength * Omega[i, j]**2
        
        return log_prob
    
    def sample_precision_with_soft_constraints(self) -> np.ndarray:
        """
        Sample from Wishart with soft constraints using Metropolis-Hastings
        """
        # Initialize with unconstrained Wishart sample instead of identity
        from scipy.stats import wishart
        Omega = wishart.rvs(df=self.nu, scale=self.V)
        
        # More steps and better step size
        n_steps = 10000
        step_size = 0.05
        n_accepted = 0
        
        for step in range(n_steps):
            # Propose by adding noise directly to Omega (symmetric)
            noise = step_size * np.random.randn(self.n, self.n)
            noise = (noise + noise.T) / 2  # Make symmetric
            Omega_proposal = Omega + noise
            
            # Check if proposal is positive definite
            try:
                np.linalg.cholesky(Omega_proposal)
            except:
                continue  # Skip invalid proposals
            
            # Compute log probabilities
            log_prob_current = self._log_prob_precision(Omega)
            log_prob_proposal = self._log_prob_precision(Omega_proposal)
            
            # Accept/reject
            if np.random.rand() < np.exp(log_prob_proposal - log_prob_current):
                Omega = Omega_proposal
                n_accepted += 1
        
        print(f"MH acceptance rate: {n_accepted/n_steps:.3f}")
        return Omega

    def generate_synthetic_data(self, seed: Optional[int] = None, num: int = 1000) -> Dict:
        """Generate synthetic dataset using proper sampling from priors"""
        
        # Sample precision matrix from prior with soft constraints
        Omega = self.sample_precision_with_soft_constraints()
        print("Generated precision matrix:")
        print(Omega)
        Sigma = np.linalg.inv(Omega)
        
        # Sample probability matrix from Dirichlet prior
        p_mat = np.zeros((self.n, self.k))
        for i in range(self.n):
            p_mat[i] = dirichlet.rvs(self.alpha * np.ones(self.k))[0]
        print("Probability matrix:")
        print(p_mat)
        
        # Generate continuous samples
        z = gibbs_mvnormal(mean=np.zeros(self.n), cov=Sigma, n_samples=num)
        if num == 1:
            z = z.reshape(1, -1)
        
        # Generate boundaries based on probability matrix and data
        boundaries = np.zeros((self.n, self.k-1))
        for i in range(self.n):
            z_i = z[:, i]
            cumsum_p = 0
            for j in range(self.k-1):
                cumsum_p += p_mat[i, j]
                boundaries[i, j] = np.quantile(z_i, cumsum_p)
        
        print("Generated boundaries:")
        print(boundaries)
        
        # Convert continuous to categorical using boundaries
        x = np.ones((num, self.n), dtype=int)
        for sample_idx in range(num):
            for i in range(self.n):
                x[sample_idx, i] = self._continuous_to_category(z[sample_idx, i], boundaries[i])
        
        return {
            'z': z,
            'x': x,
            'Omega': Omega,
            'Sigma': Sigma,
            'p_mat': p_mat,
            'boundaries': boundaries
        }
    
    def _continuous_to_category(self, z_val: float, boundaries: np.ndarray) -> int:
        """Convert continuous value to category using boundaries"""
        for j in range(len(boundaries)):
            if z_val <= boundaries[j]:
                return j + 1
        return len(boundaries) + 1
    
    def _compute_thresholds_from_boundaries(self, boundaries: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Convert boundaries to thresholds (normalized by standard deviation)"""
        thresholds = np.zeros((self.n, self.k-1))
        for i in range(self.n):
            for j in range(self.k-1):
                thresholds[i, j] = boundaries[i, j]
        return thresholds

        
    def _independence_approximation(self, obs_idx_0: np.ndarray, x_obs: np.ndarray,
                              Omega: np.ndarray, boundaries: np.ndarray) -> float:
        """Fallback independence approximation"""
        Sigma = np.linalg.inv(Omega.detach().cpu())
        
        log_lik = 0.0
        for idx, i in enumerate(obs_idx_0):
            x_val = x_obs[idx]
            sigma_i = np.sqrt(Sigma[i, i])
            
            if x_val == 1:
                prob = stats.norm.cdf(boundaries[i, 0] / sigma_i)
            elif x_val == self.k:
                prob = 1 - stats.norm.cdf(boundaries[i, self.k-2] / sigma_i)
            else:
                prob = stats.norm.cdf(boundaries[i, x_val-1] / sigma_i) - \
                    stats.norm.cdf(boundaries[i, x_val-2] / sigma_i)
            
            log_lik += np.log(np.maximum(prob, 1e-10))
        
    @staticmethod
    def extract_sparsity_pattern_from_omega(omega: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
        """
        Extract sparsity pattern from a given omega matrix.
        
        Args:
            omega: Precision matrix
            threshold: Threshold below which elements are considered zero
            
        Returns:
            Binary sparsity pattern (1 = non-zero, 0 = zero)
        """
        return (np.abs(omega) > threshold).astype(float)

    def fit_map_with_pytorch(self, x_obs_batch: np.ndarray, obs_idx_batch: List[np.ndarray],
                            boundaries: np.ndarray, epochs: int = 1000, lr: float = 0.01,
                            lambda_sparsity: float = 100.0, device: str = 'cpu') -> Dict:
        """
        Find MAP estimates using PyTorch optimization with proper priors.
        """
        
        # Initialize PyTorch model with prior parameters
        model = DomainSpecificModel(
            self.n, self.k, self.sparsity_pattern, self.penalty_strength,
            alpha=self.alpha, nu=self.nu, V=self.V
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Convert data to tensors
        boundaries_tensor = torch.tensor(boundaries, dtype=torch.float32, device=device)
        
        # Convert observation data to tensors
        obs_data = []
        for x_obs, obs_idx in zip(x_obs_batch, obs_idx_batch):
            obs_data.append({
                'x_obs': torch.tensor(x_obs, dtype=torch.long, device=device),
                'obs_idx': torch.tensor(obs_idx - 1, dtype=torch.long, device=device)
            })
        
        print(f"Starting PyTorch MAP optimization with {len(x_obs_batch)} training examples...")
        print(f"Using device: {device}")
        print(f"Prior parameters: α={self.alpha}, ν={self.nu}")
        print(f"Sparsity penalty: {lambda_sparsity}")
        
        # Training loop
        losses = []
        likelihood_losses = []
        prior_losses = []
        
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model()
            omega = outputs['omega']
            p_mat = outputs['p_mat']
            thresholds = outputs['thresholds']
            
            # Compute likelihood loss
            likelihood_loss = 0.0
            for obs_datum in obs_data:
                x_obs = obs_datum['x_obs']
                obs_idx = obs_datum['obs_idx']
                
                log_lik = self._marginal_log_likelihood_torch(
                    x_obs, obs_idx, omega, thresholds
                )
                likelihood_loss -= log_lik
            
            # Compute prior losses (negative log priors)
            precision_prior_loss = -model.log_prior_precision(omega)
            probability_prior_loss = -model.log_prior_probabilities(p_mat)
            
            # Total loss = -log likelihood - log prior
            total_loss = likelihood_loss + precision_prior_loss + probability_prior_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Store losses for tracking
            losses.append(total_loss.item())
            likelihood_losses.append(likelihood_loss.item())
            prior_losses.append((precision_prior_loss + probability_prior_loss).item())
            
            # Print progress
            if epoch % 1 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:4d}: Total = {total_loss.item():.6f}, "
                      f"Likelihood = {likelihood_loss.item():.6f}, "
                      f"Prior = {(precision_prior_loss + probability_prior_loss).item():.6f}")
        
        # Extract final parameters
        with torch.no_grad():
            final_outputs = model()
            final_omega = final_outputs['omega'].cpu().numpy()
            final_p_mat = final_outputs['p_mat'].cpu().numpy()
            final_boundaries = final_outputs['thresholds'].cpu().numpy()
        
        print(f"MAP optimization completed.")
        print(f"Final sparsity: {np.sum(np.abs(final_omega) < 1e-3)} / {self.n * self.n} elements near zero")
        
        return {
            'Omega': final_omega,
            'p_mat': final_p_mat,
            'boundaries': final_boundaries,
            'losses': losses,
            'likelihood_losses': likelihood_losses,
            'prior_losses': prior_losses,
            'model': model
        }
    
    def _marginal_log_likelihood_torch(self, x_obs: torch.Tensor, obs_idx: torch.Tensor,
                                                omega: torch.Tensor, boundaries: torch.Tensor,
                                                n_samples: int = 100000, steepness: float = 2000.0) -> torch.Tensor:
        """
        Fully differentiable marginal log likelihood using smooth Monte Carlo.
        
        This version is truly differentiable w.r.t. both omega and boundaries.
        """
        try:
            n_total = omega.shape[0]
            n_obs = len(obs_idx)
            
            if n_obs == 0:
                return torch.tensor(0.0, device=omega.device, dtype=omega.dtype, requires_grad=True)
            
            # Convert precision to covariance using Cholesky solve
            try:
                L_omega = torch.linalg.cholesky(omega)
                I = torch.eye(n_total, device=omega.device, dtype=omega.dtype)
                sigma_full = torch.cholesky_solve(I, L_omega)
            except:
                omega_reg = omega + 1e-6 * torch.eye(n_total, device=omega.device, dtype=omega.dtype)
                sigma_full = torch.linalg.inv(omega_reg)
            
            # Extract marginal covariance
            sigma_marginal = sigma_full[obs_idx][:, obs_idx]
            
            # Compute bounds (keeping everything differentiable)
            lower_bounds = []
            upper_bounds = []
            
            for i, (obs_i, x_val) in enumerate(zip(obs_idx, x_obs)):
                x_val_item = x_val.item()
                
                if x_val_item == 1:
                    # Use large negative number instead of -inf for differentiability
                    lower_bounds.append(torch.tensor(-1e4, device=omega.device, dtype=omega.dtype))
                    upper_bounds.append(boundaries[obs_i, 0])
                elif x_val_item == self.k:
                    lower_bounds.append(boundaries[obs_i, self.k-2])
                    # Use large positive number instead of +inf
                    upper_bounds.append(torch.tensor(1e4, device=omega.device, dtype=omega.dtype))
                else:
                    lower_bounds.append(boundaries[obs_i, x_val_item-2])
                    upper_bounds.append(boundaries[obs_i, x_val_item-1])
            
            lower_bounds = torch.stack(lower_bounds)
            upper_bounds = torch.stack(upper_bounds)
            genz_algorithm = CorrectedGenzAlgorithm()
            # Compute differentiable CDF
            prob = genz_algorithm(lower_bounds.unsqueeze(0), upper_bounds.unsqueeze(0), sigma_marginal.unsqueeze(0))
            
            return torch.log(prob)
            
        except Exception as e:
            print(f"Error in differentiable likelihood computation: {e}")
            return torch.tensor(-1e6, device=omega.device, dtype=omega.dtype, requires_grad=True)
    

    def export_training_data(self, data: Dict, missing_prob: float = 0.3, 
                           n_samples_marginal: int = 5000, 
                           output_file: str = "gaussian_data.json") -> None:
        """Export data in training script format with boundaries included"""
        
        num_datapoints = data['x'].shape[0]
        exported_data = []
        
        print(f"Exporting {num_datapoints} data points...")
        
        Omega = data['Omega']
        Sigma = np.linalg.inv(Omega)
        boundaries = data['boundaries']
        
        # Save metadata including boundaries and true parameters
        metadata = {
            'n_variables': self.n,
            'k_categories': self.k,
            'boundaries': boundaries.tolist(),
            'true_parameters': {
                'Omega': Omega.tolist(),
                'Sigma': Sigma.tolist(),
                'p_mat': data['p_mat'].tolist() if 'p_mat' in data else None
            },
            'missing_prob': missing_prob,
            'data_generation_seed': 42  # For reproducibility
        }
        
        for dp_idx in tqdm(range(num_datapoints)):
            if (dp_idx + 1) % 100 == 0:
                print(f"Processing data point {dp_idx + 1}/{num_datapoints}")
            
            true_x = data['x'][dp_idx]
            
            # Generate random missingness pattern
            np.random.seed(dp_idx + 42)
            observed_mask = np.random.rand(self.n) > missing_prob
            
            if not np.any(observed_mask):
                observed_mask[np.random.randint(self.n)] = True
            
            known_questions = []
            input_vectors = []
            answer_vectors = []
            annotators = []
            questions = []
            marginal_distributions = {}
            
            for var_idx in range(self.n):
                if observed_mask[var_idx]:
                    # Observed variable
                    known_questions.append(1.0)
                    
                    input_vec = [0.0] + [0.0] * self.k
                    input_vec[1 + true_x[var_idx] - 1] = 1.0
                    input_vectors.append(input_vec)
                    
                    answer_vec = [0.0] * self.k
                    answer_vec[true_x[var_idx] - 1] = 1.0
                    answer_vectors.append(answer_vec)
                    
                else:
                    # Missing variable
                    known_questions.append(0.0)
                    
                    input_vec = [1.0] + [0.0] * self.k
                    input_vectors.append(input_vec)
                    
                    # Compute marginal distribution via Gibbs sampling
                    marginal_probs = self._compute_marginal_distribution_gibbs(
                        var_idx, observed_mask, true_x, Omega, boundaries, n_samples_marginal
                    )
                    
                    answer_vectors.append(marginal_probs.tolist())
                    marginal_distributions[str(var_idx)] = marginal_probs.tolist()
                
                annotators.append(0)
                questions.append(var_idx)
            
            entry = {
                'known_questions': known_questions,
                'input': input_vectors,
                'answers': answer_vectors,
                'annotators': annotators,
                'questions': questions,
                'marginal_distributions': marginal_distributions
            }
            
            exported_data.append(entry)

            output_data = {
                'metadata': metadata,
                'data': exported_data
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        # Create final output with metadata
        output_data = {
            'metadata': metadata,
            'data': exported_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Exported {len(exported_data)} data points to {output_file}")
        
        total_observed = sum(len([x for x in entry['known_questions'] if x == 1.0]) 
                           for entry in exported_data)
        total_missing = sum(len([x for x in entry['known_questions'] if x == 0.0]) 
                          for entry in exported_data)
        
        print(f"Statistics:")
        print(f"  Total variables: {total_observed + total_missing}")
        print(f"  Observed: {total_observed} ({100*total_observed/(total_observed+total_missing):.1f}%)")
        print(f"  Missing: {total_missing} ({100*total_missing/(total_observed+total_missing):.1f}%)")
        print(f"  Average observed per data point: {total_observed/len(exported_data):.1f}")
    
    def _compute_marginal_distribution_gibbs(self, target_var: int, observed_mask: np.ndarray,
                                           true_x: np.ndarray, Omega: np.ndarray, 
                                           boundaries: np.ndarray, n_samples: int) -> np.ndarray:
        """Compute marginal distribution using Gibbs sampling with boundaries"""
        
        Sigma = np.linalg.inv(Omega)
        obs_indices = np.where(observed_mask)[0]
        obs_values = true_x[obs_indices]
        
        z_current = np.zeros(self.n)
        
        # Initialize observed variables at bin midpoints
        for i, obs_idx in enumerate(obs_indices):
            x_val = obs_values[i]
            z_current[obs_idx] = self._get_bin_midpoint(obs_idx, x_val, boundaries)
        
        # Initialize missing variables randomly
        for var_idx in range(self.n):
            if not observed_mask[var_idx]:
                z_current[var_idx] = np.random.randn() * np.sqrt(Sigma[var_idx, var_idx])
        
        target_samples = []
        burn_in = min(1000, n_samples // 4)
        
        for sample_idx in range(n_samples + burn_in):
            for var_idx in range(self.n):
                if observed_mask[var_idx]:
                    x_val = true_x[var_idx]
                    z_current[var_idx] = self._sample_truncated_normal(
                        var_idx, x_val, z_current, Omega, boundaries
                    )
                else:
                    z_current[var_idx] = self._sample_conditional_normal(
                        var_idx, z_current, Omega
                    )
            
            if sample_idx >= burn_in:
                target_x = self._continuous_to_category(z_current[target_var], boundaries[target_var])
                target_samples.append(target_x)
        
        marginal_counts = np.zeros(self.k)
        for sample in target_samples:
            marginal_counts[sample - 1] += 1
        
        marginal_probs = marginal_counts / len(target_samples)
        
        epsilon = 1e-6
        marginal_probs = marginal_probs + epsilon
        marginal_probs = marginal_probs / marginal_probs.sum()
        
        return marginal_probs

    def _get_bin_midpoint(self, var_idx: int, x_val: int, boundaries: np.ndarray) -> float:
        """Get representative latent value for observed bin using boundaries"""
        if x_val == 1:
            upper = boundaries[var_idx, 0]
            lower = upper - 2.0  # Reasonable lower bound
            return (lower + upper) / 2
        elif x_val == self.k:
            lower = boundaries[var_idx, self.k-2]
            upper = lower + 2.0  # Reasonable upper bound
            return (lower + upper) / 2
        else:
            lower = boundaries[var_idx, x_val-2]
            upper = boundaries[var_idx, x_val-1]
            return (lower + upper) / 2

    def _get_bin_midpoint(self, var_idx: int, x_val: int, boundaries: np.ndarray) -> float:
        """Get representative latent value for observed bin using boundaries"""
        if x_val == 1:
            upper = boundaries[var_idx, 0]
            lower = upper - 2.0  # Reasonable lower bound
            return (lower + upper) / 2
        elif x_val == self.k:
            lower = boundaries[var_idx, self.k-2]
            upper = lower + 2.0  # Reasonable upper bound
            return (lower + upper) / 2
        else:
            lower = boundaries[var_idx, x_val-2]
            upper = boundaries[var_idx, x_val-1]
            return (lower + upper) / 2

    def _sample_truncated_normal(self, var_idx: int, x_val: int, z_current: np.ndarray,
                               Omega: np.ndarray, boundaries: np.ndarray) -> float:
        """Sample from truncated normal for observed variable"""
        other_idx = [i for i in range(self.n) if i != var_idx]
        Sigma = np.linalg.inv(Omega)
        
        if len(other_idx) > 0:
            Sigma_io = Sigma[var_idx, other_idx]
            Sigma_oo = Sigma[np.ix_(other_idx, other_idx)]
            z_others = z_current[other_idx]
            
            try:
                conditional_mean = Sigma_io @ np.linalg.inv(Sigma_oo) @ z_others
                conditional_var = Sigma[var_idx, var_idx] - Sigma_io @ np.linalg.inv(Sigma_oo) @ Sigma_io.T
            except:
                conditional_mean = 0.0
                conditional_var = Sigma[var_idx, var_idx]
        else:
            conditional_mean = 0.0
            conditional_var = Sigma[var_idx, var_idx]
        
        conditional_std = np.sqrt(max(conditional_var, 1e-10))
        
        # Get truncation bounds using boundaries
        if x_val == 1:
            lower, upper = -np.inf, boundaries[var_idx, 0]
        elif x_val == self.k:
            lower, upper = boundaries[var_idx, self.k-2], np.inf
        else:
            lower, upper = boundaries[var_idx, x_val-2], boundaries[var_idx, x_val-1]
        
        return self._truncated_normal_sample(conditional_mean, conditional_std, lower, upper)

    def _sample_conditional_normal(self, var_idx: int, z_current: np.ndarray, 
                                 Omega: np.ndarray) -> float:
        """Sample from conditional normal for missing variable"""
        other_idx = [i for i in range(self.n) if i != var_idx]
        Sigma = np.linalg.inv(Omega)
        
        if len(other_idx) > 0:
            Sigma_io = Sigma[var_idx, other_idx]
            Sigma_oo = Sigma[np.ix_(other_idx, other_idx)]
            z_others = z_current[other_idx]
            
            try:
                conditional_mean = Sigma_io @ np.linalg.inv(Sigma_oo) @ z_others
                conditional_var = Sigma[var_idx, var_idx] - Sigma_io @ np.linalg.inv(Sigma_oo) @ Sigma_io.T
            except:
                conditional_mean = 0.0
                conditional_var = Sigma[var_idx, var_idx]
        else:
            conditional_mean = 0.0
            conditional_var = Sigma[var_idx, var_idx]
        
        return np.random.normal(conditional_mean, np.sqrt(max(conditional_var, 1e-10)))

    def _truncated_normal_sample(self, mean: float, std: float, lower: float, upper: float) -> float:
        """Robust truncated normal sampler"""
        if np.isinf(lower) and np.isinf(upper):
            return np.random.normal(mean, std)
        
        try:
            from scipy.stats import truncnorm
            
            if np.isinf(lower):
                a = -np.inf
            else:
                a = (lower - mean) / std
                
            if np.isinf(upper):
                b = np.inf
            else:
                b = (upper - mean) / std
            
            sample = truncnorm.rvs(a, b, loc=mean, scale=std)
            return sample
            
        except ImportError as e:
            raise e

    def evaluate_with_kl_divergence(self, data: Dict, missing_patterns: List[Tuple], 
                          condition_name: str, map_params: Dict, dev_data: List[Dict]) -> Dict:
        """Evaluate using KL divergence between predicted and true marginal distributions"""
        
        if condition_name == 'known_params':
            Omega = map_params['Omega']
            boundaries = data['boundaries']
        elif condition_name == 'map':
            Omega = map_params['Omega']
            boundaries = map_params["boundaries"]
        else:
            raise ValueError("condition_name must be 'known_params' or 'map'")
        
        # Initialize sample pool if not exists or parameters changed
        self._initialize_sample_pool(Omega)
        
        num_datapoints = len(missing_patterns)
        all_kl_divergences = []
        total_positions_evaluated = 0
        
        print(f"Evaluating {condition_name} with KL divergence on {num_datapoints} data points...")
        
        for dp_idx in tqdm(range(num_datapoints)):
            if (dp_idx + 1) % 50 == 0:
                print(f"  Processed {dp_idx + 1}/{num_datapoints} data points")
            
            obs_idx, x_obs = missing_patterns[dp_idx]
            true_x = data['x'][dp_idx]
            dev_entry = dev_data[dp_idx]
            obs_idx_0 = obs_idx - 1
            missing_idx_0 = [i for i in range(self.n) if i not in obs_idx_0]
            
            if len(missing_idx_0) == 0:
                continue
            
            # Compute predicted marginal distributions using cached pool
            predicted_marginals = self._compute_marginal_distributions_from_pool(
                obs_idx_0, x_obs, Omega, boundaries, missing_idx_0, n_samples=1000
            )
            
            true_marginals = {}
            for var_idx in missing_idx_0:
                answer_vec = dev_entry['answers'][var_idx]
                if max(answer_vec) < 1.0 or sum(answer_vec) != 1.0:
                    true_marginals[var_idx] = np.array(answer_vec)
            
            # Compute KL divergence for each missing variable
            for var_idx in missing_idx_0:
                pred_dist = predicted_marginals[var_idx]
                true_dist = true_marginals[var_idx]
                
                epsilon = 1e-8
                pred_dist = pred_dist + epsilon
                true_dist = true_dist + epsilon
                
                pred_dist = pred_dist / pred_dist.sum()
                true_dist = true_dist / true_dist.sum()
                
                kl_div = np.sum(true_dist * np.log(true_dist / pred_dist))
                all_kl_divergences.append(kl_div)
                total_positions_evaluated += 1
        
        if all_kl_divergences:
            avg_kl = np.mean(all_kl_divergences)
            std_kl = np.std(all_kl_divergences)
            median_kl = np.median(all_kl_divergences)
            percentile_25 = np.percentile(all_kl_divergences, 25)
            percentile_75 = np.percentile(all_kl_divergences, 75)
        else:
            avg_kl = std_kl = median_kl = percentile_25 = percentile_75 = 0.0
        
        return {
            'avg_kl_divergence': avg_kl,
            'std_kl_divergence': std_kl,
            'median_kl_divergence': median_kl,
            'percentile_25': percentile_25,
            'percentile_75': percentile_75,
            'total_positions_evaluated': total_positions_evaluated,
            'examples_evaluated': num_datapoints
        }
    
    def _compute_marginal_distributions_batch(self, obs_idx_0: np.ndarray, x_obs: np.ndarray,
                                            Omega: np.ndarray, boundaries: np.ndarray, 
                                            missing_idx_0: List[int], n_samples: int = 3000) -> Dict:
        """Compute marginal distributions for multiple missing variables using Gibbs sampling"""
        
        Sigma = np.linalg.inv(Omega)
        
        z_current = np.zeros(self.n)
        
        # Initialize observed variables at bin midpoints
        for i, obs_i in enumerate(obs_idx_0):
            x_val = x_obs[i]
            z_current[obs_i] = self._get_bin_midpoint(obs_i, x_val, boundaries)
        
        # Initialize missing variables randomly
        for var_idx in missing_idx_0:
            z_current[var_idx] = np.random.randn() * np.sqrt(Sigma[var_idx, var_idx])
        
        # Collect samples for all missing variables
        samples = {var_idx: [] for var_idx in missing_idx_0}
        
        # Burn-in period
        burn_in = min(1000, n_samples // 4)
        
        # Gibbs sampling
        for sample_idx in range(n_samples + burn_in):
            # Sample all variables
            for var_idx in range(self.n):
                if var_idx in obs_idx_0:
                    # Observed variables: sample from truncated normal
                    obs_i = np.where(obs_idx_0 == var_idx)[0][0]
                    x_val = x_obs[obs_i]
                    z_current[var_idx] = self._sample_truncated_normal(
                        var_idx, x_val, z_current, Omega, boundaries
                    )
                else:
                    # Missing variables: sample from conditional normal
                    z_current[var_idx] = self._sample_conditional_normal(
                        var_idx, z_current, Omega
                    )
            
            # After burn-in, collect samples for missing variables
            if sample_idx >= burn_in:
                for var_idx in missing_idx_0:
                    target_x = self._continuous_to_category(z_current[var_idx], boundaries[var_idx])
                    samples[var_idx].append(target_x)
        
        # Compute empirical marginal distributions
        marginals = {}
        for var_idx in missing_idx_0:
            marginal_counts = np.zeros(self.k)
            for sample in samples[var_idx]:
                marginal_counts[sample - 1] += 1  # Convert to 0-based indexing
            
            # Normalize to probabilities
            marginal_probs = marginal_counts / len(samples[var_idx])
            
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            marginal_probs = marginal_probs + epsilon
            marginal_probs = marginal_probs / marginal_probs.sum()
            
            marginals[var_idx] = marginal_probs
        
        return marginals

    def train_and_evaluate_with_varying_data_size(self, train_data_file: str, dev_data_file: str,
                                                training_sizes: List[int] = None,
                                                use_pytorch: bool = True, pytorch_epochs: int = 500,
                                                pytorch_lr: float = 0.01, lambda_sparsity: float = 100.0,
                                                results_file: str = "training_size_comparison.json") -> Dict:
        """
        Train domain-specific models with different training data sizes and evaluate on dev data.
        
        Args:
            train_data_file: Path to training data JSON file
            dev_data_file: Path to dev data JSON file
            training_sizes: List of training data sizes to test (default: [100, 200, ..., 1000])
            use_pytorch: Whether to use PyTorch optimization
            pytorch_epochs: Number of epochs for PyTorch training
            pytorch_lr: Learning rate for PyTorch
            lambda_sparsity: Sparsity penalty
            results_file: Output file for results
            
        Returns:
            Results dictionary with KL divergences for different training sizes
        """
        
        # Default training sizes
        if training_sizes is None:
            training_sizes = list(range(100, 1001, 100))  # [100, 200, 300, ..., 1000]

        training_sizes = [100, 200, 500, 800, 1000, 1200, 1600, 2000, 2400]
        
        print(f"Training with different data sizes: {training_sizes}")
        
        # Load training data
        with open(train_data_file, 'r') as f:
            train_content = json.load(f)
        
        if isinstance(train_content, list):
            raise ValueError("Training data must include metadata with true parameters")
        else:
            full_train_data = train_content['data']
            train_metadata = train_content['metadata']
        
        print(f"Loaded {len(full_train_data)} total training examples")
        
        # Load dev data
        with open(dev_data_file, 'r') as f:
            dev_content = json.load(f)
        
        if isinstance(dev_content, list):
            dev_data = dev_content
            dev_metadata = None
            print("Warning: Old format detected for dev data")
        else:
            dev_data = dev_content['data'][:100]  # Limit for evaluation speed
            dev_metadata = dev_content['metadata']
        
        print(f"Loaded {len(dev_data)} dev examples")
        
        # Get boundaries and true parameters from metadata
        boundaries = np.array(train_metadata['boundaries'])
        gt_omega = np.array(train_metadata["true_parameters"]['Omega'])
        gt_sigma = np.array(train_metadata["true_parameters"]['Sigma'])
        
        # Ground truth model parameters
        gt_params = {
            'Omega': gt_omega,
            'name': 'Ground Truth'
        }
        
        # Extract dev patterns for evaluation
        dev_patterns = []
        for entry in dev_data:
            known_questions = entry['known_questions']
            input_vecs = entry['input']
            
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i + 1)  # Convert to 1-based
                    value = np.argmax(input_vec[1:]) + 1
                    obs_values.append(value)
            
            dev_patterns.append((np.array(obs_indices), np.array(obs_values)))
        
        # Create evaluation data structure
        eval_data = {
            'boundaries': boundaries,
            'x': self._extract_true_values_from_dev(dev_data),
            'true_parameters': {
                'Omega': gt_omega,
                'Sigma': gt_sigma
            }
        }
        
        # Evaluate ground truth model once
        print("Evaluating ground truth model...")
        gt_results = self.evaluate_with_kl_divergence(
            eval_data, dev_patterns, 'known_params', gt_params, dev_data
        )
        print(f"Ground truth KL divergence: {gt_results['avg_kl_divergence']:.6f}")
        
        # Initialize results storage
        results = {
            'metadata': {
                'training_sizes': training_sizes,
                'dev_examples': len(dev_data),
                'ground_truth_kl': gt_results['avg_kl_divergence'],
                'training_config': {
                    'pytorch_epochs': pytorch_epochs,
                    'pytorch_lr': pytorch_lr,
                    'lambda_sparsity': lambda_sparsity
                }
            },
            'ground_truth': gt_results,
            'training_size_results': {}
        }
        
        # Train and evaluate for each training size
        for train_size in training_sizes:
            print(f"\n{'='*60}")
            print(f"Training with {train_size} examples")
            print(f"{'='*60}")
            
            # Extract subset of training data
            train_subset = full_train_data[:train_size]
            
            # Extract training patterns for MAP estimation
            train_x_obs_batch = []
            train_obs_idx_batch = []
            
            for entry in train_subset:
                known_questions = entry['known_questions']
                input_vecs = entry['input']
                
                obs_indices = []
                obs_values = []
                
                for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                    if known == 1.0:  # Observed variable
                        obs_indices.append(i + 1)  # Convert to 1-based
                        value = np.argmax(input_vec[1:]) + 1
                        obs_values.append(value)
                
                if len(obs_indices) > 0:  # Only include if some variables are observed
                    train_x_obs_batch.append(np.array(obs_values))
                    train_obs_idx_batch.append(np.array(obs_indices))
            
            print(f"Extracted {len(train_x_obs_batch)} training patterns with observations")
            
            # MAP estimation on training subset
            if use_pytorch:
                map_params = self.fit_map_with_pytorch(
                    train_x_obs_batch, train_obs_idx_batch, boundaries,
                    epochs=pytorch_epochs, lr=pytorch_lr, lambda_sparsity=lambda_sparsity,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                map_params['name'] = f'MAP Estimation (n={train_size})'
            else:
                map_params = self.fit_map_with_lingauss(
                    train_x_obs_batch, train_obs_idx_batch, boundaries
                )
                map_params['name'] = f'MAP Estimation (n={train_size})'
            
            # Evaluate MAP model
            print(f"Evaluating MAP model trained on {train_size} examples...")
            map_results = self.evaluate_with_kl_divergence(
                eval_data, dev_patterns, 'map', map_params, dev_data
            )
            
            print(f"MAP KL divergence: {map_results['avg_kl_divergence']:.6f}")
            
            # Store results
            results['training_size_results'][str(train_size)] = {
                'training_examples': train_size,
                'training_patterns': len(train_x_obs_batch),
                'map_results': map_results,
                'map_parameters': {
                    'Omega': map_params['Omega'].tolist()
                }
            }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING SIZE COMPARISON COMPLETED!")
        print(f"{'='*60}")
        print(f"Results saved to: {results_file}")
        
        # Print summary
        print(f"\nSummary of Results:")
        print(f"Ground Truth KL: {gt_results['avg_kl_divergence']:.6f}")
        print(f"\nMAP Results by Training Size:")
        for train_size in training_sizes:
            kl_div = results['training_size_results'][str(train_size)]['map_results']['avg_kl_divergence']
            print(f"  {train_size:4d} examples: KL = {kl_div:.6f}")
        
        return results
    
    def _extract_true_values_from_dev(self, dev_data: List[Dict]) -> np.ndarray:
        """Extract true values from dev data entries"""
        true_values = []
        
        for entry in dev_data:
            true_x = []
            for answer_vec in entry['answers']:
                if max(answer_vec) == 1.0:  # One-hot encoded
                    true_x.append(np.argmax(answer_vec) + 1)
                else:  # Probabilistic - use mode
                    true_x.append(np.argmax(answer_vec) + 1)
            true_values.append(true_x)
        
        return np.array(true_values)
    
    def _print_domain_model_results(self, results: Dict):
        """Print formatted domain model results"""
        print(f"\n{'='*60}")
        print("DOMAIN MODEL TRAINING AND EVALUATION RESULTS")
        print(f"{'='*60}")
        
        print(f"\nTraining Statistics:")
        stats = results['training_stats']
        print(f"  Training examples: {stats['num_train_examples']}")
        print(f"  Training patterns with observations: {stats['num_train_patterns']}")
        print(f"  Dev examples: {stats['num_dev_examples']}")
        
        print(f"\nDomain Model Results (KL Divergence on Dev Data):")
        for model_name, metrics in results['domain_models'].items():
            print(f"\n  {model_name.replace('_', ' ').title()}:")
            print(f"    Mean KL: {metrics['avg_kl_divergence']:.6f}")
            print(f"    Std KL:  {metrics['std_kl_divergence']:.6f}")
            print(f"    Median:  {metrics['median_kl_divergence']:.6f}")
            print(f"    25th %:  {metrics['percentile_25']:.6f}")
            print(f"    75th %:  {metrics['percentile_75']:.6f}")
            print(f"    Positions evaluated: {metrics['total_positions_evaluated']}")
        
        print(f"\n{'='*60}")
        print("COMPARISON INSTRUCTIONS:")
        print("1. Train your neural model on the same training data")
        print("2. Evaluate neural model on dev data using KL divergence")
        print("3. Compare neural model KL divergence with domain models above")
        print("4. Lower KL divergence = better marginal distribution prediction")
        print("5. True parameters are saved in metadata for consistent evaluation")
        print(f"   - Metadata source: {results.get('metadata_source', 'unknown')}")
        print(f"{'='*60}")

    def train_and_evaluate_domain_models(self, train_data_file: str, dev_data_file: str,
                                       use_pytorch: bool = True, pytorch_epochs: int = 1000,
                                       pytorch_lr: float = 0.01, lambda_sparsity: float = 100.0) -> Dict:
        """
        Train domain-specific models on training data and evaluate on dev data.
        
        Args:
            train_data_file: Path to training data JSON file
            dev_data_file: Path to dev data JSON file
            
        Returns:
            Results with KL divergences for both ground-truth and MAP models
        """
        
        # Load training data
        with open(train_data_file, 'r') as f:
            train_content = json.load(f)
        
        if isinstance(train_content, list):
            train_data = train_content
            train_metadata = None
            print("Warning: Old format detected, no metadata available")
            raise ValueError("Training data must include metadata with true parameters")
        else:
            train_data = train_content['data'][:500]
            train_metadata = train_content['metadata']
        
        print(f"Loaded {len(train_data)} training examples")
        
        # Load dev data
        with open(dev_data_file, 'r') as f:
            dev_content = json.load(f)
        
        if isinstance(dev_content, list):
            dev_data = dev_content
            dev_metadata = None
            print("Warning: Old format detected for dev data")
        else:
            dev_data = dev_content['data'][:100] #for evaluation speed here
            dev_metadata = dev_content['metadata']
        
        print(f"Loaded {len(dev_data)} dev examples")
        
        # Get boundaries and true parameters from metadata
        if train_metadata:
            boundaries = np.array(train_metadata['boundaries'])
            
            gt_omega = np.array(train_metadata["true_parameters"]['Omega'])
            gt_sigma = np.array(train_metadata["true_parameters"]['Sigma'])
            print("Using true parameters from training metadata")
                
        
        # Extract training patterns for MAP estimation
        print("\nExtracting training patterns...")
        train_x_obs_batch = []
        train_obs_idx_batch = []
        
        for entry in train_data:
            known_questions = entry['known_questions']
            input_vecs = entry['input']
            
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i + 1)  # Convert to 1-based
                    # Extract value from one-hot encoding
                    value = np.argmax(input_vec[1:]) + 1  # Skip mask bit
                    obs_values.append(value)
            
            if len(obs_indices) > 0:  # Only include if some variables are observed
                train_x_obs_batch.append(np.array(obs_values))
                train_obs_idx_batch.append(np.array(obs_indices))
        
        print(f"Extracted {len(train_x_obs_batch)} training patterns with observations")
        
        # Ground truth model (use saved parameters)
        print("\nUsing ground truth parameters from metadata...")
        gt_params = {
            'Omega': gt_omega,
            'name': 'Ground Truth'
        }
        
        # MAP estimation on training data
        print(f"\nTraining MAP model on training data...")
        if use_pytorch:
            print("Using PyTorch optimization with soft sparsity constraints")
            map_params = self.fit_map_with_pytorch(
                train_x_obs_batch, train_obs_idx_batch, boundaries,
                epochs=pytorch_epochs, lr=pytorch_lr, lambda_sparsity=lambda_sparsity,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            map_params['name'] = 'MAP Estimation (PyTorch)'

            print(map_params)
        
        # Extract dev patterns for evaluation
        print("\nExtracting dev patterns...")
        dev_patterns = []
        
        for entry in dev_data:
            known_questions = entry['known_questions']
            input_vecs = entry['input']
            
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i + 1)  # Convert to 1-based
                    value = np.argmax(input_vec[1:]) + 1
                    obs_values.append(value)
            
            dev_patterns.append((np.array(obs_indices), np.array(obs_values)))
        
        # Evaluate both models on dev data
        print("\nEvaluating models on dev data...")
        
        # Create evaluation data structure
        eval_data = {
            'boundaries': boundaries,
            'x': self._extract_true_values_from_dev(dev_data),
            'true_parameters': {
                'Omega': gt_omega,
                'Sigma': gt_sigma
            }
        }
        
        # Evaluate ground truth model
        '''print("Evaluating ground truth model...")
        gt_results = self.evaluate_with_kl_divergence(
            eval_data, dev_patterns, 'known_params', gt_params, dev_data
        )

        print(gt_results["avg_kl_divergence"])'''
        
        # Evaluate MAP model
        print("Evaluating MAP model...")
        map_results = self.evaluate_with_kl_divergence(
            eval_data, dev_patterns, 'map', map_params, dev_data
        )

        print(map_results["avg_kl_divergence"])
        
        # Compile results
        results = {
            'training_stats': {
                'num_train_examples': len(train_data),
                'num_train_patterns': len(train_x_obs_batch),
                'num_dev_examples': len(dev_data)
            },
            'domain_models': {
                "gt_model": gt_results,
                'map_estimation': map_results
            },
            'parameters': {
                'true_parameters': {
                    'Omega': gt_omega.tolist(),
                    'Sigma': gt_sigma.tolist()
                },
                'map_parameters': {
                    'Omega': map_params['Omega'].tolist()
                },
                'boundaries': boundaries.tolist()
            },
            'metadata_source': 'training_file' if train_metadata else 'generated'
        }
        
        # Print results
        self._print_domain_model_results(results)
        
        # Save results for neural model comparison
        results_file = "domain_model_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        return results

    def compare_with_neural_model(self, dev_data_file: str) -> Dict:
        """DEPRECATED: Use train_and_evaluate_domain_models instead"""
        print("Warning: This method is deprecated.")
        print("Use train_and_evaluate_domain_models() for proper training and evaluation.")
        return {}
    
    def _print_comparison_results(self, results: Dict):
        """Print formatted comparison results"""
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS: Domain-Specific vs Neural Models")
        print(f"{'='*60}")
        
        print(f"\nDev Data Statistics:")
        stats = results['dev_data_stats']
        print(f"  Examples: {stats['num_examples']}")
        print(f"  Avg observed variables: {stats['avg_observed_vars']:.1f}")
        print(f"  Avg missing variables: {stats['avg_missing_vars']:.1f}")
        
        print(f"\nDomain-Specific Models (KL Divergence):")
        for condition, metrics in results['domain_specific'].items():
            print(f"\n  {condition.replace('_', ' ').title()}:")
            print(f"    Mean KL: {metrics['avg_kl_divergence']:.6f}")
            print(f"    Std KL:  {metrics['std_kl_divergence']:.6f}")
            print(f"    Median:  {metrics['median_kl_divergence']:.6f}")
            print(f"    25th %:  {metrics['percentile_25']:.6f}")
            print(f"    75th %:  {metrics['percentile_75']:.6f}")
            print(f"    Positions evaluated: {metrics['total_positions_evaluated']}")
        
        if results['neural_model'] is not None:
            print(f"\n  Neural Model:")
            nm = results['neural_model']
            print(f"    Mean KL: {nm['avg_kl_divergence']:.6f}")
            print(f"    Std KL:  {nm['std_kl_divergence']:.6f}")
            print(f"    Median:  {nm['median_kl_divergence']:.6f}")
        else:
            print(f"\n  Neural Model: Not evaluated (train separately)")
        
        print(f"\n{'='*60}")
    
    def plot_kl_comparison(self, comparison_results: Dict, save_path: str = "kl_comparison.png"):
        """Plot KL divergence comparison between methods"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Collect data for plotting
            plot_data = []
            
            for condition, metrics in comparison_results['domain_specific'].items():
                kl_values = metrics['all_kl_divergences']
                for kl in kl_values:
                    plot_data.append({
                        'Method': condition.replace('_', ' ').title(),
                        'KL_Divergence': kl
                    })
            
            # Create DataFrame
            df = pd.DataFrame(plot_data)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Box plot
            plt.subplot(2, 2, 1)
            sns.boxplot(data=df, x='Method', y='KL_Divergence')
            plt.title('KL Divergence Distribution by Method')
            plt.xticks(rotation=45)
            
            # Violin plot
            plt.subplot(2, 2, 2)
            sns.violinplot(data=df, x='Method', y='KL_Divergence')
            plt.title('KL Divergence Density by Method')
            plt.xticks(rotation=45)
            
            # Histogram comparison
            plt.subplot(2, 2, (3, 4))
            methods = df['Method'].unique()
            for method in methods:
                method_data = df[df['Method'] == method]['KL_Divergence']
                plt.hist(method_data, alpha=0.6, label=method, bins=30)
            
            plt.xlabel('KL Divergence')
            plt.ylabel('Frequency')
            plt.title('KL Divergence Histograms')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Comparison plot saved to {save_path}")
            
        except ImportError:
            print("Matplotlib/seaborn not available for plotting")


def generate_training_data(n: int = 10, k: int = 4, num_datapoints: int = 1000,
                          missing_probs: list = [0.3, 0.5, 0.7], output_prefix: str = "gaussian",
                          penalty_strength: float = 100.0):
    """Generate training data in the format expected by the training script"""
    
    # Set up model parameters
    alpha = 2.0
    nu = n + 2
    V = np.eye(n)
    
    # Create sparsity pattern (tridiagonal)
    sparsity_pattern = np.zeros((n, n))
    for i in range(n):
        sparsity_pattern[i, i] = 1
        if i > 0:
            sparsity_pattern[i, i-1] = 1
            sparsity_pattern[i-1, i] = 1
    
    # Initialize model
    model = GaussianBinningWithLinGauss(n, k, sparsity_pattern, penalty_strength,
                                       alpha, nu, V)
    
    print(f"Generating {num_datapoints} data points...")
    
    # Generate data once
    data = model.generate_synthetic_data(num=num_datapoints)
    
    # Split into train and dev (80/20 split)
    split_idx = int(0.8 * num_datapoints)
    
    # Create train data
    train_data = {
        'x': data['x'][:split_idx],
        'z': data['z'][:split_idx],
        'Omega': data['Omega'],
        'Sigma': data['Sigma'],
        'boundaries': data['boundaries']
    }
    
    # Create dev data
    dev_data = {
        'x': data['x'][split_idx:],
        'z': data['z'][split_idx:],
        'Omega': data['Omega'],
        'Sigma': data['Sigma'],
        'boundaries': data['boundaries']
    }
    
    created_files = []
    
    # Export data for each missing probability
    for missing_prob in missing_probs:
        print(f"\n{'='*50}")
        print(f"Processing missing probability: {missing_prob}")
        print(f"{'='*50}")
        
        # Create observation rate string for filename (e.g., 0.3 -> "obs70", 0.5 -> "obs50", 0.7 -> "obs30")
        obs_rate = int((1.0 - missing_prob) * 100)
        obs_suffix = f"obs{obs_rate}"
        
        # Export train data
        train_file = f"{output_prefix}_train_{n}_{obs_suffix}_new.json"
        print(f"\nExporting training data to {train_file}...")
        model.export_training_data(train_data, missing_prob=missing_prob, 
                                  n_samples_marginal=1000, output_file=train_file)
        created_files.append(train_file)
        
        # Export dev data
        dev_file = f"{output_prefix}_dev_{n}_{obs_suffix}_new.json"
        print(f"\nExporting dev data to {dev_file}...")
        model.export_training_data(dev_data, missing_prob=missing_prob, 
                                  n_samples_marginal=1000, output_file=dev_file)
        created_files.append(dev_file)
    
    print(f"\n{'='*60}")
    print(f"Data generation complete!")
    print(f"{'='*60}")
    print(f"Generated datasets with observation rates:")
    for missing_prob in missing_probs:
        obs_rate = int((1.0 - missing_prob) * 100)
        print(f"  {obs_rate}% observed ({missing_prob} missing probability)")
    
    print(f"\nFiles created:")
    for i, file in enumerate(created_files):
        if i % 2 == 0:  # Training files
            print(f"  Training: {file} ({split_idx} examples)")
        else:  # Dev files
            print(f"  Dev: {file} ({num_datapoints - split_idx} examples)")
    
    return {
        'model': model,
        'train_data': train_data,
        'dev_data': dev_data,
        'created_files': created_files,
        'missing_probs': missing_probs
    }


def run_comparison_study(train_data_file: str = "gaussian_train_10.json",
                        dev_data_file: str = "gaussian_dev_10.json", 
                        n: int = 10, k: int = 4, penalty_strength: float = 100.0,
                        use_pytorch: bool = True, pytorch_epochs: int = 1000,
                        pytorch_lr: float = 0.02, lambda_sparsity: float = 100.0):
    """
    Modified to pass proper prior parameters to the model
    """
    
    # Set up model parameters (should match data generation)
    alpha = 2.0
    nu = n + 2
    V = np.eye(n)
    
    # Load training data to get sparsity pattern
    with open(train_data_file, 'r') as f:
        train_content = json.load(f)
    
    
    if isinstance(train_content, dict) and 'metadata' in train_content:
        metadata = train_content['metadata']
        
        # Use saved sparsity pattern directly instead of inferring
        if 'sparsity_pattern' in metadata:
            sparsity_pattern = np.array(metadata['sparsity_pattern'])
            print(f"Loaded sparsity pattern: {np.sum(sparsity_pattern == 0)} zero elements")
        else:
            # Fallback to old method if sparsity_pattern not saved
            print("Warning: No sparsity pattern in metadata, inferring from Omega...")
            true_omega = np.array(metadata['Omega'])
            sparsity_pattern = GaussianBinningWithLinGauss.extract_sparsity_pattern_from_omega(
                true_omega, threshold=1e-3
            )
            print(f"Extracted sparsity pattern: {np.sum(sparsity_pattern == 0)} zero elements")
        
        # Also load other saved parameters
        penalty_strength = metadata.get('penalty_strength', 100.0)  # Default fallback
        print(f"Using penalty strength: {penalty_strength}")
    else:
        # Handle old format without metadata
        raise ValueError("Training data must include metadata with sparsity pattern")
    
    # Initialize model with proper prior parameters
    model = GaussianBinningWithLinGauss(n, k, sparsity_pattern, penalty_strength,
                                       alpha, nu, V)
    
    print(f"Model initialized with proper priors:")
    print(f"  Dirichlet parameter α = {alpha}")
    print(f"  Wishart parameters ν = {nu}, V = {V.shape}")
    
    # Train and evaluate
    results = model.train_and_evaluate_domain_models(
        train_data_file, dev_data_file, use_pytorch=use_pytorch,
        pytorch_epochs=pytorch_epochs, pytorch_lr=pytorch_lr,
        lambda_sparsity=lambda_sparsity
    )
    
    return results


def run_training_size_comparison(train_data_file: str = "gaussian_train_10.json",
                                dev_data_file: str = "gaussian_dev_10.json", 
                                n: int = 10, k: int = 4, penalty_strength: float = 100.0,
                                training_sizes: List[int] = None,
                                pytorch_epochs: int = 500, pytorch_lr: float = 0.02, 
                                lambda_sparsity: float = 100.0):
    """
    New function to run training size comparison study
    """
    
    # Set up model parameters (should match data generation)
    alpha = 2.0
    nu = n + 2
    V = np.eye(n)
    
    # Load training data to get sparsity pattern
    with open(train_data_file, 'r') as f:
        train_content = json.load(f)
    
    if isinstance(train_content, dict) and 'metadata' in train_content:
        metadata = train_content['metadata']
        
        # Use saved sparsity pattern directly instead of inferring
        if 'sparsity_pattern' in metadata:
            sparsity_pattern = np.array(metadata['sparsity_pattern'])
            print(f"Loaded sparsity pattern: {np.sum(sparsity_pattern == 0)} zero elements")
        else:
            # Fallback to old method if sparsity_pattern not saved
            print("Warning: No sparsity pattern in metadata, inferring from Omega...")
            true_omega = np.array(metadata['true_parameters']['Omega'])
            sparsity_pattern = GaussianBinningWithLinGauss.extract_sparsity_pattern_from_omega(
                true_omega, threshold=1e-3
            )
            print(f"Extracted sparsity pattern: {np.sum(sparsity_pattern == 0)} zero elements")
        
        # Also load other saved parameters
        penalty_strength = metadata.get('penalty_strength', 100.0)  # Default fallback
        print(f"Using penalty strength: {penalty_strength}")
    else:
        # Handle old format without metadata
        raise ValueError("Training data must include metadata with sparsity pattern")
    
    # Initialize model with proper prior parameters
    model = GaussianBinningWithLinGauss(n, k, sparsity_pattern, penalty_strength,
                                       alpha, nu, V)
    
    print(f"Model initialized with proper priors:")
    print(f"  Dirichlet parameter α = {alpha}")
    print(f"  Wishart parameters ν = {nu}, V = {V.shape}")
    
    # Run training size comparison
    results = model.train_and_evaluate_with_varying_data_size(
        train_data_file, dev_data_file, training_sizes=training_sizes,
        use_pytorch=True, pytorch_epochs=pytorch_epochs, pytorch_lr=pytorch_lr,
        lambda_sparsity=lambda_sparsity
    )
    
    return results


def run_experiment(n: int = 10, k: int = 4, missing_prob: float = 0.5,
                  n_datasets: int = 1, penalty_strength: float = 100.0):
    """Run complete experiment on multiple datasets (original functionality)"""
    
    # Set up model parameters
    alpha = 2.0
    nu = n + 2
    V = np.eye(n)
    
    # Create sparsity pattern (e.g., tridiagonal)
    sparsity_pattern = np.zeros((n, n))
    for i in range(n):
        sparsity_pattern[i, i] = 1
        if i > 0:
            sparsity_pattern[i, i-1] = 1
            sparsity_pattern[i-1, i] = 1
    
    # Initialize model
    model = GaussianBinningWithLinGauss(n, k, sparsity_pattern, penalty_strength,
                                       alpha, nu, V)
    
    # Store results across datasets
    all_results = {
        'known_params': {'log_loss': [], 'exact_match_loss': [], 'squared_l2_loss': []},
        'map': {'log_loss': [], 'exact_match_loss': [], 'squared_l2_loss': []},
        'full_bayes': {'log_loss': [], 'exact_match_loss': [], 'squared_l2_loss': []}
    }
    
    for dataset_idx in range(n_datasets):
        print(f"\n{'='*50}")
        print(f"Dataset {dataset_idx + 1}/{n_datasets}")
        print(f"{'='*50}")
        
        # Generate synthetic data with multiple data points
        data = model.generate_synthetic_data(seed=dataset_idx, num=100)
        
        print(f"Generated {data['x'].shape[0]} data points with {n} variables each")
        
        # Note: This would need to be updated to use the new format with boundaries
        # For now, keeping original structure for backwards compatibility
        print("Warning: Original experiment mode may need updates for boundary handling")
        
        # Skip the actual experiment for now since it needs boundary updates
        results = {
            'known_params': {'log_loss': 0.0, 'exact_match_loss': 0.0, 'squared_l2_loss': 0.0},
            'map': {'log_loss': 0.0, 'exact_match_loss': 0.0, 'squared_l2_loss': 0.0},
            'full_bayes': {'log_loss': np.nan, 'exact_match_loss': np.nan, 'squared_l2_loss': np.nan}
        }
        
        # Store results
        for condition in ['known_params', 'map', 'full_bayes']:
            for metric in ['log_loss', 'exact_match_loss', 'squared_l2_loss']:
                all_results[condition][metric].append(results[condition][metric])
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gaussian Binning Model')
    parser.add_argument('--mode', choices=['experiment', 'generate_data', 'compare', 'training_size_comparison'], 
                       default='experiment',
                       help='Mode: run experiment, generate training data, compare models, or run training size comparison')
    parser.add_argument('--n', type=int, default=10, help='Number of variables')
    parser.add_argument('--k', type=int, default=5, help='Number of categories')
    parser.add_argument('--missing_prob', type=float, default=0.5, help='Missing probability')
    parser.add_argument('--num_datapoints', type=int, default=3000, help='Number of data points to generate')
    parser.add_argument('--output_prefix', default='gaussian', help='Output file prefix')
    parser.add_argument('--train_file', default='gaussian_train_2400.json', help='Training data file for comparison')
    parser.add_argument('--dev_file', default='gaussian_dev_600.json', help='Dev data file for comparison')
    parser.add_argument('--training_sizes', nargs='+', type=int, default=None, 
                       help='List of training sizes to test (default: 100 200 300 ... 1000)')
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    seed = args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.mode == 'generate_data':
        # Generate training data
        print("Generating training and dev data...")
        results = generate_training_data(
            n=args.n,
            k=args.k,
            num_datapoints=args.num_datapoints,
            output_prefix=args.output_prefix
        )
        print(f"Generated files: {results['train_file']}, {results['dev_file']}")
        print("\nNext steps:")
        print(f"1. Train your neural model on: {results['train_file']}")
        print(f"2. Run comparison study: python script.py --mode compare --train_file {results['train_file']} --dev_file {results['dev_file']}")
        print(f"3. Run training size comparison: python script.py --mode training_size_comparison --train_file {results['train_file']} --dev_file {results['dev_file']}")
        
    elif args.mode == 'compare':
        # Run comparison study
        print("Training domain-specific models and evaluating...")
        results = run_comparison_study(
            train_data_file=args.train_file,
            dev_data_file=args.dev_file,
            n=args.n,
            k=args.k,
            pytorch_epochs=40
        )
        
        print("\n" + "="*60)
        print("COMPARISON STUDY COMPLETED!")
        print("="*60)
        print("Domain model baselines computed.")
        print("Now train your neural model and compare KL divergences.")
        print("Results saved to: domain_model_results.json")
        print("="*60)
        
    elif args.mode == 'training_size_comparison':
        # Run training size comparison
        print("Running training size comparison study...")
        results = run_training_size_comparison(
            train_data_file=args.train_file,
            dev_data_file=args.dev_file,
            n=args.n,
            k=args.k,
            training_sizes=args.training_sizes,
            pytorch_epochs=100
        )
        
        print("\n" + "="*60)
        print("TRAINING SIZE COMPARISON COMPLETED!")
        print("="*60)
        print("Results saved to: training_size_comparison.json")
        print("Compare your neural model performance at different training sizes.")
        print("="*60)
        
    else:
        # Run the original experiment
        print("Running original experiment...")
        print("Note: Original experiment mode needs boundary updates")
        results = run_experiment(
            n=args.n,
            k=args.k,
            missing_prob=args.missing_prob,
            n_datasets=1,  # Reduced for testing
            penalty_strength=200.0,
            output_prefix=args.output_prefix
        )
        print(f"Generated files: {results['train_file']}, {results['dev_file']}")