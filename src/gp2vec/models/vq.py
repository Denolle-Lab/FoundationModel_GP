"""
Vector Quantization for Wav2Vec2-style Training

This module implements both Gumbel-Softmax and EMA-based K-means quantization
for creating discrete targets in self-supervised learning.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel-Softmax Vector Quantizer.
    
    Uses Gumbel-Softmax for differentiable discrete sampling with
    temperature annealing during training.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_vars: int,
        codebook_size: int,
        temp_schedule: Tuple[float, float, float] = (2.0, 0.5, 0.999995),
        diversity_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
    ):
        """
        Initialize Gumbel Vector Quantizer.
        
        Args:
            feature_dim: Input feature dimension
            num_vars: Number of codebook variables/groups
            codebook_size: Size of each codebook
            temp_schedule: (start_temp, end_temp, decay_factor)
            diversity_loss_weight: Weight for codebook diversity loss
            commitment_loss_weight: Weight for commitment loss
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_vars = num_vars
        self.codebook_size = codebook_size
        self.diversity_loss_weight = diversity_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        
        # Temperature scheduling
        self.start_temp, self.end_temp, self.temp_decay = temp_schedule
        self.register_buffer("temperature", torch.tensor(self.start_temp))
        self.register_buffer("step_count", torch.tensor(0))
        
        # Learnable codebooks
        self.codebooks = nn.Parameter(
            torch.randn(num_vars, codebook_size, feature_dim // num_vars)
        )
        
        # Projection layers
        self.project_q = nn.Linear(feature_dim, feature_dim)
        self.project_out = nn.Linear(feature_dim, feature_dim)
        
        # Initialize codebooks
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.codebooks, -1 / self.codebook_size, 1 / self.codebook_size)
        nn.init.xavier_uniform_(self.project_q.weight)
        nn.init.xavier_uniform_(self.project_out.weight)
    
    def update_temperature(self):
        """Update temperature for annealing."""
        if self.training:
            self.step_count += 1
            self.temperature = max(
                self.end_temp,
                self.start_temp * (self.temp_decay ** self.step_count.item())
            )
    
    def forward(
        self, 
        features: torch.Tensor,
        produce_targets: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantizer.
        
        Args:
            features: Input features (B, T, D)
            produce_targets: Whether to produce quantized targets
            
        Returns:
            Tuple of (quantized_features, targets, extra_losses)
        """
        batch_size, seq_len, _ = features.shape
        
        # Project input
        q = self.project_q(features)  # (B, T, D)
        
        # Reshape for codebook lookup: (B, T, num_vars, D//num_vars)
        q = q.view(batch_size, seq_len, self.num_vars, -1)
        
        # Compute distances to codebooks
        # q: (B, T, num_vars, D//num_vars)
        # codebooks: (num_vars, codebook_size, D//num_vars)
        
        distances = torch.cdist(
            q.view(batch_size * seq_len * self.num_vars, -1),
            self.codebooks.view(self.num_vars * self.codebook_size, -1)
        )  # (B*T*num_vars, num_vars*codebook_size)
        
        # Reshape distances: (B, T, num_vars, num_vars, codebook_size)
        distances = distances.view(
            batch_size, seq_len, self.num_vars, self.num_vars, self.codebook_size
        )
        
        # Extract relevant distances (diagonal)
        # We want distances[b, t, v, v, :] for each variable v
        var_distances = torch.diagonal(distances, dim1=2, dim2=3).transpose(-1, -2)
        # var_distances: (B, T, num_vars, codebook_size)
        
        # Convert distances to logits (negative distances)
        logits = -var_distances
        
        if produce_targets:
            # Sample using Gumbel-Softmax
            self.update_temperature()
            
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits_with_noise = (logits + gumbel_noise) / self.temperature
            
            # Softmax sampling
            soft_samples = F.softmax(logits_with_noise, dim=-1)  # (B, T, num_vars, codebook_size)
            
            # Hard sampling (straight-through estimator)
            hard_indices = soft_samples.argmax(dim=-1)  # (B, T, num_vars)
            hard_samples = F.one_hot(hard_indices, self.codebook_size).float()
            
            # Straight-through: use hard for forward, soft for backward
            samples = hard_samples + soft_samples - soft_samples.detach()
            
        else:
            # Deterministic selection (during inference)
            hard_indices = logits.argmax(dim=-1)  # (B, T, num_vars)
            samples = F.one_hot(hard_indices, self.codebook_size).float()
        
        # Get quantized features
        # samples: (B, T, num_vars, codebook_size)  
        # codebooks: (num_vars, codebook_size, D//num_vars)
        quantized_vars = torch.einsum(
            'btvk,vkd->btvd', samples, self.codebooks
        )  # (B, T, num_vars, D//num_vars)
        
        # Concatenate variables
        quantized = quantized_vars.view(batch_size, seq_len, -1)  # (B, T, D)
        
        # Project output
        quantized = self.project_out(quantized)
        
        # Compute losses
        extra_losses = self._compute_losses(q, samples, hard_indices if produce_targets else None)
        
        # Create targets (indices for contrastive learning)
        if produce_targets:
            targets = self._create_targets(hard_indices)
        else:
            targets = None
        
        return quantized, targets, extra_losses
    
    def _compute_losses(
        self, 
        features: torch.Tensor,
        samples: torch.Tensor, 
        indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute auxiliary losses."""
        losses = torch.tensor(0.0, device=features.device)
        
        # Diversity loss - encourage uniform codebook usage
        if self.diversity_loss_weight > 0 and indices is not None:
            # Compute codebook usage frequency
            batch_size, seq_len, num_vars = indices.shape
            
            # Flatten indices and compute histogram
            flat_indices = indices.view(-1, num_vars)  # (B*T, num_vars)
            
            diversity_losses = []
            for v in range(num_vars):
                var_indices = flat_indices[:, v]  # (B*T,)
                
                # Compute probability distribution
                counts = torch.bincount(var_indices, minlength=self.codebook_size)
                probs = counts.float() / counts.sum()
                
                # Entropy loss (maximize entropy = minimize negative entropy)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                max_entropy = math.log(self.codebook_size)
                diversity_loss = -(entropy / max_entropy)  # Negative because we want to maximize
                
                diversity_losses.append(diversity_loss)
            
            diversity_loss = torch.stack(diversity_losses).mean()
            losses = losses + self.diversity_loss_weight * diversity_loss
        
        return losses
    
    def _create_targets(self, indices: torch.Tensor) -> torch.Tensor:
        """Create targets for contrastive learning."""
        # Combine indices from all variables into single target
        # indices: (B, T, num_vars)
        batch_size, seq_len, num_vars = indices.shape
        
        # Simple approach: sum weighted indices
        weights = torch.arange(num_vars, device=indices.device) * self.codebook_size
        targets = torch.sum(indices * weights.view(1, 1, -1), dim=-1)  # (B, T)
        
        return targets
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Get current codebook usage statistics."""
        with torch.no_grad():
            # This would typically be called with accumulated statistics
            return torch.ones(self.num_vars, self.codebook_size, device=self.codebooks.device)


class EMAVectorQuantizer(nn.Module):
    """
    EMA-based Vector Quantizer (like VQ-VAE).
    
    Uses exponential moving averages to update codebook embeddings.
    """
    
    def __init__(
        self,
        feature_dim: int,
        codebook_size: int,
        commitment_loss_weight: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """
        Initialize EMA Vector Quantizer.
        
        Args:
            feature_dim: Input feature dimension
            codebook_size: Size of codebook
            commitment_loss_weight: Weight for commitment loss
            ema_decay: EMA decay factor
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.commitment_loss_weight = commitment_loss_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.register_buffer("embeddings", torch.randn(codebook_size, feature_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", torch.randn(codebook_size, feature_dim))
        
        # Initialize embeddings
        self.embeddings.uniform_(-1 / codebook_size, 1 / codebook_size)
        self.ema_weight.copy_(self.embeddings)
    
    def forward(
        self, 
        features: torch.Tensor,
        produce_targets: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through EMA quantizer.
        
        Args:
            features: Input features (B, T, D)
            produce_targets: Whether to produce targets
            
        Returns:
            Tuple of (quantized_features, targets, extra_losses)
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Flatten features for quantization
        flat_features = features.view(-1, feature_dim)  # (B*T, D)
        
        # Compute distances to embeddings
        distances = torch.cdist(flat_features, self.embeddings)  # (B*T, K)
        
        # Find nearest embeddings
        indices = distances.argmin(dim=1)  # (B*T,)
        
        # Get quantized features
        quantized_flat = F.embedding(indices, self.embeddings)  # (B*T, D)
        quantized = quantized_flat.view(batch_size, seq_len, feature_dim)  # (B, T, D)
        
        # Straight-through estimator
        quantized = features + (quantized - features).detach()
        
        # Update embeddings with EMA (only during training)
        if self.training:
            self._update_embeddings(flat_features, indices)
        
        # Compute commitment loss
        commitment_loss = F.mse_loss(features, quantized.detach())
        extra_losses = self.commitment_loss_weight * commitment_loss
        
        # Create targets
        if produce_targets:
            targets = indices.view(batch_size, seq_len)  # (B, T)
        else:
            targets = None
        
        return quantized, targets, extra_losses
    
    def _update_embeddings(self, features: torch.Tensor, indices: torch.Tensor):
        """Update embeddings using EMA."""
        # Compute cluster assignments
        encodings = F.one_hot(indices, self.codebook_size).float()  # (B*T, K)
        
        # Update cluster sizes
        cluster_size = encodings.sum(dim=0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        
        # Update embeddings
        embed_sum = encodings.t() @ features  # (K, D)
        self.ema_weight.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )
        
        # Normalize embeddings
        cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (self.ema_cluster_size.sum() + self.codebook_size * self.epsilon)
        )
        self.embeddings.copy_(self.ema_weight / cluster_size.unsqueeze(1))
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook usage statistics."""
        return self.ema_cluster_size / self.ema_cluster_size.sum()


def create_vector_quantizer(
    quantizer_type: str = "gumbel",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating vector quantizers.
    
    Args:
        quantizer_type: Type of quantizer ('gumbel', 'ema')
        **kwargs: Arguments for quantizer initialization
        
    Returns:
        Vector quantizer module
    """
    if quantizer_type == "gumbel":
        return GumbelVectorQuantizer(**kwargs)
    elif quantizer_type == "ema":
        return EMAVectorQuantizer(**kwargs)
    else:
        raise ValueError(f"Unknown quantizer type: {quantizer_type}")


# Default configurations
DEFAULT_VQ_CONFIGS = {
    "gumbel_small": {
        "num_vars": 2,
        "codebook_size": 320,
        "temp_schedule": (2.0, 0.5, 0.999995),
        "diversity_loss_weight": 0.1,
    },
    "gumbel_base": {
        "num_vars": 2,
        "codebook_size": 8192,
        "temp_schedule": (2.0, 0.5, 0.999995),
        "diversity_loss_weight": 0.1,
    },
    "ema_small": {
        "codebook_size": 640,
        "ema_decay": 0.99,
        "commitment_loss_weight": 0.25,
    },
    "ema_base": {
        "codebook_size": 8192,
        "ema_decay": 0.99,
        "commitment_loss_weight": 0.25,
    },
}