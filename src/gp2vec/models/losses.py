"""
Loss Functions for GP2Vec Training

This module implements contrastive losses, masking utilities, and auxiliary losses
for Wav2Vec2-style self-supervised learning on seismic data.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for self-supervised learning.
    
    Computes contrastive loss between predicted and target representations
    at masked positions, using negatives from the same batch.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        negative_sampling: str = "batch",  # 'batch', 'random', 'hard'
        num_negatives: int = 100,
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
            negative_sampling: Strategy for sampling negatives
            num_negatives: Number of negative samples
        """
        super().__init__()
        
        self.temperature = temperature
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        target_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss.
        
        Args:
            predictions: Predicted features at masked positions (B, T, D)
            targets: Target quantized representations (B, T, D) or indices (B, T)
            mask: Binary mask indicating masked positions (B, T)
            target_features: Optional target features for computing similarity
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract masked predictions and targets
        mask_flat = mask.view(-1)  # (B*T,)
        masked_indices = mask_flat.bool()
        
        if not masked_indices.any():
            return torch.tensor(0.0, device=predictions.device), {}
        
        # Get predictions at masked positions
        pred_flat = predictions.view(-1, predictions.size(-1))  # (B*T, D)
        masked_preds = pred_flat[masked_indices]  # (M, D) where M = number of masked positions
        
        # Handle different target formats
        if targets.dim() == 3:  # Target features (B, T, D)
            target_flat = targets.view(-1, targets.size(-1))  # (B*T, D)
            masked_targets = target_flat[masked_indices]  # (M, D)
            
            # Compute similarities
            logits = torch.matmul(
                masked_preds, masked_targets.transpose(0, 1)
            ) / self.temperature  # (M, M)
            
        elif targets.dim() == 2:  # Target indices (B, T)
            target_flat = targets.view(-1)  # (B*T,)
            masked_target_indices = target_flat[masked_indices]  # (M,)
            
            # For quantized targets, we need the actual embeddings
            if target_features is None:
                raise ValueError("target_features required when targets are indices")
            
            # Get unique target embeddings
            unique_indices, inverse_indices = torch.unique(
                masked_target_indices, return_inverse=True
            )
            target_embeds = target_features[unique_indices]  # (U, D) where U = unique targets
            
            # Compute similarities to all unique targets
            logits = torch.matmul(
                masked_preds, target_embeds.transpose(0, 1)
            ) / self.temperature  # (M, U)
            
            # Create labels for correct targets
            labels = inverse_indices  # (M,)
        
        else:
            raise ValueError(f"Invalid target shape: {targets.shape}")
        
        # Sample negatives if using batch sampling
        if self.negative_sampling == "batch" and targets.dim() == 3:
            # Use all other positions in batch as negatives
            labels = torch.arange(masked_preds.size(0), device=predictions.device)
            
        elif self.negative_sampling == "random" and targets.dim() == 3:
            # Sample random negatives from the batch
            batch_size, seq_len, feat_dim = targets.shape
            
            # Sample negative positions
            neg_indices = torch.randint(
                0, batch_size * seq_len, 
                (masked_preds.size(0), self.num_negatives),
                device=predictions.device
            )
            
            # Get negative samples
            all_targets = targets.view(-1, feat_dim)  # (B*T, D)
            negatives = all_targets[neg_indices]  # (M, N, D)
            
            # Combine positives and negatives
            positives = masked_targets.unsqueeze(1)  # (M, 1, D)
            all_samples = torch.cat([positives, negatives], dim=1)  # (M, 1+N, D)
            
            # Compute logits
            logits = torch.sum(
                masked_preds.unsqueeze(1) * all_samples, dim=-1
            ) / self.temperature  # (M, 1+N)
            
            # Labels are always 0 (first position is positive)
            labels = torch.zeros(masked_preds.size(0), device=predictions.device, dtype=torch.long)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()
            
            # Top-k accuracy
            if logits.size(1) >= 5:
                top5_acc = (logits.topk(5, dim=1)[1] == labels.unsqueeze(1)).any(dim=1).float().mean()
            else:
                top5_acc = accuracy
        
        metrics = {
            'contrastive_loss': loss,
            'contrastive_accuracy': accuracy,
            'contrastive_top5_accuracy': top5_acc,
            'num_masked_positions': torch.tensor(masked_preds.size(0), dtype=torch.float),
        }
        
        return loss, metrics


class MaskingStrategy:
    """
    Masking strategy for self-supervised learning.
    
    Implements span masking similar to Wav2Vec2.0 with configurable parameters.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        min_masks: int = 2,
        no_overlap: bool = False,
        min_space: int = 1,
    ):
        """
        Initialize masking strategy.
        
        Args:
            mask_prob: Probability of masking each position
            mask_length: Length of each mask span
            min_masks: Minimum number of masks per sequence
            no_overlap: Whether to prevent overlapping masks
            min_space: Minimum space between masks
        """
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.no_overlap = no_overlap
        self.min_space = min_space
    
    def generate_mask(
        self, 
        shape: Tuple[int, int], 
        device: torch.device,
        mask_prob: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate random masks for a batch.
        
        Args:
            shape: (batch_size, seq_len)
            device: Device to create mask on
            mask_prob: Override default mask probability
            
        Returns:
            Binary mask tensor (batch_size, seq_len)
        """
        batch_size, seq_len = shape
        
        if mask_prob is None:
            mask_prob = self.mask_prob
        
        masks = []
        
        for _ in range(batch_size):
            mask = self._generate_single_mask(seq_len, mask_prob, device)
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_single_mask(
        self, 
        seq_len: int, 
        mask_prob: float, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate mask for single sequence."""
        mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        
        if seq_len <= self.mask_length:
            return mask
        
        # Calculate number of masks needed
        num_masked = int(mask_prob * seq_len)
        if num_masked < self.min_masks:
            num_masked = self.min_masks
        
        # Generate mask starts
        num_masks = math.ceil(num_masked / self.mask_length)
        
        mask_starts = []
        for _ in range(num_masks):
            if self.no_overlap:
                # Find valid start positions (avoiding overlap)
                valid_starts = []
                for start in range(seq_len - self.mask_length + 1):
                    # Check if this start position would overlap
                    valid = True
                    for existing_start in mask_starts:
                        if (start < existing_start + self.mask_length + self.min_space and 
                            start + self.mask_length + self.min_space > existing_start):
                            valid = False
                            break
                    
                    if valid:
                        valid_starts.append(start)
                
                if valid_starts:
                    start = random.choice(valid_starts)
                    mask_starts.append(start)
                else:
                    break
            else:
                # Random start position
                start = random.randint(0, seq_len - self.mask_length)
                mask_starts.append(start)
        
        # Apply masks
        for start in mask_starts:
            end = min(start + self.mask_length, seq_len)
            mask[start:end] = True
        
        return mask
    
    def apply_channel_masking(
        self,
        mask: torch.Tensor,
        num_channels: int,
        channel_mask_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply channel-wise masking in addition to time masking.
        
        Args:
            mask: Time mask (B, T)
            num_channels: Number of channels
            channel_mask_prob: Probability of masking entire channels
            
        Returns:
            Combined mask (B, C, T)
        """
        batch_size, seq_len = mask.shape
        device = mask.device
        
        # Expand time mask to all channels
        expanded_mask = mask.unsqueeze(1).expand(batch_size, num_channels, seq_len)
        
        # Generate channel masks
        for b in range(batch_size):
            for c in range(num_channels):
                if random.random() < channel_mask_prob:
                    expanded_mask[b, c, :] = True
        
        return expanded_mask


class GP2VecLoss(nn.Module):
    """
    Combined loss function for GP2Vec training.
    
    Includes contrastive loss, quantizer diversity loss, and optional auxiliary losses.
    """
    
    def __init__(
        self,
        contrastive_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.1,
        feature_penalty_weight: float = 10.0,
        temperature: float = 0.1,
        **contrastive_kwargs
    ):
        """
        Initialize combined loss.
        
        Args:
            contrastive_loss_weight: Weight for contrastive loss
            diversity_loss_weight: Weight for codebook diversity loss  
            feature_penalty_weight: Weight for feature penalty
            temperature: Temperature for contrastive loss
            **contrastive_kwargs: Additional arguments for ContrastiveLoss
        """
        super().__init__()
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.feature_penalty_weight = feature_penalty_weight
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature, **contrastive_kwargs
        )
        self.masking_strategy = MaskingStrategy()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        quantizer_losses: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions (B, T, D)
            targets: Quantized targets (B, T, D) or (B, T)
            mask: Mask indicating prediction positions (B, T)
            features: Raw features for penalty (B, T, D)
            quantizer_losses: Additional losses from quantizer
            target_features: Target feature embeddings for contrastive loss
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}
        
        # Contrastive loss
        if self.contrastive_loss_weight > 0:
            contrastive_loss, contrastive_metrics = self.contrastive_loss(
                predictions, targets, mask, target_features
            )
            total_loss += self.contrastive_loss_weight * contrastive_loss
            metrics.update(contrastive_metrics)
        
        # Quantizer diversity loss
        if quantizer_losses is not None and self.diversity_loss_weight > 0:
            total_loss += self.diversity_loss_weight * quantizer_losses
            metrics['diversity_loss'] = quantizer_losses
        
        # Feature penalty (variance regularization)
        if features is not None and self.feature_penalty_weight > 0:
            feature_penalty = self._compute_feature_penalty(features, mask)
            total_loss += self.feature_penalty_weight * feature_penalty
            metrics['feature_penalty'] = feature_penalty
        
        metrics['total_loss'] = total_loss
        
        return total_loss, metrics
    
    def _compute_feature_penalty(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature penalty to encourage diversity.
        
        Penalizes low variance in features to prevent mode collapse.
        """
        # Only compute penalty on non-masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(features)
        masked_features = features.masked_fill(mask_expanded.bool(), 0)
        
        # Compute variance across the feature dimension
        feature_var = torch.var(masked_features, dim=-1, keepdim=True)
        
        # Penalty is negative log variance (encourages higher variance)
        penalty = -torch.log(feature_var + 1e-8).mean()
        
        return penalty


def create_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    mask_prob: float = 0.65,
    mask_length: int = 10,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function to create masks.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device for mask tensor
        mask_prob: Masking probability
        mask_length: Mask span length
        **kwargs: Additional arguments for MaskingStrategy
        
    Returns:
        Binary mask tensor (batch_size, seq_len)
    """
    masking = MaskingStrategy(
        mask_prob=mask_prob, 
        mask_length=mask_length, 
        **kwargs
    )
    return masking.generate_mask((batch_size, seq_len), device)


# Default loss configurations
DEFAULT_LOSS_CONFIGS = {
    "base": {
        "contrastive_loss_weight": 1.0,
        "diversity_loss_weight": 0.1,
        "feature_penalty_weight": 10.0,
        "temperature": 0.1,
        "negative_sampling": "batch",
    },
    "strong_regularization": {
        "contrastive_loss_weight": 1.0,
        "diversity_loss_weight": 0.5,
        "feature_penalty_weight": 50.0,
        "temperature": 0.05,
        "negative_sampling": "batch",
    },
}