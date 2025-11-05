"""
Transformer Context Encoder

This module implements the Transformer encoder that processes masked latent features
and learns contextual representations for self-supervised learning.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :].transpose(0, 1)


class TransformerContextEncoder(nn.Module):
    """
    Transformer encoder for learning contextual representations.
    
    Processes masked latent features and learns to predict missing information
    through self-attention mechanisms.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_positional_encoding: bool = True,
        max_positions: int = 10000,
    ):
        """
        Initialize transformer encoder.
        
        Args:
            feature_dim: Feature dimension (d_model)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (if None, uses 4 * feature_dim)
            dropout: General dropout rate
            attention_dropout: Attention dropout rate
            activation: Activation function in feed-forward layers
            layer_norm_eps: Layer normalization epsilon
            use_positional_encoding: Whether to add positional encoding
            max_positions: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        
        if ff_dim is None:
            ff_dim = 4 * feature_dim
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(feature_dim, max_positions)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,  # Pre-norm like GPT
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim, eps=layer_norm_eps)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            features: Input features (B, T, D)
            attention_mask: Attention mask for self-attention (T, T)
            key_padding_mask: Key padding mask (B, T)
            
        Returns:
            Contextualized features (B, T, D)
        """
        # Add positional encoding
        if self.use_positional_encoding:
            features = features.transpose(0, 1)  # (T, B, D)
            features = self.pos_encoding(features)
            features = features.transpose(0, 1)  # (B, T, D)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Transform attention mask format if provided
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert from (T, T) to the format expected by nn.TransformerEncoder
            # Need to expand for all samples in batch and all heads
            batch_size = features.size(0)
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer
        output = self.transformer(
            features,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask
        )
        
        return output


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for transformer.
    
    More flexible than absolute positional encoding for variable length sequences.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 512):
        """
        Initialize relative positional encoding.
        
        Args:
            d_model: Model dimension  
            max_relative_position: Maximum relative position to encode
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Learnable relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position embeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position embeddings (seq_len, seq_len, d_model)
        """
        device = self.embeddings.weight.device
        
        # Create relative position matrix
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip to maximum range
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to positive indices
        relative_positions = relative_positions + self.max_relative_position
        
        # Get embeddings
        embeddings = self.embeddings(relative_positions)  # (seq_len, seq_len, d_model)
        
        return embeddings


class ConformerEncoder(nn.Module):
    """
    Conformer encoder combining CNN and Transformer.
    
    Integrates convolution and self-attention for processing sequential data
    with both local and global dependencies.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize Conformer encoder.
        
        Args:
            feature_dim: Feature dimension
            num_layers: Number of conformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            conv_kernel_size: Convolution kernel size
            dropout: Dropout rate
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        if ff_dim is None:
            ff_dim = 4 * feature_dim
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self, 
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Conformer encoder."""
        x = features
        
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        
        x = self.layer_norm(x)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block with feed-forward, attention, and convolution."""
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        ff_dim: int,
        conv_kernel_size: int,
        dropout: float = 0.1,
    ):
        """Initialize Conformer block."""
        super().__init__()
        
        # Feed-forward modules
        self.ff1 = FeedForward(feature_dim, ff_dim, dropout)
        self.ff2 = FeedForward(feature_dim, ff_dim, dropout)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(feature_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv = ConvolutionModule(feature_dim, conv_kernel_size, dropout)
        
        # Layer norms
        self.ff1_norm = nn.LayerNorm(feature_dim)
        self.ff2_norm = nn.LayerNorm(feature_dim)
        self.conv_norm = nn.LayerNorm(feature_dim)
        
        # Scaling factors
        self.ff_scale = 0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Conformer block."""
        # First feed-forward
        x = x + self.ff_scale * self.ff1(self.ff1_norm(x))
        
        # Multi-head self-attention
        residual = x
        x = self.attention_norm(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + self.attention_dropout(attn_out)
        
        # Convolution
        x = x + self.conv(self.conv_norm(x))
        
        # Second feed-forward
        x = x + self.ff_scale * self.ff2(self.ff2_norm(x))
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with Swish activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward network."""
        super().__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.dropout(self.w_2(self.activation(self.w_1(x))))


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(
        self, 
        d_model: int, 
        kernel_size: int, 
        dropout: float = 0.1
    ):
        """Initialize convolution module."""
        super().__init__()
        
        # Pointwise convolutions
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        
        # First pointwise convolution + GLU
        x = self.pointwise_conv1(x)  # (B, 2*D, T)
        x = F.glu(x, dim=1)  # (B, D, T)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Second pointwise convolution
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (B, T, D)
        return x


def create_context_encoder(
    encoder_type: str = "transformer",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating context encoders.
    
    Args:
        encoder_type: Type of encoder ('transformer', 'conformer')
        **kwargs: Arguments for encoder initialization
        
    Returns:
        Context encoder module
    """
    if encoder_type == "transformer":
        return TransformerContextEncoder(**kwargs)
    elif encoder_type == "conformer":
        return ConformerEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Default configurations
DEFAULT_TRANSFORMER_CONFIGS = {
    "small": {
        "feature_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ff_dim": 1024,
        "dropout": 0.1,
    },
    "base": {
        "feature_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "ff_dim": 3072,
        "dropout": 0.1,
    },
    "large": {
        "feature_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "ff_dim": 4096,
        "dropout": 0.1,
    },
}