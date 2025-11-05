"""
1D CNN Feature Encoder for Seismic Waveforms

This module implements the feature encoder that converts raw seismic waveforms
into latent representations, supporting both single-channel and multi-component inputs.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """1D Convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = True,
        norm_type: str = "group",
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        """
        Initialize convolution block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels  
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding (if None, uses kernel_size // 2)
            groups: Convolution groups
            bias: Whether to use bias
            norm_type: Normalization type ('group', 'layer', 'batch', None)
            activation: Activation function ('gelu', 'relu', 'swish')
            dropout: Dropout rate
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        
        # Normalization
        if norm_type == "group":
            num_groups = min(32, out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        
        if isinstance(self.norm, nn.LayerNorm):
            # LayerNorm expects (B, T, C)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FeatureEncoder(nn.Module):
    """
    1D CNN encoder for seismic waveforms.
    
    Converts raw waveforms to latent features with progressive downsampling.
    Supports both single-component (Z) and multi-component (Z, N, E) inputs.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        conv_layers: List[Tuple[int, int, int]] = None,  # (out_channels, kernel_size, stride)
        feature_dim: int = 768,
        dropout: float = 0.1,
        norm_type: str = "group",
        activation: str = "gelu",
    ):
        """
        Initialize feature encoder.
        
        Args:
            input_channels: Number of input channels (1 for Z, 3 for ZNE)
            conv_layers: List of (out_channels, kernel_size, stride) for each layer
            feature_dim: Final feature dimension
            dropout: Dropout rate
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        if conv_layers is None:
            # Default architecture similar to Wav2Vec2
            conv_layers = [
                (512, 10, 5),    # ~5ms kernel at 100Hz, stride=5 -> 20Hz
                (512, 3, 2),     # stride=2 -> 10Hz  
                (512, 3, 2),     # stride=2 -> 5Hz
                (512, 3, 2),     # stride=2 -> 2.5Hz
                (512, 3, 2),     # stride=2 -> 1.25Hz
                (512, 2, 2),     # stride=2 -> 0.625Hz
                (512, 2, 2),     # stride=2 -> 0.3125Hz
            ]
        
        self.input_channels = input_channels
        self.conv_layers = conv_layers
        self.feature_dim = feature_dim
        
        # Build convolution layers
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(conv_layers):
            layers.append(Conv1DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
            ))
            in_channels = out_channels
        
        self.conv_blocks = nn.ModuleList(layers)
        
        # Final projection to feature dimension
        final_channels = conv_layers[-1][0] if conv_layers else input_channels
        self.projection = nn.Linear(final_channels, feature_dim)
        
        # Calculate total stride for reference
        self.total_stride = 1
        for _, _, stride in conv_layers:
            self.total_stride *= stride
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            waveforms: Input waveforms (B, C, T)
            
        Returns:
            Features: (B, T', D) where T' = T // total_stride
        """
        x = waveforms  # (B, C, T)
        
        # Apply convolution blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # (B, C', T')
        
        # Transpose to (B, T', C') for projection
        x = x.transpose(1, 2)  # (B, T', C')
        
        # Project to feature dimension
        x = self.projection(x)  # (B, T', D)
        
        return x
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return input_length // self.total_stride
    
    def get_receptive_field(self) -> int:
        """Calculate receptive field of the encoder."""
        receptive_field = 1
        for _, kernel_size, stride in reversed(self.conv_layers):
            receptive_field = (receptive_field - 1) * stride + kernel_size
        return receptive_field


class AdaptiveFeatureEncoder(nn.Module):
    """
    Adaptive feature encoder that can handle variable input channels.
    
    Uses channel-wise processing followed by fusion for multi-component data.
    """
    
    def __init__(
        self,
        max_channels: int = 3,
        conv_layers: List[Tuple[int, int, int]] = None,
        feature_dim: int = 768,
        fusion_method: str = "concatenate",  # 'concatenate', 'attention', 'average'
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize adaptive encoder.
        
        Args:
            max_channels: Maximum number of input channels
            conv_layers: Convolution layer specifications  
            feature_dim: Output feature dimension
            fusion_method: Method for fusing multi-channel features
            dropout: Dropout rate
            **kwargs: Additional arguments for FeatureEncoder
        """
        super().__init__()
        
        self.max_channels = max_channels
        self.fusion_method = fusion_method
        self.feature_dim = feature_dim
        
        # Single-channel encoder
        self.channel_encoder = FeatureEncoder(
            input_channels=1,
            conv_layers=conv_layers,
            feature_dim=feature_dim,
            dropout=dropout,
            **kwargs
        )
        
        # Channel fusion layers
        if fusion_method == "concatenate":
            self.fusion_proj = nn.Linear(feature_dim * max_channels, feature_dim)
        elif fusion_method == "attention":
            self.channel_attention = nn.MultiheadAttention(
                feature_dim, num_heads=8, dropout=dropout, batch_first=True
            )
        elif fusion_method == "average":
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, waveforms: torch.Tensor, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive channel processing.
        
        Args:
            waveforms: Input waveforms (B, C, T)
            channel_mask: Binary mask for valid channels (B, C)
            
        Returns:
            Features: (B, T', D)
        """
        batch_size, n_channels, seq_len = waveforms.shape
        
        # Encode each channel separately
        channel_features = []
        for c in range(n_channels):
            # Extract single channel
            channel_waveform = waveforms[:, c:c+1, :]  # (B, 1, T)
            
            # Encode
            features = self.channel_encoder(channel_waveform)  # (B, T', D)
            channel_features.append(features)
        
        # Pad with zeros if needed
        while len(channel_features) < self.max_channels:
            zeros = torch.zeros_like(channel_features[0])
            channel_features.append(zeros)
        
        # Truncate if too many channels
        channel_features = channel_features[:self.max_channels]
        
        # Fuse channels
        if self.fusion_method == "concatenate":
            # Concatenate along feature dimension
            fused = torch.cat(channel_features, dim=-1)  # (B, T', D*C)
            fused = self.fusion_proj(fused)  # (B, T', D)
            
        elif self.fusion_method == "attention":
            # Stack channels as sequence dimension
            stacked = torch.stack(channel_features, dim=2)  # (B, T', C, D)
            B, T, C, D = stacked.shape
            
            # Reshape for attention: (B*T, C, D)
            stacked_flat = stacked.view(B * T, C, D)
            
            # Apply self-attention over channels
            attended, _ = self.channel_attention(
                stacked_flat, stacked_flat, stacked_flat
            )  # (B*T, C, D)
            
            # Average over channels and reshape
            fused = attended.mean(dim=1).view(B, T, D)  # (B, T', D)
            
        elif self.fusion_method == "average":
            # Simple average
            stacked = torch.stack(channel_features, dim=0)  # (C, B, T', D)
            fused = stacked.mean(dim=0)  # (B, T', D)
        
        return fused


def create_feature_encoder(
    encoder_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating feature encoders.
    
    Args:
        encoder_type: Type of encoder ('standard', 'adaptive')
        **kwargs: Arguments for encoder initialization
        
    Returns:
        Feature encoder module
    """
    if encoder_type == "standard":
        return FeatureEncoder(**kwargs)
    elif encoder_type == "adaptive":
        return AdaptiveFeatureEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Default configurations
DEFAULT_CONFIGS = {
    "small": {
        "conv_layers": [
            (256, 10, 5),
            (256, 3, 2),
            (256, 3, 2),
            (256, 3, 2),
            (512, 2, 2),
        ],
        "feature_dim": 256,
    },
    "base": {
        "conv_layers": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "feature_dim": 768,
    },
    "large": {
        "conv_layers": [
            (512, 10, 5),
            (512, 8, 4),
            (512, 4, 2),
            (512, 4, 2),
            (512, 4, 2),
            (512, 2, 2),
        ],
        "feature_dim": 1024,
    },
}