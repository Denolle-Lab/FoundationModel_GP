"""
GP2Vec: Wav2Vec2-style Self-Supervised Model for Seismic Data

This module implements the complete GP2Vec model that combines:
- Feature encoder (CNN)
- Vector quantizer 
- Context encoder (Transformer)
- Station metadata conditioning
- Contrastive learning objectives
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder, create_feature_encoder
from .transformer import TransformerContextEncoder, create_context_encoder
from .vq import GumbelVectorQuantizer, create_vector_quantizer
from .losses import GP2VecLoss, MaskingStrategy

logger = logging.getLogger(__name__)


class MetadataEmbedder(nn.Module):
    """
    Station metadata embedding module.
    
    Converts station metadata (categorical and continuous features) into
    dense embeddings for conditioning the waveform model.
    """
    
    def __init__(
        self,
        categorical_features: Dict[str, Dict[str, int]],
        continuous_features: List[str],
        embedding_dim: int = 128,
        continuous_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize metadata embedder.
        
        Args:
            categorical_features: Dict mapping feature names to vocab info
                Format: {feature_name: {'vocab_size': int, 'embed_dim': int}}
            continuous_features: List of continuous feature names
            embedding_dim: Output embedding dimension
            continuous_dim: Dimension for continuous feature processing
            dropout: Dropout rate
        """
        super().__init__()
        
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_dim = embedding_dim
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        total_categorical_dim = 0
        
        for feature_name, config in categorical_features.items():
            vocab_size = config['vocab_size']
            embed_dim = config['embed_dim']
            
            self.categorical_embeddings[feature_name] = nn.Embedding(
                vocab_size, embed_dim
            )
            total_categorical_dim += embed_dim
        
        # Continuous feature processing
        if continuous_features:
            self.continuous_mlp = nn.Sequential(
                nn.Linear(len(continuous_features), continuous_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(continuous_dim, continuous_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            continuous_dim = 0
            self.continuous_mlp = None
        
        # Final projection
        total_input_dim = total_categorical_dim + continuous_dim
        if total_input_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(total_input_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            # No metadata features - create dummy embedding
            self.projection = nn.Parameter(torch.zeros(1, embedding_dim))
    
    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through metadata embedder.
        
        Args:
            metadata: Dictionary of metadata tensors
            
        Returns:
            Metadata embeddings (B, embedding_dim)
        """
        if not self.categorical_features and not self.continuous_features:
            # No metadata - return dummy embeddings
            batch_size = 1  # Will be broadcast later
            return self.projection.expand(batch_size, -1)
        
        embeddings = []
        
        # Process categorical features
        for feature_name in self.categorical_features:
            if feature_name in metadata:
                indices = metadata[feature_name].long()
                emb = self.categorical_embeddings[feature_name](indices)
                embeddings.append(emb)
        
        # Process continuous features
        if self.continuous_features and any(f in metadata for f in self.continuous_features):
            continuous_values = []
            for feature_name in self.continuous_features:
                if feature_name in metadata:
                    continuous_values.append(metadata[feature_name].unsqueeze(-1))
                else:
                    # Fill missing features with zeros
                    batch_size = list(metadata.values())[0].size(0)
                    zeros = torch.zeros(batch_size, 1, device=list(metadata.values())[0].device)
                    continuous_values.append(zeros)
            
            if continuous_values:
                continuous_tensor = torch.cat(continuous_values, dim=-1)
                continuous_emb = self.continuous_mlp(continuous_tensor)
                embeddings.append(continuous_emb)
        
        # Combine embeddings
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            return self.projection(combined)
        else:
            # Fallback to dummy embedding
            batch_size = list(metadata.values())[0].size(0)
            return torch.zeros(batch_size, self.embedding_dim, 
                             device=list(metadata.values())[0].device)


class GP2Vec(nn.Module):
    """
    Complete GP2Vec model for self-supervised seismic representation learning.
    
    Architecture:
    1. Feature encoder (CNN) converts waveforms to latent features
    2. Vector quantizer creates discrete targets 
    3. Metadata embedder processes station information
    4. Context encoder (Transformer) learns representations from masked features
    5. Contrastive loss trains the model to predict quantized targets
    """
    
    def __init__(
        self,
        # Feature encoder config
        input_channels: int = 3,
        encoder_config: Optional[Dict] = None,
        encoder_type: str = "standard",
        
        # Vector quantizer config
        quantizer_config: Optional[Dict] = None,
        quantizer_type: str = "gumbel",
        
        # Context encoder config
        context_config: Optional[Dict] = None,
        context_type: str = "transformer",
        
        # Metadata config
        metadata_config: Optional[Dict] = None,
        metadata_fusion: str = "add",  # 'add', 'concat', 'cross_attention', 'film'
        
        # Training config
        loss_config: Optional[Dict] = None,
        masking_config: Optional[Dict] = None,
        
        # General config
        feature_dim: int = 768,
        dropout: float = 0.1,
    ):
        """
        Initialize GP2Vec model.
        
        Args:
            input_channels: Number of input channels (1 for Z, 3 for ZNE)
            encoder_config: Configuration for feature encoder
            encoder_type: Type of feature encoder
            quantizer_config: Configuration for vector quantizer
            quantizer_type: Type of vector quantizer
            context_config: Configuration for context encoder
            context_type: Type of context encoder
            metadata_config: Configuration for metadata embedder
            metadata_fusion: Method for fusing metadata with features
            loss_config: Configuration for loss function
            masking_config: Configuration for masking strategy
            feature_dim: Model feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.metadata_fusion = metadata_fusion
        
        # Feature encoder
        encoder_config = encoder_config or {}
        encoder_config.update({
            'input_channels': input_channels,
            'feature_dim': feature_dim,
            'dropout': dropout,
        })
        self.feature_encoder = create_feature_encoder(encoder_type, **encoder_config)
        
        # Vector quantizer
        quantizer_config = quantizer_config or {}
        quantizer_config.update({
            'feature_dim': feature_dim,
        })
        self.quantizer = create_vector_quantizer(quantizer_type, **quantizer_config)
        
        # Metadata embedder
        if metadata_config:
            self.metadata_embedder = MetadataEmbedder(
                dropout=dropout,
                embedding_dim=feature_dim,
                **metadata_config
            )
            self.use_metadata = True
        else:
            self.metadata_embedder = None
            self.use_metadata = False
        
        # Metadata fusion layers
        if self.use_metadata:
            if metadata_fusion == "concat":
                # Concatenation requires projection back to feature_dim
                self.metadata_projection = nn.Linear(2 * feature_dim, feature_dim)
            elif metadata_fusion == "cross_attention":
                # Cross-attention between features and metadata
                self.cross_attention = nn.MultiheadAttention(
                    feature_dim, num_heads=8, dropout=dropout, batch_first=True
                )
            elif metadata_fusion == "film":
                # FiLM conditioning
                self.film_scale = nn.Linear(feature_dim, feature_dim)
                self.film_shift = nn.Linear(feature_dim, feature_dim)
            # 'add' fusion requires no additional parameters
        
        # Context encoder  
        context_config = context_config or {}
        context_config.update({
            'feature_dim': feature_dim,
            'dropout': dropout,
        })
        self.context_encoder = create_context_encoder(context_type, **context_config)
        
        # Prediction head
        self.prediction_head = nn.Linear(feature_dim, feature_dim)
        
        # Loss function
        loss_config = loss_config or {}
        self.loss_fn = GP2VecLoss(**loss_config)
        
        # Masking strategy
        masking_config = masking_config or {}
        self.masking_strategy = MaskingStrategy(**masking_config)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
    
    def load_wav2vec_weights(
        self,
        weights_path: Union[str, Path],
        strict: bool = False,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Load pre-trained Wav2Vec2 weights into GP2Vec model.
        
        Args:
            weights_path: Path to extracted Wav2Vec2 weights
            strict: Whether to require exact parameter matching
            verbose: Whether to print loading progress
            
        Returns:
            Dictionary with loading statistics
        """
        try:
            from ..utils.wav2vec_transfer import load_wav2vec_weights, initialize_gp2vec_from_wav2vec
        except ImportError:
            logger.error("Could not import wav2vec_transfer utilities")
            raise ImportError("wav2vec_transfer module not available")
        
        if verbose:
            logger.info(f"🔄 Loading Wav2Vec2 weights from {weights_path}")
        
        # Load weights
        wav2vec_weights = load_wav2vec_weights(str(weights_path))
        
        # Initialize model with weights
        updated_params = initialize_gp2vec_from_wav2vec(
            self, wav2vec_weights, strict=strict
        )
        
        stats = {
            'total_params': len(list(self.parameters())),
            'updated_params': len(updated_params),
            'update_ratio': len(updated_params) / len(list(self.parameters()))
        }
        
        if verbose:
            logger.info(f"✅ Loaded {len(updated_params)} parameters from Wav2Vec2")
            logger.info(f"   - Update ratio: {stats['update_ratio']:.1%}")
        
        return stats
    
    def forward(
        self,
        waveforms: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GP2Vec model.
        
        Args:
            waveforms: Input waveforms (B, C, T)
            metadata: Station metadata dictionary
            mask: Optional pre-computed mask (B, T')
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing model outputs and losses
        """
        batch_size = waveforms.size(0)
        device = waveforms.device
        
        # 1. Feature encoding
        features = self.feature_encoder(waveforms)  # (B, T', D)
        seq_len = features.size(1)
        
        # 2. Vector quantization (for targets)
        with torch.no_grad():
            quantized_features, targets, quantizer_losses = self.quantizer(
                features, produce_targets=True
            )
        
        # 3. Generate mask if not provided
        if mask is None:
            mask = self.masking_strategy.generate_mask(
                (batch_size, seq_len), device
            )
        
        # 4. Metadata embedding and fusion
        if self.use_metadata and metadata is not None:
            metadata_emb = self.metadata_embedder(metadata)  # (B, D)
            
            # Expand metadata to sequence length
            metadata_emb = metadata_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T', D)
            
            # Fuse metadata with features
            features = self._fuse_metadata(features, metadata_emb)
        
        # 5. Apply masking to features
        mask_expanded = mask.unsqueeze(-1).expand_as(features)
        masked_features = features.masked_fill(mask_expanded.bool(), 0.0)
        
        # 6. Context encoding
        contextualized = self.context_encoder(masked_features)  # (B, T', D)
        
        # 7. Prediction head
        predictions = self.prediction_head(contextualized)  # (B, T', D)
        
        # 8. Compute loss
        loss, metrics = self.loss_fn(
            predictions=predictions,
            targets=targets,
            mask=mask,
            features=features,
            quantizer_losses=quantizer_losses,
            target_features=None,  # Could pass quantizer codebook here
        )
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'mask': mask,
            'metrics': metrics,
        }
        
        if return_features:
            outputs.update({
                'raw_features': features,
                'quantized_features': quantized_features,
                'contextualized_features': contextualized,
            })
        
        return outputs
    
    def _fuse_metadata(
        self, 
        features: torch.Tensor, 
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuse metadata embeddings with waveform features."""
        if self.metadata_fusion == "add":
            return features + metadata_emb
        
        elif self.metadata_fusion == "concat":
            combined = torch.cat([features, metadata_emb], dim=-1)
            return self.metadata_projection(combined)
        
        elif self.metadata_fusion == "cross_attention":
            # Use metadata as query, features as key/value
            attended, _ = self.cross_attention(
                metadata_emb, features, features
            )
            return features + attended
        
        elif self.metadata_fusion == "film":
            # Feature-wise Linear Modulation
            scale = self.film_scale(metadata_emb)
            shift = self.film_shift(metadata_emb)
            return scale * features + shift
        
        else:
            raise ValueError(f"Unknown metadata fusion method: {self.metadata_fusion}")
    
    def encode(
        self,
        waveforms: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract features for downstream tasks.
        
        Args:
            waveforms: Input waveforms (B, C, T)
            metadata: Station metadata
            layer: Which layer to extract features from (-1 = final)
            
        Returns:
            Extracted features (B, T', D)
        """
        with torch.no_grad():
            # Feature encoding
            features = self.feature_encoder(waveforms)
            
            # Metadata fusion
            if self.use_metadata and metadata is not None:
                metadata_emb = self.metadata_embedder(metadata)
                seq_len = features.size(1)
                metadata_emb = metadata_emb.unsqueeze(1).expand(-1, seq_len, -1)
                features = self._fuse_metadata(features, metadata_emb)
            
            # Context encoding
            contextualized = self.context_encoder(features)
            
            return contextualized
    
    def get_targets(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Get quantized targets for a given input."""
        with torch.no_grad():
            features = self.feature_encoder(waveforms)
            _, targets, _ = self.quantizer(features, produce_targets=True)
            return targets


def create_gp2vec_model(
    model_size: str = "base",
    input_channels: int = 3,
    metadata_config: Optional[Dict] = None,
    **kwargs
) -> GP2Vec:
    """
    Factory function for creating GP2Vec models.
    
    Args:
        model_size: Model size ('small', 'base', 'large')
        input_channels: Number of input channels
        metadata_config: Metadata configuration
        **kwargs: Additional model arguments
        
    Returns:
        GP2Vec model instance
    """
    # Default configurations for different model sizes
    size_configs = {
        "small": {
            "feature_dim": 256,
            "encoder_config": {
                "conv_layers": [(256, 10, 5), (256, 3, 2), (256, 3, 2), (512, 2, 2)],
            },
            "quantizer_config": {"num_vars": 2, "codebook_size": 320},
            "context_config": {"num_layers": 6, "num_heads": 8, "ff_dim": 1024},
        },
        "base": {
            "feature_dim": 768,
            "encoder_config": {
                "conv_layers": [(512, 10, 5), (512, 3, 2), (512, 3, 2), 
                               (512, 3, 2), (512, 2, 2), (512, 2, 2)],
            },
            "quantizer_config": {"num_vars": 2, "codebook_size": 8192},
            "context_config": {"num_layers": 12, "num_heads": 12, "ff_dim": 3072},
        },
        "large": {
            "feature_dim": 1024,
            "encoder_config": {
                "conv_layers": [(512, 10, 5), (512, 8, 4), (512, 4, 2), 
                               (512, 4, 2), (512, 2, 2)],
            },
            "quantizer_config": {"num_vars": 2, "codebook_size": 8192},
            "context_config": {"num_layers": 24, "num_heads": 16, "ff_dim": 4096},
        },
    }
    
    config = size_configs.get(model_size, size_configs["base"])
    config.update(kwargs)
    
    return GP2Vec(
        input_channels=input_channels,
        metadata_config=metadata_config,
        **config
    )