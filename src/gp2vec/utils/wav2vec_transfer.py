"""
Utilities for transferring pre-trained Wav2Vec2 weights to GP2Vec seismic models.

This module provides functions to:
1. Load pre-trained Wav2Vec2 models from Hugging Face
2. Extract and adapt weights for seismic data (3-component, different sampling rates)
3. Initialize GP2Vec models with transferred weights
4. Handle dimension mismatches gracefully

Author: GP2Vec Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
import warnings

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Transformers library not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class Wav2Vec2WeightExtractor:
    """Extract and adapt Wav2Vec2 weights for GP2Vec seismic models."""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize the weight extractor.
        
        Args:
            model_name: Hugging Face model name/path
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")
            
        self.model_name = model_name
        self.wav2vec_model = None
        self.wav2vec_config = None
        
    def load_pretrained_model(self) -> None:
        """Load the pre-trained Wav2Vec2 model."""
        logger.info(f"Loading pre-trained Wav2Vec2 model: {self.model_name}")
        
        try:
            self.wav2vec_config = Wav2Vec2Config.from_pretrained(self.model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.wav2vec_model.eval()
            
            logger.info(f"✅ Loaded Wav2Vec2 model with {sum(p.numel() for p in self.wav2vec_model.parameters()):,} parameters")
            logger.info(f"   - Conv layers: {len(self.wav2vec_model.feature_extractor.conv_layers)}")
            logger.info(f"   - Hidden size: {self.wav2vec_config.hidden_size}")
            logger.info(f"   - Transformer layers: {self.wav2vec_config.num_hidden_layers}")
            logger.info(f"   - Attention heads: {self.wav2vec_config.num_attention_heads}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {e}")
            raise
            
    def extract_feature_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract CNN feature encoder weights from Wav2Vec2.
        
        Returns:
            Dictionary of layer weights adapted for seismic data
        """
        if self.wav2vec_model is None:
            self.load_pretrained_model()
            
        logger.info("🔧 Extracting feature encoder weights...")
        
        extracted_weights = {}
        feature_extractor = self.wav2vec_model.feature_extractor
        
        # Extract convolutional layers
        for i, conv_layer in enumerate(feature_extractor.conv_layers):
            layer_name = f"conv_{i}"
            
            # Get original conv1d weights: (out_channels, in_channels, kernel_size)
            conv_weight = conv_layer.conv.weight.data.clone()
            conv_bias = conv_layer.conv.bias.data.clone() if conv_layer.conv.bias is not None else None
            
            # Adapt for 3-component seismic input (only for first layer)
            if i == 0:
                # Original: (512, 1, 10) for audio
                # Target: (64, 3, 10) for seismic (E, N, Z components)
                original_out, original_in, kernel_size = conv_weight.shape
                
                logger.info(f"   - Layer {i}: Adapting input from {original_in} to 3 channels")
                logger.info(f"   - Layer {i}: Reducing output from {original_out} to 64 channels")
                
                # Replicate weights across 3 input channels and reduce output channels
                adapted_weight = conv_weight[:64].repeat(1, 3, 1) / 3.0  # Average to maintain magnitude
                adapted_bias = conv_bias[:64] if conv_bias is not None else None
                
                extracted_weights[f"{layer_name}.weight"] = adapted_weight
                if adapted_bias is not None:
                    extracted_weights[f"{layer_name}.bias"] = adapted_bias
            else:
                # For subsequent layers, we may need to adapt channel dimensions
                # This depends on the target architecture
                out_ch, in_ch, kernel_size = conv_weight.shape
                
                # Reduce channels to match typical seismic model sizes
                target_out = min(out_ch, 256)  # Cap at reasonable size
                target_in = min(in_ch, 256)
                
                if target_out < out_ch or target_in < in_ch:
                    logger.info(f"   - Layer {i}: Reducing channels from ({out_ch}, {in_ch}) to ({target_out}, {target_in})")
                    adapted_weight = conv_weight[:target_out, :target_in, :]
                    adapted_bias = conv_bias[:target_out] if conv_bias is not None else None
                else:
                    adapted_weight = conv_weight
                    adapted_bias = conv_bias
                
                extracted_weights[f"{layer_name}.weight"] = adapted_weight
                if adapted_bias is not None:
                    extracted_weights[f"{layer_name}.bias"] = adapted_bias
            
            # Extract normalization weights if present
            if hasattr(conv_layer, 'layer_norm') and conv_layer.layer_norm is not None:
                norm_weight = conv_layer.layer_norm.weight.data.clone()
                norm_bias = conv_layer.layer_norm.bias.data.clone()
                
                # Adapt normalization for reduced channels
                target_features = extracted_weights[f"{layer_name}.weight"].shape[0]
                if len(norm_weight) > target_features:
                    norm_weight = norm_weight[:target_features]
                    norm_bias = norm_bias[:target_features]
                
                extracted_weights[f"{layer_name}.norm.weight"] = norm_weight
                extracted_weights[f"{layer_name}.norm.bias"] = norm_bias
        
        logger.info(f"✅ Extracted {len(extracted_weights)} feature encoder weight tensors")
        return extracted_weights
    
    def extract_transformer_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract Transformer encoder weights from Wav2Vec2.
        
        Returns:
            Dictionary of transformer weights
        """
        if self.wav2vec_model is None:
            self.load_pretrained_model()
            
        logger.info("🔧 Extracting transformer weights...")
        
        extracted_weights = {}
        transformer = self.wav2vec_model.encoder
        
        # Position embeddings (if present)
        if hasattr(transformer, 'pos_conv_embed'):
            pos_embed = transformer.pos_conv_embed
            if hasattr(pos_embed, 'conv'):
                extracted_weights['pos_embed.weight'] = pos_embed.conv.weight.data.clone()
                if pos_embed.conv.bias is not None:
                    extracted_weights['pos_embed.bias'] = pos_embed.conv.bias.data.clone()
        
        # Layer norm
        if hasattr(transformer, 'layer_norm'):
            extracted_weights['layer_norm.weight'] = transformer.layer_norm.weight.data.clone()
            extracted_weights['layer_norm.bias'] = transformer.layer_norm.bias.data.clone()
        
        # Transformer layers
        for i, layer in enumerate(transformer.layers):
            layer_prefix = f"transformer.{i}"
            
            # Self-attention
            if hasattr(layer, 'attention'):
                attn = layer.attention
                
                # Query, Key, Value projections
                if hasattr(attn, 'q_proj'):
                    extracted_weights[f"{layer_prefix}.self_attn.q_proj.weight"] = attn.q_proj.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.self_attn.q_proj.bias"] = attn.q_proj.bias.data.clone()
                
                if hasattr(attn, 'k_proj'):
                    extracted_weights[f"{layer_prefix}.self_attn.k_proj.weight"] = attn.k_proj.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.self_attn.k_proj.bias"] = attn.k_proj.bias.data.clone()
                
                if hasattr(attn, 'v_proj'):
                    extracted_weights[f"{layer_prefix}.self_attn.v_proj.weight"] = attn.v_proj.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.self_attn.v_proj.bias"] = attn.v_proj.bias.data.clone()
                
                # Output projection
                if hasattr(attn, 'out_proj'):
                    extracted_weights[f"{layer_prefix}.self_attn.out_proj.weight"] = attn.out_proj.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.self_attn.out_proj.bias"] = attn.out_proj.bias.data.clone()
            
            # Feed-forward network
            if hasattr(layer, 'feed_forward'):
                ffn = layer.feed_forward
                
                if hasattr(ffn, 'intermediate_dense'):
                    extracted_weights[f"{layer_prefix}.ffn.intermediate.weight"] = ffn.intermediate_dense.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.ffn.intermediate.bias"] = ffn.intermediate_dense.bias.data.clone()
                
                if hasattr(ffn, 'output_dense'):
                    extracted_weights[f"{layer_prefix}.ffn.output.weight"] = ffn.output_dense.weight.data.clone()
                    extracted_weights[f"{layer_prefix}.ffn.output.bias"] = ffn.output_dense.bias.data.clone()
            
            # Layer norms
            if hasattr(layer, 'layer_norm'):
                extracted_weights[f"{layer_prefix}.layer_norm.weight"] = layer.layer_norm.weight.data.clone()
                extracted_weights[f"{layer_prefix}.layer_norm.bias"] = layer.layer_norm.bias.data.clone()
            
            if hasattr(layer, 'final_layer_norm'):
                extracted_weights[f"{layer_prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.data.clone()
                extracted_weights[f"{layer_prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.data.clone()
        
        logger.info(f"✅ Extracted {len(extracted_weights)} transformer weight tensors")
        return extracted_weights
    
    def extract_quantizer_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract vector quantizer weights from Wav2Vec2.
        
        Returns:
            Dictionary of quantizer weights
        """
        if self.wav2vec_model is None:
            self.load_pretrained_model()
            
        logger.info("🔧 Extracting quantizer weights...")
        
        extracted_weights = {}
        
        # Wav2Vec2 uses Gumbel vector quantization
        if hasattr(self.wav2vec_model, 'quantizer'):
            quantizer = self.wav2vec_model.quantizer
            
            # Weight projection (if present)
            if hasattr(quantizer, 'weight_proj') and quantizer.weight_proj is not None:
                extracted_weights['weight_proj.weight'] = quantizer.weight_proj.weight.data.clone()
                if quantizer.weight_proj.bias is not None:
                    extracted_weights['weight_proj.bias'] = quantizer.weight_proj.bias.data.clone()
            
            # Codevectors (codebook)
            if hasattr(quantizer, 'codevectors'):
                # Shape: (num_groups, entries_per_group, embed_dim)
                codevectors = quantizer.codevectors.data.clone()
                extracted_weights['codevectors'] = codevectors
                logger.info(f"   - Codebook shape: {codevectors.shape}")
        
        logger.info(f"✅ Extracted {len(extracted_weights)} quantizer weight tensors")
        return extracted_weights
    
    def save_weights(self, output_path: str) -> None:
        """
        Save all extracted weights to a file.
        
        Args:
            output_path: Path to save the weights
        """
        logger.info(f"💾 Saving extracted weights to {output_path}")
        
        all_weights = {
            'feature_encoder': self.extract_feature_encoder_weights(),
            'transformer': self.extract_transformer_weights(),
            'quantizer': self.extract_quantizer_weights(),
            'config': {
                'model_name': self.model_name,
                'hidden_size': self.wav2vec_config.hidden_size,
                'num_layers': self.wav2vec_config.num_hidden_layers,
                'num_heads': self.wav2vec_config.num_attention_heads,
                'intermediate_size': getattr(self.wav2vec_config, 'intermediate_size', 
                                           self.wav2vec_config.hidden_size * 4)
            }
        }
        
        torch.save(all_weights, output_path)
        logger.info(f"✅ Saved weights with {sum(len(v) for v in all_weights.values() if isinstance(v, dict))} tensors")


def load_wav2vec_weights(weights_path: str) -> Dict:
    """
    Load previously extracted Wav2Vec2 weights.
    
    Args:
        weights_path: Path to the saved weights
        
    Returns:
        Dictionary of extracted weights and config
    """
    logger.info(f"📂 Loading Wav2Vec2 weights from {weights_path}")
    
    try:
        weights = torch.load(weights_path, map_location='cpu')
        logger.info(f"✅ Loaded weights with keys: {list(weights.keys())}")
        return weights
    except Exception as e:
        logger.error(f"❌ Failed to load weights: {e}")
        raise


def initialize_gp2vec_from_wav2vec(gp2vec_model, wav2vec_weights: Dict, strict: bool = False) -> None:
    """
    Initialize a GP2Vec model with Wav2Vec2 weights.
    
    Args:
        gp2vec_model: GP2Vec model instance
        wav2vec_weights: Dictionary of extracted Wav2Vec2 weights
        strict: Whether to require exact parameter matching
    """
    logger.info("🔄 Initializing GP2Vec model with Wav2Vec2 weights...")
    
    # Get model state dict
    model_state = gp2vec_model.state_dict()
    updated_params = []
    
    # Initialize feature encoder
    if 'feature_encoder' in wav2vec_weights:
        feature_weights = wav2vec_weights['feature_encoder']
        
        for param_name, param_tensor in feature_weights.items():
            # Try to match parameter names (may need adaptation)
            gp2vec_param_name = f"feature_encoder.{param_name}"
            
            if gp2vec_param_name in model_state:
                target_shape = model_state[gp2vec_param_name].shape
                source_shape = param_tensor.shape
                
                if target_shape == source_shape:
                    model_state[gp2vec_param_name] = param_tensor
                    updated_params.append(gp2vec_param_name)
                    logger.debug(f"   ✓ {gp2vec_param_name}: {source_shape}")
                elif not strict:
                    # Try to adapt dimensions
                    adapted_tensor = adapt_tensor_dimensions(param_tensor, target_shape)
                    if adapted_tensor is not None:
                        model_state[gp2vec_param_name] = adapted_tensor
                        updated_params.append(gp2vec_param_name)
                        logger.debug(f"   ~ {gp2vec_param_name}: {source_shape} -> {target_shape}")
                    else:
                        logger.warning(f"   ❌ Could not adapt {gp2vec_param_name}: {source_shape} vs {target_shape}")
                else:
                    logger.warning(f"   ❌ Shape mismatch for {gp2vec_param_name}: {source_shape} vs {target_shape}")
    
    # Initialize transformer (similar process)
    if 'transformer' in wav2vec_weights:
        transformer_weights = wav2vec_weights['transformer']
        
        for param_name, param_tensor in transformer_weights.items():
            # Map parameter names to GP2Vec convention
            gp2vec_param_name = f"context_encoder.{param_name}"
            
            if gp2vec_param_name in model_state:
                target_shape = model_state[gp2vec_param_name].shape
                source_shape = param_tensor.shape
                
                if target_shape == source_shape:
                    model_state[gp2vec_param_name] = param_tensor
                    updated_params.append(gp2vec_param_name)
                elif not strict:
                    adapted_tensor = adapt_tensor_dimensions(param_tensor, target_shape)
                    if adapted_tensor is not None:
                        model_state[gp2vec_param_name] = adapted_tensor
                        updated_params.append(gp2vec_param_name)
    
    # Initialize quantizer
    if 'quantizer' in wav2vec_weights:
        quantizer_weights = wav2vec_weights['quantizer']
        
        for param_name, param_tensor in quantizer_weights.items():
            gp2vec_param_name = f"quantizer.{param_name}"
            
            if gp2vec_param_name in model_state:
                target_shape = model_state[gp2vec_param_name].shape
                if target_shape == param_tensor.shape:
                    model_state[gp2vec_param_name] = param_tensor
                    updated_params.append(gp2vec_param_name)
    
    # Load the updated state dict
    gp2vec_model.load_state_dict(model_state, strict=False)
    
    logger.info(f"✅ Initialized {len(updated_params)} parameters from Wav2Vec2")
    logger.info(f"   - Updated parameters: {len(updated_params)}/{len(model_state)}")
    
    return updated_params


def adapt_tensor_dimensions(source_tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    """
    Adapt tensor dimensions to match target shape.
    
    Args:
        source_tensor: Source tensor to adapt
        target_shape: Target shape
        
    Returns:
        Adapted tensor or None if not possible
    """
    source_shape = source_tensor.shape
    
    if len(source_shape) != len(target_shape):
        return None
    
    # Simple dimension reduction/expansion
    adapted = source_tensor
    
    for dim, (source_size, target_size) in enumerate(zip(source_shape, target_shape)):
        if source_size == target_size:
            continue
        elif source_size > target_size:
            # Truncate
            indices = torch.arange(target_size)
            adapted = torch.index_select(adapted, dim, indices)
        elif source_size < target_size:
            # Pad with zeros or repeat
            pad_size = target_size - source_size
            if dim == 0:  # Usually output dimension
                padding = torch.zeros(pad_size, *adapted.shape[1:], dtype=adapted.dtype, device=adapted.device)
                adapted = torch.cat([adapted, padding], dim=0)
            else:
                # For other dimensions, we might repeat or pad differently
                # This is a simple strategy - might need refinement
                repeat_factor = target_size // source_size
                remainder = target_size % source_size
                
                if repeat_factor > 1:
                    repeated = adapted.repeat_interleave(repeat_factor, dim=dim)
                    if remainder > 0:
                        extra = torch.narrow(adapted, dim, 0, remainder)
                        repeated = torch.cat([repeated, extra], dim=dim)
                    adapted = repeated
                else:
                    # Just pad with zeros
                    pad_dims = [0] * (2 * len(adapted.shape))
                    pad_dims[2 * (len(adapted.shape) - 1 - dim) + 1] = pad_size
                    adapted = torch.nn.functional.pad(adapted, pad_dims)
        
    return adapted if adapted.shape == target_shape else None


# Example usage functions
def extract_wav2vec_weights_cli():
    """Command line interface for extracting Wav2Vec2 weights."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 weights for GP2Vec")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h", 
                       help="Hugging Face model name")
    parser.add_argument("--output", default="wav2vec_weights.pth",
                       help="Output file for weights")
    
    args = parser.parse_args()
    
    extractor = Wav2Vec2WeightExtractor(args.model)
    extractor.save_weights(args.output)
    
    print(f"✅ Extracted weights saved to {args.output}")


if __name__ == "__main__":
    extract_wav2vec_weights_cli()