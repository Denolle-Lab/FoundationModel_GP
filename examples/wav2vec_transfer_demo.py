#!/usr/bin/env python3
"""
Demonstration of Wav2Vec2 to GP2Vec weight transfer.

This script shows the complete workflow:
1. Extract weights from pre-trained Wav2Vec2
2. Create GP2Vec model
3. Transfer and adapt weights
4. Verify the transfer worked correctly
"""

import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_weight_transfer():
    """Demonstrate the complete Wav2Vec2 -> GP2Vec weight transfer process."""
    
    print("🚀 GP2Vec Weight Transfer Demonstration")
    print("=" * 50)
    
    # Step 1: Extract Wav2Vec2 weights
    print("\n📦 Step 1: Extract Wav2Vec2 Weights")
    
    try:
        from src.gp2vec.utils.wav2vec_transfer import Wav2Vec2WeightExtractor
        
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / "wav2vec2_demo.pth"
        
        if not weights_path.exists():
            print("   Extracting weights from facebook/wav2vec2-base-960h...")
            extractor = Wav2Vec2WeightExtractor("facebook/wav2vec2-base-960h")
            extractor.save_weights(str(weights_path))
            print(f"   ✅ Weights saved to {weights_path}")
        else:
            print(f"   ✅ Using existing weights from {weights_path}")
            
        # Load and inspect weights
        weights = torch.load(weights_path, map_location='cpu')
        print(f"\n   📊 Weight Statistics:")
        print(f"      - Source model: {weights['config']['model_name']}")
        print(f"      - Hidden size: {weights['config']['hidden_size']}")
        print(f"      - Transformer layers: {weights['config']['num_layers']}")
        print(f"      - File size: {weights_path.stat().st_size / (1024*1024):.1f} MB")
        
        for component in ['feature_encoder', 'transformer', 'quantizer']:
            if component in weights:
                count = len(weights[component]) if isinstance(weights[component], dict) else 0
                print(f"      - {component.replace('_', ' ').title()}: {count} tensors")
        
    except ImportError:
        print("   ❌ transformers library not available")
        print("   Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"   ❌ Error extracting weights: {e}")
        return False
    
    # Step 2: Create GP2Vec model
    print("\n🏗️  Step 2: Create GP2Vec Model")
    
    try:
        from src.gp2vec.models.gp2vec import create_gp2vec_model
        
        # Create small model for demo
        model = create_gp2vec_model(
            model_size="small",
            input_channels=3,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Created GP2Vec model with {total_params:,} parameters")
        
        # Show model architecture
        print(f"\n   📋 Model Architecture:")
        for name, module in model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            print(f"      - {name}: {param_count:,} parameters")
        
    except Exception as e:
        print(f"   ❌ Error creating GP2Vec model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Transfer weights
    print("\n🔄 Step 3: Transfer Wav2Vec2 Weights to GP2Vec")
    
    try:
        # Record initial weights for comparison
        initial_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Transfer weights
        stats = model.load_wav2vec_weights(weights_path, strict=False, verbose=True)
        
        print(f"\n   📊 Transfer Statistics:")
        print(f"      - Total model parameters: {stats['total_params']}")
        print(f"      - Updated parameters: {stats['updated_params']}")
        print(f"      - Update ratio: {stats['update_ratio']:.1%}")
        
        # Check which parameters were updated
        updated_count = 0
        unchanged_count = 0
        
        for name, param in model.named_parameters():
            if not torch.equal(initial_state[name], param):
                updated_count += 1
            else:
                unchanged_count += 1
        
        print(f"      - Parameters changed: {updated_count}")
        print(f"      - Parameters unchanged: {unchanged_count}")
        
    except Exception as e:
        print(f"   ❌ Error transferring weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Verify functionality
    print("\n✅ Step 4: Verify Model Functionality")
    
    try:
        # Create dummy input (3-component seismic data)
        batch_size = 4
        seq_length = 3000  # 30 seconds at 100 Hz
        dummy_input = torch.randn(batch_size, 3, seq_length)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model.forward(dummy_input)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   - Input shape: {dummy_input.shape}")
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key} shape: {value.shape}")
                else:
                    print(f"   - {key}: {value}")
        
        # Test feature extraction
        features = model.extract_features(dummy_input)
        print(f"   - Extracted features shape: {features.shape}")
        
    except Exception as e:
        print(f"   ❌ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Weight Transfer Demonstration Complete!")
    print("   ✅ Successfully transferred Wav2Vec2 weights to GP2Vec")
    print("   ✅ Model is ready for seismic data training")
    print(f"\n💡 Next Steps:")
    print(f"   - Use this initialized model for training on seismic data")
    print(f"   - Expect faster convergence compared to random initialization")
    print(f"   - Fine-tune the transferred representations for seismic tasks")
    
    return True

def compare_architectures():
    """Compare Wav2Vec2 and GP2Vec architectures."""
    
    print("\n🔍 Architecture Comparison: Wav2Vec2 vs GP2Vec")
    print("=" * 50)
    
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        from src.gp2vec.models.gp2vec import create_gp2vec_model
        
        # Load Wav2Vec2 config
        wav2vec_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Create models (don't load weights to save time)
        gp2vec_model = create_gp2vec_model("base", input_channels=3)
        
        print(f"\n📊 Model Comparison:")
        print(f"   {'Component':<20} {'Wav2Vec2':<15} {'GP2Vec':<15} {'Adaptation'}")
        print(f"   {'-' * 20} {'-' * 15} {'-' * 15} {'-' * 20}")
        
        # Input channels
        print(f"   {'Input Channels':<20} {'1 (audio)':<15} {'3 (seismic)':<15} {'Replicated'}")
        
        # Hidden dimension
        wav2vec_hidden = wav2vec_config.hidden_size
        gp2vec_hidden = gp2vec_model.feature_dim
        print(f"   {'Hidden Dimension':<20} {wav2vec_hidden:<15} {gp2vec_hidden:<15} {'Matched' if wav2vec_hidden == gp2vec_hidden else 'Adapted'}")
        
        # Transformer layers
        wav2vec_layers = wav2vec_config.num_hidden_layers
        gp2vec_layers = len(gp2vec_model.context_encoder.transformer.layers) if hasattr(gp2vec_model.context_encoder, 'transformer') else 'N/A'
        print(f"   {'Transformer Layers':<20} {wav2vec_layers:<15} {gp2vec_layers:<15} {'Matched' if wav2vec_layers == gp2vec_layers else 'Adapted'}")
        
        # Attention heads  
        wav2vec_heads = wav2vec_config.num_attention_heads
        print(f"   {'Attention Heads':<20} {wav2vec_heads:<15} {'8':<15} {'Adapted'}")
        
        # Parameters
        wav2vec_params = "95M"  # Approximate for base model
        gp2vec_params = f"{sum(p.numel() for p in gp2vec_model.parameters()) / 1e6:.1f}M"
        print(f"   {'Parameters':<20} {wav2vec_params:<15} {gp2vec_params:<15} {'Scaled'}")
        
        print(f"\n🔄 Adaptation Strategy:")
        print(f"   1. Feature Encoder: Adapt input from 1D audio to 3D seismic")
        print(f"   2. Vector Quantizer: Transfer codebook and projection layers")
        print(f"   3. Transformer: Transfer attention weights and layer norms")
        print(f"   4. Dimensions: Adapt layer sizes while preserving learned patterns")
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

if __name__ == "__main__":
    success = demonstrate_weight_transfer()
    
    if success:
        compare_architectures()
        print(f"\n🌟 Demonstration completed successfully!")
    else:
        print(f"\n❌ Demonstration failed. Check error messages above.")
        sys.exit(1)