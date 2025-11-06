#!/usr/bin/env python3
"""
Script to extract pre-trained Wav2Vec2 weights for GP2Vec initialization.

Usage:
    python scripts/extract_wav2vec_weights.py --model facebook/wav2vec2-base-960h --output weights/wav2vec2_base.pth
    python scripts/extract_wav2vec_weights.py --model facebook/wav2vec2-large-960h --output weights/wav2vec2_large.pth
"""

import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gp2vec.utils.wav2vec_transfer import Wav2Vec2WeightExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(
        description="Extract Wav2Vec2 weights for GP2Vec initialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", 
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face Wav2Vec2 model name or path"
    )
    
    parser.add_argument(
        "--output",
        default="weights/wav2vec2_weights.pth",
        help="Output path for extracted weights"
    )
    
    parser.add_argument(
        "--create-dir",
        action="store_true",
        help="Create output directory if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    if args.create_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Extracting weights from {args.model}")
    print(f"📁 Output path: {args.output}")
    
    try:
        # Extract weights
        extractor = Wav2Vec2WeightExtractor(args.model)
        extractor.save_weights(args.output)
        
        print(f"✅ Successfully extracted weights to {args.output}")
        print(f"📊 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Print summary of extracted components
        import torch
        weights = torch.load(args.output, map_location='cpu')
        
        print(f"\n📋 Extraction Summary:")
        print(f"   - Source model: {weights['config']['model_name']}")
        print(f"   - Hidden size: {weights['config']['hidden_size']}")
        print(f"   - Transformer layers: {weights['config']['num_layers']}")
        print(f"   - Attention heads: {weights['config']['num_heads']}")
        
        for component in ['feature_encoder', 'transformer', 'quantizer']:
            if component in weights and isinstance(weights[component], dict):
                print(f"   - {component.replace('_', ' ').title()}: {len(weights[component])} parameters")
        
    except Exception as e:
        print(f"❌ Error extracting weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()