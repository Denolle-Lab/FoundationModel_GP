"""
GP2Vec Model Usage Example

This example shows how to use a pretrained GP2Vec model
for feature extraction and downstream tasks.
"""

import logging
from pathlib import Path

import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate GP2Vec model usage."""
    logger.info("GP2Vec Model Usage Example")
    
    # Import GP2Vec modules
    from gp2vec.models.gp2vec import GP2Vec
    from gp2vec.train.evaluate_downstream import DownstreamEvaluator, load_pretrained_model
    
    # 1. Create a model (or load pretrained)
    logger.info("Step 1: Creating/loading model")
    
    # Option A: Create new model with default config
    model = GP2Vec()
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Option B: Load pretrained model (uncomment if you have a checkpoint)
    # checkpoint_path = "path/to/your/checkpoint.ckpt" 
    # model = load_pretrained_model(checkpoint_path)
    
    # 2. Create some example data
    logger.info("Step 2: Creating example data")
    
    batch_size = 4
    num_channels = 3  # Z, N, E components
    sequence_length = 3000  # 30 seconds at 100 Hz
    
    # Random waveform data (in practice, this would come from your data pipeline)
    waveforms = torch.randn(batch_size, num_channels, sequence_length)
    
    # Example metadata (coordinates, instrument info, etc.)
    metadata = {
        'latitude': torch.tensor([34.05, 34.10, 34.15, 34.20]),
        'longitude': torch.tensor([-118.25, -118.30, -118.35, -118.40]),
        'elevation': torch.tensor([100.0, 150.0, 200.0, 250.0]),
        'sampling_rate': torch.tensor([100.0, 100.0, 100.0, 100.0]),
    }
    
    logger.info(f"Example waveforms shape: {waveforms.shape}")
    logger.info(f"Metadata keys: {list(metadata.keys())}")
    
    # 3. Extract features using the model
    logger.info("Step 3: Extracting features")
    
    model.eval()
    with torch.no_grad():
        # Extract features (representations)
        features = model.encode(waveforms, metadata)
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
        
        # You can also get intermediate representations
        cnn_features = model.feature_encoder(waveforms)
        logger.info(f"CNN features shape: {cnn_features.shape}")
        
        # Quantized features
        quantized_features, vq_loss, perplexity = model.quantizer(cnn_features)
        logger.info(f"Quantized features shape: {quantized_features.shape}")
        logger.info(f"Codebook perplexity: {perplexity:.2f}")
    
    # 4. Demonstrate training forward pass (with loss)
    logger.info("Step 4: Training forward pass")
    
    model.train()
    
    # Forward pass for training (includes masking and contrastive loss)
    output = model(waveforms, metadata)
    
    logger.info(f"Training output keys: {list(output.keys())}")
    logger.info(f"Contrastive loss: {output['contrastive_loss']:.4f}")
    logger.info(f"VQ loss: {output['vq_loss']:.4f}")
    logger.info(f"Total loss: {output['loss']:.4f}")
    
    # 5. Demonstrate downstream evaluation setup
    logger.info("Step 5: Downstream evaluation setup")
    
    # Create evaluator
    evaluator = DownstreamEvaluator(model)
    
    # Create dummy downstream task data
    n_samples = 100
    downstream_waveforms = torch.randn(n_samples, num_channels, sequence_length)
    
    # Binary classification labels (e.g., phase picking: 0=noise, 1=event)
    phase_labels = torch.randint(0, 2, (n_samples,))
    
    # Create dummy metadata for each sample
    dummy_metadata = [{
        'latitude': np.random.uniform(32, 42),
        'longitude': np.random.uniform(-125, -114),
        'elevation': np.random.uniform(0, 3000),
        'sampling_rate': 100.0,
    } for _ in range(n_samples)]
    
    logger.info(f"Downstream data: {n_samples} samples")
    logger.info(f"Phase labels distribution: {phase_labels.bincount()}")
    
    # Evaluate on phase picking task
    try:
        results = evaluator.evaluate_phase_picking(
            downstream_waveforms, 
            phase_labels, 
            dummy_metadata,
            test_size=0.3
        )
        
        logger.info("Phase picking evaluation results:")
        for method, metrics in results.items():
            logger.info(f"  {method}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
    
    except Exception as e:
        logger.warning(f"Downstream evaluation failed (expected with dummy data): {e}")
    
    # 6. Show how to save/load model state
    logger.info("Step 6: Model serialization")
    
    # Save model
    output_path = Path("./example_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),  # If you implement this method
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")
    
    # Load model
    checkpoint = torch.load(output_path, map_location='cpu')
    new_model = GP2Vec()  # Create with same config
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully")
    
    # Verify they're the same
    with torch.no_grad():
        original_output = model.encode(waveforms[:1], {k: v[:1] for k, v in metadata.items()})
        loaded_output = new_model.encode(waveforms[:1], {k: v[:1] for k, v in metadata.items()})
        
        diff = torch.abs(original_output - loaded_output).max()
        logger.info(f"Max difference between original and loaded model: {diff:.6f}")
    
    # Clean up
    output_path.unlink()
    
    logger.info("Model usage example completed successfully!")


if __name__ == "__main__":
    main()