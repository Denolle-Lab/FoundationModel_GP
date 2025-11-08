"""
Example: Loading Real Seismic Data from SCEDC S3

This example demonstrates how to use the SCEDCSeismicDataset to load
real continuous seismic waveforms from the SCEDC S3 bucket.

The SCEDCSeismicDataset is part of the s3_manifest module and provides
direct access to SCEDC data with proper filename parsing.
"""

import logging

import torch
from torch.utils.data import DataLoader

from gp2vec.data.s3_manifest import SCEDCSeismicDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate SCEDC dataset usage."""
    logger.info("SCEDC Real Data Example")
    
    # Create dataset
    logger.info("Creating SCEDC dataset...")
    dataset = SCEDCSeismicDataset(
        start_date="2023-01-01",
        num_days=2,  # Just 2 days for quick testing
        networks=["CI"],  # Southern California Seismic Network
        stations=["ADE", "ADO", "BAR"],  # Select a few stations
        channels=["BHE", "BHN", "BHZ"],  # 3-component broadband
        sample_length_sec=30.0,
        sample_rate=100.0,
        samples_per_day=5
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Create DataLoader
    logger.info("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Process a few batches
    logger.info("Processing batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just process first 3 batches
            break
            
        waveforms = batch['waveform']
        metadata = batch['metadata']
        station_ids = batch['station_id']
        dates = batch['date']
        channels = batch['channels']
        
        logger.info(f"\nBatch {i+1}:")
        logger.info(f"  Waveforms shape: {waveforms.shape}")
        logger.info(f"  Metadata shape: {metadata.shape}")
        logger.info(f"  Stations: {station_ids}")
        logger.info(f"  Dates: {dates}")
        logger.info(f"  Channels: {channels}")
        
        # Check waveform statistics
        logger.info(f"  Waveform mean: {waveforms.mean():.4f}")
        logger.info(f"  Waveform std: {waveforms.std():.4f}")
        logger.info(f"  Waveform min: {waveforms.min():.4f}")
        logger.info(f"  Waveform max: {waveforms.max():.4f}")
    
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
