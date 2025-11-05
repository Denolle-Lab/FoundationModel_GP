"""
GP2Vec Data Pipeline Example

This example demonstrates how to use the GP2Vec data pipeline
to load and process seismic data from S3.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate GP2Vec data pipeline usage."""
    logger.info("GP2Vec Data Pipeline Example")
    
    # Import GP2Vec modules
    from gp2vec.data.s3_manifest import S3ManifestBuilder
    from gp2vec.data.metadata import StationMetadataManager  
    from gp2vec.data.datapipes import SeismicDataPipeline
    
    # Configuration
    bucket = "scedc-pds"
    prefix = "continuous_waveforms/"
    cache_dir = "./cache"
    
    # 1. Build manifest (or load existing one)
    logger.info("Step 1: Building/loading manifest")
    
    manifest_path = Path(cache_dir) / "manifest_example.parquet"
    
    if not manifest_path.exists():
        logger.info("Building new manifest...")
        builder = S3ManifestBuilder(bucket=bucket, prefix=prefix)
        
        # Build manifest with filters for smaller dataset
        filters = {
            'networks': ['CI'],  # Just CI network for example
            'start_date': '2023-01-01',
            'end_date': '2023-01-07',  # Just one week
            'max_files_per_station': 10,  # Very limited
        }
        
        manifest_df = builder.build_manifest(
            output_path=str(manifest_path),
            filters=filters,
            output_format='parquet'
        )
        logger.info(f"Built manifest with {len(manifest_df)} files")
    else:
        logger.info(f"Loading existing manifest from {manifest_path}")
        import pandas as pd
        manifest_df = pd.read_parquet(manifest_path)
        logger.info(f"Loaded manifest with {len(manifest_df)} files")
    
    # 2. Set up metadata manager
    logger.info("Step 2: Setting up metadata manager")
    
    metadata_manager = StationMetadataManager(
        client_name="IRIS",
        cache_dir=str(Path(cache_dir) / "metadata")
    )
    
    # 3. Create data pipeline
    logger.info("Step 3: Creating data pipeline")
    
    pipeline = SeismicDataPipeline(
        manifest_path=str(manifest_path),
        metadata_manager=metadata_manager,
        target_sampling_rate=100.0,
        window_length=30.0,
        cache_dir=cache_dir
    )
    
    # 4. Create WebDataset
    logger.info("Step 4: Creating WebDataset")
    
    dataset = pipeline.create_webdataset(
        shard_size=100,  # Small shards for example
        output_pattern=str(Path(cache_dir) / "shards" / "shard_{000000..000010}.tar.gz")
    )
    
    # 5. Create DataLoader
    logger.info("Step 5: Creating DataLoader")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True
    )
    
    # 6. Iterate through data
    logger.info("Step 6: Processing batches")
    
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Just show first 5 batches
            break
        
        waveforms = batch['waveform']  # (B, C, T)
        metadata = batch.get('metadata', {})
        
        logger.info(f"Batch {i}:")
        logger.info(f"  Waveforms shape: {waveforms.shape}")
        logger.info(f"  Waveforms range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
        
        if metadata:
            logger.info(f"  Metadata keys: {list(metadata.keys())}")
        
        # Example processing
        # Apply some simple preprocessing
        waveforms_normalized = torch.nn.functional.normalize(waveforms, dim=-1)
        
        logger.info(f"  Normalized range: [{waveforms_normalized.min():.3f}, {waveforms_normalized.max():.3f}]")
    
    logger.info("Data pipeline example completed successfully!")


if __name__ == "__main__":
    main()