#!/usr/bin/env python3
"""
Fetch and cache station metadata for GP2Vec training.

This script downloads station metadata from FDSN web services
and caches it locally for use during training.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gp2vec.data.metadata import StationMetadataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch station metadata for GP2Vec"
    )
    
    # FDSN configuration
    parser.add_argument(
        "--client",
        default="IRIS",
        help="FDSN client (IRIS, SCEDC, NCEDC, etc.)"
    )
    parser.add_argument(
        "--networks", 
        nargs="+",
        default=["CI", "AZ", "US", "TA"],
        help="Network codes to fetch"
    )
    
    # Geographic filtering
    parser.add_argument(
        "--min-latitude",
        type=float,
        default=32.0,
        help="Minimum latitude"
    )
    parser.add_argument(
        "--max-latitude", 
        type=float,
        default=42.0,
        help="Maximum latitude"
    )
    parser.add_argument(
        "--min-longitude",
        type=float, 
        default=-125.0,
        help="Minimum longitude"
    )
    parser.add_argument(
        "--max-longitude",
        type=float,
        default=-114.0,
        help="Maximum longitude"
    )
    
    # Temporal filtering
    parser.add_argument(
        "--start-time",
        type=str,
        help="Start time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end-time",
        type=str,
        help="End time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    
    # Channel filtering
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Channel codes to include (e.g., BHZ, HHZ)"
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        help="Location codes to include"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: cache directory)"
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache/metadata",
        help="Cache directory for metadata"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json", "pickle"],
        default="parquet",
        help="Output format"
    )
    
    # Processing options
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of cached metadata"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing stations"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries"
    )
    
    # Feature extraction
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract derived features from metadata"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=[
            "coordinates", "instrument", "site_conditions", 
            "temporal", "geographic_region", "elevation_band"
        ],
        default=["coordinates", "instrument", "temporal"],
        help="Features to extract"
    )
    
    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate metadata after fetching"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true", 
        help="Verbose logging"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Build query parameters
    query_params = {
        'minlatitude': args.min_latitude,
        'maxlatitude': args.max_latitude,  
        'minlongitude': args.min_longitude,
        'maxlongitude': args.max_longitude,
    }
    
    if args.start_time:
        query_params['starttime'] = args.start_time
    
    if args.end_time:
        query_params['endtime'] = args.end_time
    
    if args.channels:
        query_params['channel'] = ','.join(args.channels)
    
    if args.locations:
        query_params['location'] = ','.join(args.locations)
    
    logger.info("Starting metadata fetch...")
    logger.info(f"FDSN Client: {args.client}")
    logger.info(f"Networks: {args.networks}")
    logger.info(f"Geographic bounds: {args.min_latitude}°N to {args.max_latitude}°N, {args.min_longitude}°E to {args.max_longitude}°E")
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        # Initialize metadata manager
        manager = StationMetadataManager(
            client_name=args.client,
            cache_dir=str(cache_dir),
            timeout=args.timeout,
            max_retries=args.max_retries
        )
        
        # Fetch metadata for each network
        all_metadata = []
        
        for network in args.networks:
            logger.info(f"Fetching metadata for network: {network}")
            
            network_params = query_params.copy()
            network_params['network'] = network
            
            try:
                metadata = manager.get_station_metadata(
                    **network_params,
                    force_refresh=args.force_refresh
                )
                
                if metadata is not None and not metadata.empty:
                    all_metadata.append(metadata)
                    logger.info(f"  Found {len(metadata)} stations")
                else:
                    logger.warning(f"  No stations found for {network}")
                    
            except Exception as e:
                logger.error(f"  Failed to fetch {network}: {e}")
                if args.verbose:
                    logger.exception("Full traceback:")
                continue
        
        if not all_metadata:
            logger.error("No metadata was successfully fetched")
            sys.exit(1)
        
        # Combine all metadata
        import pandas as pd
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(combined_metadata)
        combined_metadata = combined_metadata.drop_duplicates(
            subset=['network', 'station', 'location', 'channel']
        )
        final_count = len(combined_metadata)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate entries")
        
        # Extract features if requested
        if args.extract_features:
            logger.info("Extracting metadata features...")
            feature_metadata = manager.extract_metadata_features(
                combined_metadata,
                features=args.features
            )
            combined_metadata = feature_metadata
        
        # Validate if requested
        if args.validate:
            logger.info("Validating metadata...")
            is_valid = manager.validate_metadata(combined_metadata)
            if not is_valid:
                logger.warning("Metadata validation found issues")
        
        # Save metadata
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = cache_dir / f"stations_{args.client.lower()}.{args.format}"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in requested format
        if args.format == "parquet":
            combined_metadata.to_parquet(output_path, index=False)
        elif args.format == "csv":
            combined_metadata.to_csv(output_path, index=False)
        elif args.format == "json":
            combined_metadata.to_json(output_path, orient='records', indent=2)
        elif args.format == "pickle":
            combined_metadata.to_pickle(output_path)
        
        # Print summary
        logger.info("Metadata fetch completed!")
        logger.info(f"Total stations: {final_count:,}")
        logger.info(f"Unique networks: {combined_metadata['network'].nunique()}")
        logger.info(f"Unique stations: {combined_metadata['station'].nunique()}")
        
        if 'network' in combined_metadata.columns:
            network_counts = combined_metadata['network'].value_counts()
            logger.info("Stations per network:")
            for network, count in network_counts.items():
                logger.info(f"  {network}: {count:,}")
        
        logger.info(f"Metadata saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()