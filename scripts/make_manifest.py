#!/usr/bin/env python3
"""
Build S3 manifest file for GP2Vec training data.

This script scans S3 for miniSEED files and creates a manifest
that can be used for efficient data loading during training.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gp2vec.data.s3_manifest import S3ManifestBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build S3 manifest for GP2Vec training data"
    )
    
    # S3 configuration
    parser.add_argument(
        "--bucket", 
        default="scedc-pds",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--prefix",
        default="continuous_waveforms/",
        help="S3 prefix to scan"
    )
    parser.add_argument(
        "--region",
        default="us-west-2", 
        help="S3 region"
    )
    
    # Filtering options
    parser.add_argument(
        "--networks",
        nargs="+",
        default=["CI", "AZ", "US", "TA"],
        help="Network codes to include"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-files-per-station",
        type=int,
        default=10000,
        help="Maximum files per station"
    )
    
    # Geographic filtering
    parser.add_argument(
        "--min-latitude",
        type=float,
        help="Minimum latitude"
    )
    parser.add_argument(
        "--max-latitude",
        type=float,
        help="Maximum latitude" 
    )
    parser.add_argument(
        "--min-longitude",
        type=float,
        help="Minimum longitude"
    )
    parser.add_argument(
        "--max-longitude",
        type=float,
        help="Maximum longitude"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output manifest file path"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json"],
        default="parquet",
        help="Output format"
    )
    
    # Performance options
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory for caching intermediate results"
    )
    
    # Metadata options
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include station metadata in manifest"
    )
    parser.add_argument(
        "--fdsn-client",
        default="IRIS",
        help="FDSN client for metadata"
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
    
    # Build filters dictionary
    filters = {}
    
    if args.networks:
        filters['networks'] = args.networks
    
    if args.start_date:
        filters['start_date'] = args.start_date
    
    if args.end_date:
        filters['end_date'] = args.end_date
    
    if args.max_files_per_station:
        filters['max_files_per_station'] = args.max_files_per_station
    
    # Geographic filters
    geo_filters = {}
    if args.min_latitude is not None:
        geo_filters['min_latitude'] = args.min_latitude
    if args.max_latitude is not None:
        geo_filters['max_latitude'] = args.max_latitude
    if args.min_longitude is not None:
        geo_filters['min_longitude'] = args.min_longitude  
    if args.max_longitude is not None:
        geo_filters['max_longitude'] = args.max_longitude
    
    if geo_filters:
        filters['geographic'] = geo_filters
    
    logger.info("Starting manifest building...")
    logger.info(f"S3 Location: s3://{args.bucket}/{args.prefix}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Filters: {filters}")
    
    try:
        # Initialize manifest builder
        builder = S3ManifestBuilder(
            bucket=args.bucket,
            prefix=args.prefix,
            region=args.region,
            cache_dir=args.cache_dir,
            num_workers=args.workers,
        )
        
        # Build manifest
        manifest_df = builder.build_manifest(
            output_path=args.output,
            filters=filters,
            include_metadata=args.include_metadata,
            fdsn_client=args.fdsn_client if args.include_metadata else None,
            batch_size=args.batch_size,
            output_format=args.format,
        )
        
        # Print summary
        logger.info("Manifest building completed!")
        logger.info(f"Total files: {len(manifest_df):,}")
        logger.info(f"Unique stations: {manifest_df['station'].nunique():,}")
        logger.info(f"Date range: {manifest_df['date'].min()} to {manifest_df['date'].max()}")
        
        if 'network' in manifest_df.columns:
            network_counts = manifest_df['network'].value_counts()
            logger.info("Files per network:")
            for network, count in network_counts.items():
                logger.info(f"  {network}: {count:,}")
        
        logger.info(f"Manifest saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to build manifest: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()