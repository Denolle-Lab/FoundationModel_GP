"""
S3 Manifest Builder for EarthScope miniSEED Data

This module provides functionality to crawl EarthScope S3 buckets and build
a Parquet manifest of available miniSEED files with metadata.
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import s3fs
from obspy import read, UTCDateTime
from tqdm import tqdm

from ..utils.io import S3Client

logger = logging.getLogger(__name__)


class S3ManifestBuilder:
    """Build a manifest of miniSEED files available on EarthScope S3."""
    
    def __init__(
        self,
        bucket: str = "earthscope-data",
        prefix: str = "miniseed",
        fs: Optional[s3fs.S3FileSystem] = None,
        max_workers: int = 8,
    ):
        """
        Initialize the manifest builder.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix/path to miniSEED data
            fs: S3 filesystem instance (if None, will create one)
            max_workers: Number of parallel workers for metadata extraction
        """
        self.bucket = bucket
        self.prefix = prefix
        self.fs = fs or s3fs.S3FileSystem(anon=True)
        self.max_workers = max_workers
        self.s3_client = S3Client()
    
    def list_mseed_files(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """
        List miniSEED files in S3 bucket with optional filtering.
        
        Args:
            network: Network code filter (e.g., 'IU')
            station: Station code filter (e.g., 'ANMO')
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            List of S3 object keys
        """
        search_prefix = f"{self.bucket}/{self.prefix}"
        
        if network:
            search_prefix += f"/{network}"
        if station:
            search_prefix += f"/{station}"
            
        logger.info(f"Searching for miniSEED files in {search_prefix}")
        
        try:
            all_keys = self.fs.glob(f"{search_prefix}/**/*.mseed")
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
        
        # Filter by date if specified
        if start_date or end_date:
            all_keys = self._filter_by_date(all_keys, start_date, end_date)
        
        logger.info(f"Found {len(all_keys)} miniSEED files")
        return all_keys
    
    def _filter_by_date(
        self, 
        keys: List[str], 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> List[str]:
        """Filter file keys by date range."""
        filtered_keys = []
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        for key in keys:
            # Extract date from path (assumes format: .../YYYY/DOY/...)
            try:
                parts = key.split('/')
                year_part = None
                doy_part = None
                
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4:  # Year
                        year_part = part
                        if i + 1 < len(parts) and parts[i + 1].isdigit():
                            doy_part = parts[i + 1]
                        break
                
                if year_part and doy_part:
                    file_date = datetime.strptime(f"{year_part}{doy_part:0>3}", "%Y%j")
                    
                    if start_dt and file_date < start_dt:
                        continue
                    if end_dt and file_date > end_dt:
                        continue
                        
                filtered_keys.append(key)
                
            except (ValueError, IndexError):
                # If date parsing fails, include the file
                filtered_keys.append(key)
                
        return filtered_keys
    
    def _extract_file_metadata(self, key: str) -> Optional[Dict]:
        """Extract metadata from a single miniSEED file."""
        try:
            # Get file info
            info = self.fs.info(key)
            size = info.get('size', 0)
            
            # Parse file path components
            path_parts = key.replace(f"{self.bucket}/{self.prefix}/", "").split('/')
            
            metadata = {
                'key': key,
                'size_bytes': size,
                'network': None,
                'station': None,
                'location': None,
                'channel': None,
                'year': None,
                'doy': None,
                'starttime': None,
                'endtime': None,
                'sampling_rate': None,
                'npts': None,
            }
            
            # Extract network/station from path
            if len(path_parts) >= 2:
                metadata['network'] = path_parts[0]
                metadata['station'] = path_parts[1]
            
            # Extract year/DOY if present
            for part in path_parts:
                if part.isdigit() and len(part) == 4:
                    metadata['year'] = int(part)
                elif part.isdigit() and len(part) == 3:
                    metadata['doy'] = int(part)
            
            # Try to read file header for timing/sampling info
            # Note: Only read header to avoid downloading entire file
            try:
                with self.fs.open(key, 'rb') as f:
                    # Read only first few KB for header
                    header_data = f.read(8192)
                    
                if header_data:
                    stream = read(io.BytesIO(header_data), headonly=True)
                    if stream:
                        tr = stream[0]
                        metadata.update({
                            'location': tr.stats.location,
                            'channel': tr.stats.channel,
                            'starttime': tr.stats.starttime.datetime.isoformat(),
                            'endtime': tr.stats.endtime.datetime.isoformat(), 
                            'sampling_rate': tr.stats.sampling_rate,
                            'npts': tr.stats.npts,
                        })
            except Exception as e:
                logger.debug(f"Could not read header for {key}: {e}")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {key}: {e}")
            return None
    
    def build_manifest(
        self,
        output_path: str,
        network: Optional[str] = None,
        station: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> str:
        """
        Build and save a manifest of miniSEED files.
        
        Args:
            output_path: Path to save the Parquet manifest
            network: Network code filter
            station: Station code filter  
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            Path to the saved manifest file
        """
        # List files
        keys = self.list_mseed_files(network, station, start_date, end_date)
        
        if max_files:
            keys = keys[:max_files]
        
        if not keys:
            logger.warning("No files found matching criteria")
            return output_path
        
        # Extract metadata in parallel
        logger.info(f"Extracting metadata from {len(keys)} files...")
        metadata_list = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self._extract_file_metadata, key): key 
                for key in keys
            }
            
            for future in tqdm(as_completed(future_to_key), total=len(keys)):
                metadata = future.result()
                if metadata:
                    metadata_list.append(metadata)
        
        # Create DataFrame and save
        df = pd.DataFrame(metadata_list)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved manifest with {len(df)} files to {output_path}")
        
        # Print summary
        logger.info(f"Manifest summary:")
        logger.info(f"  Total files: {len(df)}")
        logger.info(f"  Total size: {df['size_bytes'].sum() / 1e9:.1f} GB")
        logger.info(f"  Networks: {df['network'].nunique()}")
        logger.info(f"  Stations: {df['station'].nunique()}")
        if 'starttime' in df.columns:
            logger.info(f"  Date range: {df['starttime'].min()} to {df['starttime'].max()}")
        
        return output_path


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load a manifest from Parquet file."""
    return pd.read_parquet(manifest_path)


def filter_manifest(
    manifest: pd.DataFrame,
    networks: Optional[List[str]] = None,
    stations: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    min_duration_hours: Optional[float] = None,
    sampling_rates: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Filter a manifest DataFrame based on various criteria.
    
    Args:
        manifest: Input manifest DataFrame
        networks: List of network codes to include
        stations: List of station codes to include  
        channels: List of channel codes to include
        min_duration_hours: Minimum duration in hours
        sampling_rates: List of acceptable sampling rates
        
    Returns:
        Filtered DataFrame
    """
    df = manifest.copy()
    
    if networks:
        df = df[df['network'].isin(networks)]
    
    if stations:
        df = df[df['station'].isin(stations)]
        
    if channels:
        df = df[df['channel'].isin(channels)]
    
    if sampling_rates:
        df = df[df['sampling_rate'].isin(sampling_rates)]
    
    if min_duration_hours:
        if 'starttime' in df.columns and 'endtime' in df.columns:
            df['duration_hours'] = (
                pd.to_datetime(df['endtime']) - pd.to_datetime(df['starttime'])
            ).dt.total_seconds() / 3600
            df = df[df['duration_hours'] >= min_duration_hours]
    
    return df