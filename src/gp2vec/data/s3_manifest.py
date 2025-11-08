"""
S3 Manifest Builder for EarthScope miniSEED Data

This module provides functionality to crawl EarthScope S3 buckets and build
a Parquet manifest of available miniSEED files with metadata. It also includes
a PyTorch dataset class for loading real seismic data from SCEDC S3 bucket.
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import torch
from obspy import read, UTCDateTime
from torch.utils.data import Dataset
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


class SCEDCSeismicDataset(Dataset):
    """
    PyTorch dataset for loading real seismic data from SCEDC S3 bucket.
    
    The SCEDC provides continuous seismic waveforms organized as:
    s3://scedc-pds/continuous_waveforms/{year}/{year}_{julian_day:03d}/
    
    Files are named following the pattern:
    {network}{station}{channel}{location}{year}{julian_day}.ms
    
    Examples:
        - CIGMR__LHN___2022002.ms (CI network, GMR station, LHN channel, no location)
        - CE13884HNZ10_2022002.ms (CE network, 13884 station, HNZ channel, location 10)
    
    This format follows the SCEDC S3 bucket structure and is compatible with
    the SCEDCS3DataStore class from gp2vec.data.datastore.
    """
    
    def __init__(
        self,
        start_date: str = "2023-01-01",
        num_days: int = 3,
        networks: List[str] = None,
        stations: List[str] = None,
        channels: List[str] = None,
        sample_length_sec: float = 30.0,
        sample_rate: float = 100.0,
        samples_per_day: int = 10,
        transform: Optional[Callable] = None,
        fs: Optional[s3fs.S3FileSystem] = None,
        bucket: str = "scedc-pds",
        base_path: str = "continuous_waveforms"
    ):
        """
        Initialize SCEDC dataset.
        
        Args:
            start_date: Start date in format "YYYY-MM-DD"
            num_days: Number of days to include
            networks: List of network codes (e.g., ["CI", "AZ"])
            stations: List of station codes (e.g., ["ADE", "ADO", "BAR"])
            channels: List of channel codes (e.g., ["BHE", "BHN", "BHZ"])
            sample_length_sec: Length of waveform samples in seconds
            sample_rate: Target sampling rate in Hz
            samples_per_day: Number of samples to generate per station-day
            transform: Optional transform to apply to samples
            fs: Optional S3 filesystem instance
            bucket: S3 bucket name
            base_path: Base path within bucket
        """
        super().__init__()
        
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.num_days = num_days
        self.networks = networks or ["CI"]
        self.stations = stations or ["ADE", "ADO", "BAR"]
        self.channels = channels or ["BHE", "BHN", "BHZ"]
        self.sample_length_sec = sample_length_sec
        self.sample_rate = sample_rate
        self.samples_per_day = samples_per_day
        self.transform = transform
        self.sample_length_pts = int(sample_length_sec * sample_rate)
        
        # Initialize S3 filesystem (anonymous access)
        self.fs = fs or s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.base_path = base_path
        
        # Build list of available data files
        self._build_file_list()
        
        # Station metadata cache
        self.station_metadata = {}
    
    def _parse_channel_from_filename(self, filename: str) -> Dict[str, any]:
        """
        Parse SCEDC filename format.
        
        Format: {network}{station}{channel}{location}{year}{julian_day}.ms
        
        Examples:
            - CIGMR__LHN___2022002.ms
              → network=CI, station=GMR, channel=LHN, location='', year=2022, day=002
            - CE13884HNZ10_2022002.ms
              → network=CE, station=13884, channel=HNZ, location='10', year=2022, day=002
        
        This follows the same pattern as SCEDCS3DataStore._parse_channel():
            - Network: characters 0-2
            - Station: characters 2-7 (with trailing underscores stripped)
            - Channel: characters 7-10
            - Location: characters 10-12 (with underscores stripped)
            - Year: last 7 characters (positions -7:-3)
            - Julian day: last 3 characters
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary with parsed components
        """
        # Remove .ms extension and path
        basename = filename.split('/')[-1].replace('.ms', '')
        
        # Extract components following SCEDCS3DataStore pattern
        network = basename[:2]
        station = basename[2:7].rstrip("_")
        channel = basename[7:10]
        location = basename[10:12].strip("_")
        year = int(basename[-7:-3])
        day = int(basename[-3:])
        
        return {
            'network': network,
            'station': station,
            'channel': channel,
            'location': location,
            'year': year,
            'julian_day': day
        }
        
    def _build_file_list(self):
        """Build list of available miniSEED files for the specified parameters."""
        self.file_list = []
        
        for day_offset in range(self.num_days):
            current_date = self.start_date + timedelta(days=day_offset)
            year = current_date.year
            julian_day = current_date.timetuple().tm_yday
            
            # Use SCEDCS3DataStore path pattern: {year}/{year}_{julian_day:03d}/
            day_path = f"{self.base_path}/{year}/{year}_{julian_day:03d}"
            
            try:
                # List all files for this day
                full_path = f"{self.bucket}/{day_path}"
                logger.info(f"Searching: {full_path}")
                files = self.fs.glob(f"{full_path}/*.ms")
                logger.info(f"Found {len(files)} total files on {current_date.strftime('%Y-%m-%d')}")
                
                # Group by station and location
                station_files = {}
                
                for file_path in files:
                    filename = file_path.split('/')[-1]
                    
                    try:
                        parsed = self._parse_channel_from_filename(filename)
                        
                        # Filter by network and station
                        if parsed['network'] not in self.networks:
                            continue
                        if parsed['station'] not in self.stations:
                            continue
                        
                        # Create station key
                        key = f"{parsed['network']}_{parsed['station']}_{parsed['location']}"
                        
                        if key not in station_files:
                            station_files[key] = {
                                'network': parsed['network'],
                                'station': parsed['station'],
                                'location': parsed['location'],
                                'channels': {}
                            }
                        
                        # Store file path for this channel
                        station_files[key]['channels'][parsed['channel']] = f"s3://{file_path}"
                        
                    except Exception as e:
                        logger.warning(f"Could not parse {filename}: {e}")
                        continue
                
                # Add complete 3-component sets
                for station_key, station_info in station_files.items():
                    available_channels = set(station_info['channels'].keys())
                    target_channels = set(self.channels)
                    
                    # Check if we have at least 2 components (can work with incomplete sets)
                    if len(available_channels.intersection(target_channels)) >= 2:
                        self.file_list.append({
                            'date': current_date,
                            'station_key': station_key,
                            'files': station_info['channels'],
                            'network': station_info['network'],
                            'station': station_info['station'],
                            'location': station_info['location']
                        })
                        logger.info(f"  Added {station_key} with channels: {list(station_info['channels'].keys())}")
                        
            except Exception as e:
                logger.warning(f"Could not access {day_path}: {e}")
                continue
        
        logger.info(f"Total: {len(self.file_list)} station-days with multi-component data")
        
    def _get_station_metadata(self, network: str, station: str) -> Dict[str, any]:
        """
        Get station metadata from SCEDC FDSN.
        
        Args:
            network: Network code
            station: Station code
            
        Returns:
            Dictionary with station metadata
        """
        key = f"{network}.{station}"
        
        if key not in self.station_metadata:
            try:
                # Use ObsPy to get station metadata
                from obspy.clients.fdsn import Client
                client = Client("SCEDC")
                
                # Get station info (try recent date)
                inventory = client.get_stations(
                    network=network, 
                    station=station,
                    starttime=UTCDateTime("2023-01-01"),
                    level="station"
                )
                
                if inventory:
                    net = inventory[0]
                    sta = net[0]
                    self.station_metadata[key] = {
                        'latitude': sta.latitude,
                        'longitude': sta.longitude,
                        'elevation': sta.elevation,
                        'creation_date': sta.creation_date,
                        'site_name': sta.site.name if sta.site else "Unknown"
                    }
                else:
                    # Fallback metadata
                    self.station_metadata[key] = {
                        'latitude': 34.0,  # Rough SoCal center
                        'longitude': -118.0,
                        'elevation': 100.0,
                        'creation_date': UTCDateTime("2000-01-01"),
                        'site_name': "Unknown"
                    }
                    
            except Exception as e:
                logger.warning(f"Could not get metadata for {key}: {e}")
                # Fallback metadata with some randomization
                self.station_metadata[key] = {
                    'latitude': 34.0 + np.random.uniform(-2, 2),
                    'longitude': -118.0 + np.random.uniform(-2, 2),
                    'elevation': 100.0 + np.random.uniform(-50, 500),
                    'creation_date': UTCDateTime("2000-01-01"),
                    'site_name': "Unknown"
                }
        
        return self.station_metadata[key]
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.file_list) * self.samples_per_day
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - waveform: 3-component waveform data (torch.Tensor)
                - metadata: Station metadata (torch.Tensor)
                - station_id: Station identifier (str)
                - date: Date string (str)
                - channels: List of channel names (List[str])
        """
        # Determine which station-day and which sample within that day
        file_idx = idx // self.samples_per_day
        sample_idx = idx % self.samples_per_day
        
        if file_idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range")
            
        file_info = self.file_list[file_idx]
        
        try:
            # Load waveform data
            waveforms = []
            channel_names = []
            
            for channel in self.channels:
                if channel in file_info['files']:
                    file_path = file_info['files'][channel]
                    
                    # Read miniSEED file from S3
                    with self.fs.open(file_path, 'rb') as f:
                        st = read(f, format='MSEED')
                    
                    if st and len(st) > 0:
                        tr = st[0]  # Take first trace
                        
                        # Basic preprocessing
                        tr.detrend('linear')
                        tr.filter('bandpass', freqmin=1.0, freqmax=45.0)
                        
                        # Resample if needed
                        if abs(tr.stats.sampling_rate - self.sample_rate) > 0.1:
                            tr.resample(self.sample_rate)
                        
                        waveforms.append(tr.data)
                        channel_names.append(channel)
                
            if len(waveforms) == 0:
                raise ValueError("No valid waveforms found")
            
            # Ensure all traces have the same length
            min_length = min(len(w) for w in waveforms)
            if min_length < self.sample_length_pts:
                # Pad with zeros if too short
                for i in range(len(waveforms)):
                    if len(waveforms[i]) < self.sample_length_pts:
                        waveforms[i] = np.pad(
                            waveforms[i], 
                            (0, self.sample_length_pts - len(waveforms[i])), 
                            'constant'
                        )
                min_length = self.sample_length_pts
            
            # Extract random window
            if min_length > self.sample_length_pts:
                max_start = min_length - self.sample_length_pts
                start_idx = np.random.randint(0, max_start + 1)
                waveforms = [w[start_idx:start_idx + self.sample_length_pts] for w in waveforms]
            
            # Stack to create 3-component array (pad with zeros if needed)
            while len(waveforms) < 3:
                waveforms.append(np.zeros(self.sample_length_pts))
                channel_names.append("PAD")
            
            # Take only first 3 components
            waveforms = waveforms[:3]
            channel_names = channel_names[:3]
            
            data = np.stack(waveforms, axis=0).astype(np.float32)
            
            # Normalize each component
            for i in range(data.shape[0]):
                std = np.std(data[i])
                if std > 0:
                    data[i] = data[i] / std
            
            # Get station metadata
            metadata = self._get_station_metadata(file_info['network'], file_info['station'])
            
            # Create metadata tensor (normalized coordinates)
            metadata_tensor = torch.tensor([
                (metadata['latitude'] - 34.0) / 5.0,  # Normalize around SoCal
                (metadata['longitude'] + 118.0) / 5.0,
                metadata['elevation'] / 1000.0,  # Convert to km
                float(metadata['creation_date'].timestamp) / 1e9  # Normalize timestamp
            ], dtype=torch.float32)
            
            sample = {
                'waveform': torch.from_numpy(data),
                'metadata': metadata_tensor,
                'station_id': f"{file_info['network']}.{file_info['station']}",
                'date': file_info['date'].strftime("%Y-%m-%d"),
                'channels': channel_names
            }
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
            
        except Exception as e:
            logger.error(f"Error loading data for index {idx}: {e}")
            # Return dummy data in case of error
            dummy_data = np.random.randn(3, self.sample_length_pts).astype(np.float32)
            dummy_metadata = torch.zeros(4, dtype=torch.float32)
            
            return {
                'waveform': torch.from_numpy(dummy_data),
                'metadata': dummy_metadata,
                'station_id': "DUMMY.DUMMY",
                'date': "2023-01-01",
                'channels': ["ERR", "ERR", "ERR"]
            }