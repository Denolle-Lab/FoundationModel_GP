"""
Station Metadata Management

This module handles fetching, parsing, and caching of station metadata
from FDSN services using ObsPy, with support for creating embeddings.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory, Network, Station, Channel

logger = logging.getLogger(__name__)


class StationMetadataManager:
    """Manage station metadata fetching, parsing, and embedding creation."""
    
    def __init__(
        self,
        fdsn_client: str = "IRIS",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize metadata manager.
        
        Args:
            fdsn_client: FDSN client name ('IRIS', 'USGS', etc.)
            cache_dir: Directory for caching metadata (optional)
        """
        self.client = Client(fdsn_client)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_station_metadata(
        self,
        networks: Union[str, List[str]],
        stations: Optional[Union[str, List[str]]] = None,
        locations: Optional[Union[str, List[str]]] = None,
        channels: Optional[Union[str, List[str]]] = None,
        starttime: Optional[Union[str, UTCDateTime]] = None,
        endtime: Optional[Union[str, UTCDateTime]] = None,
        level: str = "response",
        include_restricted: bool = False,
    ) -> Inventory:
        """
        Fetch station metadata from FDSN service.
        
        Args:
            networks: Network code(s) (e.g., 'IU' or ['IU', 'TA'])
            stations: Station code(s) (optional)
            locations: Location code(s) (optional)
            channels: Channel code(s) (optional)
            starttime: Start time for metadata query
            endtime: End time for metadata query
            level: Metadata level ('network', 'station', 'channel', 'response')
            include_restricted: Whether to include restricted stations
            
        Returns:
            ObsPy Inventory object
        """
        # Convert inputs to proper format
        if isinstance(networks, str):
            networks = [networks]
        if isinstance(stations, str):
            stations = [stations] 
        if isinstance(locations, str):
            locations = [locations]
        if isinstance(channels, str):
            channels = [channels]
        
        if isinstance(starttime, str):
            starttime = UTCDateTime(starttime)
        if isinstance(endtime, str):
            endtime = UTCDateTime(endtime)
        
        network_str = ','.join(networks)
        station_str = ','.join(stations) if stations else '*'
        location_str = ','.join(locations) if locations else '*'
        channel_str = ','.join(channels) if channels else '*'
        
        logger.info(f"Fetching metadata: {network_str}.{station_str}.{location_str}.{channel_str}")
        
        try:
            inventory = self.client.get_stations(
                network=network_str,
                station=station_str,
                location=location_str,
                channel=channel_str,
                starttime=starttime,
                endtime=endtime,
                level=level,
                includerestricted=include_restricted,
            )
            
            logger.info(f"Fetched metadata for {len(inventory.networks)} networks")
            return inventory
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return Inventory(networks=[], source="")
    
    def inventory_to_dataframe(
        self, 
        inventory: Inventory,
        include_response: bool = True,
    ) -> pd.DataFrame:
        """
        Convert ObsPy Inventory to pandas DataFrame.
        
        Args:
            inventory: ObsPy Inventory object
            include_response: Whether to include response information
            
        Returns:
            DataFrame with station metadata
        """
        records = []
        
        for network in inventory.networks:
            for station in network.stations:
                for channel in station.channels:
                    record = self._extract_channel_metadata(
                        network, station, channel, include_response
                    )
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add derived features
        if len(df) > 0:
            df = self._add_derived_features(df)
        
        return df
    
    def _extract_channel_metadata(
        self, 
        network: Network, 
        station: Station, 
        channel: Channel,
        include_response: bool = True,
    ) -> Dict:
        """Extract metadata from a single channel."""
        record = {
            # Basic identifiers
            'network': network.code,
            'station': station.code,
            'location': channel.location_code or '',
            'channel': channel.code,
            
            # Geographic coordinates
            'latitude': station.latitude,
            'longitude': station.longitude,
            'elevation_m': station.elevation,
            
            # Channel-specific geometry
            'azimuth_deg': channel.azimuth if channel.azimuth is not None else 0.0,
            'dip_deg': channel.dip if channel.dip is not None else 0.0,
            'depth_m': channel.depth if channel.depth is not None else 0.0,
            
            # Timing
            'start_date': channel.start_date.isoformat() if channel.start_date else None,
            'end_date': channel.end_date.isoformat() if channel.end_date else None,
            'sampling_rate': channel.sample_rate,
            
            # Equipment
            'sensor_description': channel.sensor.description if channel.sensor else '',
            'datalogger_description': channel.data_logger.description if channel.data_logger else '',
            
            # Site information
            'site_name': station.site.name if station.site else '',
            'creation_date': station.creation_date.isoformat() if station.creation_date else None,
        }
        
        # Response information
        if include_response and channel.response:
            try:
                sensitivity = channel.response.instrument_sensitivity
                if sensitivity:
                    record.update({
                        'response_gain': sensitivity.value,
                        'response_frequency': sensitivity.frequency,
                        'input_units': sensitivity.input_units,
                        'output_units': sensitivity.output_units,
                    })
                else:
                    record.update({
                        'response_gain': None,
                        'response_frequency': None,
                        'input_units': '',
                        'output_units': '',
                    })
            except Exception as e:
                logger.debug(f"Could not extract response for {network.code}.{station.code}.{channel.code}: {e}")
                record.update({
                    'response_gain': None,
                    'response_frequency': None,
                    'input_units': '',
                    'output_units': '',
                })
        
        return record
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to metadata DataFrame."""
        df = df.copy()
        
        # Channel type (last character)
        df['channel_type'] = df['channel'].str[-1]
        
        # Instrument band (first character)
        df['instrument_band'] = df['channel'].str[0]
        
        # Instrument type (second character)  
        df['instrument_type'] = df['channel'].str[1]
        
        # Location tier
        df['location_tier'] = df['location'].apply(
            lambda x: 'primary' if x == '' or x == '00' else 'secondary'
        )
        
        # Distance from equator
        df['distance_from_equator'] = np.abs(df['latitude'])
        
        # Hemisphere
        df['hemisphere'] = df['latitude'].apply(
            lambda x: 'north' if x >= 0 else 'south'
        )
        
        # Elevation category
        df['elevation_category'] = pd.cut(
            df['elevation_m'],
            bins=[-np.inf, 0, 500, 1500, np.inf],
            labels=['below_sea_level', 'low', 'medium', 'high']
        )
        
        # Sampling rate category
        df['sampling_rate_category'] = pd.cut(
            df['sampling_rate'],
            bins=[0, 20, 50, 100, 250, np.inf],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def build_metadata_table(
        self,
        networks: Union[str, List[str]],
        starttime: Optional[str] = None,
        endtime: Optional[str] = None,
        cache_file: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Build complete metadata table with caching.
        
        Args:
            networks: Network code(s) to fetch
            starttime: Start time for metadata query
            endtime: End time for metadata query
            cache_file: Path to cache file (optional)
            force_refresh: Whether to force refresh from FDSN
            
        Returns:
            DataFrame with station metadata
        """
        # Check cache first
        if cache_file and not force_refresh:
            cache_path = Path(cache_file)
            if cache_path.exists():
                logger.info(f"Loading cached metadata from {cache_file}")
                return pd.read_parquet(cache_file)
        
        # Fetch from FDSN
        inventory = self.fetch_station_metadata(
            networks=networks,
            starttime=starttime,
            endtime=endtime,
            level="response"
        )
        
        df = self.inventory_to_dataframe(inventory)
        
        # Cache result
        if cache_file and len(df) > 0:
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached metadata to {cache_file}")
        
        return df
    
    def create_embeddings_config(self, df: pd.DataFrame) -> Dict:
        """
        Create embedding configuration from metadata DataFrame.
        
        Args:
            df: Metadata DataFrame
            
        Returns:
            Dictionary with embedding dimensions for categorical variables
        """
        config = {
            'categorical_features': {},
            'continuous_features': [
                'latitude', 'longitude', 'elevation_m', 
                'azimuth_deg', 'dip_deg', 'sampling_rate',
                'response_gain', 'distance_from_equator'
            ]
        }
        
        # Calculate embedding dimensions for categorical features
        categorical_cols = [
            'network', 'station', 'location', 'channel', 
            'channel_type', 'instrument_band', 'instrument_type',
            'location_tier', 'hemisphere', 'elevation_category',
            'sampling_rate_category', 'sensor_description', 
            'datalogger_description'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                n_unique = df[col].nunique()
                # Use embedding dimension based on number of unique values
                embed_dim = min(32, max(4, int(np.log2(n_unique) * 4)))
                config['categorical_features'][col] = {
                    'vocab_size': n_unique,
                    'embed_dim': embed_dim
                }
        
        return config
    
    def normalize_continuous_features(
        self, 
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize continuous features and return normalization stats.
        
        Args:
            df: Input DataFrame
            features: List of features to normalize (if None, uses default set)
            
        Returns:
            Tuple of (normalized_df, normalization_stats)
        """
        if features is None:
            features = [
                'latitude', 'longitude', 'elevation_m',
                'azimuth_deg', 'dip_deg', 'sampling_rate',
                'response_gain', 'distance_from_equator'
            ]
        
        df_norm = df.copy()
        stats = {}
        
        for feature in features:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    if std_val > 0:
                        df_norm[feature] = (df[feature] - mean_val) / std_val
                        stats[feature] = {'mean': mean_val, 'std': std_val}
                    else:
                        stats[feature] = {'mean': mean_val, 'std': 1.0}
        
        return df_norm, stats


def build_metadata_table(
    networks: Union[str, List[str]],
    starttime: Optional[str] = None, 
    endtime: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Convenience function to build metadata table."""
    manager = StationMetadataManager()
    return manager.build_metadata_table(
        networks=networks,
        starttime=starttime,
        endtime=endtime,
        **kwargs
    )


def load_metadata_cache(cache_file: str) -> pd.DataFrame:
    """Load cached metadata from Parquet file."""
    return pd.read_parquet(cache_file)


def get_station_info(
    network: str,
    station: str,
    starttime: Optional[str] = None,
    endtime: Optional[str] = None,
) -> Dict:
    """Get information for a specific station."""
    manager = StationMetadataManager()
    inventory = manager.fetch_station_metadata(
        networks=[network],
        stations=[station],
        starttime=starttime,
        endtime=endtime,
        level="response"
    )
    
    if inventory.networks and inventory.networks[0].stations:
        df = manager.inventory_to_dataframe(inventory)
        return df.to_dict('records')
    else:
        return {}


def extract_response_curve(
    channel: Channel,
    frequencies: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract instrument response curve for a channel.
    
    Args:
        channel: ObsPy Channel object with response
        frequencies: Frequency array (if None, uses default log-spaced)
        
    Returns:
        Tuple of (frequencies, log_amplitude_response)
    """
    if frequencies is None:
        # Default: 64 log-spaced frequencies from 0.01 to 50 Hz
        frequencies = np.logspace(-2, np.log10(50), 64)
    
    if not channel.response:
        return frequencies, np.zeros_like(frequencies)
    
    try:
        # Evaluate response
        response = channel.response.get_evalresp_response(
            t_samp=1.0,  # Dummy sampling rate
            nfft=len(frequencies) * 2,
            output='VEL'
        )[0]
        
        # Calculate log amplitude
        log_amplitude = np.log10(np.abs(response[:len(frequencies)]))
        
        return frequencies, log_amplitude
        
    except Exception as e:
        logger.warning(f"Could not extract response curve: {e}")
        return frequencies, np.zeros_like(frequencies)