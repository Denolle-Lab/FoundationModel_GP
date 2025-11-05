"""
Geographic Utilities for Seismic Station Processing

This module provides utilities for geographic coordinate processing,
region bucketing, distance calculations, and spatial feature engineering
for seismic station metadata.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, lon1: float, 
    lat2: float, lon2: float,
    radius_km: float = 6371.0
) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        radius_km: Earth radius in kilometers
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2)
    
    c = 2 * math.asin(math.sqrt(a))
    
    return radius_km * c


def azimuth_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> Tuple[float, float]:
    """
    Calculate azimuth (bearing) and distance between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Tuple of (azimuth_degrees, distance_km)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate distance
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Calculate azimuth
    dlon = lon2_rad - lon1_rad
    
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
    
    azimuth_rad = math.atan2(y, x)
    azimuth_deg = math.degrees(azimuth_rad)
    
    # Normalize to 0-360
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg, distance


def normalize_coordinates(
    latitudes: Union[List[float], np.ndarray],
    longitudes: Union[List[float], np.ndarray],
    method: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Normalize geographic coordinates.
    
    Args:
        latitudes: Latitude values
        longitudes: Longitude values  
        method: Normalization method ('standard', 'minmax', 'mercator')
        
    Returns:
        Tuple of (normalized_lat, normalized_lon, normalization_stats)
    """
    lats = np.array(latitudes)
    lons = np.array(longitudes)
    
    stats = {}
    
    if method == "standard":
        # Z-score normalization
        lat_mean, lat_std = lats.mean(), lats.std()
        lon_mean, lon_std = lons.mean(), lons.std()
        
        norm_lats = (lats - lat_mean) / (lat_std + 1e-8)
        norm_lons = (lons - lon_mean) / (lon_std + 1e-8)
        
        stats = {
            'lat_mean': lat_mean, 'lat_std': lat_std,
            'lon_mean': lon_mean, 'lon_std': lon_std
        }
    
    elif method == "minmax":
        # Min-max normalization to [-1, 1]
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        norm_lats = 2 * (lats - lat_min) / (lat_max - lat_min + 1e-8) - 1
        norm_lons = 2 * (lons - lon_min) / (lon_max - lon_min + 1e-8) - 1
        
        stats = {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max
        }
    
    elif method == "mercator":
        # Mercator projection normalization
        # Latitude: convert to Mercator Y, then normalize
        mercator_y = np.log(np.tan(np.pi/4 + np.radians(lats)/2))
        
        lat_mean, lat_std = mercator_y.mean(), mercator_y.std()
        lon_mean, lon_std = lons.mean(), lons.std()
        
        norm_lats = (mercator_y - lat_mean) / (lat_std + 1e-8)
        norm_lons = (lons - lon_mean) / (lon_std + 1e-8)
        
        stats = {
            'mercator_lat_mean': lat_mean, 'mercator_lat_std': lat_std,
            'lon_mean': lon_mean, 'lon_std': lon_std
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return norm_lats, norm_lons, stats


class RegionBucketer:
    """
    Geographic region bucketing for seismic stations.
    
    Assigns stations to geographic regions for analysis and modeling.
    """
    
    def __init__(
        self,
        method: str = "grid",
        grid_size_deg: float = 1.0,
        custom_regions: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize region bucketer.
        
        Args:
            method: Bucketing method ('grid', 'custom', 'kmeans')
            grid_size_deg: Grid size in degrees for grid method
            custom_regions: Custom region definitions
        """
        self.method = method
        self.grid_size_deg = grid_size_deg
        self.custom_regions = custom_regions or {}
        
        # Predefined tectonic regions
        self.tectonic_regions = {
            'cascadia': {
                'lat_range': (40.0, 50.0),
                'lon_range': (-130.0, -120.0),
                'description': 'Cascadia Subduction Zone'
            },
            'san_andreas': {
                'lat_range': (32.0, 40.0),
                'lon_range': (-125.0, -115.0),
                'description': 'San Andreas Fault System'
            },
            'yellowstone': {
                'lat_range': (44.0, 46.0),
                'lon_range': (-112.0, -109.0),
                'description': 'Yellowstone Caldera'
            },
            'alaska': {
                'lat_range': (55.0, 72.0),
                'lon_range': (-170.0, -130.0),
                'description': 'Alaska Seismic Zone'
            },
            'eastern_us': {
                'lat_range': (25.0, 50.0),
                'lon_range': (-100.0, -65.0),
                'description': 'Eastern United States'
            },
        }
    
    def assign_regions(
        self,
        latitudes: Union[List[float], np.ndarray],
        longitudes: Union[List[float], np.ndarray],
    ) -> List[str]:
        """
        Assign geographic regions to coordinates.
        
        Args:
            latitudes: Latitude values
            longitudes: Longitude values
            
        Returns:
            List of region identifiers
        """
        lats = np.array(latitudes)
        lons = np.array(longitudes)
        
        if self.method == "grid":
            return self._assign_grid_regions(lats, lons)
        elif self.method == "custom":
            return self._assign_custom_regions(lats, lons)
        elif self.method == "tectonic":
            return self._assign_tectonic_regions(lats, lons)
        else:
            raise ValueError(f"Unknown bucketing method: {self.method}")
    
    def _assign_grid_regions(
        self, 
        lats: np.ndarray, 
        lons: np.ndarray
    ) -> List[str]:
        """Assign grid-based regions."""
        regions = []
        
        for lat, lon in zip(lats, lons):
            # Calculate grid indices
            lat_idx = int(lat / self.grid_size_deg)
            lon_idx = int(lon / self.grid_size_deg)
            
            region_id = f"grid_{lat_idx:+03d}_{lon_idx:+04d}"
            regions.append(region_id)
        
        return regions
    
    def _assign_custom_regions(
        self,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> List[str]:
        """Assign custom-defined regions."""
        regions = []
        
        for lat, lon in zip(lats, lons):
            assigned_region = "unknown"
            
            # Check each custom region
            for region_name, region_def in self.custom_regions.items():
                if self._point_in_region(lat, lon, region_def):
                    assigned_region = region_name
                    break
            
            regions.append(assigned_region)
        
        return regions
    
    def _assign_tectonic_regions(
        self,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> List[str]:
        """Assign tectonic regions."""
        regions = []
        
        for lat, lon in zip(lats, lons):
            assigned_region = "other"
            
            # Check each tectonic region
            for region_name, region_def in self.tectonic_regions.items():
                if self._point_in_region(lat, lon, region_def):
                    assigned_region = region_name
                    break
            
            regions.append(assigned_region)
        
        return regions
    
    def _point_in_region(
        self, 
        lat: float, 
        lon: float, 
        region_def: Dict
    ) -> bool:
        """Check if point is in defined region."""
        if 'lat_range' in region_def and 'lon_range' in region_def:
            lat_min, lat_max = region_def['lat_range']
            lon_min, lon_max = region_def['lon_range']
            
            return (lat_min <= lat <= lat_max and 
                    lon_min <= lon <= lon_max)
        
        elif 'center' in region_def and 'radius_km' in region_def:
            center_lat, center_lon = region_def['center']
            radius_km = region_def['radius_km']
            
            distance = haversine_distance(lat, lon, center_lat, center_lon)
            return distance <= radius_km
        
        else:
            return False


class SpatialFeatureExtractor:
    """
    Extract spatial features from station coordinates and metadata.
    
    Creates derived features that capture geographic and geometric relationships.
    """
    
    def __init__(
        self,
        reference_coords: Optional[Tuple[float, float]] = None,
        include_derived: bool = True,
    ):
        """
        Initialize spatial feature extractor.
        
        Args:
            reference_coords: (lat, lon) reference point for distance calculations
            include_derived: Whether to include derived spatial features
        """
        self.reference_coords = reference_coords
        self.include_derived = include_derived
        
        # Default reference point (center of CONUS)
        if reference_coords is None:
            self.reference_coords = (39.8283, -98.5795)
    
    def extract_features(
        self,
        latitudes: Union[List[float], np.ndarray],
        longitudes: Union[List[float], np.ndarray],
        elevations: Optional[Union[List[float], np.ndarray]] = None,
        azimuths: Optional[Union[List[float], np.ndarray]] = None,
        dips: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract spatial features from coordinates.
        
        Args:
            latitudes: Station latitudes
            longitudes: Station longitudes
            elevations: Station elevations (optional)
            azimuths: Sensor azimuths (optional)
            dips: Sensor dips (optional)
            
        Returns:
            Dictionary of extracted features
        """
        lats = np.array(latitudes)
        lons = np.array(longitudes)
        
        features = {
            'latitude': lats,
            'longitude': lons,
        }
        
        if elevations is not None:
            features['elevation'] = np.array(elevations)
        
        if azimuths is not None:
            features['azimuth'] = np.array(azimuths)
            
        if dips is not None:
            features['dip'] = np.array(dips)
        
        if self.include_derived:
            derived_features = self._extract_derived_features(
                lats, lons, elevations, azimuths
            )
            features.update(derived_features)
        
        return features
    
    def _extract_derived_features(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        elevations: Optional[np.ndarray],
        azimuths: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Extract derived spatial features."""
        features = {}
        
        # Distance and azimuth from reference point
        ref_lat, ref_lon = self.reference_coords
        
        distances = []
        ref_azimuths = []
        
        for lat, lon in zip(lats, lons):
            azimuth, distance = azimuth_distance(ref_lat, ref_lon, lat, lon)
            distances.append(distance)
            ref_azimuths.append(azimuth)
        
        features['distance_from_reference'] = np.array(distances)
        features['azimuth_from_reference'] = np.array(ref_azimuths)
        
        # Coordinate transformations
        features['latitude_rad'] = np.radians(lats)
        features['longitude_rad'] = np.radians(lons)
        
        # Trigonometric features for periodicity
        features['lat_sin'] = np.sin(features['latitude_rad'])
        features['lat_cos'] = np.cos(features['latitude_rad'])
        features['lon_sin'] = np.sin(features['longitude_rad'])
        features['lon_cos'] = np.cos(features['longitude_rad'])
        
        # Distance from equator and prime meridian
        features['distance_from_equator'] = np.abs(lats)
        features['distance_from_prime_meridian'] = np.abs(lons)
        
        # Elevation features if available
        if elevations is not None:
            elevs = np.array(elevations)
            features['elevation_km'] = elevs / 1000.0
            features['log_elevation'] = np.log(np.maximum(elevs, 1.0))
            
            # Elevation categories
            features['elevation_category'] = np.select(
                [elevs < 0, elevs < 500, elevs < 1500, elevs >= 1500],
                [0, 1, 2, 3]  # below_sea, low, medium, high
            )
        
        # Azimuth features if available
        if azimuths is not None:
            azims = np.array(azimuths)
            azim_rad = np.radians(azims)
            
            features['azimuth_sin'] = np.sin(azim_rad)
            features['azimuth_cos'] = np.cos(azim_rad)
            
            # Cardinal direction indicators
            features['is_north'] = (azims < 45) | (azims > 315)
            features['is_east'] = (45 <= azims) & (azims < 135)
            features['is_south'] = (135 <= azims) & (azims < 225)
            features['is_west'] = (225 <= azims) & (azims < 315)
        
        return features


def compute_station_distances(
    station_coords: List[Tuple[float, float]],
    max_distance_km: Optional[float] = None,
) -> np.ndarray:
    """
    Compute pairwise distances between stations.
    
    Args:
        station_coords: List of (lat, lon) tuples
        max_distance_km: Maximum distance to compute (for efficiency)
        
    Returns:
        Distance matrix (N, N)
    """
    n_stations = len(station_coords)
    distances = np.zeros((n_stations, n_stations))
    
    for i in range(n_stations):
        lat1, lon1 = station_coords[i]
        
        for j in range(i + 1, n_stations):
            lat2, lon2 = station_coords[j]
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            
            if max_distance_km is None or dist <= max_distance_km:
                distances[i, j] = dist
                distances[j, i] = dist
            else:
                distances[i, j] = np.inf
                distances[j, i] = np.inf
    
    return distances


def find_nearby_stations(
    target_coord: Tuple[float, float],
    station_coords: List[Tuple[float, float]],
    max_distance_km: float = 100.0,
    max_count: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Find nearby stations within specified distance.
    
    Args:
        target_coord: (lat, lon) of target location
        station_coords: List of (lat, lon) for all stations
        max_distance_km: Maximum distance to consider
        max_count: Maximum number of stations to return
        
    Returns:
        List of (station_index, distance_km) sorted by distance
    """
    target_lat, target_lon = target_coord
    
    nearby = []
    for i, (lat, lon) in enumerate(station_coords):
        distance = haversine_distance(target_lat, target_lon, lat, lon)
        
        if distance <= max_distance_km:
            nearby.append((i, distance))
    
    # Sort by distance
    nearby.sort(key=lambda x: x[1])
    
    # Limit count if specified
    if max_count is not None:
        nearby = nearby[:max_count]
    
    return nearby