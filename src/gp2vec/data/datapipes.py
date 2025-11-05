"""
Data Pipelines for Streaming Training Data

This module provides WebDataset and PyTorch DataLoader pipelines for
streaming seismic data from S3, joining with metadata, and applying augmentations.
"""

import io
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset

from .decoder import MiniSeedDecoder, process_stream_for_training
from .metadata import StationMetadataManager
from ..utils.aug import WaveformAugmenter
from ..utils.io import S3Client

logger = logging.getLogger(__name__)


class SeismicDataPipeline:
    """Main data pipeline for streaming seismic training data."""
    
    def __init__(
        self,
        manifest_path: str,
        metadata_path: str,
        window_length_sec: float = 30.0,
        overlap_sec: float = 22.5,
        target_sampling_rate: float = 100.0,
        max_channels: int = 3,
        augmenter: Optional[WaveformAugmenter] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the data pipeline.
        
        Args:
            manifest_path: Path to S3 manifest Parquet file
            metadata_path: Path to station metadata Parquet file
            window_length_sec: Length of training windows
            overlap_sec: Overlap between windows
            target_sampling_rate: Target sampling rate for resampling
            max_channels: Maximum number of channels (pad/truncate)
            augmenter: Waveform augmenter instance
            cache_dir: Directory for caching processed data
        """
        self.manifest_path = manifest_path
        self.metadata_path = metadata_path
        self.window_length_sec = window_length_sec
        self.overlap_sec = overlap_sec
        self.target_sampling_rate = target_sampling_rate
        self.max_channels = max_channels
        self.augmenter = augmenter
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata
        self.manifest_df = pd.read_parquet(manifest_path)
        self.metadata_df = pd.read_parquet(metadata_path)
        
        # Create decoder
        self.decoder = MiniSeedDecoder(target_sampling_rates=[target_sampling_rate])
        self.s3_client = S3Client()
        
        # Prepare metadata lookup
        self._prepare_metadata_lookup()
    
    def _prepare_metadata_lookup(self):
        """Prepare fast metadata lookup by station."""
        # Create lookup dictionary: (network, station) -> metadata
        self.metadata_lookup = {}
        for _, row in self.metadata_df.iterrows():
            key = (row['network'], row['station'])
            if key not in self.metadata_lookup:
                self.metadata_lookup[key] = []
            self.metadata_lookup[key].append(row.to_dict())
    
    def create_webdataset_shards(
        self,
        output_dir: str,
        shard_size_mb: int = 1000,
        max_files_per_shard: int = 1000,
        pattern: str = "shard-%06d.tar",
    ) -> List[str]:
        """
        Create WebDataset shards from manifest.
        
        Args:
            output_dir: Output directory for shards
            shard_size_mb: Target shard size in MB
            max_files_per_shard: Maximum files per shard
            pattern: Shard filename pattern
            
        Returns:
            List of created shard paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        shard_paths = []
        current_shard = 0
        current_shard_size = 0
        current_shard_files = 0
        
        # Group manifest by reasonable chunks
        manifest_shuffled = self.manifest_df.sample(frac=1.0).reset_index(drop=True)
        
        with wds.ShardWriter(
            str(output_path / (pattern % current_shard)),
            maxsize=shard_size_mb * 1024 * 1024,
            maxcount=max_files_per_shard
        ) as sink:
            
            for idx, row in manifest_shuffled.iterrows():
                try:
                    # Process file
                    samples = self._process_file_for_shard(row)
                    
                    for sample_idx, sample in enumerate(samples):
                        key = f"{idx:08d}_{sample_idx:04d}"
                        sink.write(sample, key=key)
                    
                    # Check if we need a new shard
                    current_shard_size += row.get('size_bytes', 0)
                    current_shard_files += 1
                    
                    if (current_shard_size >= shard_size_mb * 1024 * 1024 or 
                        current_shard_files >= max_files_per_shard):
                        
                        shard_paths.append(str(output_path / (pattern % current_shard)))
                        current_shard += 1
                        current_shard_size = 0
                        current_shard_files = 0
                        
                        # Create new shard writer
                        sink.close()
                        sink = wds.ShardWriter(
                            str(output_path / (pattern % current_shard)),
                            maxsize=shard_size_mb * 1024 * 1024,
                            maxcount=max_files_per_shard
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to process file {row.get('key', '')}: {e}")
                    continue
            
            # Add final shard if it has content
            if current_shard_files > 0:
                shard_paths.append(str(output_path / (pattern % current_shard)))
        
        logger.info(f"Created {len(shard_paths)} shards in {output_dir}")
        return shard_paths
    
    def _process_file_for_shard(self, manifest_row: pd.Series) -> List[Dict]:
        """Process a single file from manifest into shard samples."""
        s3_key = manifest_row['key']
        
        try:
            # Read and process stream
            stream = self.decoder.read_mseed_s3(s3_key, apply_qc=True)
            if not stream:
                return []
            
            # Process for training
            windows = process_stream_for_training(
                stream,
                target_sr=self.target_sampling_rate,
                window_length_sec=self.window_length_sec,
                overlap_sec=self.overlap_sec,
            )
            
            samples = []
            for data_array, window_meta in windows:
                # Get station metadata
                station_meta = self._get_station_metadata(window_meta)
                
                # Create sample
                sample = {
                    'waveform.npy': data_array.astype(np.float32),
                    'metadata.json': json.dumps({
                        'window': window_meta,
                        'station': station_meta,
                        'manifest': manifest_row.to_dict(),
                    }),
                }
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.warning(f"Failed to process {s3_key}: {e}")
            return []
    
    def _get_station_metadata(self, window_meta: Dict) -> Dict:
        """Get station metadata for a window."""
        network = window_meta.get('network', '')
        station = window_meta.get('station', '')
        
        key = (network, station)
        if key in self.metadata_lookup:
            # Return first matching channel metadata
            return self.metadata_lookup[key][0]
        else:
            # Return empty metadata
            return {}
    
    def create_webdataset_loader(
        self,
        shard_urls: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        buffer_size: int = 1000,
    ) -> DataLoader:
        """
        Create WebDataset DataLoader from shard URLs.
        
        Args:
            shard_urls: List of shard file URLs/paths
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            buffer_size: Shuffle buffer size
            
        Returns:
            PyTorch DataLoader
        """
        def process_sample(sample):
            """Process a single sample from WebDataset."""
            try:
                # Load waveform
                waveform = np.load(io.BytesIO(sample['waveform.npy']))
                
                # Load metadata
                metadata = json.loads(sample['metadata.json'].decode('utf-8'))
                
                # Pad/truncate channels
                if waveform.shape[0] < self.max_channels:
                    # Pad with zeros
                    pad_size = self.max_channels - waveform.shape[0]
                    waveform = np.pad(waveform, ((0, pad_size), (0, 0)), 'constant')
                elif waveform.shape[0] > self.max_channels:
                    # Truncate
                    waveform = waveform[:self.max_channels]
                
                # Apply augmentations
                if self.augmenter:
                    waveform = self.augmenter.apply_augmentations(
                        waveform, metadata['station']
                    )
                
                return {
                    'waveform': torch.from_numpy(waveform).float(),
                    'metadata': metadata,
                }
                
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                # Return dummy sample
                return {
                    'waveform': torch.zeros(self.max_channels, int(self.window_length_sec * self.target_sampling_rate)),
                    'metadata': {},
                }
        
        # Create dataset
        dataset = (
            wds.WebDataset(shard_urls)
            .shuffle(buffer_size if shuffle else 0)
            .decode()
            .map(process_sample)
            .batched(batch_size)
        )
        
        return wds.WebLoader(
            dataset,
            num_workers=num_workers,
            batch_size=None,  # Batching handled by WebDataset
        )


class SeismicIterableDataset(IterableDataset):
    """PyTorch IterableDataset for streaming seismic data."""
    
    def __init__(
        self,
        manifest_path: str,
        metadata_path: str,
        window_length_sec: float = 30.0,
        target_sampling_rate: float = 100.0,
        max_channels: int = 3,
        augmenter: Optional[WaveformAugmenter] = None,
        max_files: Optional[int] = None,
        shuffle_files: bool = True,
    ):
        """
        Initialize iterable dataset.
        
        Args:
            manifest_path: Path to S3 manifest
            metadata_path: Path to station metadata
            window_length_sec: Window length in seconds
            target_sampling_rate: Target sampling rate
            max_channels: Maximum number of channels
            augmenter: Waveform augmenter
            max_files: Maximum files to process (for testing)
            shuffle_files: Whether to shuffle file order
        """
        super().__init__()
        
        self.manifest_df = pd.read_parquet(manifest_path)
        self.metadata_df = pd.read_parquet(metadata_path)
        self.window_length_sec = window_length_sec
        self.target_sampling_rate = target_sampling_rate
        self.max_channels = max_channels
        self.augmenter = augmenter
        self.shuffle_files = shuffle_files
        
        if max_files:
            self.manifest_df = self.manifest_df.head(max_files)
        
        if shuffle_files:
            self.manifest_df = self.manifest_df.sample(frac=1.0).reset_index(drop=True)
        
        # Initialize components
        self.decoder = MiniSeedDecoder(target_sampling_rates=[target_sampling_rate])
        self.pipeline = SeismicDataPipeline(
            manifest_path, metadata_path, window_length_sec,
            target_sampling_rate=target_sampling_rate, max_channels=max_channels
        )
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples."""
        for _, row in self.manifest_df.iterrows():
            try:
                # Read and process stream
                stream = self.decoder.read_mseed_s3(row['key'], apply_qc=True)
                if not stream:
                    continue
                
                # Process for training
                windows = process_stream_for_training(
                    stream,
                    target_sr=self.target_sampling_rate,
                    window_length_sec=self.window_length_sec,
                )
                
                for data_array, window_meta in windows:
                    # Get station metadata
                    station_meta = self.pipeline._get_station_metadata(window_meta)
                    
                    # Pad/truncate channels
                    if data_array.shape[0] < self.max_channels:
                        pad_size = self.max_channels - data_array.shape[0]
                        data_array = np.pad(data_array, ((0, pad_size), (0, 0)), 'constant')
                    elif data_array.shape[0] > self.max_channels:
                        data_array = data_array[:self.max_channels]
                    
                    # Apply augmentations
                    if self.augmenter:
                        data_array = self.augmenter.apply_augmentations(
                            data_array, station_meta
                        )
                    
                    yield {
                        'waveform': torch.from_numpy(data_array).float(),
                        'metadata': {
                            'window': window_meta,
                            'station': station_meta,
                        }
                    }
            
            except Exception as e:
                logger.warning(f"Failed to process file {row.get('key', '')}: {e}")
                continue


def create_train_dataloader(
    manifest_path: str,
    metadata_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    window_length_sec: float = 30.0,
    target_sampling_rate: float = 100.0,
    max_channels: int = 3,
    use_webdataset: bool = True,
    shard_dir: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create training DataLoader.
    
    Args:
        manifest_path: Path to S3 manifest
        metadata_path: Path to station metadata  
        batch_size: Batch size
        num_workers: Number of workers
        window_length_sec: Window length
        target_sampling_rate: Sampling rate
        max_channels: Max channels
        use_webdataset: Whether to use WebDataset format
        shard_dir: Directory with WebDataset shards (if use_webdataset=True)
        **kwargs: Additional arguments
        
    Returns:
        PyTorch DataLoader
    """
    if use_webdataset and shard_dir:
        # Use WebDataset
        shard_paths = list(Path(shard_dir).glob("*.tar"))
        shard_urls = [str(p) for p in shard_paths]
        
        pipeline = SeismicDataPipeline(
            manifest_path, metadata_path,
            window_length_sec=window_length_sec,
            target_sampling_rate=target_sampling_rate,
            max_channels=max_channels,
        )
        
        return pipeline.create_webdataset_loader(
            shard_urls, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
    
    else:
        # Use IterableDataset
        dataset = SeismicIterableDataset(
            manifest_path, metadata_path,
            window_length_sec=window_length_sec,
            target_sampling_rate=target_sampling_rate,
            max_channels=max_channels,
            **kwargs
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def collate_seismic_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for seismic data batches."""
    waveforms = torch.stack([sample['waveform'] for sample in batch])
    
    # Note: metadata is not collated since it varies per sample
    # For training, you might want to extract and collate specific metadata features
    
    return {
        'waveforms': waveforms,
        'metadata': [sample['metadata'] for sample in batch],
    }