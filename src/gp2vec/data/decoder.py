"""
miniSEED Decoder for S3 Streaming

This module provides functionality to read miniSEED files from S3 streams,
perform basic quality control, and prepare waveforms for model input.
"""

import io
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import s3fs
from obspy import Stream, Trace, read
from obspy.core.util import AttribDict
from obspy.signal import filter as obs_filter

from ..utils.io import S3Client

logger = logging.getLogger(__name__)


class MiniSeedDecoder:
    """Decode miniSEED data from S3 streams with quality control."""
    
    def __init__(
        self,
        fs: Optional[s3fs.S3FileSystem] = None,
        target_sampling_rates: List[float] = [50.0, 100.0],
        max_gap_seconds: float = 300.0,
        min_trace_length_seconds: float = 30.0,
    ):
        """
        Initialize the decoder.
        
        Args:
            fs: S3 filesystem instance
            target_sampling_rates: Preferred sampling rates for resampling
            max_gap_seconds: Maximum gap to interpolate over
            min_trace_length_seconds: Minimum trace length to keep
        """
        self.fs = fs or s3fs.S3FileSystem(anon=True)
        self.target_sampling_rates = target_sampling_rates
        self.max_gap_seconds = max_gap_seconds
        self.min_trace_length_seconds = min_trace_length_seconds
        self.s3_client = S3Client()
    
    def read_mseed_s3(
        self, 
        s3_key: str,
        apply_qc: bool = True,
        merge_method: str = "interpolate",
    ) -> Optional[Stream]:
        """
        Read miniSEED file from S3 and apply basic processing.
        
        Args:
            s3_key: S3 object key (full path including bucket)
            apply_qc: Whether to apply quality control
            merge_method: Method for merging traces ('interpolate', 'mask', None)
            
        Returns:
            ObsPy Stream object or None if reading fails
        """
        try:
            with self.fs.open(s3_key, 'rb') as f:
                stream = read(io.BytesIO(f.read()))
                
            if apply_qc:
                stream = self.apply_quality_control(stream, merge_method)
                
            return stream
            
        except Exception as e:
            logger.warning(f"Failed to read {s3_key}: {e}")
            return None
    
    def apply_quality_control(
        self, 
        stream: Stream, 
        merge_method: str = "interpolate"
    ) -> Stream:
        """
        Apply quality control processing to stream.
        
        Args:
            stream: Input ObsPy Stream
            merge_method: Method for merging overlapping traces
            
        Returns:
            Processed Stream
        """
        if not stream:
            return stream
        
        # Sort traces
        stream.sort()
        
        # Remove traces that are too short
        min_npts = int(self.min_trace_length_seconds * stream[0].stats.sampling_rate)
        stream.traces = [tr for tr in stream if tr.stats.npts >= min_npts]
        
        if not stream:
            return stream
        
        # Merge overlapping traces
        if merge_method:
            try:
                stream.merge(method=1, fill_value=merge_method)
            except Exception as e:
                logger.debug(f"Merge failed: {e}")
        
        # Basic trace processing for each trace
        for trace in stream:
            self._process_trace(trace)
        
        return stream
    
    def _process_trace(self, trace: Trace) -> None:
        """Apply basic processing to individual trace."""
        # Remove mean and trend
        trace.detrend('linear')
        trace.detrend('demean')
        
        # Taper edges
        trace.taper(max_percentage=0.05, type='cosine')
        
        # Check for gaps and fill small ones
        if hasattr(trace.stats, 'processing'):
            # Handle any existing processing info
            pass
    
    def resample_to_target_rate(
        self, 
        stream: Stream, 
        target_rate: Optional[float] = None
    ) -> Stream:
        """
        Resample stream to target sampling rate.
        
        Args:
            stream: Input stream
            target_rate: Target sampling rate (if None, chooses best from list)
            
        Returns:
            Resampled stream
        """
        if not stream:
            return stream
        
        original_rate = stream[0].stats.sampling_rate
        
        if target_rate is None:
            # Choose target rate closest to original but not higher
            valid_rates = [r for r in self.target_sampling_rates if r <= original_rate]
            if valid_rates:
                target_rate = max(valid_rates)
            else:
                target_rate = min(self.target_sampling_rates)
        
        if abs(original_rate - target_rate) > 0.1:
            logger.debug(f"Resampling from {original_rate} to {target_rate} Hz")
            stream.resample(target_rate, window='hann', no_filter=False)
        
        return stream
    
    def remove_instrument_response(
        self,
        stream: Stream,
        inventory,
        output: str = "VEL",
        pre_filt: Optional[Tuple[float, float, float, float]] = None,
    ) -> Stream:
        """
        Remove instrument response using station metadata.
        
        Args:
            stream: Input stream
            inventory: ObsPy Inventory with response information
            output: Output units ('DISP', 'VEL', 'ACC')
            pre_filt: Pre-filter frequencies (f1, f2, f3, f4)
            
        Returns:
            Corrected stream
        """
        try:
            if pre_filt is None:
                # Default pre-filter based on Nyquist
                nyquist = stream[0].stats.sampling_rate / 2.0
                pre_filt = (0.005, 0.006, 0.4 * nyquist, 0.45 * nyquist)
            
            stream.remove_response(
                inventory=inventory,
                output=output,
                pre_filt=pre_filt,
                zero_mean=True,
                taper=True,
                taper_fraction=0.05,
            )
            
        except Exception as e:
            logger.warning(f"Failed to remove response: {e}")
        
        return stream
    
    def apply_bandpass_filter(
        self,
        stream: Stream,
        freqmin: float = 0.1,
        freqmax: Optional[float] = None,
        corners: int = 4,
        zerophase: bool = True,
    ) -> Stream:
        """
        Apply bandpass filter to stream.
        
        Args:
            stream: Input stream  
            freqmin: Minimum frequency
            freqmax: Maximum frequency (if None, uses 0.45 * Nyquist)
            corners: Filter corners
            zerophase: Whether to use zero-phase filter
            
        Returns:
            Filtered stream
        """
        if not stream:
            return stream
        
        try:
            if freqmax is None:
                nyquist = stream[0].stats.sampling_rate / 2.0
                freqmax = 0.45 * nyquist
            
            stream.filter(
                'bandpass',
                freqmin=freqmin,
                freqmax=freqmax,
                corners=corners,
                zerophase=zerophase
            )
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
        
        return stream
    
    def extract_windows(
        self,
        stream: Stream,
        window_length_sec: float = 30.0,
        overlap_sec: float = 22.5,
        min_window_length_sec: Optional[float] = None,
    ) -> List[Tuple[Stream, Dict]]:
        """
        Extract overlapping windows from stream.
        
        Args:
            stream: Input stream
            window_length_sec: Window length in seconds
            overlap_sec: Overlap between windows in seconds
            min_window_length_sec: Minimum window length (defaults to window_length_sec)
            
        Returns:
            List of (window_stream, metadata) tuples
        """
        if not stream:
            return []
        
        if min_window_length_sec is None:
            min_window_length_sec = window_length_sec
        
        windows = []
        
        for trace in stream:
            sampling_rate = trace.stats.sampling_rate
            window_npts = int(window_length_sec * sampling_rate)
            overlap_npts = int(overlap_sec * sampling_rate)
            step_npts = window_npts - overlap_npts
            min_npts = int(min_window_length_sec * sampling_rate)
            
            # Extract windows
            start_idx = 0
            while start_idx + min_npts <= len(trace.data):
                end_idx = min(start_idx + window_npts, len(trace.data))
                
                # Create window trace
                window_data = trace.data[start_idx:end_idx].copy()
                window_trace = trace.copy()
                window_trace.data = window_data
                window_trace.stats.npts = len(window_data)
                
                # Update timing
                start_time = trace.stats.starttime + start_idx / sampling_rate
                window_trace.stats.starttime = start_time
                
                # Create window stream
                window_stream = Stream([window_trace])
                
                # Window metadata
                metadata = {
                    'original_length_sec': len(trace.data) / sampling_rate,
                    'window_start_sec': start_idx / sampling_rate,
                    'window_length_sec': len(window_data) / sampling_rate,
                    'sampling_rate': sampling_rate,
                }
                
                windows.append((window_stream, metadata))
                
                # Move to next window
                start_idx += step_npts
        
        return windows
    
    def stream_to_array(
        self, 
        stream: Stream,
        normalize: bool = True,
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert ObsPy Stream to numpy array.
        
        Args:
            stream: Input stream
            normalize: Whether to normalize data
            channels: Expected channel order (e.g., ['Z', 'N', 'E'])
            
        Returns:
            Tuple of (data_array, metadata)
            data_array shape: (n_channels, n_samples)
        """
        if not stream:
            return np.array([]), {}
        
        # Sort traces by channel
        if channels:
            sorted_traces = []
            for ch in channels:
                traces = [tr for tr in stream if tr.stats.channel.endswith(ch)]
                if traces:
                    sorted_traces.append(traces[0])  # Take first if multiple
            traces = sorted_traces
        else:
            traces = list(stream)
        
        if not traces:
            return np.array([]), {}
        
        # Get data arrays
        data_arrays = []
        for trace in traces:
            data = trace.data.astype(np.float32)
            
            if normalize:
                # Robust normalization
                std = np.std(data)
                if std > 0:
                    data = data / std
            
            data_arrays.append(data)
        
        # Stack into array
        if len(data_arrays) == 1:
            data_array = data_arrays[0][np.newaxis, :]  # Add channel dimension
        else:
            # Handle different lengths by padding/truncating
            min_length = min(len(d) for d in data_arrays)
            data_arrays = [d[:min_length] for d in data_arrays]
            data_array = np.stack(data_arrays, axis=0)
        
        # Metadata
        trace = traces[0]
        metadata = {
            'network': trace.stats.network,
            'station': trace.stats.station,
            'location': trace.stats.location,
            'channels': [tr.stats.channel for tr in traces],
            'starttime': str(trace.stats.starttime),
            'sampling_rate': trace.stats.sampling_rate,
            'npts': data_array.shape[1],
            'n_channels': data_array.shape[0],
        }
        
        return data_array, metadata


def read_mseed_s3(s3_key: str, **kwargs) -> Optional[Stream]:
    """Convenience function to read miniSEED from S3."""
    decoder = MiniSeedDecoder()
    return decoder.read_mseed_s3(s3_key, **kwargs)


def process_stream_for_training(
    stream: Stream,
    target_sr: float = 100.0,
    window_length_sec: float = 30.0,
    overlap_sec: float = 22.5,
    apply_filter: bool = True,
    freqmin: float = 0.1,
    freqmax: Optional[float] = None,
    normalize: bool = True,
) -> List[Tuple[np.ndarray, Dict]]:
    """
    Complete processing pipeline for training data.
    
    Args:
        stream: Input ObsPy Stream
        target_sr: Target sampling rate
        window_length_sec: Window length for training
        overlap_sec: Overlap between windows
        apply_filter: Whether to apply bandpass filter
        freqmin: Minimum filter frequency
        freqmax: Maximum filter frequency
        normalize: Whether to normalize waveforms
        
    Returns:
        List of (data_array, metadata) tuples ready for training
    """
    decoder = MiniSeedDecoder(target_sampling_rates=[target_sr])
    
    # Process stream
    stream = decoder.apply_quality_control(stream)
    if not stream:
        return []
    
    stream = decoder.resample_to_target_rate(stream, target_sr)
    
    if apply_filter:
        stream = decoder.apply_bandpass_filter(stream, freqmin, freqmax)
    
    # Extract windows
    windows = decoder.extract_windows(stream, window_length_sec, overlap_sec)
    
    # Convert to arrays
    processed_windows = []
    for window_stream, window_meta in windows:
        data_array, stream_meta = decoder.stream_to_array(
            window_stream, normalize=normalize
        )
        
        if data_array.size > 0:
            # Combine metadata
            combined_meta = {**stream_meta, **window_meta}
            processed_windows.append((data_array, combined_meta))
    
    return processed_windows