"""
Waveform Augmentation for Seismic Data

This module provides various data augmentation techniques for seismic waveforms
including bandpass filtering, amplitude jittering, noise injection, time shifts,
and channel dropout for improved model robustness.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

logger = logging.getLogger(__name__)


class WaveformAugmenter:
    """
    Comprehensive waveform augmentation for seismic training data.
    
    Applies various augmentations to improve model robustness and generalization.
    """
    
    def __init__(
        self,
        sampling_rate: float = 100.0,
        augmentation_prob: float = 0.8,
        
        # Bandpass filtering
        bandpass_prob: float = 0.3,
        freq_ranges: List[Tuple[float, float]] = None,
        
        # Amplitude jittering  
        amp_jitter_prob: float = 0.5,
        amp_jitter_range: Tuple[float, float] = (0.8, 1.2),
        
        # Noise injection
        noise_prob: float = 0.4,
        noise_snr_range: Tuple[float, float] = (10.0, 50.0),
        
        # Time shifting
        time_shift_prob: float = 0.3,
        max_shift_sec: float = 2.0,
        
        # Channel dropout
        channel_dropout_prob: float = 0.2,
        max_dropped_channels: int = 1,
        
        # Polarity flipping
        polarity_flip_prob: float = 0.1,
        
        # Tapering
        taper_prob: float = 0.2,
        taper_fraction_range: Tuple[float, float] = (0.01, 0.1),
    ):
        """
        Initialize waveform augmenter.
        
        Args:
            sampling_rate: Sampling rate of input waveforms
            augmentation_prob: Overall probability of applying augmentations
            bandpass_prob: Probability of applying bandpass filter
            freq_ranges: List of (low, high) frequency ranges for bandpass
            amp_jitter_prob: Probability of amplitude jittering
            amp_jitter_range: (min, max) amplitude scaling factors
            noise_prob: Probability of noise injection
            noise_snr_range: (min, max) SNR range for noise injection
            time_shift_prob: Probability of time shifting
            max_shift_sec: Maximum time shift in seconds
            channel_dropout_prob: Probability of channel dropout
            max_dropped_channels: Maximum number of channels to drop
            polarity_flip_prob: Probability of polarity flipping
            taper_prob: Probability of applying edge tapering
            taper_fraction_range: (min, max) fraction of signal to taper
        """
        self.sampling_rate = sampling_rate
        self.augmentation_prob = augmentation_prob
        
        # Augmentation parameters
        self.bandpass_prob = bandpass_prob
        self.freq_ranges = freq_ranges or [
            (0.1, 10.0),   # Low frequency
            (1.0, 20.0),   # Mid frequency  
            (2.0, 40.0),   # High frequency
            (0.5, 15.0),   # Broad band
        ]
        
        self.amp_jitter_prob = amp_jitter_prob
        self.amp_jitter_range = amp_jitter_range
        
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        
        self.time_shift_prob = time_shift_prob
        self.max_shift_samples = int(max_shift_sec * sampling_rate)
        
        self.channel_dropout_prob = channel_dropout_prob
        self.max_dropped_channels = max_dropped_channels
        
        self.polarity_flip_prob = polarity_flip_prob
        
        self.taper_prob = taper_prob
        self.taper_fraction_range = taper_fraction_range
        
        logger.info(f"Initialized waveform augmenter (sr={sampling_rate} Hz)")
    
    def apply_augmentations(
        self, 
        waveform: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Apply augmentations to a waveform.
        
        Args:
            waveform: Input waveform (C, T) or (T,)
            metadata: Optional metadata for adaptive augmentation
            
        Returns:
            Augmented waveform
        """
        if random.random() > self.augmentation_prob:
            return waveform
        
        # Ensure 2D shape (C, T)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        
        augmented = waveform.copy()
        
        # Apply augmentations in random order
        augmentation_functions = [
            self._apply_bandpass_filter,
            self._apply_amplitude_jitter,
            self._apply_noise_injection,
            self._apply_time_shift,
            self._apply_channel_dropout,
            self._apply_polarity_flip,
            self._apply_taper,
        ]
        
        random.shuffle(augmentation_functions)
        
        for aug_func in augmentation_functions:
            try:
                augmented = aug_func(augmented)
            except Exception as e:
                logger.debug(f"Augmentation {aug_func.__name__} failed: {e}")
                continue
        
        return augmented
    
    def _apply_bandpass_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random bandpass filter."""
        if random.random() > self.bandpass_prob:
            return waveform
        
        # Choose random frequency range
        freq_low, freq_high = random.choice(self.freq_ranges)
        
        # Ensure frequencies are valid
        nyquist = self.sampling_rate / 2.0
        freq_low = max(0.01, min(freq_low, nyquist * 0.45))
        freq_high = min(freq_high, nyquist * 0.45)
        
        if freq_low >= freq_high:
            return waveform
        
        try:
            # Design Butterworth filter
            sos = signal.butter(
                4, [freq_low, freq_high], 
                btype='band', 
                fs=self.sampling_rate, 
                output='sos'
            )
            
            # Apply filter to each channel
            filtered = np.zeros_like(waveform)
            for c in range(waveform.shape[0]):
                filtered[c] = signal.sosfiltfilt(sos, waveform[c])
            
            return filtered
            
        except Exception as e:
            logger.debug(f"Bandpass filter failed: {e}")
            return waveform
    
    def _apply_amplitude_jitter(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random amplitude scaling."""
        if random.random() > self.amp_jitter_prob:
            return waveform
        
        # Sample scaling factor
        scale_min, scale_max = self.amp_jitter_range
        scale_factor = random.uniform(scale_min, scale_max)
        
        return waveform * scale_factor
    
    def _apply_noise_injection(self, waveform: np.ndarray) -> np.ndarray:
        """Inject Gaussian noise with specified SNR."""
        if random.random() > self.noise_prob:
            return waveform
        
        # Calculate signal power
        signal_power = np.mean(waveform**2, axis=1, keepdims=True)
        
        # Sample target SNR
        snr_min, snr_max = self.noise_snr_range
        target_snr_db = random.uniform(snr_min, snr_max)
        
        # Calculate noise power
        snr_linear = 10**(target_snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
        
        return waveform + noise
    
    def _apply_time_shift(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random time shift."""
        if random.random() > self.time_shift_prob:
            return waveform
        
        if self.max_shift_samples == 0:
            return waveform
        
        # Sample shift amount
        shift_samples = random.randint(-self.max_shift_samples, self.max_shift_samples)
        
        if shift_samples == 0:
            return waveform
        
        # Apply shift by rolling
        shifted = np.roll(waveform, shift_samples, axis=1)
        
        # Zero out wrapped regions to avoid artifacts
        if shift_samples > 0:
            shifted[:, :shift_samples] = 0
        else:
            shifted[:, shift_samples:] = 0
        
        return shifted
    
    def _apply_channel_dropout(self, waveform: np.ndarray) -> np.ndarray:
        """Randomly drop channels."""
        if random.random() > self.channel_dropout_prob:
            return waveform
        
        num_channels = waveform.shape[0]
        if num_channels <= 1:
            return waveform
        
        # Determine number of channels to drop
        max_drop = min(self.max_dropped_channels, num_channels - 1)
        num_drop = random.randint(1, max_drop)
        
        # Select channels to drop
        channels_to_drop = random.sample(range(num_channels), num_drop)
        
        # Zero out selected channels
        dropped = waveform.copy()
        dropped[channels_to_drop, :] = 0
        
        return dropped
    
    def _apply_polarity_flip(self, waveform: np.ndarray) -> np.ndarray:
        """Randomly flip polarity of channels."""
        if random.random() > self.polarity_flip_prob:
            return waveform
        
        # Flip each channel independently
        flipped = waveform.copy()
        for c in range(waveform.shape[0]):
            if random.random() < 0.5:
                flipped[c] = -flipped[c]
        
        return flipped
    
    def _apply_taper(self, waveform: np.ndarray) -> np.ndarray:
        """Apply edge tapering."""
        if random.random() > self.taper_prob:
            return waveform
        
        # Sample taper fraction
        frac_min, frac_max = self.taper_fraction_range
        taper_fraction = random.uniform(frac_min, frac_max)
        
        seq_len = waveform.shape[1]
        taper_samples = int(taper_fraction * seq_len)
        
        if taper_samples == 0:
            return waveform
        
        # Create Hann taper
        taper = np.ones(seq_len)
        
        # Left taper
        left_taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples) / taper_samples))
        taper[:taper_samples] = left_taper
        
        # Right taper  
        right_taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples, 0, -1) / taper_samples))
        taper[-taper_samples:] = right_taper
        
        return waveform * taper[np.newaxis, :]


class AdaptiveAugmenter(WaveformAugmenter):
    """
    Adaptive augmenter that adjusts augmentation strength based on metadata.
    
    Uses station metadata to apply more appropriate augmentations based on
    instrument characteristics, location, etc.
    """
    
    def __init__(
        self,
        base_augmenter_config: Optional[Dict] = None,
        metadata_adaptation: bool = True,
        **kwargs
    ):
        """
        Initialize adaptive augmenter.
        
        Args:
            base_augmenter_config: Base augmenter configuration
            metadata_adaptation: Whether to adapt based on metadata
            **kwargs: Additional arguments for base augmenter
        """
        config = base_augmenter_config or {}
        config.update(kwargs)
        super().__init__(**config)
        
        self.metadata_adaptation = metadata_adaptation
    
    def apply_augmentations(
        self, 
        waveform: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """Apply metadata-adaptive augmentations."""
        if not self.metadata_adaptation or metadata is None:
            return super().apply_augmentations(waveform, metadata)
        
        # Adapt augmentation parameters based on metadata
        adapted_probs = self._adapt_probabilities(metadata)
        
        # Temporarily modify probabilities
        original_probs = {
            'bandpass_prob': self.bandpass_prob,
            'amp_jitter_prob': self.amp_jitter_prob,
            'noise_prob': self.noise_prob,
            'time_shift_prob': self.time_shift_prob,
            'channel_dropout_prob': self.channel_dropout_prob,
        }
        
        # Apply adapted probabilities
        for prob_name, prob_value in adapted_probs.items():
            if hasattr(self, prob_name):
                setattr(self, prob_name, prob_value)
        
        try:
            # Apply augmentations with adapted probabilities
            augmented = super().apply_augmentations(waveform, metadata)
        finally:
            # Restore original probabilities
            for prob_name, prob_value in original_probs.items():
                setattr(self, prob_name, prob_value)
        
        return augmented
    
    def _adapt_probabilities(self, metadata: Dict) -> Dict[str, float]:
        """Adapt augmentation probabilities based on metadata."""
        adapted = {}
        
        # Get sampling rate if available
        sampling_rate = metadata.get('sampling_rate', self.sampling_rate)
        
        # Adapt based on sampling rate
        if sampling_rate < 50:
            # Low sampling rate - reduce high-frequency augmentations
            adapted['bandpass_prob'] = self.bandpass_prob * 0.5
        elif sampling_rate > 200:
            # High sampling rate - can be more aggressive
            adapted['bandpass_prob'] = min(1.0, self.bandpass_prob * 1.5)
        
        # Adapt based on instrument type
        instrument_band = metadata.get('instrument_band', '')
        if instrument_band == 'H':  # High-gain seismometer
            # More sensitive to noise
            adapted['noise_prob'] = self.noise_prob * 0.7
        elif instrument_band == 'E':  # Extremely short period
            # Can handle more aggressive augmentation
            adapted['amp_jitter_prob'] = min(1.0, self.amp_jitter_prob * 1.2)
        
        # Adapt based on channel type
        channel_type = metadata.get('channel_type', '')
        if channel_type in ['Z']:  # Vertical component
            # Less aggressive time shifting for vertical
            adapted['time_shift_prob'] = self.time_shift_prob * 0.8
        
        # Adapt based on location (e.g., vault vs surface)
        location = metadata.get('location', '')
        if location in ['00', '10']:  # Vault installations
            # Typically lower noise, can add more artificial noise
            adapted['noise_prob'] = min(1.0, self.noise_prob * 1.3)
        
        return adapted


def create_augmenter(
    augmenter_type: str = "standard",
    **kwargs
) -> WaveformAugmenter:
    """
    Factory function for creating waveform augmenters.
    
    Args:
        augmenter_type: Type of augmenter ('standard', 'adaptive')
        **kwargs: Arguments for augmenter initialization
        
    Returns:
        WaveformAugmenter instance
    """
    if augmenter_type == "standard":
        return WaveformAugmenter(**kwargs)
    elif augmenter_type == "adaptive":
        return AdaptiveAugmenter(**kwargs)
    else:
        raise ValueError(f"Unknown augmenter type: {augmenter_type}")


# Default configurations
DEFAULT_AUGMENTATION_CONFIGS = {
    "light": {
        "augmentation_prob": 0.5,
        "bandpass_prob": 0.2,
        "amp_jitter_prob": 0.3,
        "noise_prob": 0.2,
        "time_shift_prob": 0.1,
        "channel_dropout_prob": 0.1,
    },
    "standard": {
        "augmentation_prob": 0.8,
        "bandpass_prob": 0.3,
        "amp_jitter_prob": 0.5,
        "noise_prob": 0.4,
        "time_shift_prob": 0.3,
        "channel_dropout_prob": 0.2,
    },
    "aggressive": {
        "augmentation_prob": 0.9,
        "bandpass_prob": 0.5,
        "amp_jitter_prob": 0.7,
        "noise_prob": 0.6,
        "time_shift_prob": 0.5,
        "channel_dropout_prob": 0.3,
        "polarity_flip_prob": 0.2,
    },
}