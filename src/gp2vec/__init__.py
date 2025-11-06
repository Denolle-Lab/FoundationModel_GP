"""
GP2Vec: Wav2Vec2-style self-supervised model for seismic waveform representation learning.
"""

__version__ = "0.1.0"
__author__ = "GP2Vec Team"
__email__ = "mdenolle@uw.edu"

from gp2vec.data import decoder, metadata, s3_manifest
from gp2vec.models import gp2vec
from gp2vec.utils import io

__all__ = [
    "decoder",
    "metadata", 
    "s3_manifest",
    "gp2vec",
    "io",
]