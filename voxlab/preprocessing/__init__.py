"""
VoxLab Preprocessing Module

Audio preprocessing functions for VoxLab with threshold-based silence detection.
"""

from .functions import (
    resample_audio,
    convert_to_mono,
    remove_silence,
    break_into_chunks,
    normalize_audio_rms
)

from .pipeline import PreprocessingPipeline

__all__ = [
    'resample_audio',
    'convert_to_mono', 
    'remove_silence',
    'break_into_chunks',
    'normalize_audio_rms',
    'PreprocessingPipeline'
]