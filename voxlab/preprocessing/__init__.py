"""
VoxLab Preprocessing Module

Audio preprocessing functions for VoxLab including:
- Audio resampling with device preservation
- Mono/stereo conversion with channel selection
- Silence removal with fade transitions
- Audio trimming from start, end, or both ends
- Audio chunking with fade-in/fade-out
- RMS-based audio normalization
- Pipeline chaining with constraint validation

All functions support inplace/non-inplace operations and maintain device placement.
"""

from .functions import (
    resample_audio,
    convert_to_mono,
    remove_silence,
    break_into_chunks,
    normalize_audio_rms,
    trim_audio
)

from .pipeline import PreprocessingPipeline

__all__ = [
    'resample_audio',
    'convert_to_mono', 
    'remove_silence',
    'break_into_chunks',
    'normalize_audio_rms',
    'trim_audio',
    'PreprocessingPipeline'
]