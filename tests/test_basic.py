"""
Basic tests to verify the test framework and imports work.
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import resample_audio


def test_import_works():
    """Test that basic imports work."""
    assert AudioSamples is not None
    assert resample_audio is not None


def test_create_simple_audio():
    """Test creating a simple AudioSamples object."""
    # Create 1 second of mono sine wave at 440Hz
    sample_rate = 44100
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    sine_wave = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    audio_data = sine_wave.unsqueeze(0)  # Add channel dimension
    
    audio = AudioSamples(audio_data, sample_rate)
    
    assert audio.sample_rate == sample_rate
    assert audio.audio_data.shape == (1, sample_rate)
    assert audio.audio_data.dtype == torch.float32


def test_simple_resample():
    """Test basic resampling functionality."""
    # Create simple audio
    sample_rate = 44100
    audio_data = torch.randn(1, sample_rate, dtype=torch.float32)  # 1 second
    audio = AudioSamples(audio_data, sample_rate)
    
    # Resample to half the rate
    resampled = resample_audio(audio, 22050)
    
    assert resampled.sample_rate == 22050
    assert resampled.audio_data.shape[0] == 1  # Same number of channels
    assert resampled.audio_data.shape[1] > 0   # Has some samples