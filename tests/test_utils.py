"""
Test the test utilities to make sure they work properly.
"""

import pytest
import torch
from tests.utils import generate_sine_wave_audio, assert_audio_properties


def test_generate_sine_wave_mono():
    """Test generating mono sine wave."""
    audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1)
    
    assert_audio_properties(audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)


def test_generate_sine_wave_stereo():
    """Test generating stereo sine wave."""
    audio = generate_sine_wave_audio(duration_sec=0.5, sample_rate=22050, channels=2)
    
    assert_audio_properties(audio, expected_sample_rate=22050, expected_channels=2, expected_duration=0.5)


def test_assert_audio_properties():
    """Test the assert_audio_properties utility function."""
    from voxlab.core.audio_samples import AudioSamples
    
    audio_data = torch.randn(2, 88200, dtype=torch.float32)  # 2 seconds stereo at 44100Hz
    audio = AudioSamples(audio_data, 44100)
    
    # This should pass
    assert_audio_properties(audio, expected_sample_rate=44100, expected_channels=2, expected_duration=2.0)
    
    # This should fail
    with pytest.raises(AssertionError):
        assert_audio_properties(audio, expected_sample_rate=22050)  # Wrong sample rate