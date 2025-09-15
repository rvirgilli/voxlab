"""
Test utilities for VoxLab test suite.
Provides synthetic audio generation and common fixtures using AudioSamples.
"""

import torch
import numpy as np
from voxlab.core.audio_samples import AudioSamples


def generate_sine_wave_audio(duration_sec=1.0, frequency=440, sample_rate=44100, amplitude=0.5, channels=1):
    """
    Generate a sine wave AudioSamples object.
    
    Parameters:
    duration_sec (float): Duration in seconds
    frequency (int): Frequency in Hz
    sample_rate (int): Sample rate in Hz
    amplitude (float): Amplitude (0.0 to 1.0)
    channels (int): Number of channels (1 for mono, 2 for stereo)
    
    Returns:
    AudioSamples: Generated sine wave audio
    """
    num_samples = int(sample_rate * duration_sec)
    t = torch.linspace(0, duration_sec, num_samples, dtype=torch.float32)
    
    # Generate sine wave
    sine_wave = amplitude * torch.sin(2 * torch.pi * frequency * t)
    
    # Add channel dimension and duplicate for stereo if needed
    if channels == 1:
        audio_data = sine_wave.unsqueeze(0)  # Shape: (1, num_samples)
    elif channels == 2:
        audio_data = torch.stack([sine_wave, sine_wave], dim=0)  # Shape: (2, num_samples)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")
    
    return AudioSamples(audio_data, sample_rate)


def generate_white_noise_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.1, channels=1):
    """
    Generate white noise AudioSamples object.
    
    Parameters:
    duration_sec (float): Duration in seconds
    sample_rate (int): Sample rate in Hz
    amplitude (float): Amplitude (0.0 to 1.0)
    channels (int): Number of channels
    
    Returns:
    AudioSamples: Generated white noise audio
    """
    num_samples = int(sample_rate * duration_sec)
    
    # Generate white noise
    noise = amplitude * torch.randn(channels, num_samples, dtype=torch.float32)
    
    return AudioSamples(noise, sample_rate)


def generate_silence_audio(duration_sec=1.0, sample_rate=44100, channels=1):
    """
    Generate silent AudioSamples object.
    
    Parameters:
    duration_sec (float): Duration in seconds
    sample_rate (int): Sample rate in Hz
    channels (int): Number of channels
    
    Returns:
    AudioSamples: Silent audio
    """
    num_samples = int(sample_rate * duration_sec)
    audio_data = torch.zeros(channels, num_samples, dtype=torch.float32)
    
    return AudioSamples(audio_data, sample_rate)


def generate_mixed_audio(duration_sec=2.0, sample_rate=44100, channels=1):
    """
    Generate audio with sine wave + silence + sine wave pattern.
    Useful for testing silence removal and segmentation.
    
    Parameters:
    duration_sec (float): Total duration in seconds
    sample_rate (int): Sample rate in Hz
    channels (int): Number of channels
    
    Returns:
    AudioSamples: Mixed audio with signal and silence
    """
    segment_duration = duration_sec / 3
    
    # Create three segments: sine, silence, sine
    sine1 = generate_sine_wave_audio(segment_duration, 440, sample_rate, 0.5, channels)
    silence = generate_silence_audio(segment_duration, sample_rate, channels)
    sine2 = generate_sine_wave_audio(segment_duration, 880, sample_rate, 0.5, channels)
    
    # Concatenate along time axis
    combined_audio = torch.cat([sine1.audio_data, silence.audio_data, sine2.audio_data], dim=1)
    
    return AudioSamples(combined_audio, sample_rate)


def assert_audio_properties(audio, expected_sample_rate=None, expected_channels=None, expected_duration=None, tolerance=0.01):
    """
    Assert that AudioSamples object has expected properties.
    
    Parameters:
    audio (AudioSamples): Audio to check
    expected_sample_rate (int): Expected sample rate
    expected_channels (int): Expected number of channels
    expected_duration (float): Expected duration in seconds
    tolerance (float): Tolerance for duration comparison
    """
    assert isinstance(audio, AudioSamples), f"Expected AudioSamples, got {type(audio)}"
    assert isinstance(audio.audio_data, torch.Tensor), f"Expected torch.Tensor, got {type(audio.audio_data)}"
    assert audio.audio_data.dtype == torch.float32, f"Expected float32, got {audio.audio_data.dtype}"
    
    if expected_sample_rate is not None:
        assert audio.sample_rate == expected_sample_rate, f"Expected sample rate {expected_sample_rate}, got {audio.sample_rate}"
    
    if expected_channels is not None:
        assert audio.audio_data.shape[0] == expected_channels, f"Expected {expected_channels} channels, got {audio.audio_data.shape[0]}"
    
    if expected_duration is not None:
        actual_duration = audio.audio_data.shape[1] / audio.sample_rate
        assert abs(actual_duration - expected_duration) < tolerance, f"Expected duration {expected_duration}s, got {actual_duration}s"


def create_test_audio_file(file_path, audio_samples):
    """
    Create a temporary audio file for testing file I/O operations.
    
    Parameters:
    file_path (str): Path to save the audio file
    audio_samples (AudioSamples): Audio to save
    """
    audio_samples.export(file_path)