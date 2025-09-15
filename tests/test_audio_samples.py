"""
Tests for the AudioSamples core class.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from voxlab.core.audio_samples import AudioSamples
from tests.utils import generate_sine_wave_audio, assert_audio_properties


class TestAudioSamplesCreation:
    """Test AudioSamples object creation."""
    
    def test_create_from_tensor(self):
        """Test creating AudioSamples from torch tensor."""
        sample_rate = 44100
        audio_data = torch.randn(2, 44100, dtype=torch.float32)  # 1 second stereo
        
        audio = AudioSamples(audio_data, sample_rate)
        
        assert_audio_properties(audio, expected_sample_rate=sample_rate, expected_channels=2, expected_duration=1.0)
        assert torch.equal(audio.audio_data, audio_data)
    
    def test_create_mono(self):
        """Test creating mono AudioSamples."""
        sample_rate = 22050
        audio_data = torch.randn(1, 22050, dtype=torch.float32)  # 1 second mono
        
        audio = AudioSamples(audio_data, sample_rate)
        
        assert_audio_properties(audio, expected_sample_rate=sample_rate, expected_channels=1, expected_duration=1.0)
    
    def test_create_stereo(self):
        """Test creating stereo AudioSamples."""
        sample_rate = 48000
        audio_data = torch.randn(2, 48000, dtype=torch.float32)  # 1 second stereo
        
        audio = AudioSamples(audio_data, sample_rate)
        
        assert_audio_properties(audio, expected_sample_rate=sample_rate, expected_channels=2, expected_duration=1.0)


class TestAudioSamplesFileIO:
    """Test AudioSamples file loading and saving operations."""
    
    def test_export_and_load_wav(self):
        """Test exporting to WAV and loading back."""
        original_audio = generate_sine_wave_audio(duration_sec=0.5, channels=2)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                # Export audio
                original_audio.export(tmp_file.name, format='wav')
                
                # Load audio back
                loaded_audio = AudioSamples.load(tmp_file.name)
                
                # Check properties (note: loading might change some properties slightly)
                assert_audio_properties(loaded_audio, expected_sample_rate=original_audio.sample_rate, expected_channels=2)
                assert loaded_audio.audio_data.shape[1] > 0  # Has some samples
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(ValueError, match="Error loading audio file"):
            AudioSamples.load("/nonexistent/path/to/audio.wav")
    
    def test_export_formats(self):
        """Test exporting to different formats."""
        audio = generate_sine_wave_audio(duration_sec=0.1, channels=1)
        
        formats = ['wav', 'mp3', 'ogg', 'flac']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as tmp_file:
                try:
                    audio.export(tmp_file.name, format=fmt)
                    assert os.path.exists(tmp_file.name)
                    assert os.path.getsize(tmp_file.name) > 0
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_export_unsupported_format(self):
        """Test exporting to unsupported format."""
        audio = generate_sine_wave_audio(duration_sec=0.1)
        
        with tempfile.NamedTemporaryFile(suffix='.xyz') as tmp_file:
            with pytest.raises(ValueError, match="Unsupported export format"):
                audio.export(tmp_file.name, format='xyz')


class TestAudioSamplesConversion:
    """Test AudioSamples data conversion methods."""
    
    def test_to_numpy(self):
        """Test conversion to numpy array."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 44100)
        
        numpy_data = audio.to_numpy()
        
        import numpy as np
        assert isinstance(numpy_data, np.ndarray)  # numpy array
        assert numpy_data.shape == audio_data.shape
        assert torch.allclose(torch.from_numpy(numpy_data), audio_data)
    
    def test_to_tensor(self):
        """Test conversion to tensor (should return the same tensor)."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 44100)
        
        tensor_data = audio.to_tensor()
        
        assert isinstance(tensor_data, torch.Tensor)
        assert torch.equal(tensor_data, audio_data)


class TestAudioSamplesLoadBehavior:
    """Test AudioSamples.load() behavior with different audio formats."""
    
    def test_load_mono_to_stereo_conversion(self):
        """Test that mono audio gets converted to stereo."""
        # Create a mono audio file
        mono_audio = generate_sine_wave_audio(duration_sec=0.1, channels=1)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                mono_audio.export(tmp_file.name, format='wav')
                
                # Load it back - should be converted to stereo
                loaded_audio = AudioSamples.load(tmp_file.name)
                
                # Should now have 2 channels (converted from mono)
                assert loaded_audio.audio_data.shape[0] == 2
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_dtype_conversion(self):
        """Test that loaded audio is converted to float32."""
        audio = generate_sine_wave_audio(duration_sec=0.1, channels=2)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                audio.export(tmp_file.name, format='wav')
                loaded_audio = AudioSamples.load(tmp_file.name)
                
                assert loaded_audio.audio_data.dtype == torch.float32
                
            finally:
                os.unlink(tmp_file.name)