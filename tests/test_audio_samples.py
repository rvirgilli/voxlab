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


class TestAudioSamplesProperties:
    """Test AudioSamples properties."""
    
    def test_duration_property(self):
        """Test duration property calculation."""
        # Test various durations and sample rates
        audio_1s = AudioSamples(torch.randn(2, 44100), 44100)
        assert audio_1s.duration == 1.0
        
        audio_2s = AudioSamples(torch.randn(1, 88200), 44100)
        assert audio_2s.duration == 2.0
        
        audio_half = AudioSamples(torch.randn(2, 22050), 44100)
        assert audio_half.duration == 0.5
    
    def test_num_samples_property(self):
        """Test num_samples property."""
        audio = AudioSamples(torch.randn(2, 12345), 44100)
        assert audio.num_samples == 12345
        
        audio_mono = AudioSamples(torch.randn(1, 1000), 22050)
        assert audio_mono.num_samples == 1000
    
    def test_channels_property(self):
        """Test channels property."""
        mono_audio = AudioSamples(torch.randn(1, 1000), 44100)
        assert mono_audio.channels == 1
        
        stereo_audio = AudioSamples(torch.randn(2, 1000), 44100)
        assert stereo_audio.channels == 2
    
    def test_shape_property(self):
        """Test shape property."""
        audio = AudioSamples(torch.randn(2, 5000), 44100)
        assert audio.shape == (2, 5000)
        
        mono_audio = AudioSamples(torch.randn(1, 3000), 22050)
        assert mono_audio.shape == (1, 3000)
    
    def test_dtype_property(self):
        """Test dtype property."""
        audio_float32 = AudioSamples(torch.randn(2, 1000, dtype=torch.float32), 44100)
        assert audio_float32.dtype == torch.float32
        
        audio_float64 = AudioSamples(torch.randn(2, 1000, dtype=torch.float64), 44100)
        assert audio_float64.dtype == torch.float64
    
    def test_is_mono_property(self):
        """Test is_mono property."""
        mono_audio = AudioSamples(torch.randn(1, 1000), 44100)
        assert mono_audio.is_mono == True
        assert mono_audio.is_stereo == False
        
        stereo_audio = AudioSamples(torch.randn(2, 1000), 44100)
        assert stereo_audio.is_mono == False
        assert stereo_audio.is_stereo == True
    
    def test_is_stereo_property(self):
        """Test is_stereo property."""
        stereo_audio = AudioSamples(torch.randn(2, 1000), 44100)
        assert stereo_audio.is_stereo == True
        assert stereo_audio.is_mono == False
        
        mono_audio = AudioSamples(torch.randn(1, 1000), 44100)
        assert mono_audio.is_stereo == False
        assert mono_audio.is_mono == True


class TestAudioSamplesRepr:
    """Test AudioSamples string representation."""
    
    def test_repr_mono(self):
        """Test __repr__ for mono audio."""
        audio = AudioSamples(torch.randn(1, 22050), 22050)
        repr_str = repr(audio)
        
        assert "mono" in repr_str
        assert "22050Hz" in repr_str
        assert "1.000s" in repr_str
        assert "22,050 samples" in repr_str
        assert "device=" in repr_str
    
    def test_repr_stereo(self):
        """Test __repr__ for stereo audio."""
        audio = AudioSamples(torch.randn(2, 44100), 44100)
        repr_str = repr(audio)
        
        assert "2-channel" in repr_str
        assert "44100Hz" in repr_str
        assert "1.000s" in repr_str
        assert "44,100 samples" in repr_str
        assert "device=" in repr_str
    
    def test_repr_different_durations(self):
        """Test __repr__ with different durations."""
        # Test 0.5 seconds
        audio_half = AudioSamples(torch.randn(1, 11025), 22050)
        repr_str = repr(audio_half)
        assert "0.500s" in repr_str
        
        # Test 2.5 seconds
        audio_long = AudioSamples(torch.randn(2, 110250), 44100)
        repr_str = repr(audio_long)
        assert "2.500s" in repr_str


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