"""
Tests for individual preprocessing functions.
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import (
    convert_to_mono, remove_silence, break_into_chunks, normalize_audio_rms
)
from tests.utils import (
    generate_sine_wave_audio, generate_mixed_audio, generate_silence_audio,
    assert_audio_properties
)


class TestConvertToMono:
    """Test stereo to mono conversion with inplace functionality."""
    
    def test_convert_stereo_to_mono_left_inplace_true(self):
        """Test converting stereo to mono using left channel with inplace=True."""
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        original_left = stereo_audio.audio_data[0].clone()
        original_id = id(stereo_audio)
        
        mono_audio = convert_to_mono(stereo_audio, method='left', inplace=True)
        
        # Should return same object
        assert id(mono_audio) == original_id
        assert_audio_properties(mono_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        # Should equal the left channel of original
        assert torch.allclose(mono_audio.audio_data[0], original_left)
    
    def test_convert_stereo_to_mono_left_inplace_false(self):
        """Test converting stereo to mono using left channel with inplace=False."""
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        original_left = stereo_audio.audio_data[0].clone()
        original_id = id(stereo_audio)
        original_shape = stereo_audio.audio_data.shape
        
        mono_audio = convert_to_mono(stereo_audio, method='left', inplace=False)
        
        # Should return different object
        assert id(mono_audio) != original_id
        assert_audio_properties(mono_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        # Should equal the left channel of original
        assert torch.allclose(mono_audio.audio_data[0], original_left)
        # Original should be unchanged
        assert stereo_audio.audio_data.shape == original_shape
    
    def test_convert_stereo_to_mono_right_inplace_true(self):
        """Test converting stereo to mono using right channel with inplace=True."""
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        original_right = stereo_audio.audio_data[1].clone()
        original_id = id(stereo_audio)
        
        mono_audio = convert_to_mono(stereo_audio, method='right', inplace=True)
        
        # Should return same object
        assert id(mono_audio) == original_id
        assert_audio_properties(mono_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        # Should equal the right channel of original
        assert torch.allclose(mono_audio.audio_data[0], original_right)
    
    def test_convert_stereo_to_mono_right_inplace_false(self):
        """Test converting stereo to mono using right channel with inplace=False."""
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        original_right = stereo_audio.audio_data[1].clone()
        original_id = id(stereo_audio)
        original_shape = stereo_audio.audio_data.shape
        
        mono_audio = convert_to_mono(stereo_audio, method='right', inplace=False)
        
        # Should return different object
        assert id(mono_audio) != original_id
        assert_audio_properties(mono_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        # Should equal the right channel of original
        assert torch.allclose(mono_audio.audio_data[0], original_right)
        # Original should be unchanged
        assert stereo_audio.audio_data.shape == original_shape
    
    def test_convert_mono_to_mono(self):
        """Test converting mono audio with inplace=True (should return same object)."""
        mono_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        original_data = mono_audio.audio_data.clone()
        
        result_audio = convert_to_mono(mono_audio, method='left', inplace=True)
        
        # Should return same object since already mono and inplace=True
        assert result_audio is mono_audio
        assert_audio_properties(result_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        assert torch.equal(result_audio.audio_data, original_data)

    def test_convert_mono_to_mono_not_inplace(self):
        """Test converting mono audio with inplace=False (should return new object)."""
        mono_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        original_data = mono_audio.audio_data.clone()
        
        result_audio = convert_to_mono(mono_audio, method='left', inplace=False)
        
        # Should return different object since inplace=False
        assert result_audio is not mono_audio
        assert_audio_properties(result_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        assert torch.equal(result_audio.audio_data, original_data)
        # Original should remain unchanged
        assert torch.equal(mono_audio.audio_data, original_data)
    
    def test_convert_invalid_method(self):
        """Test converting with invalid method."""
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        
        with pytest.raises(ValueError, match="Unsupported method"):
            convert_to_mono(stereo_audio, method='mean')


class TestRemoveSilence:
    """Test silence removal functionality."""
    
    def test_remove_silence_basic(self):
        """Test basic silence removal with mixed audio."""
        # Create audio with signal-silence-signal pattern
        mixed_audio = generate_mixed_audio(duration_sec=3.0, sample_rate=44100, channels=1)
        
        processed_audio = remove_silence(mixed_audio, 
                                       silence_thresh=-30, 
                                       min_silence_len=500, 
                                       min_segment_len=500,
                                       fade_duration=10,
                                       silence_duration=10)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=44100, expected_channels=1)
        # Should be shorter than or equal to original due to silence removal
        assert processed_audio.audio_data.shape[1] <= mixed_audio.audio_data.shape[1]
    
    def test_remove_silence_all_signal(self):
        """Test silence removal on audio with no silence."""
        signal_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        processed_audio = remove_silence(signal_audio,
                                       silence_thresh=-30,
                                       min_silence_len=100,
                                       min_segment_len=100)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=44100, expected_channels=1)
    
    def test_remove_silence_all_silence(self):
        """Test silence removal on completely silent audio."""
        silence_audio = generate_silence_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        processed_audio = remove_silence(silence_audio,
                                       silence_thresh=-30,
                                       min_silence_len=100,
                                       min_segment_len=100)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=44100, expected_channels=1)
    
    def test_remove_silence_stereo(self):
        """Test silence removal on stereo audio."""
        mixed_audio = generate_mixed_audio(duration_sec=2.0, sample_rate=22050, channels=2)
        
        processed_audio = remove_silence(mixed_audio,
                                       silence_thresh=-30,
                                       min_silence_len=200,
                                       min_segment_len=200)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=22050, expected_channels=2)


class TestBreakIntoChunks:
    """Test audio chunking functionality."""
    
    def test_break_into_chunks_basic(self):
        """Test breaking audio into chunks."""
        # Create 10-second audio
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=44100, channels=2)
        
        chunks = break_into_chunks(audio, chunk_size=2000, fade_duration=50)  # 2-second chunks
        
        assert isinstance(chunks, list)
        assert len(chunks) == 5  # 10 seconds / 2 seconds per chunk
        
        for chunk in chunks:
            assert isinstance(chunk, AudioSamples)
            assert_audio_properties(chunk, expected_sample_rate=44100, expected_channels=2, expected_duration=2.0)
    
    def test_break_into_chunks_small_audio(self):
        """Test chunking audio smaller than chunk size."""
        # Create 0.5-second audio
        audio = generate_sine_wave_audio(duration_sec=0.5, sample_rate=44100, channels=1)
        
        chunks = break_into_chunks(audio, chunk_size=2000, fade_duration=50)  # 2-second chunks
        
        assert isinstance(chunks, list)
        assert len(chunks) == 0  # No chunks possible
    
    def test_break_into_chunks_exact_size(self):
        """Test chunking audio that exactly fits chunk size."""
        # Create 5-second audio
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=44100, channels=1)
        
        chunks = break_into_chunks(audio, chunk_size=5000, fade_duration=50)  # 5-second chunks
        
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert_audio_properties(chunks[0], expected_sample_rate=44100, expected_channels=1, expected_duration=5.0)
    
    def test_break_into_chunks_different_sizes(self):
        """Test different chunk sizes."""
        audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=44100, channels=1)
        
        # Test 1-second chunks
        chunks_1s = break_into_chunks(audio, chunk_size=1000, fade_duration=50)
        assert len(chunks_1s) == 6
        
        # Test 3-second chunks
        chunks_3s = break_into_chunks(audio, chunk_size=3000, fade_duration=50)
        assert len(chunks_3s) == 2
    
    def test_break_into_chunks_mono_and_stereo(self):
        """Test chunking both mono and stereo audio."""
        # Mono audio
        mono_audio = generate_sine_wave_audio(duration_sec=4.0, sample_rate=22050, channels=1)
        mono_chunks = break_into_chunks(mono_audio, chunk_size=1000, fade_duration=25)
        
        assert len(mono_chunks) == 4
        for chunk in mono_chunks:
            assert_audio_properties(chunk, expected_sample_rate=22050, expected_channels=1, expected_duration=1.0)
        
        # Stereo audio
        stereo_audio = generate_sine_wave_audio(duration_sec=4.0, sample_rate=22050, channels=2)
        stereo_chunks = break_into_chunks(stereo_audio, chunk_size=1000, fade_duration=25)
        
        assert len(stereo_chunks) == 4
        for chunk in stereo_chunks:
            assert_audio_properties(chunk, expected_sample_rate=22050, expected_channels=2, expected_duration=1.0)


class TestNormalizeAudioRms:
    """Test RMS normalization functionality with inplace support."""
    
    def test_normalize_audio_rms_basic_inplace_true(self):
        """Test basic RMS normalization with inplace=True."""
        # Create audio with known amplitude
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.1, channels=1)
        original_id = id(audio)
        original_data = audio.audio_data.clone()
        
        normalized_audio = normalize_audio_rms(audio, target_rms=-15, inplace=True)
        
        # Should return same object
        assert id(normalized_audio) == original_id
        assert isinstance(normalized_audio, AudioSamples)
        assert_audio_properties(normalized_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        
        # Check that RMS is closer to target
        normalized_rms = torch.sqrt(torch.mean(normalized_audio.audio_data ** 2))
        target_rms_linear = 10 ** (-15 / 20)
        assert abs(normalized_rms - target_rms_linear) < 0.01
        
        # Should be different from original
        assert not torch.equal(normalized_audio.audio_data, original_data)
    
    def test_normalize_audio_rms_basic_inplace_false(self):
        """Test basic RMS normalization with inplace=False."""
        # Create audio with known amplitude
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.1, channels=1)
        original_id = id(audio)
        original_data = audio.audio_data.clone()
        
        normalized_audio = normalize_audio_rms(audio, target_rms=-15, inplace=False)
        
        # Should return different object
        assert id(normalized_audio) != original_id
        assert isinstance(normalized_audio, AudioSamples)
        assert_audio_properties(normalized_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        
        # Check that RMS is closer to target
        normalized_rms = torch.sqrt(torch.mean(normalized_audio.audio_data ** 2))
        target_rms_linear = 10 ** (-15 / 20)
        assert abs(normalized_rms - target_rms_linear) < 0.01
        
        # Original should be unchanged
        assert torch.equal(audio.audio_data, original_data)
        # Results should be different from original
        assert not torch.equal(normalized_audio.audio_data, original_data)
    
    def test_normalize_audio_rms_stereo_inplace_true(self):
        """Test RMS normalization on stereo audio with inplace=True."""
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.2, channels=2)
        original_id = id(audio)
        original_data = audio.audio_data.clone()
        
        normalized_audio = normalize_audio_rms(audio, target_rms=-20, inplace=True)
        
        # Should return same object
        assert id(normalized_audio) == original_id
        assert isinstance(normalized_audio, AudioSamples)
        assert_audio_properties(normalized_audio, expected_sample_rate=44100, expected_channels=2, expected_duration=1.0)
        
        # Check that normalization was applied
        assert not torch.equal(normalized_audio.audio_data, original_data)
    
    def test_normalize_audio_rms_stereo_inplace_false(self):
        """Test RMS normalization on stereo audio with inplace=False."""
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.2, channels=2)
        original_id = id(audio)
        original_data = audio.audio_data.clone()
        
        normalized_audio = normalize_audio_rms(audio, target_rms=-20, inplace=False)
        
        # Should return different object
        assert id(normalized_audio) != original_id
        assert isinstance(normalized_audio, AudioSamples)
        assert_audio_properties(normalized_audio, expected_sample_rate=44100, expected_channels=2, expected_duration=1.0)
        
        # Original should be unchanged
        assert torch.equal(audio.audio_data, original_data)
        # Check that normalization was applied to result
        assert not torch.equal(normalized_audio.audio_data, original_data)
    
    def test_normalize_audio_rms_different_targets(self):
        """Test RMS normalization with different target levels."""
        audio = generate_sine_wave_audio(duration_sec=0.5, sample_rate=22050, amplitude=0.1, channels=1)
        
        # Test different target RMS levels
        targets = [-10, -15, -20, -25]
        
        for target in targets:
            normalized = normalize_audio_rms(audio, target_rms=target, inplace=False)
            assert isinstance(normalized, AudioSamples)
            assert_audio_properties(normalized, expected_sample_rate=22050, expected_channels=1, expected_duration=0.5)
    
    def test_normalize_audio_rms_silence(self):
        """Test RMS normalization on silent audio."""
        silence_audio = generate_silence_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        # Should handle silence gracefully (might result in inf or nan, but shouldn't crash)
        try:
            normalized_audio = normalize_audio_rms(silence_audio, target_rms=-15, inplace=False)
            assert isinstance(normalized_audio, AudioSamples)
            # If it succeeds, the shape should be preserved
            assert_audio_properties(normalized_audio, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
        except (ZeroDivisionError, RuntimeError):
            # It's acceptable for normalization to fail on silence
            pass
    
    def test_normalize_preserves_shape(self):
        """Test that normalization preserves audio shape."""
        original_audio = generate_sine_wave_audio(duration_sec=2.0, sample_rate=48000, amplitude=0.3, channels=2)
        
        normalized_audio = normalize_audio_rms(original_audio, target_rms=-12)
        
        # Shape should be identical
        assert normalized_audio.audio_data.shape == original_audio.audio_data.shape
        assert normalized_audio.sample_rate == original_audio.sample_rate