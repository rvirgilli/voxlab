"""
Tests for individual preprocessing functions.
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import (
    convert_to_mono, remove_silence, break_into_chunks, normalize_audio_rms, trim_audio
)
from tests.utils import (
    generate_sine_wave_audio, generate_mixed_audio, generate_silence_audio,
    assert_audio_properties
)


def calculate_expected_chunks(audio_duration_ms, chunk_duration_ms, mode, overlap_ms):
    """Calculate expected number of chunks for different modes."""
    if audio_duration_ms <= chunk_duration_ms:
        return 1
    
    if mode == 'min_overlap':
        # Min overlap: step_size = chunk_duration - min_overlap
        step_size = chunk_duration_ms - overlap_ms
        if step_size <= 0:
            step_size = max(1, chunk_duration_ms // 4)
        
        # Calculate chunks needed for full coverage
        remaining = audio_duration_ms - chunk_duration_ms
        return 1 + max(0, (remaining + step_size - 1) // step_size)
    
    elif mode == 'max_overlap':
        # Max overlap: step_size = chunk_duration - max_overlap  
        min_step = chunk_duration_ms - overlap_ms
        if min_step <= 0:
            return 1
        
        # Maximum chunks that fit
        remaining = audio_duration_ms - chunk_duration_ms
        return 1 + remaining // min_step
    
    elif mode == 'exact_count':
        return overlap_ms  # In this case, overlap_ms is actually chunk_count
    
    return 0


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
                                       _min_segment_len=500,
                                       _fade_duration=10,
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
                                       _min_segment_len=100)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=44100, expected_channels=1)
    
    def test_remove_silence_all_silence(self):
        """Test silence removal on completely silent audio."""
        silence_audio = generate_silence_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        processed_audio = remove_silence(silence_audio,
                                       silence_thresh=-30,
                                       min_silence_len=100,
                                       _min_segment_len=100)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=44100, expected_channels=1)
    
    def test_remove_silence_stereo(self):
        """Test silence removal on stereo audio."""
        mixed_audio = generate_mixed_audio(duration_sec=2.0, sample_rate=22050, channels=2)
        
        processed_audio = remove_silence(mixed_audio,
                                       silence_thresh=-30,
                                       min_silence_len=200,
                                       _min_segment_len=200)
        
        assert isinstance(processed_audio, AudioSamples)
        assert_audio_properties(processed_audio, expected_sample_rate=22050, expected_channels=2)


class TestBreakIntoChunks:
    """Test the new unified audio chunking functionality."""
    
    def test_break_into_chunks_exact_count_basic(self):
        """Test exact count mode with precise calculations."""
        # Create 10-second audio
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)
        
        # Test exact count: 3 chunks of 4s each  
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=3, chunk_duration=4000, fade_duration=50)
        expected_chunks = calculate_expected_chunks(10000, 4000, 'exact_count', 3)
        
        assert isinstance(chunks, list)
        assert len(chunks) == expected_chunks == 3
        
        # All chunks should be 4 seconds and evenly spaced from 0 to 10s
        for chunk in chunks:
            assert isinstance(chunk, AudioSamples)
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
    
    def test_break_into_chunks_min_overlap_your_case(self):
        """Test your specific use case: 11s audio, 4s chunks, 2s min overlap."""
        # Create 11-second audio
        audio = generate_sine_wave_audio(duration_sec=11.0, sample_rate=16000, channels=1)
        
        chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=4000, min_overlap=2000, fade_duration=50)
        expected_chunks = calculate_expected_chunks(11000, 4000, 'min_overlap', 2000)
        
        print(f"Expected: {expected_chunks}, Got: {len(chunks)}")  # Debug
        assert len(chunks) == expected_chunks == 5  # Should be 5 chunks
        
        # All chunks should be 4 seconds
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
    
    def test_break_into_chunks_min_overlap_calculations(self):
        """Test min overlap with different calculated scenarios."""
        test_cases = [
            # (audio_ms, chunk_ms, min_overlap_ms, expected_chunks)
            (8000, 3000, 1000, 3),  # 8s audio, 3s chunks, 1s overlap → step=2s → (8-3)/2+1 = 3.5 → 4 chunks
            (6000, 4000, 0, 2),     # 6s audio, 4s chunks, 0s overlap → step=4s → (6-4)/4+1 = 1.5 → 2 chunks  
            (12000, 5000, 2000, 3), # 12s audio, 5s chunks, 2s overlap → step=3s → (12-5)/3+1 = 3.33 → 4 chunks
        ]
        
        for audio_ms, chunk_ms, overlap_ms, expected in test_cases:
            audio = generate_sine_wave_audio(duration_sec=audio_ms/1000, sample_rate=16000, channels=1)
            chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=chunk_ms, min_overlap=overlap_ms)
            calculated_expected = calculate_expected_chunks(audio_ms, chunk_ms, 'min_overlap', overlap_ms)
            
            print(f"Audio: {audio_ms}ms, Chunk: {chunk_ms}ms, Overlap: {overlap_ms}ms")
            print(f"Expected: {calculated_expected}, Got: {len(chunks)}")
            
            assert len(chunks) == calculated_expected
    
    def test_break_into_chunks_max_overlap_calculations(self):
        """Test max overlap with precise calculations."""
        test_cases = [
            # (audio_ms, chunk_ms, max_overlap_ms, expected_chunks)
            (10000, 4000, 1000, 3),  # 10s audio, 4s chunks, max 1s overlap → step=3s → (10-4)/3+1 = 3 chunks
            (8000, 3000, 0, 2),      # 8s audio, 3s chunks, max 0s overlap → step=3s → (8-3)/3+1 = 2.66 → 2 chunks
            (12000, 4000, 2000, 5),  # 12s audio, 4s chunks, max 2s overlap → step=2s → (12-4)/2+1 = 5 chunks
        ]
        
        for audio_ms, chunk_ms, overlap_ms, _ in test_cases:
            audio = generate_sine_wave_audio(duration_sec=audio_ms/1000, sample_rate=16000, channels=1)
            chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=chunk_ms, max_overlap=overlap_ms)
            calculated_expected = calculate_expected_chunks(audio_ms, chunk_ms, 'max_overlap', overlap_ms)
            
            print(f"Audio: {audio_ms}ms, Chunk: {chunk_ms}ms, Max overlap: {overlap_ms}ms")
            print(f"Expected: {calculated_expected}, Got: {len(chunks)}")
            
            assert len(chunks) == calculated_expected
            
            # All chunks should be the expected duration
            for chunk in chunks:
                assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=chunk_ms/1000)
    
    def test_break_into_chunks_edge_cases(self):
        """Test edge cases for all modes."""
        # Short audio (shorter than chunk size)
        short_audio = generate_sine_wave_audio(duration_sec=2.0, sample_rate=16000, channels=1)
        
        # Test exact_count mode: should return exactly chunk_count copies of padded audio
        chunks_exact = break_into_chunks(short_audio, mode='exact_count', chunk_count=5, chunk_duration=4000)
        assert len(chunks_exact) == 5  # Exactly 5 chunks as requested
        for chunk in chunks_exact:
            assert abs(chunk.duration - 4.0) < 0.1  # Each chunk is 4.0s (padded)
        
        # Min/max overlap modes should return 1 chunk of padded audio
        chunks_min = break_into_chunks(short_audio, mode='min_overlap', chunk_duration=4000, min_overlap=1000)  
        chunks_max = break_into_chunks(short_audio, mode='max_overlap', chunk_duration=4000, max_overlap=1000)
        
        assert len(chunks_min) == 1  
        assert len(chunks_max) == 1
        
        # Verify min/max overlap return padded audio (4.0s, not original 2.0s)
        for chunks in [chunks_min, chunks_max]:
            assert abs(chunks[0].duration - 4.0) < 0.1  # Padded to chunk_duration
    
    def test_break_into_chunks_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        
        # Invalid mode
        with pytest.raises(ValueError, match="Unknown mode"):
            break_into_chunks(audio, mode='invalid_mode')
        
        # Missing required parameters for exact_count
        with pytest.raises(ValueError, match="requires 'chunk_count' and 'chunk_duration'"):
            break_into_chunks(audio, mode='exact_count', chunk_count=3)
        
        with pytest.raises(ValueError, match="requires 'chunk_count' and 'chunk_duration'"):
            break_into_chunks(audio, mode='exact_count', chunk_duration=4000)
    
    def test_break_into_chunks_stereo_audio(self):
        """Test all modes work with stereo audio."""
        stereo_audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=22050, channels=2)
        
        chunks_exact = break_into_chunks(stereo_audio, mode='exact_count', chunk_count=2, chunk_duration=3000)
        chunks_min = break_into_chunks(stereo_audio, mode='min_overlap', chunk_duration=3000, min_overlap=500)
        chunks_max = break_into_chunks(stereo_audio, mode='max_overlap', chunk_duration=3000, max_overlap=1000)
        
        # All should handle stereo correctly
        for chunks in [chunks_exact, chunks_min, chunks_max]:
            assert len(chunks) > 0
            for chunk in chunks:
                assert_audio_properties(chunk, expected_sample_rate=22050, expected_channels=2, expected_duration=3.0)


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


class TestTrimAudio:
    """Test the trim_audio function."""
    
    def test_trim_both_ends_basic(self):
        """Test trimming silence from both ends."""
        # Create audio with silence on both ends: silence + signal + silence
        silence_start = generate_silence_audio(duration_sec=0.5, sample_rate=44100, channels=1)
        signal = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.5, channels=1)
        silence_end = generate_silence_audio(duration_sec=0.3, sample_rate=44100, channels=1)
        
        # Concatenate: 0.5s silence + 1s signal + 0.3s silence = 1.8s total
        full_audio_data = torch.cat([silence_start.audio_data, signal.audio_data, silence_end.audio_data], dim=1)
        full_audio = AudioSamples(full_audio_data, 44100)
        
        # Trim both ends
        trimmed_audio = trim_audio(full_audio, silence_thresh=-40, mode='both', inplace=False)
        
        # Should be approximately 1 second (the signal part)
        assert_audio_properties(trimmed_audio, expected_sample_rate=44100, expected_channels=1)
        assert abs(trimmed_audio.duration - 1.0) < 0.1  # Allow some tolerance
        # Original should be unchanged
        assert abs(full_audio.duration - 1.8) < 0.1
    
    def test_trim_start_only(self):
        """Test trimming silence from start only."""
        # Create audio: silence + signal
        silence_start = generate_silence_audio(duration_sec=0.5, sample_rate=44100, channels=1)
        signal = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.5, channels=1)
        
        full_audio_data = torch.cat([silence_start.audio_data, signal.audio_data], dim=1)
        full_audio = AudioSamples(full_audio_data, 44100)
        
        # Trim start only
        trimmed_audio = trim_audio(full_audio, silence_thresh=-40, mode='start', inplace=False)
        
        # Should be approximately 1 second (removed start silence)
        assert_audio_properties(trimmed_audio, expected_sample_rate=44100, expected_channels=1)
        assert abs(trimmed_audio.duration - 1.0) < 0.1
    
    def test_trim_end_only(self):
        """Test trimming silence from end only."""
        # Create audio: signal + silence
        signal = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.5, channels=1)
        silence_end = generate_silence_audio(duration_sec=0.3, sample_rate=44100, channels=1)
        
        full_audio_data = torch.cat([signal.audio_data, silence_end.audio_data], dim=1)
        full_audio = AudioSamples(full_audio_data, 44100)
        
        # Trim end only
        trimmed_audio = trim_audio(full_audio, silence_thresh=-40, mode='end', inplace=False)
        
        # Should be approximately 1 second (removed end silence)
        assert_audio_properties(trimmed_audio, expected_sample_rate=44100, expected_channels=1)
        assert abs(trimmed_audio.duration - 1.0) < 0.1
    
    def test_trim_inplace_true(self):
        """Test trim_audio with inplace=True."""
        silence_start = generate_silence_audio(duration_sec=0.2, sample_rate=22050, channels=2)
        signal = generate_sine_wave_audio(duration_sec=0.5, sample_rate=22050, amplitude=0.5, channels=2)
        
        full_audio_data = torch.cat([silence_start.audio_data, signal.audio_data], dim=1)
        audio = AudioSamples(full_audio_data, 22050)
        original_id = id(audio)
        
        result = trim_audio(audio, silence_thresh=-40, mode='start', inplace=True)
        
        # Should return same object
        assert result is audio
        assert id(result) == original_id
        # Duration should be reduced
        assert abs(result.duration - 0.5) < 0.1
    
    def test_trim_inplace_false(self):
        """Test trim_audio with inplace=False."""
        silence_start = generate_silence_audio(duration_sec=0.2, sample_rate=22050, channels=2)
        signal = generate_sine_wave_audio(duration_sec=0.5, sample_rate=22050, amplitude=0.5, channels=2)
        
        full_audio_data = torch.cat([silence_start.audio_data, signal.audio_data], dim=1)
        audio = AudioSamples(full_audio_data, 22050)
        original_duration = audio.duration
        
        result = trim_audio(audio, silence_thresh=-40, mode='start', inplace=False)
        
        # Should return different object
        assert result is not audio
        # Original should be unchanged
        assert abs(audio.duration - original_duration) < 0.01
        # Result should be trimmed
        assert abs(result.duration - 0.5) < 0.1
    
    def test_trim_all_silence(self):
        """Test trimming audio that is all silence."""
        silence_audio = generate_silence_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        trimmed_audio = trim_audio(silence_audio, silence_thresh=-40, mode='both', inplace=False)
        
        # Should return minimal audio (1 sample)
        assert trimmed_audio.num_samples == 1
        assert_audio_properties(trimmed_audio, expected_sample_rate=44100, expected_channels=1)
    
    def test_trim_no_silence(self):
        """Test trimming audio with no silence."""
        signal_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, amplitude=0.5, channels=1)
        
        trimmed_audio = trim_audio(signal_audio, silence_thresh=-40, mode='both', inplace=False)
        
        # Should be mostly unchanged (might lose a few samples due to floating point precision)
        assert_audio_properties(trimmed_audio, expected_sample_rate=44100, expected_channels=1)
        assert abs(trimmed_audio.duration - 1.0) < 0.01  # Allow small tolerance
        # Should retain most of the audio
        assert trimmed_audio.num_samples >= signal_audio.num_samples - 10
    
    def test_trim_stereo_audio(self):
        """Test trimming stereo audio."""
        silence_start = generate_silence_audio(duration_sec=0.3, sample_rate=48000, channels=2)
        signal = generate_sine_wave_audio(duration_sec=1.0, sample_rate=48000, amplitude=0.5, channels=2)
        silence_end = generate_silence_audio(duration_sec=0.2, sample_rate=48000, channels=2)
        
        full_audio_data = torch.cat([silence_start.audio_data, signal.audio_data, silence_end.audio_data], dim=1)
        full_audio = AudioSamples(full_audio_data, 48000)
        
        trimmed_audio = trim_audio(full_audio, silence_thresh=-40, mode='both', inplace=False)
        
        # Should preserve stereo and remove silence
        assert_audio_properties(trimmed_audio, expected_sample_rate=48000, expected_channels=2)
        assert abs(trimmed_audio.duration - 1.0) < 0.1
    
    def test_trim_invalid_mode(self):
        """Test trim_audio with invalid mode."""
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1)
        
        with pytest.raises(ValueError, match="Invalid mode 'invalid'"):
            trim_audio(audio, mode='invalid')
    
    def test_trim_different_thresholds(self):
        """Test trim_audio with different silence thresholds."""
        # Create audio with very quiet signal on ends
        quiet_start = generate_sine_wave_audio(duration_sec=0.2, sample_rate=44100, amplitude=0.001, channels=1)  # Very quiet
        loud_signal = generate_sine_wave_audio(duration_sec=0.5, sample_rate=44100, amplitude=0.5, channels=1)   # Loud
        quiet_end = generate_sine_wave_audio(duration_sec=0.2, sample_rate=44100, amplitude=0.001, channels=1)   # Very quiet
        
        full_audio_data = torch.cat([quiet_start.audio_data, loud_signal.audio_data, quiet_end.audio_data], dim=1)
        audio = AudioSamples(full_audio_data, 44100)
        
        # With strict threshold (-60dB), should keep quiet parts
        trimmed_strict = trim_audio(audio, silence_thresh=-60, mode='both', inplace=False)
        
        # With loose threshold (-20dB), should remove quiet parts  
        trimmed_loose = trim_audio(audio, silence_thresh=-20, mode='both', inplace=False)
        
        # Loose threshold should result in shorter audio
        assert trimmed_loose.duration < trimmed_strict.duration


class TestBreakIntoChunksExactCount:
    """Test exact_count mode with comprehensive edge cases."""
    
    def test_exact_count_basic_cases(self):
        """Test basic exact count functionality with calculated expectations."""
        test_cases = [
            # (audio_sec, chunk_count, chunk_duration_ms, expected_chunks)
            (10.0, 3, 4000, 3),  # 10s audio, 3 chunks of 4s each
            (8.0, 2, 5000, 2),   # 8s audio, 2 chunks of 5s each  
            (12.0, 4, 3000, 4),  # 12s audio, 4 chunks of 3s each
        ]
        
        for audio_sec, chunk_count, chunk_duration_ms, expected in test_cases:
            audio = generate_sine_wave_audio(duration_sec=audio_sec, sample_rate=16000, channels=1)
            chunks = break_into_chunks(audio, mode='exact_count', 
                                     chunk_count=chunk_count, 
                                     chunk_duration=chunk_duration_ms, 
                                     fade_duration=50)
            
            assert len(chunks) == expected == chunk_count
            
            # All chunks should have the specified duration
            for chunk in chunks:
                expected_duration = chunk_duration_ms / 1000
                assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=expected_duration)
    
    def test_exact_count_single_chunk(self):
        """Test exact count with single chunk (chunk_count=1)."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=1, chunk_duration=3000)
        
        assert len(chunks) == 1
        # Single chunk should be 3s and centered in the 8s audio
        assert_audio_properties(chunks[0], expected_sample_rate=16000, expected_channels=1, expected_duration=3.0)
    
    def test_exact_count_chunk_longer_than_audio(self):
        """Test exact count when chunk duration > audio duration."""
        # 3s audio with 5s chunk duration
        audio = generate_sine_wave_audio(duration_sec=3.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=2, chunk_duration=5000)
        
        assert len(chunks) == 2
        # Each chunk should still be the requested duration, with padding/overlap as needed
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=5.0)
    
    def test_exact_count_zero_chunks(self):
        """Test exact count with chunk_count=0."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=0, chunk_duration=2000)
        
        assert len(chunks) == 0
    
    def test_exact_count_stereo(self):
        """Test exact count with stereo audio."""
        audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=22050, channels=2)
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=3, chunk_duration=2000)
        
        assert len(chunks) == 3
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=22050, expected_channels=2, expected_duration=2.0)


class TestBreakIntoChunksMinOverlap:
    """Test min_overlap mode with comprehensive edge cases."""
    
    def test_min_overlap_your_specific_case(self):
        """Test your specific use case: 11s audio, 4s chunks, 2s min overlap."""
        audio = generate_sine_wave_audio(duration_sec=11.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=4000, min_overlap=2000)
        
        # Calculate expected: step_size = 4000 - 2000 = 2000ms
        # remaining = 11000 - 4000 = 7000ms  
        # chunks = 1 + ceil(7000/2000) = 1 + 4 = 5 chunks
        expected_chunks = calculate_expected_chunks(11000, 4000, 'min_overlap', 2000)
        assert len(chunks) == expected_chunks == 5
        
        # All chunks should be 4 seconds
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
    
    def test_min_overlap_calculated_cases(self):
        """Test min overlap with precisely calculated expectations."""
        test_cases = [
            # (audio_ms, chunk_ms, min_overlap_ms, expected_chunks)
            (8000, 3000, 1000, 4),   # step=2000, remaining=5000, chunks=1+ceil(5000/2000)=4
            (6000, 4000, 0, 2),      # step=4000, remaining=2000, chunks=1+ceil(2000/4000)=2  
            (9000, 2000, 500, 6),    # Even spacing with 600ms overlap (>500ms min)
            (12000, 5000, 2000, 4),  # step=3000, remaining=7000, chunks=1+ceil(7000/3000)=4
        ]
        
        for audio_ms, chunk_ms, overlap_ms, expected in test_cases:
            audio = generate_sine_wave_audio(duration_sec=audio_ms/1000, sample_rate=16000, channels=1)
            chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=chunk_ms, min_overlap=overlap_ms)
            calculated_expected = calculate_expected_chunks(audio_ms, chunk_ms, 'min_overlap', overlap_ms)
            
            assert len(chunks) == calculated_expected == expected
            
            # All chunks should have the specified duration
            for chunk in chunks:
                assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=chunk_ms/1000)
    
    def test_min_overlap_zero_overlap(self):
        """Test min overlap with 0 overlap (still ensures full coverage)."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=3000, min_overlap=0)
        
        # With 0 min overlap, should still create overlapping chunks for full coverage
        # step_size = 3000, remaining = 7000, chunks = 1 + ceil(7000/3000) = 4
        expected = calculate_expected_chunks(10000, 3000, 'min_overlap', 0)
        assert len(chunks) == expected
    
    def test_min_overlap_exceeds_chunk_duration(self):
        """Test min overlap when overlap >= chunk duration."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=3000, min_overlap=3500)
        
        # Should still work with heavily overlapping chunks
        assert len(chunks) >= 2  # Should create multiple chunks
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=3.0)
    
    def test_min_overlap_short_audio(self):
        """Test min overlap with audio shorter than chunk duration."""
        audio = generate_sine_wave_audio(duration_sec=2.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=5000, min_overlap=1000)
        
        assert len(chunks) == 1
        assert_audio_properties(chunks[0], expected_sample_rate=16000, expected_channels=1, expected_duration=5.0)  # Padded to chunk_duration


class TestBreakIntoChunksMaxOverlap:
    """Test max_overlap mode with comprehensive edge cases."""
    
    def test_max_overlap_calculated_cases(self):
        """Test max overlap with precisely calculated expectations."""  
        test_cases = [
            # (audio_ms, chunk_ms, max_overlap_ms, expected_chunks)
            (10000, 4000, 1000, 3),  # step=3000, remaining=6000, chunks=1+floor(6000/3000)=3
            (8000, 3000, 0, 2),      # step=3000, remaining=5000, chunks=1+floor(5000/3000)=2
            (12000, 4000, 2000, 5),  # step=2000, remaining=8000, chunks=1+floor(8000/2000)=5
            (15000, 5000, 1500, 3),  # step=3500, remaining=10000, chunks=1+floor(10000/3500)=3
        ]
        
        for audio_ms, chunk_ms, overlap_ms, expected in test_cases:
            audio = generate_sine_wave_audio(duration_sec=audio_ms/1000, sample_rate=16000, channels=1)
            chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=chunk_ms, max_overlap=overlap_ms)
            calculated_expected = calculate_expected_chunks(audio_ms, chunk_ms, 'max_overlap', overlap_ms)
            
            assert len(chunks) == calculated_expected == expected
            
            # All chunks should have the specified duration
            for chunk in chunks:
                assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=chunk_ms/1000)
    
    def test_max_overlap_zero_overlap(self):
        """Test max overlap with 0 overlap (non-overlapping chunks)."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=4000, max_overlap=0)
        
        # step = 4000, remaining = 6000, chunks = 1 + floor(6000/4000) = 2 
        expected = calculate_expected_chunks(10000, 4000, 'max_overlap', 0)
        assert len(chunks) == expected == 2
    
    def test_max_overlap_exceeds_chunk_duration(self):
        """Test max overlap when overlap >= chunk duration."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=3000, max_overlap=3500)
        
        # Should return only 1 chunk when max_overlap >= chunk_duration
        assert len(chunks) == 1
        assert_audio_properties(chunks[0], expected_sample_rate=16000, expected_channels=1, expected_duration=3.0)
    
    def test_max_overlap_short_audio(self):
        """Test max overlap with audio shorter than chunk duration."""
        audio = generate_sine_wave_audio(duration_sec=2.0, sample_rate=16000, channels=1)
        chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=5000, max_overlap=1000)
        
        assert len(chunks) == 1
        assert_audio_properties(chunks[0], expected_sample_rate=16000, expected_channels=1, expected_duration=5.0)  # Padded to chunk_duration


class TestBreakIntoChunksReturnTimings:
    """Test return_timings functionality and verify spacing/overlap properties."""
    
    def test_exact_count_even_spacing_and_coverage(self):
        """Test exact_count has even spacing and full coverage."""
        audio = generate_sine_wave_audio(duration_sec=12.0, sample_rate=16000, channels=1)
        chunks, timings = break_into_chunks(audio, mode='exact_count', chunk_count=4, chunk_duration=3000, return_timings=True)
        
        assert len(chunks) == len(timings) == 4
        
        # All chunks should be exactly 3 seconds
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=3.0)
        
        # First chunk starts at 0, last ends at 12
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[-1][1] - 12.0) < 0.01
        
        # Check even spacing - calculate step size between chunk starts
        starts = [timing[0] for timing in timings]
        steps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        
        # All steps should be equal (even spacing)
        for i in range(1, len(steps)):
            assert abs(steps[i] - steps[0]) < 0.01, f"Uneven spacing: steps = {steps}"
    
    def test_min_overlap_constraints_and_coverage(self):
        """Test min_overlap respects minimum overlap and has full coverage."""
        audio = generate_sine_wave_audio(duration_sec=11.0, sample_rate=16000, channels=1)  
        chunks, timings = break_into_chunks(audio, mode='min_overlap', chunk_duration=4000, min_overlap=2000, return_timings=True)
        
        assert len(chunks) == 5
        
        # All chunks should be exactly 4 seconds
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
        
        # Full coverage: first starts at 0, last ends at 11
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[-1][1] - 11.0) < 0.01
        
        # Check minimum overlap constraint
        for i in range(1, len(timings)):
            prev_end = timings[i-1][1]
            curr_start = timings[i][0]
            actual_overlap = prev_end - curr_start
            assert actual_overlap >= 2.0 - 0.01, f"Overlap {actual_overlap:.3f}s < minimum 2.0s between chunks {i-1} and {i}"
        
        # Check even spacing of chunk starts
        starts = [timing[0] for timing in timings]
        steps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        
        # All steps should be equal (even spacing)
        for i in range(1, len(steps)):
            assert abs(steps[i] - steps[0]) < 0.01, f"Uneven spacing in min_overlap: steps = {steps}"
    
    def test_max_overlap_constraints_and_coverage(self):
        """Test max_overlap respects maximum overlap constraint."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)
        chunks, timings = break_into_chunks(audio, mode='max_overlap', chunk_duration=4000, max_overlap=1000, return_timings=True)
        
        assert len(chunks) == 3  # Should create 3 chunks with this configuration
        
        # All chunks should be exactly 4 seconds
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
        
        # First starts at 0, last ends at 10
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[-1][1] - 10.0) < 0.01
        
        # Check maximum overlap constraint
        for i in range(1, len(timings)):
            prev_end = timings[i-1][1]
            curr_start = timings[i][0]
            actual_overlap = prev_end - curr_start
            assert actual_overlap <= 1.0 + 0.01, f"Overlap {actual_overlap:.3f}s > maximum 1.0s between chunks {i-1} and {i}"
        
        # Check even spacing of chunk starts
        starts = [timing[0] for timing in timings]
        steps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        
        # All steps should be equal (even spacing)
        for i in range(1, len(steps)):
            assert abs(steps[i] - steps[0]) < 0.01, f"Uneven spacing in max_overlap: steps = {steps}"
    
    def test_zero_overlap_max_overlap(self):
        """Test max_overlap with 0 overlap produces non-overlapping chunks."""
        audio = generate_sine_wave_audio(duration_sec=9.0, sample_rate=16000, channels=1)
        chunks, timings = break_into_chunks(audio, mode='max_overlap', chunk_duration=3000, max_overlap=0, return_timings=True)
        
        assert len(chunks) == 3  # Should fit 3 non-overlapping 3s chunks in 9s
        
        # Check no overlap (actually gap between chunks)
        for i in range(1, len(timings)):
            prev_end = timings[i-1][1]
            curr_start = timings[i][0]
            gap = curr_start - prev_end
            assert gap >= -0.01, f"Overlap detected: gap = {gap:.3f}s (should be >= 0)"
    
    def test_your_specific_case_with_timing_verification(self):
        """Test your specific case: 11s audio, 4s chunks, 2s min overlap with detailed verification."""
        audio = generate_sine_wave_audio(duration_sec=11.0, sample_rate=16000, channels=1)
        chunks, timings = break_into_chunks(audio, mode='min_overlap', chunk_duration=4000, min_overlap=2000, return_timings=True)
        
        # Should create exactly 5 chunks
        assert len(chunks) == 5
        
        # Print timing details for verification
        print("\\nYour specific case - Timing verification:")
        for i, (start, end) in enumerate(timings):
            overlap = 0 if i == 0 else max(0, timings[i-1][1] - start)
            print(f"  Chunk {i+1}: {start:.1f}s to {end:.1f}s (overlap: {overlap:.1f}s)")
        
        # Verify properties:
        # 1. All chunks are exactly 4s
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=4.0)
        
        # 2. Full coverage: 0 to 11s
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[-1][1] - 11.0) < 0.01
        
        # 3. Minimum overlap of 2s maintained
        for i in range(1, len(timings)):
            prev_end = timings[i-1][1]
            curr_start = timings[i][0]
            actual_overlap = prev_end - curr_start
            assert actual_overlap >= 2.0 - 0.01, f"Chunk {i}: overlap {actual_overlap:.3f}s < minimum 2.0s"
        
        # 4. Even spacing
        starts = [timing[0] for timing in timings]
        steps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        for i in range(1, len(steps)):
            assert abs(steps[i] - steps[0]) < 0.01, f"Uneven spacing: {steps}"
    
    def test_return_timings_false_default(self):
        """Test that return_timings=False returns only chunks."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)
        result = break_into_chunks(audio, mode='exact_count', chunk_count=2, chunk_duration=3000)
        
        # Should return list, not tuple
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(chunk, AudioSamples) for chunk in result)


class TestBreakIntoChunksEdgeCases:
    """Test edge cases and error conditions for all modes."""
    
    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        
        with pytest.raises(ValueError, match="Unknown mode"):
            break_into_chunks(audio, mode='invalid_mode')
    
    def test_exact_count_missing_parameters(self):
        """Test error handling for missing exact_count parameters."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        
        with pytest.raises(ValueError, match="requires 'chunk_count' and 'chunk_duration'"):
            break_into_chunks(audio, mode='exact_count', chunk_count=3)
        
        with pytest.raises(ValueError, match="requires 'chunk_count' and 'chunk_duration'"):
            break_into_chunks(audio, mode='exact_count', chunk_duration=4000)
        
        with pytest.raises(ValueError, match="requires 'chunk_count' and 'chunk_duration'"):
            break_into_chunks(audio, mode='exact_count')
    
    def test_very_short_audio(self):
        """Test all modes with extremely short audio."""
        # 0.1 second audio
        audio = generate_sine_wave_audio(duration_sec=0.1, sample_rate=16000, channels=1)
        
        # exact_count mode will still create the requested number of chunks
        chunks_exact = break_into_chunks(audio, mode='exact_count', chunk_count=5, chunk_duration=2000)
        assert len(chunks_exact) == 5  # Should create exactly 5 chunks as requested
        # Each chunk will be padded to chunk_duration
        for chunk in chunks_exact:
            assert abs(chunk.duration - 2.0) < 0.01  # Padded to chunk_duration
        
        # min/max_overlap modes should return single chunk for short audio
        chunks_min = break_into_chunks(audio, mode='min_overlap', chunk_duration=2000, min_overlap=500)
        chunks_max = break_into_chunks(audio, mode='max_overlap', chunk_duration=2000, max_overlap=500)
        
        assert len(chunks_min) == 1
        assert len(chunks_max) == 1
        
        # Should be padded to chunk_duration
        assert abs(chunks_min[0].duration - 2.0) < 0.01  # Padded to chunk_duration
        assert abs(chunks_max[0].duration - 2.0) < 0.01  # Padded to chunk_duration
    
    def test_very_large_chunk_count(self):
        """Test exact_count with very large chunk_count."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        
        # Request 100 chunks of 1s each from 5s audio - creates heavy overlap
        chunks = break_into_chunks(audio, mode='exact_count', chunk_count=100, chunk_duration=1000)
        
        assert len(chunks) == 100
        
        # All chunks should be exactly 1s (now that we fixed the implementation)
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=16000, expected_channels=1, expected_duration=1.0)
    
    def test_different_sample_rates(self):
        """Test all modes work with different sample rates."""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for sr in sample_rates:
            audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=sr, channels=1)
            
            chunks_exact = break_into_chunks(audio, mode='exact_count', chunk_count=2, chunk_duration=2000)
            chunks_min = break_into_chunks(audio, mode='min_overlap', chunk_duration=2000, min_overlap=500)
            chunks_max = break_into_chunks(audio, mode='max_overlap', chunk_duration=2000, max_overlap=500)
            
            # All should work and preserve sample rate
            for chunks in [chunks_exact, chunks_min, chunks_max]:
                assert len(chunks) > 0
                for chunk in chunks:
                    assert chunk.sample_rate == sr
    
    def test_mono_and_stereo_consistency(self):
        """Test that mono and stereo audio produce consistent chunk counts."""
        # Same duration and parameters for mono and stereo
        mono_audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)
        stereo_audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=2)
        
        for mode_params in [
            {'mode': 'exact_count', 'chunk_count': 3, 'chunk_duration': 3000},
            {'mode': 'min_overlap', 'chunk_duration': 3000, 'min_overlap': 1000},
            {'mode': 'max_overlap', 'chunk_duration': 3000, 'max_overlap': 1500},
        ]:
            mono_chunks = break_into_chunks(mono_audio, **mode_params)
            stereo_chunks = break_into_chunks(stereo_audio, **mode_params)
            
            # Should produce same number of chunks
            assert len(mono_chunks) == len(stereo_chunks)
            
            # Verify channel counts
            for chunk in mono_chunks:
                assert chunk.audio_data.shape[0] == 1
            for chunk in stereo_chunks:
                assert chunk.audio_data.shape[0] == 2


class TestBreakIntoChunksSplitByTime:
    """Test split_by_time mode with comprehensive cases."""

    def test_split_by_time_basic(self):
        """Test basic split by time with no overlap."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # Split at 3s and 7s: creates chunks [0-3s], [3-7s], [7-10s]
        chunks = break_into_chunks(audio, mode='split_by_time', split_points=[3000, 7000], overlap=0)

        assert len(chunks) == 3
        assert isinstance(chunks[0], AudioSamples)

        # Verify chunk durations approximately match expected
        assert abs(chunks[0].duration - 3.0) < 0.1  # 0 to 3s
        assert abs(chunks[1].duration - 4.0) < 0.1  # 3 to 7s
        assert abs(chunks[2].duration - 3.0) < 0.1  # 7 to 10s

    def test_split_by_time_with_overlap(self):
        """Test split by time with 1s overlap."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # Split at 3s and 7s with 1s overlap
        # Expected: [0-3.5s], [2.5-7.5s], [6.5-10s]
        chunks = break_into_chunks(audio, mode='split_by_time', split_points=[3000, 7000], overlap=1000)

        assert len(chunks) == 3

        # Verify chunk durations with overlap
        assert abs(chunks[0].duration - 3.5) < 0.1  # Extended by 0.5s
        assert abs(chunks[1].duration - 5.0) < 0.1  # 2.5 to 7.5s
        assert abs(chunks[2].duration - 3.5) < 0.1  # Extended by 0.5s

    def test_split_by_time_with_timings(self):
        """Test split by time returns correct timing information."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)

        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[3000, 5000],
                                           overlap=1000,
                                           return_timings=True)

        assert len(chunks) == len(timings) == 3

        # Verify timing boundaries with 1s overlap (500ms on each side)
        # Chunk 0: 0 to 3.5s (split at 3s + 500ms)
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[0][1] - 3.5) < 0.01

        # Chunk 1: 2.5s to 5.5s (split at 3s - 500ms to 5s + 500ms)
        assert abs(timings[1][0] - 2.5) < 0.01
        assert abs(timings[1][1] - 5.5) < 0.01

        # Chunk 2: 4.5s to 8s (split at 5s - 500ms to end)
        assert abs(timings[2][0] - 4.5) < 0.01
        assert abs(timings[2][1] - 8.0) < 0.01

        # Verify actual overlap between chunks
        overlap_1_2 = timings[0][1] - timings[1][0]
        overlap_2_3 = timings[1][1] - timings[2][0]
        assert abs(overlap_1_2 - 1.0) < 0.01  # 1s overlap
        assert abs(overlap_2_3 - 1.0) < 0.01  # 1s overlap

    def test_split_by_time_single_split(self):
        """Test split by time with single split point."""
        audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=16000, channels=1)

        chunks = break_into_chunks(audio, mode='split_by_time', split_points=[3000], overlap=0)

        assert len(chunks) == 2
        assert abs(chunks[0].duration - 3.0) < 0.1  # 0 to 3s
        assert abs(chunks[1].duration - 3.0) < 0.1  # 3 to 6s

    def test_split_by_time_multiple_splits(self):
        """Test split by time with many split points."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # Split every 2 seconds
        chunks = break_into_chunks(audio, mode='split_by_time',
                                  split_points=[2000, 4000, 6000, 8000],
                                  overlap=0)

        assert len(chunks) == 5  # 5 segments from 4 split points

        # Each chunk should be approximately 2 seconds
        for chunk in chunks:
            assert abs(chunk.duration - 2.0) < 0.1

    def test_split_by_time_unsorted_splits(self):
        """Test that split points are sorted automatically."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # Provide unsorted split points
        chunks = break_into_chunks(audio, mode='split_by_time',
                                  split_points=[7000, 3000, 5000],
                                  overlap=0)

        assert len(chunks) == 4
        # Should handle sorting internally

    def test_split_by_time_duplicate_splits(self):
        """Test that duplicate split points are removed."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=16000, channels=1)

        # Provide duplicate split points
        chunks = break_into_chunks(audio, mode='split_by_time',
                                  split_points=[3000, 3000, 5000, 5000],
                                  overlap=0)

        assert len(chunks) == 3  # Only 2 unique splits = 3 chunks

    def test_split_by_time_overlap_at_boundaries(self):
        """Test that overlap doesn't extend beyond audio boundaries."""
        audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=16000, channels=1)

        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[3000],
                                           overlap=2000,
                                           return_timings=True)

        # First chunk should start at 0, not negative
        assert timings[0][0] >= 0.0

        # Last chunk should end at 6s, not beyond
        assert timings[-1][1] <= 6.0 + 0.01

    def test_split_by_time_large_overlap(self):
        """Test split by time with very large overlap."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # 4s overlap on 2s segments = heavy overlap
        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[2000, 4000, 6000, 8000],
                                           overlap=4000,
                                           return_timings=True)

        assert len(chunks) == 5

        # Verify chunks have significant overlap
        for i in range(1, len(timings)):
            overlap = timings[i-1][1] - timings[i][0]
            assert overlap >= 4.0 - 0.1  # Should have at least 4s overlap (minus tolerance)

    def test_split_by_time_stereo(self):
        """Test split by time with stereo audio."""
        audio = generate_sine_wave_audio(duration_sec=8.0, sample_rate=22050, channels=2)

        chunks = break_into_chunks(audio, mode='split_by_time',
                                  split_points=[3000, 6000],
                                  overlap=500)

        assert len(chunks) == 3

        # All chunks should be stereo
        for chunk in chunks:
            assert_audio_properties(chunk, expected_sample_rate=22050, expected_channels=2)

    def test_split_by_time_invalid_split_points(self):
        """Test error handling for invalid split points."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        # Split point beyond audio duration
        with pytest.raises(ValueError, match="outside audio range"):
            break_into_chunks(audio, mode='split_by_time', split_points=[6000], overlap=0)

        # Negative split point
        with pytest.raises(ValueError, match="outside audio range"):
            break_into_chunks(audio, mode='split_by_time', split_points=[-1000, 3000], overlap=0)

    def test_split_by_time_missing_split_points(self):
        """Test error handling for missing split_points parameter."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="requires 'split_points' parameter"):
            break_into_chunks(audio, mode='split_by_time', overlap=1000)

    def test_split_by_time_invalid_split_points_type(self):
        """Test error handling for invalid split_points type."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        # split_points must be a list or tuple
        with pytest.raises(ValueError, match="must be a list or tuple"):
            break_into_chunks(audio, mode='split_by_time', split_points=3000, overlap=0)

    def test_split_by_time_negative_overlap(self):
        """Test error handling for negative overlap."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="overlap must be non-negative"):
            break_into_chunks(audio, mode='split_by_time', split_points=[2500], overlap=-1000)

    def test_split_by_time_empty_split_points(self):
        """Test split by time with empty split points list."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        # Empty list should return single chunk (entire audio)
        chunks = break_into_chunks(audio, mode='split_by_time', split_points=[], overlap=0)

        assert len(chunks) == 1
        assert abs(chunks[0].duration - 5.0) < 0.1

    def test_split_by_time_device_preservation(self):
        """Test that split by time preserves device placement."""
        audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=16000, channels=1)

        # Test on CPU (GPU test would require CUDA)
        chunks = break_into_chunks(audio, mode='split_by_time',
                                  split_points=[2000, 4000],
                                  overlap=500)

        # All chunks should be on same device as original
        for chunk in chunks:
            assert chunk.device == audio.device

    def test_split_by_time_different_sample_rates(self):
        """Test split by time with different sample rates."""
        sample_rates = [8000, 16000, 22050, 44100, 48000]

        for sr in sample_rates:
            audio = generate_sine_wave_audio(duration_sec=6.0, sample_rate=sr, channels=1)

            chunks = break_into_chunks(audio, mode='split_by_time',
                                      split_points=[2000, 4000],
                                      overlap=500)

            assert len(chunks) == 3
            for chunk in chunks:
                assert chunk.sample_rate == sr

    def test_split_by_time_example_case(self):
        """Test the example case from documentation: 0-3s, 3-5s with 1s overlap."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[3000],
                                           overlap=1000,
                                           return_timings=True)

        assert len(chunks) == 2

        # First chunk: 0 to 3.5s (extended 500ms past split point)
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[0][1] - 3.5) < 0.01

        # Second chunk: 2.5s to 5s (started 500ms before split point)
        assert abs(timings[1][0] - 2.5) < 0.01
        assert abs(timings[1][1] - 5.0) < 0.01

        # Verify 1s overlap
        overlap = timings[0][1] - timings[1][0]
        assert abs(overlap - 1.0) < 0.01

    def test_split_by_time_zero_overlap(self):
        """Test split by time with explicitly zero overlap."""
        audio = generate_sine_wave_audio(duration_sec=9.0, sample_rate=16000, channels=1)

        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[3000, 6000],
                                           overlap=0,
                                           return_timings=True)

        assert len(chunks) == 3

        # With 0 overlap, chunks should be exactly adjacent
        assert abs(timings[0][1] - timings[1][0]) < 0.01  # No gap/overlap
        assert abs(timings[1][1] - timings[2][0]) < 0.01  # No gap/overlap

    def test_split_by_time_odd_overlap(self):
        """Test split by time with odd overlap value (tests integer division)."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        # 1001ms overlap = 500ms on each side (floor division)
        chunks, timings = break_into_chunks(audio, mode='split_by_time',
                                           split_points=[5000],
                                           overlap=1001,
                                           return_timings=True)

        assert len(chunks) == 2

        # Verify overlap is approximately 1s (due to floor division of 1001/2 = 500)
        overlap = timings[0][1] - timings[1][0]
        assert abs(overlap - 1.0) < 0.05  # Allow some tolerance for rounding

    def test_extract_intervals_basic(self):
        """Test basic extract_intervals functionality."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        intervals = [[1000, 3000], [5000, 8000]]
        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

        assert len(chunks) == 2
        assert chunks[0].duration == 2.0
        assert chunks[1].duration == 3.0
        assert all(isinstance(c, AudioSamples) for c in chunks)

    def test_extract_intervals_with_timings(self):
        """Test extract_intervals with return_timings=True."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        intervals = [[1000, 3000], [5000, 7000]]
        chunks, timings = break_into_chunks(audio, mode='extract_intervals',
                                           intervals=intervals,
                                           return_timings=True)

        assert len(chunks) == 2
        assert len(timings) == 2
        assert abs(timings[0][0] - 1.0) < 0.01
        assert abs(timings[0][1] - 3.0) < 0.01
        assert abs(timings[1][0] - 5.0) < 0.01
        assert abs(timings[1][1] - 7.0) < 0.01

    def test_extract_intervals_non_contiguous(self):
        """Test extracting non-contiguous intervals (with gaps)."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        intervals = [[500, 1000], [3000, 4000], [7000, 9000]]
        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

        assert len(chunks) == 3
        assert abs(chunks[0].duration - 0.5) < 0.01
        assert abs(chunks[1].duration - 1.0) < 0.01
        assert abs(chunks[2].duration - 2.0) < 0.01

    def test_extract_intervals_overlapping(self):
        """Test extracting overlapping intervals."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        intervals = [[1000, 4000], [3000, 6000], [5000, 8000]]
        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

        assert len(chunks) == 3
        assert all(isinstance(c, AudioSamples) for c in chunks)
        assert chunks[0].duration == 3.0
        assert chunks[1].duration == 3.0
        assert chunks[2].duration == 3.0

    def test_extract_intervals_single_interval(self):
        """Test extracting a single interval."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=[[2000, 5000]])

        assert len(chunks) == 1
        assert abs(chunks[0].duration - 3.0) < 0.01

    def test_extract_intervals_empty_list(self):
        """Test extract_intervals with empty intervals list."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=[])

        assert len(chunks) == 0
        assert isinstance(chunks, list)

    def test_extract_intervals_exceeds_audio_end(self):
        """Test extract_intervals with interval exceeding audio length."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        intervals = [[3000, 7000]]

        with pytest.raises(ValueError, match="exceeds audio length"):
            break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

    def test_extract_intervals_start_beyond_audio(self):
        """Test extract_intervals with interval starting beyond audio."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        intervals = [[6000, 8000]]

        with pytest.raises(ValueError, match="is beyond audio length"):
            break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

    def test_extract_intervals_invalid_interval_format(self):
        """Test extract_intervals with invalid interval format."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="must be a \\[start_ms, end_ms\\] pair"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[[1000, 2000, 3000]])

        with pytest.raises(ValueError, match="must be a \\[start_ms, end_ms\\] pair"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[[1000]])

        with pytest.raises(ValueError, match="must be a \\[start_ms, end_ms\\] pair"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[1000, 2000])

    def test_extract_intervals_negative_start(self):
        """Test extract_intervals with negative start time."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="cannot be negative"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[[-500, 2000]])

    def test_extract_intervals_invalid_range(self):
        """Test extract_intervals with end <= start."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="must be greater than start time"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[[3000, 2000]])

        with pytest.raises(ValueError, match="must be greater than start time"):
            break_into_chunks(audio, mode='extract_intervals', intervals=[[2000, 2000]])

    def test_extract_intervals_missing_parameter(self):
        """Test extract_intervals without intervals parameter."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="requires 'intervals' parameter"):
            break_into_chunks(audio, mode='extract_intervals')

    def test_extract_intervals_invalid_type(self):
        """Test extract_intervals with non-list intervals."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        with pytest.raises(ValueError, match="must be a list or tuple"):
            break_into_chunks(audio, mode='extract_intervals', intervals="not a list")

    def test_extract_intervals_stereo(self):
        """Test extract_intervals with stereo audio."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=2)

        intervals = [[1000, 3000], [5000, 7000]]
        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

        assert len(chunks) == 2
        assert chunks[0].audio_data.shape[0] == 2
        assert chunks[1].audio_data.shape[0] == 2
        assert chunks[0].duration == 2.0
        assert chunks[1].duration == 2.0

    def test_extract_intervals_device_preservation(self):
        """Test extract_intervals preserves device."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)
        original_device = audio.device

        intervals = [[1000, 3000], [4000, 5000]]
        chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

        assert all(c.device == original_device for c in chunks)

    def test_extract_intervals_different_sample_rates(self):
        """Test extract_intervals with different sample rates."""
        for sample_rate in [8000, 16000, 44100]:
            audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=sample_rate, channels=1)

            intervals = [[500, 1500], [2000, 3000]]
            chunks = break_into_chunks(audio, mode='extract_intervals', intervals=intervals)

            assert len(chunks) == 2
            assert all(c.sample_rate == sample_rate for c in chunks)
            assert abs(chunks[0].duration - 1.0) < 0.01
            assert abs(chunks[1].duration - 1.0) < 0.01

    def test_extract_intervals_precise_boundaries(self):
        """Test extract_intervals respects exact boundaries."""
        audio = generate_sine_wave_audio(duration_sec=10.0, sample_rate=16000, channels=1)

        intervals = [[0, 1000], [1000, 2000], [9000, 10000]]
        chunks, timings = break_into_chunks(audio, mode='extract_intervals',
                                           intervals=intervals,
                                           return_timings=True)

        assert len(chunks) == 3
        assert abs(timings[0][0] - 0.0) < 0.01
        assert abs(timings[0][1] - 1.0) < 0.01
        assert abs(timings[1][0] - 1.0) < 0.01
        assert abs(timings[1][1] - 2.0) < 0.01
        assert abs(timings[2][0] - 9.0) < 0.01
        assert abs(timings[2][1] - 10.0) < 0.01

    def test_extract_intervals_custom_fade_duration(self):
        """Test extract_intervals with custom fade duration."""
        audio = generate_sine_wave_audio(duration_sec=5.0, sample_rate=16000, channels=1)

        intervals = [[1000, 3000]]
        chunks = break_into_chunks(audio, mode='extract_intervals',
                                  intervals=intervals,
                                  fade_duration=100)

        assert len(chunks) == 1
        assert abs(chunks[0].duration - 2.0) < 0.01


