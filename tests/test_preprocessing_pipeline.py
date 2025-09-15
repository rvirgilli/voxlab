"""
Tests for the PreprocessingPipeline class.
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.pipeline import PreprocessingPipeline
from voxlab.preprocessing.functions import (
    resample_audio, convert_to_mono, remove_silence, 
    break_into_chunks, normalize_audio_rms
)
from tests.utils import generate_sine_wave_audio, assert_audio_properties


class TestPreprocessingPipeline:
    """Test the PreprocessingPipeline class functionality."""
    
    def test_empty_pipeline(self):
        """Test pipeline with no steps."""
        pipeline = PreprocessingPipeline()
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        
        result = pipeline.process(audio)
        
        # Should return the same audio unchanged
        assert isinstance(result, AudioSamples)
        assert torch.equal(result.audio_data, audio.audio_data)
        assert result.sample_rate == audio.sample_rate
    
    def test_single_step_pipeline(self):
        """Test pipeline with a single step."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(convert_to_mono, method='left')
        
        stereo_audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2)
        result = pipeline.process(stereo_audio)
        
        assert isinstance(result, AudioSamples)
        assert_audio_properties(result, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
    
    def test_multiple_steps_pipeline(self):
        """Test pipeline with multiple steps."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=22050)
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(normalize_audio_rms, target_rms=-15)
        
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=2, amplitude=0.1)
        result = pipeline.process(audio)
        
        assert isinstance(result, AudioSamples)
        assert_audio_properties(result, expected_sample_rate=22050, expected_channels=1, expected_duration=1.0)
        
        # Check that normalization was applied
        rms = torch.sqrt(torch.mean(result.audio_data ** 2))
        target_rms_linear = 10 ** (-15 / 20)
        assert abs(rms - target_rms_linear) < 0.01
    
    def test_break_into_chunks_as_last_step(self):
        """Test that break_into_chunks can be added as the last step."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=44100)
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(break_into_chunks, chunk_size=1000, fade_duration=50)  # 1-second chunks
        
        audio = generate_sine_wave_audio(duration_sec=3.0, sample_rate=44100, channels=2)
        result = pipeline.process(audio)
        
        # Result should be a list of AudioSamples (not a single AudioSamples)
        assert isinstance(result, list)
        assert len(result) == 3  # 3 seconds -> 3 chunks of 1 second each
        
        for chunk in result:
            assert isinstance(chunk, AudioSamples)
            assert_audio_properties(chunk, expected_sample_rate=44100, expected_channels=1, expected_duration=1.0)
    
    def test_break_into_chunks_only_once(self):
        """Test that break_into_chunks can only be added once."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(break_into_chunks, chunk_size=1000)
        
        # Should raise error when trying to add it again
        with pytest.raises(ValueError, match="break_into_chunks can only be added once"):
            pipeline.add_step(break_into_chunks, chunk_size=2000)
    
    def test_no_steps_after_break_into_chunks(self):
        """Test that no steps can be added after break_into_chunks."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(break_into_chunks, chunk_size=1000)
        
        # Should raise error when trying to add any step after break_into_chunks
        with pytest.raises(ValueError, match="No steps can be added after break_into_chunks"):
            pipeline.add_step(convert_to_mono, method='left')
    
    def test_break_into_chunks_must_be_last(self):
        """Test that break_into_chunks must be the last step."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(break_into_chunks, chunk_size=1000)
        
        # Should raise error when trying to add any step after break_into_chunks
        with pytest.raises(ValueError, match="No steps can be added after break_into_chunks"):
            pipeline.add_step(normalize_audio_rms, target_rms=-15)
    
    def test_pipeline_with_step_specific_kwargs(self):
        """Test pipeline where steps have specific parameters overridden during processing."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=22050)  # Default rate
        pipeline.add_step(normalize_audio_rms, target_rms=-20)    # Default target
        
        audio = generate_sine_wave_audio(duration_sec=1.0, sample_rate=44100, channels=1, amplitude=0.1)
        
        # Override parameters during processing
        result = pipeline.process(audio, 
                                resample_audio={'new_sample_rate': 48000},  # Override sample rate
                                normalize_audio_rms={'target_rms': -10})     # Override target RMS
        
        assert isinstance(result, AudioSamples)
        assert result.sample_rate == 48000  # Should use overridden value
        
        # Check that normalization used overridden target
        rms = torch.sqrt(torch.mean(result.audio_data ** 2))
        target_rms_linear = 10 ** (-10 / 20)  # -10 dB instead of -20 dB
        assert abs(rms - target_rms_linear) < 0.01


class TestPreprocessingPipelineEdgeCases:
    """Test edge cases and error conditions for PreprocessingPipeline."""
    
    def test_pipeline_with_invalid_function(self):
        """Test that pipeline handles invalid functions gracefully."""
        pipeline = PreprocessingPipeline()
        
        # This should work (adding a valid function)
        pipeline.add_step(convert_to_mono, method='left')
        
        # Note: We don't test invalid functions here as the current implementation
        # doesn't validate function types at add_step time
    
    def test_pipeline_preserves_audio_properties(self):
        """Test that pipeline preserves expected audio properties through multiple steps."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(normalize_audio_rms, target_rms=-15)
        pipeline.add_step(resample_audio, new_sample_rate=22050)
        
        original_audio = generate_sine_wave_audio(duration_sec=2.0, sample_rate=44100, channels=1, amplitude=0.05)
        result = pipeline.process(original_audio)
        
        assert isinstance(result, AudioSamples)
        assert result.sample_rate == 22050
        assert result.audio_data.shape[0] == 1  # Still mono
        # Duration should be approximately preserved
        expected_samples = int(2.0 * 22050)
        assert abs(result.audio_data.shape[1] - expected_samples) < 100
    
    def test_empty_audio_through_pipeline(self):
        """Test pipeline with very short/empty audio."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(normalize_audio_rms, target_rms=-15)
        
        # Very short audio
        short_audio_data = torch.randn(2, 10, dtype=torch.float32) * 0.1
        short_audio = AudioSamples(short_audio_data, 44100)
        
        # Should not crash
        result = pipeline.process(short_audio)
        assert isinstance(result, AudioSamples)
        assert result.audio_data.shape[0] == 1  # Converted to mono