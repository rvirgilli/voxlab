#!/usr/bin/env python3
"""
Comprehensive silence detection tests
Covers edge cases, long silences, and algorithm correctness
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import detect_silence, detect_nonsilent, remove_silence
from tests.utils import assert_audio_properties


class TestSilenceDetectionComprehensive:
    """Comprehensive silence detection test cases."""
    
    def test_long_silence_detection(self):
        """Test detection of long silences (12+ seconds)."""
        sample_rate = 16000
        
        # Create: 5s tone + 12s silence + 5s tone
        tone_5s = 0.2 * torch.sin(2 * torch.pi * 440 * torch.linspace(0, 5, 5 * sample_rate))
        silence_12s = torch.zeros(12 * sample_rate)
        audio_data = torch.cat([tone_5s, silence_12s, tone_5s]).unsqueeze(0)
        audio = AudioSamples(audio_data, sample_rate)
        
        # Test silence detection with parameters that should catch the 12s silence
        silent_ranges = detect_silence(audio, min_silence_len=5000, silence_thresh=-30)
        
        # Should find exactly one silence range around 12s long
        assert len(silent_ranges) == 1, f"Expected 1 silent range, got {len(silent_ranges)}"
        
        start_ms, end_ms = silent_ranges[0]
        silence_duration = end_ms - start_ms
        
        # Should be approximately 12 seconds (allow some tolerance for detection boundaries)
        expected_duration = 12000  # 12s in ms
        tolerance = 500  # 500ms tolerance
        assert abs(silence_duration - expected_duration) < tolerance, \
            f"Expected ~{expected_duration}ms silence, got {silence_duration}ms"
    
    def test_multiple_silence_patterns(self):
        """Test detection of multiple silences of varying lengths."""
        sample_rate = 16000
        
        # Create: 2s tone + 1s silence + 2s tone + 3s silence + 2s tone + 8s silence + 2s tone
        tone_2s = 0.15 * torch.sin(2 * torch.pi * 300 * torch.linspace(0, 2, 2 * sample_rate))
        silence_1s = torch.zeros(sample_rate)
        silence_3s = torch.zeros(3 * sample_rate)
        silence_8s = torch.zeros(8 * sample_rate)
        
        audio_data = torch.cat([
            tone_2s, silence_1s, tone_2s, silence_3s, tone_2s, silence_8s, tone_2s
        ]).unsqueeze(0)
        audio = AudioSamples(audio_data, sample_rate)
        
        # Test with 2s minimum silence length - should find 3s and 8s silences
        silent_ranges = detect_silence(audio, min_silence_len=2000, silence_thresh=-30)
        
        # Should find 2 silences (3s and 8s), not the 1s silence
        assert len(silent_ranges) == 2, f"Expected 2 silent ranges, got {len(silent_ranges)}"
        
        # Verify durations (allow tolerance for detection boundaries)
        durations = [end - start for start, end in silent_ranges]
        durations.sort()  # Sort to match expected order
        
        # Should be approximately 3s and 8s
        assert abs(durations[0] - 3000) < 300, f"Expected ~3000ms, got {durations[0]}ms"
        assert abs(durations[1] - 8000) < 300, f"Expected ~8000ms, got {durations[1]}ms"
    
    def test_variable_amplitude_silence_detection(self):
        """Test silence detection with variable amplitude audio."""
        sample_rate = 16000
        
        # Create: 3s loud tone + 2s quiet tone + 4s silence + 3s loud tone
        loud_tone = 0.4 * torch.sin(2 * torch.pi * 400 * torch.linspace(0, 3, 3 * sample_rate))
        quiet_tone = 0.05 * torch.sin(2 * torch.pi * 400 * torch.linspace(0, 2, 2 * sample_rate))
        silence_4s = torch.zeros(4 * sample_rate)
        
        audio_data = torch.cat([loud_tone, quiet_tone, silence_4s, loud_tone]).unsqueeze(0)
        audio = AudioSamples(audio_data, sample_rate)
        
        # Test with threshold that should detect silence but not quiet tone
        # Quiet tone is 0.05 amplitude = ~-26dB, so use -30dB threshold
        silent_ranges = detect_silence(audio, min_silence_len=2000, silence_thresh=-30)
        
        # Should find exactly one silence range (the 4s true silence)
        assert len(silent_ranges) == 1, f"Expected 1 silent range, got {len(silent_ranges)}"
        
        start_ms, end_ms = silent_ranges[0]
        silence_duration = end_ms - start_ms
        
        # Should be approximately 4 seconds
        assert abs(silence_duration - 4000) < 300, \
            f"Expected ~4000ms silence, got {silence_duration}ms"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU implementations produce identical results."""
        sample_rate = 16000
        
        # Create test audio with known silence pattern
        tone_3s = 0.3 * torch.sin(2 * torch.pi * 440 * torch.linspace(0, 3, 3 * sample_rate))
        silence_6s = torch.zeros(6 * sample_rate)
        audio_data = torch.cat([tone_3s, silence_6s, tone_3s]).unsqueeze(0)
        audio_cpu = AudioSamples(audio_data, sample_rate)
        audio_gpu = audio_cpu.cuda()
        
        # Test parameters
        params = {"min_silence_len": 2000, "silence_thresh": -30, "seek_step": 10}
        
        # Run on CPU and GPU
        cpu_result = detect_silence(audio_cpu, **params)
        gpu_result = detect_silence(audio_gpu, **params)
        
        # Results should be identical
        assert cpu_result == gpu_result, \
            f"CPU and GPU results differ: CPU={cpu_result}, GPU={gpu_result}"
        
        # Test nonsilent detection consistency
        cpu_nonsilent = detect_nonsilent(audio_cpu, **params)
        gpu_nonsilent = detect_nonsilent(audio_gpu, **params)
        
        assert cpu_nonsilent == gpu_nonsilent, \
            f"CPU and GPU nonsilent results differ: CPU={cpu_nonsilent}, GPU={gpu_nonsilent}"
    
    def test_silence_removal_preserves_content(self):
        """Test that silence removal preserves audio content properly."""
        sample_rate = 16000
        
        # Create audio with significant silence that should be removed
        tone_2s = 0.25 * torch.sin(2 * torch.pi * 440 * torch.linspace(0, 2, 2 * sample_rate))
        silence_10s = torch.zeros(10 * sample_rate)  # Long silence to be removed
        
        audio_data = torch.cat([tone_2s, silence_10s, tone_2s]).unsqueeze(0)
        original_audio = AudioSamples(audio_data, sample_rate)
        
        # Remove long silences (keep short silences)
        processed_audio = remove_silence(
            original_audio, 
            min_silence_len=5000,  # Remove silences longer than 5s
            silence_thresh=-25, 
            keep_silence=100,  # Keep 100ms of silence
            inplace=False
        )
        
        # Verify properties
        assert isinstance(processed_audio, AudioSamples)
        assert processed_audio.sample_rate == original_audio.sample_rate
        assert processed_audio.audio_data.shape[0] == original_audio.audio_data.shape[0]  # Same channels
        
        # Should be significantly shorter due to silence removal
        original_duration = original_audio.duration
        final_duration = processed_audio.duration
        reduction = original_duration - final_duration
        
        assert reduction > 8, f"Expected significant reduction (>8s), got {reduction:.1f}s"
        assert final_duration > 3, f"Final audio too short ({final_duration:.1f}s), content may be lost"
        
        # Check that audio content is preserved (amplitude should be similar)
        max_amp_orig = torch.max(torch.abs(original_audio.audio_data)).item()
        max_amp_final = torch.max(torch.abs(processed_audio.audio_data)).item()
        
        # Should preserve significant amplitude
        assert max_amp_final > max_amp_orig * 0.8, \
            f"Audio content may be corrupted (amplitude dropped from {max_amp_orig:.3f} to {max_amp_final:.3f})"
    
    def test_edge_case_very_short_audio(self):
        """Test silence detection on very short audio."""
        sample_rate = 16000
        
        # Create 100ms of random audio
        short_audio_data = torch.randn(1, int(0.1 * sample_rate)) * 0.1
        short_audio = AudioSamples(short_audio_data, sample_rate)
        
        # Should handle gracefully without errors
        result = detect_silence(short_audio, min_silence_len=50, silence_thresh=-20)
        assert isinstance(result, list)  # Should return a list, even if empty
    
    def test_edge_case_all_silence(self):
        """Test silence detection on completely silent audio."""
        sample_rate = 16000
        
        # Create 5 seconds of complete silence
        silence_data = torch.zeros(1, 5 * sample_rate)
        silence_audio = AudioSamples(silence_data, sample_rate)
        
        # Should detect one large silence range
        result = detect_silence(silence_audio, min_silence_len=1000, silence_thresh=-30)
        assert len(result) == 1, f"Expected 1 silence range for all-silent audio, got {len(result)}"
        
        start_ms, end_ms = result[0]
        duration = end_ms - start_ms
        
        # Should cover most of the audio (allow small tolerance for boundaries)
        expected_duration = 5000  # 5s
        assert abs(duration - expected_duration) < 200, \
            f"Expected ~{expected_duration}ms, got {duration}ms"
    
    def test_edge_case_no_silence(self):
        """Test silence detection on audio with no silence."""
        sample_rate = 16000
        
        # Create 3 seconds of continuous loud audio
        loud_audio_data = torch.randn(1, 3 * sample_rate) * 0.5
        loud_audio = AudioSamples(loud_audio_data, sample_rate)
        
        # Should find no silence with reasonable threshold
        result = detect_silence(loud_audio, min_silence_len=500, silence_thresh=-10)
        assert len(result) == 0, f"Expected no silence in loud audio, got {len(result)} ranges"