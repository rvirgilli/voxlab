"""
Tests for device-aware functionality in VoxLab.

Tests GPU/CUDA device placement, device preservation through pipelines,
and device-aware tensor operations.
"""

import pytest
import torch
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import (
    resample_audio, convert_to_mono, normalize_audio_rms, break_into_chunks
)
from voxlab.preprocessing.pipeline import PreprocessingPipeline
from .utils import generate_white_noise_audio


class TestDeviceAwareness:
    """Test device awareness for AudioSamples class."""
    
    def test_device_property(self):
        """Test that AudioSamples correctly reports its device."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 16000)
        
        assert audio.device.type == 'cpu'
        assert isinstance(audio.device, torch.device)
    
    def test_cpu_methods(self):
        """Test .cpu() method."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 16000)
        
        cpu_audio = audio.cpu()
        assert cpu_audio.device.type == 'cpu'
        assert cpu_audio is not audio  # Should return new instance
        assert torch.equal(cpu_audio.audio_data, audio.audio_data)
        assert cpu_audio.sample_rate == audio.sample_rate
    
    def test_to_method(self):
        """Test .to(device) method."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 16000)
        
        # Test .to('cpu')
        cpu_audio = audio.to('cpu')
        assert cpu_audio.device.type == 'cpu'
        assert cpu_audio is not audio  # Should return new instance
    
    def test_cuda_methods(self):
        """Test CUDA device methods (or CPU fallback if CUDA unavailable)."""
        audio_data = torch.randn(2, 1000, dtype=torch.float32)
        audio = AudioSamples(audio_data, 16000)
        
        if torch.cuda.is_available():
            # Test .cuda() with actual CUDA
            cuda_audio = audio.cuda()
            assert cuda_audio.device.type == 'cuda'
            assert cuda_audio is not audio  # Should return new instance
            assert cuda_audio.sample_rate == audio.sample_rate
            
            # Test .to('cuda')
            cuda_audio2 = audio.to('cuda:0')
            assert cuda_audio2.device.type == 'cuda'
            assert cuda_audio2 is not audio
        else:
            # Test that methods exist and work (will stay on CPU)
            try:
                cuda_audio = audio.cuda()
                # Should either work (on CPU) or raise a helpful error
                assert cuda_audio.sample_rate == audio.sample_rate
            except RuntimeError as e:
                # Expected behavior when CUDA not available
                assert "CUDA" in str(e) or "cuda" in str(e)


class TestPreprocessingDevicePreservation:
    """Test that preprocessing functions preserve device placement."""
    
    def test_cpu_device_preservation(self):
        """Test that CPU operations preserve CPU device."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        assert audio.device.type == 'cpu'
        
        # Test resample_audio
        resampled = resample_audio(audio, 16000, inplace=False)
        assert resampled.device.type == 'cpu'
        assert resampled.sample_rate == 16000
        
        # Test convert_to_mono
        mono = convert_to_mono(audio, method='left', inplace=False)
        assert mono.device.type == 'cpu'
        assert mono.audio_data.shape[0] == 1
        
        # Test normalize_audio_rms
        normalized = normalize_audio_rms(audio, target_rms=-20, inplace=False)
        assert normalized.device.type == 'cpu'
    
    def test_gpu_device_preservation(self):
        """Test that GPU operations preserve GPU device (or CPU fallback)."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        
        if torch.cuda.is_available():
            # Test with actual GPU
            gpu_audio = audio.cuda()
            assert gpu_audio.device.type == 'cuda'
            expected_device = 'cuda'
        else:
            # Test with CPU (GPU methods should gracefully handle this)
            gpu_audio = audio  # Stay on CPU
            expected_device = 'cpu'
        
        # Test resample_audio preserves device
        resampled = resample_audio(gpu_audio, 16000, inplace=False)
        assert resampled.device.type == expected_device
        assert resampled.sample_rate == 16000
        
        # Test convert_to_mono preserves device
        mono = convert_to_mono(gpu_audio, method='left', inplace=False)
        assert mono.device.type == expected_device
        assert mono.audio_data.shape[0] == 1
        
        # Test normalize_audio_rms preserves device
        normalized = normalize_audio_rms(gpu_audio, target_rms=-20, inplace=False)
        assert normalized.device.type == expected_device
        
        # Test break_into_chunks preserves device for all chunks
        chunks = break_into_chunks(gpu_audio, mode='exact_count', chunk_count=2, chunk_duration=500, inplace=False)
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.device.type == expected_device, f"Chunk {i} should be on {expected_device}"
    
    def test_inplace_operations(self):
        """Test that in-place operations preserve the same object."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        original_id = id(audio)
        
        # Test in-place operations return same object
        result = resample_audio(audio, 16000, inplace=True)
        assert id(result) == original_id
        assert result.sample_rate == 16000
        
        result = convert_to_mono(audio, method='left', inplace=True)
        assert id(result) == original_id
        assert result.audio_data.shape[0] == 1
        
        result = normalize_audio_rms(audio, target_rms=-15, inplace=True)
        assert id(result) == original_id
    
    def test_not_inplace_operations(self):
        """Test that non-in-place operations create new objects."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        original_id = id(audio)
        original_sample_rate = audio.sample_rate
        original_shape = audio.audio_data.shape
        
        # Test non-in-place operations create new objects
        result = resample_audio(audio, 16000, inplace=False)
        assert id(result) != original_id
        assert result.sample_rate == 16000
        assert audio.sample_rate == original_sample_rate  # Original unchanged
        
        result = convert_to_mono(audio, method='left', inplace=False)
        assert id(result) != original_id
        assert result.audio_data.shape[0] == 1
        assert audio.audio_data.shape == original_shape  # Original unchanged


class TestPipelineDeviceAwareness:
    """Test that preprocessing pipelines preserve device placement."""
    
    def test_cpu_pipeline_device_preservation(self):
        """Test CPU pipeline preserves CPU device throughout."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        assert audio.device.type == 'cpu'
        
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=16000)
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(normalize_audio_rms, target_rms=-20)
        
        result = pipeline.process(audio)
        assert result.device.type == 'cpu'
        assert result.sample_rate == 16000
        assert result.audio_data.shape[0] == 1
    
    def test_gpu_pipeline_device_preservation(self):
        """Test GPU pipeline preserves GPU device throughout (or CPU fallback)."""
        audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        
        if torch.cuda.is_available():
            gpu_audio = audio.cuda()
            assert gpu_audio.device.type == 'cuda'
            expected_device = 'cuda'
        else:
            gpu_audio = audio  # Stay on CPU
            expected_device = 'cpu'
        
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=16000)
        pipeline.add_step(convert_to_mono, method='left')
        pipeline.add_step(normalize_audio_rms, target_rms=-20)
        
        result = pipeline.process(gpu_audio)
        assert result.device.type == expected_device
        assert result.sample_rate == 16000
        assert result.audio_data.shape[0] == 1
    
    def test_pipeline_with_chunking_device_preservation(self):
        """Test pipeline with chunking preserves device for all chunks."""
        audio = generate_white_noise_audio(duration_sec=2.0, sample_rate=48000, channels=2)
        
        if torch.cuda.is_available():
            gpu_audio = audio.cuda()
            expected_device = 'cuda'
        else:
            gpu_audio = audio  # Stay on CPU
            expected_device = 'cpu'
        
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=16000)
        pipeline.add_step(break_into_chunks, mode='exact_count', chunk_count=2, chunk_duration=1000)  # 1 second chunks
        
        chunks = pipeline.process(gpu_audio)
        assert len(chunks) > 0
        
        for i, chunk in enumerate(chunks):
            assert chunk.device.type == expected_device, f"Chunk {i} should be on {expected_device}"
            assert chunk.sample_rate == 16000, f"Chunk {i} should be resampled"
    
    def test_pipeline_device_consistency(self):
        """Test that all pipeline operations maintain device consistency."""
        # Create audio on CPU
        cpu_audio = generate_white_noise_audio(duration_sec=1.0, sample_rate=48000, channels=2)
        
        pipeline = PreprocessingPipeline()
        pipeline.add_step(resample_audio, new_sample_rate=22050)
        pipeline.add_step(convert_to_mono, method='right')
        pipeline.add_step(normalize_audio_rms, target_rms=-15)
        
        result = pipeline.process(cpu_audio)
        
        # Verify final result properties
        assert result.device.type == 'cpu'
        assert result.sample_rate == 22050
        assert result.audio_data.shape[0] == 1  # mono
        
        # Verify result is on same device as input
        assert result.device == cpu_audio.device