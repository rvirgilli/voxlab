import torchaudio
import torch
from pathlib import Path


class AudioSamples:
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
    
    def __repr__(self):
        """String representation of AudioSamples."""
        channel_desc = "mono" if self.is_mono else f"{self.channels}-channel"
        return (f"AudioSamples({channel_desc}, {self.duration:.3f}s, "
                f"{self.sample_rate}Hz, {self.num_samples:,} samples, "
                f"device={self.device})")

    @classmethod
    def load(cls, file_path):
        file_path = Path(file_path)
        
        try:
            # Load audio file
            audio_data, sample_rate = torchaudio.load(str(file_path))
            
            # Ensure audio is in float32 format
            if audio_data.dtype != torch.float32:
                audio_data = audio_data.to(torch.float32)
            
            # If audio is not stereo, convert to stereo
            if audio_data.shape[0] == 1:
                audio_data = audio_data.repeat(2, 1)
            elif audio_data.shape[0] > 2:
                audio_data = audio_data[:2, :]  # Keep only first two channels if more than stereo
            
            return cls(audio_data, sample_rate)
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {e}")

    def to_numpy(self):
        return self.audio_data.numpy()

    def to_tensor(self):
        return self.audio_data

    @property
    def device(self):
        """Get the device of the audio data tensor."""
        return self.audio_data.device
    
    @property
    def duration(self):
        """Get the duration of the audio in seconds."""
        return self.audio_data.shape[1] / self.sample_rate
    
    @property
    def num_samples(self):
        """Get the number of audio samples."""
        return self.audio_data.shape[1]
    
    @property
    def channels(self):
        """Get the number of audio channels."""
        return self.audio_data.shape[0]
    
    @property
    def shape(self):
        """Get the shape of the audio tensor (channels, samples)."""
        return self.audio_data.shape
    
    @property
    def dtype(self):
        """Get the data type of the audio tensor."""
        return self.audio_data.dtype
    
    @property
    def is_mono(self):
        """Check if audio is mono (single channel)."""
        return self.channels == 1
    
    @property
    def is_stereo(self):
        """Check if audio is stereo (two channels)."""
        return self.channels == 2

    def to(self, device):
        """Move audio data to specified device. Returns new AudioSamples instance."""
        return AudioSamples(self.audio_data.to(device), self.sample_rate)

    def cuda(self):
        """Move audio data to CUDA device. Returns new AudioSamples instance."""
        return self.to('cuda')

    def cpu(self):
        """Move audio data to CPU device. Returns new AudioSamples instance."""
        return self.to('cpu')

    def export(self, export_path, format='wav'):
        if format not in ['wav', 'mp3', 'ogg', 'flac']:
            raise ValueError(f"Unsupported export format: {format}")

        export_path = Path(export_path)
        if isinstance(self.audio_data, list):
            export_path.mkdir(parents=True, exist_ok=True)
            for i, chunk in enumerate(self.audio_data):
                chunk_file_path = export_path / f"{i:04d}.{format}"
                torchaudio.save(str(chunk_file_path), chunk, self.sample_rate, format=format)
        else:
            torchaudio.save(str(export_path), self.audio_data, self.sample_rate, format=format)