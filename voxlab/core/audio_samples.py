import torchaudio
import torch
import librosa
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
        
        # Use librosa for WebM files, torchaudio for others
        if file_path.suffix.lower() == '.webm':
            return cls._load_with_librosa(file_path)
        
        try:
            # Load audio file with torchaudio
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
            # For MP4 files, try librosa fallback since torchaudio has limited MP4/AAC support
            if file_path.suffix.lower() == '.mp4':
                try:
                    return cls._load_with_librosa(file_path)
                except Exception as librosa_e:
                    raise ValueError(f"Error loading MP4 file {file_path}. TorchAudio error: {e}. Librosa error: {librosa_e}")
            else:
                raise ValueError(f"Error loading audio file {file_path}: {e}")
    
    @classmethod
    def _load_with_librosa(cls, file_path):
        """Load audio files using librosa (for WebM, MP4, and other formats not fully supported by torchaudio)."""
        # Load with librosa, preserving original sample rate and channels
        audio_data, sample_rate = librosa.load(str(file_path), sr=None, mono=False)
        
        # Convert numpy array to torch tensor
        if audio_data.ndim == 1:
            # Mono audio - convert to stereo by duplicating channel
            audio_data = torch.from_numpy(audio_data).unsqueeze(0).repeat(2, 1)
        else:
            # Multi-channel audio
            audio_data = torch.from_numpy(audio_data)
            if audio_data.shape[0] == 1:
                # Mono in multi-channel format - convert to stereo
                audio_data = audio_data.repeat(2, 1)
            elif audio_data.shape[0] > 2:
                # More than stereo - keep only first two channels
                audio_data = audio_data[:2, :]
        
        # Ensure float32 format for consistency with torchaudio loading
        if audio_data.dtype != torch.float32:
            audio_data = audio_data.to(torch.float32)
        
        return cls(audio_data, sample_rate)

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