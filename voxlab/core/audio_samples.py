import torchaudio
import torch
from pathlib import Path


class AudioSamples:
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate

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