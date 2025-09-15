# VoxLab

A comprehensive Python toolbox for audio processing using PyTorch. VoxLab provides a clean, device-aware architecture for audio manipulation with PyTorch tensors and supports GPU acceleration.

## Features

- **Device-Aware Audio Processing**: CPU/GPU operations with automatic device preservation
- **Memory-Efficient Operations**: In-place processing options to reduce memory usage
- **Comprehensive Preprocessing Pipeline**: Resampling, mono conversion, silence removal, chunking, and RMS normalization
<!-- - **Voice Embedding Extraction**: ECAPA2 model support via Hugging Face Hub (Coming Soon) -->
- **Extensive Testing**: 62 passing tests covering all functionality

## Installation

### CUDA Installation (Recommended)
```bash
# Create conda environment with Python 3.11.13
conda create -n voxlab python=3.11.13 -y
conda activate voxlab

# Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install in editable mode
pip install -e .
```

### CPU-Only Installation
```bash
# For CPU-only usage (no CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Quick Start

### Basic Audio Processing
```python
from voxlab.core.audio_samples import AudioSamples
from voxlab.preprocessing.functions import resample_audio, convert_to_mono

# Load and process audio
audio = AudioSamples.load("input.wav")
audio = resample_audio(audio, 16000, inplace=True)  # Memory efficient
audio = convert_to_mono(audio, method='left', inplace=True)
audio.export("output.wav")
```

### GPU-Accelerated Workflow
```python
# Move to GPU for processing
audio = AudioSamples.load("input.wav").cuda()
print(f"Audio device: {audio.device}")  # cuda:0

# All operations preserve GPU device
audio = resample_audio(audio, 16000, inplace=True)  # Stays on GPU
audio = normalize_audio_rms(audio, target_rms=-20, inplace=True)  # Stays on GPU
```

### Pipeline Processing
```python
from voxlab.preprocessing.pipeline import PreprocessingPipeline
from voxlab.preprocessing.functions import *

# Create pipeline
pipeline = PreprocessingPipeline()
pipeline.add_step(resample_audio, new_sample_rate=16000)
pipeline.add_step(convert_to_mono, method='left')
pipeline.add_step(normalize_audio_rms, target_rms=-15)

# Process (maintains device throughout)
audio = AudioSamples.load("input.wav").cuda()
result = pipeline.process(audio)  # Result stays on GPU
```

## Core Components

### AudioSamples Class
- Central data structure using PyTorch tensors
- Device-aware operations (`.cuda()`, `.cpu()`, `.to()`)
- Automatic format conversions and stereo handling
- Export to multiple formats (wav, mp3, ogg, flac)

### Preprocessing Functions
- **`resample_audio()`**: Device-preserving resampling with configurable sample rates
- **`convert_to_mono()`**: Stereo-to-mono conversion with channel selection
- **`remove_silence()`**: Intelligent silence removal with fade transitions
- **`break_into_chunks()`**: Audio segmentation with fade-in/fade-out
- **`normalize_audio_rms()`**: RMS-based normalization to target dB levels

<!-- ### Embedding Extraction (Coming Soon)
- **ECAPA2Model**: State-of-the-art speaker embedding model
- **Extractor**: High-level interface for embedding extraction
- **GPU acceleration** with automatic device detection -->

## Testing

Run tests using pytest:
```bash
source venv/bin/activate  # or conda activate voxlab
pytest tests/ -v
```

**Current Status: ✅ 62 tests passing**
- AudioSamples core functionality (11 tests)
- Device awareness and GPU operations (12 tests)  
- Preprocessing functions (23 tests)
- Pipeline system (11 tests)
- Utilities and infrastructure (5 tests)

## Requirements

### Core Dependencies
- Python >= 3.11.13
- PyTorch >= 2.8.0 (with torchaudio)
- scipy >= 1.16.2
- numpy >= 2.1.2
- pytest >= 8.4.2 (for testing)

<!-- ### ML Dependencies (Coming Soon)
- Hugging Face Hub >= 0.10.0
- transformers >= 4.0.0 -->

## License

MIT License

## Author

Rafaello Virgilli (rvirgilli@gmail.com)