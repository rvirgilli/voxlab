# VoxLab

A comprehensive Python toolbox for audio processing using PyTorch. VoxLab provides a clean, device-aware architecture for audio manipulation with PyTorch tensors and supports GPU acceleration.

## Features

- **Device-Aware Audio Processing**: CPU/GPU operations with automatic device preservation
- **Memory-Efficient Operations**: In-place processing options to reduce memory usage
- **Comprehensive Preprocessing Pipeline**: Resampling, mono conversion, silence removal, chunking, and RMS normalization
<!-- - **Voice Embedding Extraction**: ECAPA2 model support via Hugging Face Hub (Coming Soon) -->
- **WebM Format Support**: Full support for WebM audio files via librosa fallback
- **Mathematical Audio Chunking**: Precise chunk positioning using range covering algorithm with multiple modes
- **Extensive Testing**: 141 passing tests covering all functionality

## Installation

### CUDA Installation (Recommended)
```bash
# Create conda environment with Python 3.11
conda create -n voxlab python=3.11 -y
conda activate voxlab

# Install PyTorch with CUDA 12.9 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

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

# Load and process audio (supports wav, mp3, ogg, flac, webm, mp4)
audio = AudioSamples.load("input.webm")  # WebM and MP4 files supported!
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

### Audio Chunking
```python
from voxlab.preprocessing.functions import break_into_chunks

# Exact number of chunks with precise mathematical positioning
chunks = break_into_chunks(audio, mode='exact_count', chunk_count=5, chunk_duration=4000)

# Minimum chunks with maximum overlap constraint
chunks = break_into_chunks(audio, mode='min_overlap', chunk_duration=3000, min_overlap=1000)

# Maximum chunks with minimum spacing constraint
chunks = break_into_chunks(audio, mode='max_overlap', chunk_duration=4000, max_overlap=2000)

# Split at specific time points with optional overlap
chunks = break_into_chunks(audio, mode='split_by_time',
                          split_points=[3000, 7000], overlap=1000)

# Get timing information for each chunk
chunks, timings = break_into_chunks(audio, mode='exact_count',
                                   chunk_count=3, chunk_duration=5000,
                                   return_timings=True)
print(f"Generated {len(chunks)} chunks")
for i, (start, end) in enumerate(timings):
    print(f"  Chunk {i}: {start:.1f}s to {end:.1f}s")
```

## Memory Management: In-Place vs Off-Place Operations

VoxLab offers flexible memory management through `inplace` parameters in all preprocessing functions. Choose the approach that best fits your workflow:

### Memory-Efficient In-Place Operations (Default)
Perfect for GPU workflows and memory-constrained environments:

```python
# Pipeline approach (recommended)
from voxlab.preprocessing.pipeline import PreprocessingPipeline
from voxlab.preprocessing.functions import *

pipeline = PreprocessingPipeline()
pipeline.add_step(resample_audio, new_sample_rate=16000)  # inplace=True default
pipeline.add_step(convert_to_mono, method='left')
pipeline.add_step(normalize_audio_rms, target_rms=-20)
pipeline.add_step(trim_audio, mode='both')

audio = AudioSamples.load("input.wav").cuda()
original_id = id(audio)
result = pipeline.process(audio)  # Single "run pipeline" action
assert id(result) == original_id  # Same object through entire pipeline!

print(f"Memory efficient: {audio.device}")  # Stays on GPU
```

### Immutable Off-Place Operations
Ideal for functional programming and data preservation:

```python
# Off-place operations (inplace=False)
original_audio = AudioSamples.load("input.wav")
resampled = resample_audio(original_audio, 16000, inplace=False)
mono = convert_to_mono(resampled, method='left', inplace=False)  
normalized = normalize_audio_rms(mono, target_rms=-20, inplace=False)

# Each operation creates a new object
assert id(original_audio) != id(resampled)
assert id(resampled) != id(mono) 
assert id(mono) != id(normalized)

# Original remains unchanged
print(f"Original: {original_audio.sample_rate}Hz, {original_audio.channels} channels")
print(f"Result: {normalized.sample_rate}Hz, {normalized.channels} channels")
```

### Mixed Workflow
Combine both approaches as needed:

```python
# Load and preserve original
original = AudioSamples.load("input.wav")

# Create working copy for in-place operations
working_copy = resample_audio(original, 16000, inplace=False)  # New object
working_copy = convert_to_mono(working_copy, inplace=True)     # Modify copy
working_copy = normalize_audio_rms(working_copy, inplace=True) # Modify copy

# Original untouched, working_copy efficiently processed
assert original.sample_rate != working_copy.sample_rate
```

### Pipeline Memory Behavior
Pipelines respect individual step `inplace` parameters:

```python
# Memory-efficient pipeline (default inplace=True)
pipeline = PreprocessingPipeline()
pipeline.add_step(resample_audio, new_sample_rate=16000)  # inplace=True default
pipeline.add_step(convert_to_mono, method='left')         # inplace=True default

audio = AudioSamples.load("input.wav")
original_id = id(audio)
result = pipeline.process(audio)
assert id(result) == original_id  # Same object through entire pipeline

# Immutable pipeline
pipeline = PreprocessingPipeline()
pipeline.add_step(resample_audio, new_sample_rate=16000, inplace=False)
pipeline.add_step(convert_to_mono, method='left', inplace=False)

result = pipeline.process(audio)
assert id(result) != id(audio)  # New object created
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
- **`break_into_chunks()`**: Mathematical audio segmentation with precise positioning and four chunking modes
- **`normalize_audio_rms()`**: RMS-based normalization to target dB levels
- **`trim_audio()`**: Silence trimming from start, end, or both ends with configurable threshold

### Audio Chunking Algorithm
VoxLab uses a mathematical range covering algorithm for precise audio chunking with four modes:

- **`exact_count`**: Create exactly N chunks with evenly-distributed positioning and calculated spacing
- **`min_overlap`**: Find minimum number of chunks needed while satisfying minimum overlap constraints
- **`max_overlap`**: Generate maximum number of chunks possible while respecting maximum overlap limits
- **`split_by_time`**: Split at explicit time points with optional overlap centered around boundaries

All modes handle positive spacing (gaps), zero spacing (touching), and negative spacing (overlaps) automatically. Chunks maintain exact duration with fade-in/fade-out transitions and preserve device placement.

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

**Current Status: ✅ 141 tests passing**
- AudioSamples core functionality (23 tests)
- Device awareness and GPU operations (12 tests)
- Preprocessing functions (87 tests) including comprehensive chunking tests
- Pipeline system (11 tests)
- Silence detection and utilities (8 tests)

## Requirements

### Core Dependencies
- Python >= 3.11
- PyTorch >= 2.8.0 (with torchaudio)
- scipy >= 1.16.2
- numpy >= 2.1.2
- librosa >= 0.10.0 (for WebM and MP4 support)
- pytest >= 8.4.2 (for testing)

<!-- ### ML Dependencies (Coming Soon)
- Hugging Face Hub >= 0.10.0
- transformers >= 4.0.0 -->

## License

MIT License

## Author

Rafaello Virgilli (rvirgilli@gmail.com)