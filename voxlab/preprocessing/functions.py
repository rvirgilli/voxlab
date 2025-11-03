from ..core.audio_samples import AudioSamples
import torch
from torchaudio import transforms as T

def resample_audio(audio: AudioSamples, new_sample_rate: int, inplace: bool = True) -> AudioSamples:
    """
    Resamples an AudioSamples object to the new sample rate using torchaudio.

    Parameters:
    audio (AudioSamples): The audio to resample.
    new_sample_rate (int): The target sample rate.
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.

    Returns:
    AudioSamples: The resampled audio.
    """
    # Create a resampler and move it to the same device as audio
    resampler = T.Resample(audio.sample_rate, new_sample_rate).to(audio.device)

    # Resample the audio
    resampled_audio = resampler(audio.to_tensor())

    if inplace:
        audio.audio_data = resampled_audio
        audio.sample_rate = new_sample_rate
        return audio
    else:
        return AudioSamples(resampled_audio, new_sample_rate)

def convert_to_mono(audio: AudioSamples, method='left', inplace: bool = True) -> AudioSamples:
    """
    Converts stereo audio to mono.

    Parameters:
    audio (Audio): The audio to convert.
    method (str): The method to convert to mono ('left' or 'right').
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.

    Returns:
    Audio: The mono audio.

    Raises:
    ValueError: If the method is unsupported or if the audio has more than two channels.
    """
    if audio.audio_data.shape[0] == 1:
        # Already mono - respect inplace parameter for consistency
        if inplace:
            return audio
        else:
            return AudioSamples(audio.audio_data.clone(), audio.sample_rate)

    if audio.audio_data.shape[0] > 2:
        raise ValueError("Audio has more than two channels.")

    if method == 'left':
        mono_audio = audio.audio_data[0].unsqueeze(0)
    elif method == 'right':
        mono_audio = audio.audio_data[1].unsqueeze(0)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'left' or 'right'.")

    if inplace:
        audio.audio_data = mono_audio
        return audio
    else:
        return AudioSamples(mono_audio, audio.sample_rate)

def detect_silence(audio: AudioSamples, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Detect silent sections in audio using pure PyTorch operations.
    
    All operations stay on the original device - no device transfers except final result.
    
    Parameters:
    audio (AudioSamples): The audio to analyze
    min_silence_len (int): Minimum silence duration in milliseconds
    silence_thresh (float): Silence threshold in dBFS (e.g. -16.0)
    seek_step (int): Step size in milliseconds for iteration
    
    Returns:
    list: List of [start_ms, end_ms] ranges representing silent sections
    """
    device = audio.device
    
    # Convert parameters to samples (using torch tensors)
    min_silence_samples = torch.tensor(min_silence_len * audio.sample_rate / 1000, device=device, dtype=torch.long)
    seek_step_samples = torch.clamp(torch.tensor(seek_step * audio.sample_rate / 1000, device=device, dtype=torch.long), min=1)
    window_size = torch.clamp(torch.tensor(10 * audio.sample_rate / 1000, device=device, dtype=torch.long), min=1)
    
    # Convert dBFS threshold to linear amplitude (keep as tensor)
    silence_thresh_linear = torch.tensor(10 ** (silence_thresh / 20), device=device)
    
    # Get audio data (work with mono for simplicity, take max across channels)
    if audio.audio_data.shape[0] > 1:
        audio_mono = torch.max(torch.abs(audio.audio_data), dim=0)[0]
    else:
        audio_mono = torch.abs(audio.audio_data.squeeze(0))
    
    # audio_length = torch.tensor(audio_mono.shape[0], device=device, dtype=torch.long)  # Not needed
    
    # Use vectorized approach for all cases - more GPU-friendly
    return _detect_silence_pure_torch(audio_mono, min_silence_samples, silence_thresh_linear, 
                                    seek_step_samples, window_size, audio.sample_rate)


def _detect_silence_pure_torch(audio_mono, min_silence_samples, silence_thresh_linear, 
                              seek_step_samples, window_size, sample_rate):
    """Pure PyTorch implementation that stays on device throughout"""
    device = audio_mono.device
    audio_length = audio_mono.shape[0]
    
    # Create all window positions as tensor
    max_positions = (audio_length - window_size) // seek_step_samples + 1
    if max_positions <= 0:
        return []
    
    # Generate all window start positions
    positions = torch.arange(0, max_positions, device=device, dtype=torch.long) * seek_step_samples
    
    # Use unfold to create all sliding windows efficiently on GPU/CPU
    # This creates a [num_windows, window_size] tensor
    try:
        windows = audio_mono.unfold(0, window_size.item(), seek_step_samples.item())
    except:
        # Fallback if unfold fails
        windows = torch.stack([audio_mono[pos:pos + window_size] for pos in positions])
    
    # Compute RMS for all windows in parallel (pure torch operations)
    rms_values = torch.sqrt(torch.mean(windows ** 2, dim=1))  # [num_windows]
    
    # Find silent windows (all computation stays on device)
    is_silent = rms_values < silence_thresh_linear  # [num_windows] boolean tensor
    
    # Find consecutive silent regions using pure torch operations
    silent_ranges = _find_silent_ranges_torch(is_silent, positions, window_size, 
                                            min_silence_samples, sample_rate, device)
    
    return silent_ranges


def _find_silent_ranges_torch(is_silent, positions, window_size, min_silence_samples, sample_rate, device):
    """Find silent ranges using pure torch operations"""
    if not torch.any(is_silent):
        return []
    
    # Find transitions from non-silent to silent and vice versa
    # Pad with False to handle edge cases
    padded_silent = torch.cat([torch.tensor([False], device=device), is_silent, torch.tensor([False], device=device)])
    
    # Find start and end indices of silent regions
    diff = padded_silent[1:].long() - padded_silent[:-1].long()
    starts = torch.where(diff == 1)[0]  # Indices where silence starts
    ends = torch.where(diff == -1)[0]   # Indices where silence ends
    
    # Convert to sample positions with bounds checking
    start_positions = positions[starts]
    
    # Handle end positions carefully to avoid index out of bounds
    max_pos_idx = len(positions) - 1
    safe_ends = torch.clamp(ends, 0, max_pos_idx)
    end_positions = torch.where(ends <= max_pos_idx, 
                               positions[safe_ends], 
                               positions[-1] + window_size)
    
    # Filter by minimum silence length
    durations = end_positions - start_positions
    valid_mask = durations >= min_silence_samples
    
    if not torch.any(valid_mask):
        return []
    
    # Get valid ranges
    valid_starts = start_positions[valid_mask]
    valid_ends = end_positions[valid_mask]
    
    # Convert to milliseconds (only these final values need to be transferred to CPU)
    start_ms = (valid_starts.float() * 1000 / sample_rate).long()
    end_ms = (valid_ends.float() * 1000 / sample_rate).long()
    
    # Convert to Python list only at the very end
    if device.type == 'cuda':
        start_ms_cpu = start_ms.cpu()
        end_ms_cpu = end_ms.cpu()
    else:
        start_ms_cpu = start_ms
        end_ms_cpu = end_ms
    
    # Build final list
    silent_ranges = []
    for i in range(len(start_ms_cpu)):
        silent_ranges.append([start_ms_cpu[i].item(), end_ms_cpu[i].item()])
    
    return silent_ranges


def detect_nonsilent(audio: AudioSamples, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Detect non-silent sections in audio (inverse of detect_silence).
    
    Returns:
    list: List of [start_ms, end_ms] ranges representing non-silent sections
    """
    silent_ranges = detect_silence(audio, min_silence_len, silence_thresh, seek_step)
    
    # Calculate non-silent ranges as inverse of silent ranges
    audio_duration_ms = int(audio.duration * 1000)
    nonsilent_ranges = []
    
    last_end = 0
    for start, end in silent_ranges:
        if start > last_end:
            nonsilent_ranges.append([last_end, start])
        last_end = end
    
    # Add final segment if there's audio after the last silence
    if last_end < audio_duration_ms:
        nonsilent_ranges.append([last_end, audio_duration_ms])
    
    return nonsilent_ranges


def remove_silence(audio: AudioSamples, min_silence_len=1000, silence_thresh=-16, keep_silence=100, inplace: bool = True, 
                  _min_segment_len=None, _fade_duration=None, silence_duration=None) -> AudioSamples:
    """
    Remove silence from audio by splitting on silent sections and rejoining segments.
    
    Based on pydub's split_on_silence but optimized with incremental RMS calculation.
    
    Parameters:
    audio (AudioSamples): The audio to process
    min_silence_len (int): Minimum silence duration to split on (milliseconds)
    silence_thresh (float): Silence threshold in dBFS (e.g. -16.0)  
    keep_silence (int): Amount of silence to keep at split points (milliseconds)
    inplace (bool): If True, modifies audio in-place; if False, returns new AudioSamples
    _min_segment_len (int): Legacy parameter, ignored for backward compatibility
    _fade_duration (int): Legacy parameter, ignored for backward compatibility  
    silence_duration (int): Legacy parameter, maps to keep_silence for backward compatibility
    
    Returns:
    AudioSamples: Audio with long silences removed
    """
    # Handle legacy parameters for backward compatibility
    if silence_duration is not None:
        keep_silence = silence_duration
    # _min_segment_len and _fade_duration are legacy parameters, ignored for backward compatibility
    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len, silence_thresh)
    
    if not nonsilent_ranges:
        # No non-silent audio found, return silence
        silent_audio = torch.zeros_like(audio.audio_data[:, :1])  # Single silent sample
        if inplace:
            audio.audio_data = silent_audio
            return audio
        else:
            return AudioSamples(silent_audio, audio.sample_rate)
    
    # Convert keep_silence to samples
    keep_silence_samples = int(keep_silence * audio.sample_rate / 1000)
    
    # Build output by concatenating non-silent segments with minimal silence
    segments = []
    
    for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
        # Convert to samples
        start_sample = int(start_ms * audio.sample_rate / 1000)
        end_sample = int(end_ms * audio.sample_rate / 1000)
        
        # Extract segment
        segment = audio.audio_data[:, start_sample:end_sample]
        segments.append(segment)
        
        # Add small silence between segments (except after last segment)
        if i < len(nonsilent_ranges) - 1 and keep_silence_samples > 0:
            silence_segment = torch.zeros(audio.audio_data.shape[0], keep_silence_samples, 
                                        dtype=audio.audio_data.dtype, device=audio.audio_data.device)
            segments.append(silence_segment)
    
    # Concatenate all segments
    if segments:
        output_audio = torch.cat(segments, dim=1)
    else:
        output_audio = torch.zeros_like(audio.audio_data[:, :1])
    
    if inplace:
        audio.audio_data = output_audio
        return audio
    else:
        return AudioSamples(output_audio, audio.sample_rate)

def _apply_chunk_fades(chunk: torch.Tensor, fade_duration: int, sample_rate: int, device: torch.device) -> torch.Tensor:
    """
    Apply fade-in and fade-out to a chunk tensor.
    
    Parameters:
    chunk (torch.Tensor): Audio chunk tensor [channels, samples]
    fade_duration (int): Duration of fade in/out in milliseconds
    sample_rate (int): Sample rate of the audio
    device (torch.device): Device to create fade tensors on
    
    Returns:
    torch.Tensor: Chunk with fades applied
    """
    fade_length = int(fade_duration * sample_rate / 1000)
    chunk_length = chunk.shape[1]
    
    # Ensure fade length doesn't exceed chunk length
    fade_length = min(fade_length, chunk_length // 2)
    
    if fade_length > 0:
        fade_in = torch.linspace(0, 1, steps=fade_length, device=device)
        fade_out = torch.linspace(1, 0, steps=fade_length, device=device)
        chunk = chunk.clone()  # Avoid modifying original
        chunk[:, :fade_length] *= fade_in
        chunk[:, -fade_length:] *= fade_out
    
    return chunk

def _pad_audio_to_duration(audio_tensor: torch.Tensor, target_duration_samples: int) -> torch.Tensor:
    """
    Pad audio tensor with zeros to reach target duration.
    
    Parameters:
    audio_tensor (torch.Tensor): Audio tensor [channels, samples]
    target_duration_samples (int): Target duration in samples
    
    Returns:
    torch.Tensor: Padded audio tensor [channels, target_duration_samples]
    """
    current_samples = audio_tensor.shape[1]
    if current_samples >= target_duration_samples:
        return audio_tensor[:, :target_duration_samples]
    
    padding_needed = target_duration_samples - current_samples
    padding = torch.zeros(audio_tensor.shape[0], padding_needed,
                        dtype=audio_tensor.dtype, device=audio_tensor.device)
    return torch.cat([audio_tensor, padding], dim=1)

# Audio chunking using range covering functions - maintains device integrity

def _break_into_chunks_exact_count(audio: AudioSamples, chunk_count: int, chunk_duration: int, fade_duration=50, return_timings=False):
    """
    Break audio into exactly N evenly spaced chunks using range covering functions.
    All operations maintain device integrity - no device transfers.
    
    Parameters:
    audio (AudioSamples): Audio object to process
    chunk_count (int): Exact number of chunks to create
    chunk_duration (int): Duration of each chunk in milliseconds
    fade_duration (int): Duration of fade in/out in milliseconds
    return_timings (bool): If True, return timing information
    
    Returns:
    list[AudioSamples] or tuple[list[AudioSamples], list[tuple[float, float]]]
    """
    from .range_covering_functions import get_chunks_exactly
    
    if chunk_count <= 0:
        return [] if not return_timings else ([], [])
    
    audio_length_samples = audio.audio_data.shape[1]
    chunk_duration_samples = int(chunk_duration * audio.sample_rate / 1000)
    
    # Handle edge case: audio shorter than chunk duration
    if audio_length_samples < chunk_duration_samples:
        # Return n copies of padded audio 
        padded_chunk = _pad_audio_to_duration(audio.audio_data, chunk_duration_samples)
        chunk = _apply_chunk_fades(padded_chunk, fade_duration, audio.sample_rate, audio.device)
        chunks = [AudioSamples(chunk, audio.sample_rate) for _ in range(chunk_count)]
        if return_timings:
            timings = [(0.0, chunk_duration / 1000.0) for _ in range(chunk_count)]
            return chunks, timings
        return chunks
    
    # Use range covering function to get chunk positions
    try:
        positions = get_chunks_exactly(audio_length_samples, chunk_duration_samples, chunk_count)
    except ValueError:
        # For single chunk case where chunk_duration < audio_length, position chunk at start
        if chunk_count == 1:
            positions = [[0, chunk_duration_samples]]
        else:
            # Fallback for other impossible cases
            if return_timings:
                return [], []
            return []
    
    # Extract chunks at calculated positions - maintain device throughout
    chunks = []
    timings = []
    
    for start_samples, end_samples in positions:
        start_pos = int(round(start_samples))
        end_pos = int(round(end_samples))
        
        # Ensure chunk is exactly chunk_duration_samples long
        if end_pos > audio_length_samples:
            # Extract available audio and pad with zeros (on same device)
            available_chunk = audio.audio_data[:, start_pos:audio_length_samples]
            padding_needed = chunk_duration_samples - (audio_length_samples - start_pos)
            padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                dtype=audio.audio_data.dtype, device=audio.audio_data.device)
            chunk = torch.cat([available_chunk, padding], dim=1)
            # Timing reflects actual audio content
            actual_end_sec = audio_length_samples / audio.sample_rate
        else:
            # Extract exact chunk duration
            actual_end = min(start_pos + chunk_duration_samples, end_pos)
            chunk = audio.audio_data[:, start_pos:actual_end]
            # Pad if needed to reach exact chunk duration
            if chunk.shape[1] < chunk_duration_samples:
                padding_needed = chunk_duration_samples - chunk.shape[1]
                padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                    dtype=audio.audio_data.dtype, device=audio.audio_data.device)
                chunk = torch.cat([chunk, padding], dim=1)
            actual_end_sec = actual_end / audio.sample_rate
        
        # Apply fades (maintains device)
        chunk = _apply_chunk_fades(chunk, fade_duration, audio.sample_rate, audio.device)
        chunks.append(AudioSamples(chunk, audio.sample_rate))
        
        # Timing info in seconds
        start_sec = start_pos / audio.sample_rate
        timings.append((start_sec, actual_end_sec))
    
    if return_timings:
        return chunks, timings
    return chunks

def _break_into_chunks_min_overlap(audio: AudioSamples, chunk_duration=4000, min_overlap=2000, fade_duration=50, return_timings=False):
    """
    Creates evenly spaced chunks with minimum overlap constraint using range covering functions.
    All operations maintain device integrity - no device transfers.
    
    Parameters:
    audio (AudioSamples): Audio object to process
    chunk_duration (int): Duration of each chunk in milliseconds
    min_overlap (int): Minimum overlap between consecutive chunks in milliseconds
    fade_duration (int): Duration of fade in/out in milliseconds
    return_timings (bool): If True, return timing information
    
    Returns:
    list[AudioSamples] or tuple[list[AudioSamples], list[tuple[float, float]]]
    """
    from .range_covering_functions import get_chunks_max_spacing
    
    audio_length_samples = audio.audio_data.shape[1]
    chunk_duration_samples = int(chunk_duration * audio.sample_rate / 1000)
    min_overlap_samples = int(min_overlap * audio.sample_rate / 1000)
    
    if audio_length_samples <= chunk_duration_samples:
        # Audio is shorter than chunk size, return single chunk padded to chunk_duration
        padded_chunk = _pad_audio_to_duration(audio.audio_data, chunk_duration_samples)
        chunk = _apply_chunk_fades(padded_chunk, fade_duration, audio.sample_rate, audio.device)
        if return_timings:
            timing = (0.0, chunk_duration / 1000.0)
            return [AudioSamples(chunk, audio.sample_rate)], [timing]
        return [AudioSamples(chunk, audio.sample_rate)]
    
    # Convert min_overlap to max_spacing: max_spacing = -min_overlap
    max_spacing_samples = -min_overlap_samples
    
    # Use range covering function to get chunk positions
    try:
        positions = get_chunks_max_spacing(audio_length_samples, chunk_duration_samples, max_spacing_samples)
    except ValueError as e:
        # Handle extreme overlap cases - when min_overlap >= chunk_duration,
        # fall back to overlapping chunks with smaller step size
        if min_overlap_samples >= chunk_duration_samples:
            # Create heavily overlapping chunks with step size = chunk_duration // 2
            step_size = max(1, chunk_duration_samples // 2)
            num_chunks = max(1, (audio_length_samples - chunk_duration_samples) // step_size + 1)
            positions = []
            for i in range(num_chunks):
                start = i * step_size
                end = min(start + chunk_duration_samples, audio_length_samples)
                positions.append([start, end])
        else:
            # Other validation errors: fallback to single chunk
            padded_chunk = _pad_audio_to_duration(audio.audio_data, chunk_duration_samples)
            chunk = _apply_chunk_fades(padded_chunk, fade_duration, audio.sample_rate, audio.device)
            if return_timings:
                timing = (0.0, chunk_duration / 1000.0)
                return [AudioSamples(chunk, audio.sample_rate)], [timing]
            return [AudioSamples(chunk, audio.sample_rate)]
    
    # Extract chunks at calculated positions - maintain device throughout
    chunks = []
    timings = []
    
    for start_samples, end_samples in positions:
        start_pos = int(round(start_samples))
        end_pos = int(round(end_samples))
        
        # Ensure chunk is exactly chunk_duration_samples long
        if end_pos > audio_length_samples:
            # Extract available audio and pad with zeros (on same device)
            available_chunk = audio.audio_data[:, start_pos:audio_length_samples]
            padding_needed = chunk_duration_samples - (audio_length_samples - start_pos)
            padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                dtype=audio.audio_data.dtype, device=audio.audio_data.device)
            chunk = torch.cat([available_chunk, padding], dim=1)
            # Timing reflects actual audio content
            actual_end_sec = audio_length_samples / audio.sample_rate
        else:
            # Extract exact chunk duration  
            actual_end = min(start_pos + chunk_duration_samples, end_pos)
            chunk = audio.audio_data[:, start_pos:actual_end]
            # Pad if needed to reach exact chunk duration
            if chunk.shape[1] < chunk_duration_samples:
                padding_needed = chunk_duration_samples - chunk.shape[1]
                padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                    dtype=audio.audio_data.dtype, device=audio.audio_data.device)
                chunk = torch.cat([chunk, padding], dim=1)
            actual_end_sec = actual_end / audio.sample_rate
        
        # Apply fades (maintains device)
        chunk = _apply_chunk_fades(chunk, fade_duration, audio.sample_rate, audio.device)
        chunks.append(AudioSamples(chunk, audio.sample_rate))
        
        # Timing info in seconds
        start_sec = start_pos / audio.sample_rate
        timings.append((start_sec, actual_end_sec))
    
    if return_timings:
        return chunks, timings
    return chunks

def _break_into_chunks_max_overlap(audio: AudioSamples, chunk_duration=4000, max_overlap=3000, fade_duration=50, return_timings=False):
    """
    Creates maximum evenly spaced chunks with overlap constraint using range covering functions.
    All operations maintain device integrity - no device transfers.
    
    Parameters:
    audio (AudioSamples): Audio object to process  
    chunk_duration (int): Duration of each chunk in milliseconds
    max_overlap (int): Maximum allowed overlap between consecutive chunks in milliseconds
    fade_duration (int): Duration of fade in/out in milliseconds
    return_timings (bool): If True, return timing information
    
    Returns:
    list[AudioSamples] or tuple[list[AudioSamples], list[tuple[float, float]]]
    """
    from .range_covering_functions import get_chunks_min_spacing
    
    audio_length_samples = audio.audio_data.shape[1]
    chunk_duration_samples = int(chunk_duration * audio.sample_rate / 1000)
    max_overlap_samples = int(max_overlap * audio.sample_rate / 1000)
    
    if audio_length_samples <= chunk_duration_samples:
        # Audio is shorter than chunk size, return single chunk padded to chunk_duration
        padded_chunk = _pad_audio_to_duration(audio.audio_data, chunk_duration_samples)
        chunk = _apply_chunk_fades(padded_chunk, fade_duration, audio.sample_rate, audio.device)
        if return_timings:
            timing = (0.0, chunk_duration / 1000.0)
            return [AudioSamples(chunk, audio.sample_rate)], [timing]
        return [AudioSamples(chunk, audio.sample_rate)]
    
    # Convert max_overlap to min_spacing: min_spacing = -max_overlap
    min_spacing_samples = -max_overlap_samples
    
    # Use range covering function to get chunk positions
    try:
        positions = get_chunks_min_spacing(audio_length_samples, chunk_duration_samples, min_spacing_samples)
    except ValueError:
        # Fallback: single chunk
        padded_chunk = _pad_audio_to_duration(audio.audio_data, chunk_duration_samples)
        chunk = _apply_chunk_fades(padded_chunk, fade_duration, audio.sample_rate, audio.device)
        if return_timings:
            timing = (0.0, chunk_duration / 1000.0)
            return [AudioSamples(chunk, audio.sample_rate)], [timing]
        return [AudioSamples(chunk, audio.sample_rate)]
    
    # Extract chunks at calculated positions - maintain device throughout
    chunks = []
    timings = []
    
    for start_samples, end_samples in positions:
        start_pos = int(round(start_samples))
        end_pos = int(round(end_samples))
        
        # Ensure chunk is exactly chunk_duration_samples long
        if end_pos > audio_length_samples:
            # Extract available audio and pad with zeros (on same device)
            available_chunk = audio.audio_data[:, start_pos:audio_length_samples]
            padding_needed = chunk_duration_samples - (audio_length_samples - start_pos)
            padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                dtype=audio.audio_data.dtype, device=audio.audio_data.device)
            chunk = torch.cat([available_chunk, padding], dim=1)
            # Timing reflects actual audio content
            actual_end_sec = audio_length_samples / audio.sample_rate
        else:
            # Extract exact chunk duration
            actual_end = min(start_pos + chunk_duration_samples, end_pos)
            chunk = audio.audio_data[:, start_pos:actual_end]
            # Pad if needed to reach exact chunk duration
            if chunk.shape[1] < chunk_duration_samples:
                padding_needed = chunk_duration_samples - chunk.shape[1]
                padding = torch.zeros(audio.audio_data.shape[0], padding_needed,
                                    dtype=audio.audio_data.dtype, device=audio.audio_data.device)
                chunk = torch.cat([chunk, padding], dim=1)
            actual_end_sec = actual_end / audio.sample_rate
        
        # Apply fades (maintains device)
        chunk = _apply_chunk_fades(chunk, fade_duration, audio.sample_rate, audio.device)
        chunks.append(AudioSamples(chunk, audio.sample_rate))
        
        # Timing info in seconds
        start_sec = start_pos / audio.sample_rate
        timings.append((start_sec, actual_end_sec))
    
    if return_timings:
        return chunks, timings
    return chunks


def _break_into_chunks_extract_intervals(audio: AudioSamples, intervals, fade_duration=50, return_timings=False):
    """
    Extract audio chunks at exact time intervals.
    All operations maintain device integrity - no device transfers.

    Parameters:
    audio (AudioSamples): Audio object to process
    intervals (list): List of [start_ms, end_ms] intervals to extract
    fade_duration (int): Duration of fade in/out in milliseconds (default: 50)
    return_timings (bool): If True, return timing information

    Returns:
    list[AudioSamples] or tuple[list[AudioSamples], list[tuple[float, float]]]

    Raises:
    ValueError: If intervals is not a list/tuple, if any interval is invalid,
               or if any interval is outside audio bounds

    Example:
    # Extract specific time ranges
    chunks = break_into_chunks(audio, mode='extract_intervals',
                              intervals=[[1000, 3000], [5000, 8000]])
    # Returns 2 chunks: [1s-3s] and [5s-8s]
    """
    if not isinstance(intervals, (list, tuple)):
        raise ValueError("intervals must be a list or tuple of [start_ms, end_ms] pairs")

    if len(intervals) == 0:
        return [] if not return_timings else ([], [])

    audio_length_samples = audio.audio_data.shape[1]
    audio_length_ms = audio_length_samples * 1000 / audio.sample_rate

    chunks = []
    timings = []

    for i, interval in enumerate(intervals):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Each interval must be a [start_ms, end_ms] pair, got {interval}")

        start_ms, end_ms = interval

        if start_ms < 0:
            raise ValueError(f"Interval {i} start time {start_ms}ms cannot be negative")

        if end_ms <= start_ms:
            raise ValueError(f"Interval {i} end time {end_ms}ms must be greater than start time {start_ms}ms")

        if start_ms >= audio_length_ms:
            raise ValueError(f"Interval {i} start time {start_ms}ms is beyond audio length {audio_length_ms:.2f}ms")

        if end_ms > audio_length_ms:
            raise ValueError(f"Interval {i} end time {end_ms}ms exceeds audio length {audio_length_ms:.2f}ms")

        # Convert to samples
        start_samples = int(start_ms * audio.sample_rate / 1000)
        end_samples = int(end_ms * audio.sample_rate / 1000)

        # Extract chunk
        chunk = audio.audio_data[:, start_samples:end_samples]

        # Apply fades (maintains device)
        chunk = _apply_chunk_fades(chunk, fade_duration, audio.sample_rate, audio.device)
        chunks.append(AudioSamples(chunk, audio.sample_rate))

        # Timing info in seconds
        start_sec = start_samples / audio.sample_rate
        end_sec = end_samples / audio.sample_rate
        timings.append((start_sec, end_sec))

    if return_timings:
        return chunks, timings
    return chunks


def _break_into_chunks_split_by_time(audio: AudioSamples, split_points, overlap=0, fade_duration=50, return_timings=False):
    """
    Break audio into chunks at explicit split points with optional overlap.
    All operations maintain device integrity - no device transfers.

    Parameters:
    audio (AudioSamples): Audio object to process
    split_points (list): List of split positions in milliseconds (start of audio and end are implicit)
    overlap (int): Overlap duration in milliseconds (default: 0)
    fade_duration (int): Duration of fade in/out in milliseconds
    return_timings (bool): If True, return timing information

    Returns:
    list[AudioSamples] or tuple[list[AudioSamples], list[tuple[float, float]]]

    Example:
    # Split at 3s and 5s with 1s overlap
    chunks = break_into_chunks(audio, mode='split_by_time', split_points=[3000, 5000], overlap=1000)
    # Creates chunks: [0-3500ms], [2500-5500ms], [4500-end]
    """
    if not isinstance(split_points, (list, tuple)):
        raise ValueError("split_points must be a list or tuple")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    audio_length_samples = audio.audio_data.shape[1]
    audio_length_ms = audio_length_samples * 1000 / audio.sample_rate

    # Convert everything to samples
    split_points_samples = [int(sp * audio.sample_rate / 1000) for sp in split_points]
    overlap_samples = int(overlap * audio.sample_rate / 1000)

    # Sort split points and remove duplicates
    split_points_samples = sorted(set(split_points_samples))

    # Validate split points
    for sp in split_points_samples:
        if sp < 0 or sp > audio_length_samples:
            raise ValueError(f"Split point {sp} samples is outside audio range [0, {audio_length_samples}]")

    # Build segment boundaries: [0, split1, split2, ..., end]
    boundaries = [0] + split_points_samples + [audio_length_samples]

    # Calculate overlap extension (half on each side of split point)
    half_overlap = overlap_samples // 2

    chunks = []
    timings = []

    for i in range(len(boundaries) - 1):
        # Base segment boundaries
        segment_start = boundaries[i]
        segment_end = boundaries[i + 1]

        # Extend boundaries for overlap (except at audio edges)
        if i > 0:  # Not the first chunk
            chunk_start = max(0, segment_start - half_overlap)
        else:
            chunk_start = segment_start

        if i < len(boundaries) - 2:  # Not the last chunk
            chunk_end = min(audio_length_samples, segment_end + half_overlap)
        else:
            chunk_end = segment_end

        # Extract chunk
        chunk = audio.audio_data[:, chunk_start:chunk_end]

        # Apply fades (maintains device)
        chunk = _apply_chunk_fades(chunk, fade_duration, audio.sample_rate, audio.device)
        chunks.append(AudioSamples(chunk, audio.sample_rate))

        # Timing info in seconds
        start_sec = chunk_start / audio.sample_rate
        end_sec = chunk_end / audio.sample_rate
        timings.append((start_sec, end_sec))

    if return_timings:
        return chunks, timings
    return chunks


def break_into_chunks(audio: AudioSamples, mode='exact_count', fade_duration=50, return_timings=False, **kwargs):
    """
    Break audio into chunks using different strategies.

    Parameters:
    audio (AudioSamples): Audio object to process
    mode (str): Chunking strategy - 'exact_count', 'min_overlap', 'max_overlap', 'split_by_time', or 'extract_intervals'
    fade_duration (int): Duration of fade in/out in milliseconds (default: 50)
    return_timings (bool): If True, return (chunks, timings) tuple instead of just chunks
    **kwargs: Mode-specific parameters

    Mode-specific parameters:
    - 'exact_count': chunk_count (int), chunk_duration (int in ms)
    - 'min_overlap': chunk_duration (int in ms), min_overlap (int in ms)
    - 'max_overlap': chunk_duration (int in ms), max_overlap (int in ms)
    - 'split_by_time': split_points (list in ms), overlap (int in ms, default 0)
    - 'extract_intervals': intervals (list of [start_ms, end_ms] pairs)

    Returns:
    list[AudioSamples] OR tuple[list[AudioSamples], list[tuple[float, float]]]:
        If return_timings=False: List of audio chunks
        If return_timings=True: (chunks, timings) where timings is [(start_sec, end_sec), ...]

    Examples:
    # Create exactly 5 chunks of 4000ms each
    chunks = break_into_chunks(audio, mode='exact_count', chunk_count=5, chunk_duration=4000)

    # Create chunks with timing info
    chunks, timings = break_into_chunks(audio, mode='min_overlap', chunk_duration=4000, min_overlap=2000, return_timings=True)
    print(f"Chunk 1: {timings[0][0]:.1f}s to {timings[0][1]:.1f}s")

    # Split at specific times with overlap
    chunks = break_into_chunks(audio, mode='split_by_time', split_points=[3000, 5000], overlap=1000)

    # Extract exact time intervals
    chunks = break_into_chunks(audio, mode='extract_intervals', intervals=[[1000, 3000], [5000, 8000]])
    """
    if mode == 'exact_count':
        chunk_count = kwargs.get('chunk_count')
        chunk_duration = kwargs.get('chunk_duration')

        if chunk_count is None or chunk_duration is None:
            raise ValueError("mode='exact_count' requires 'chunk_count' and 'chunk_duration' parameters")

        result = _break_into_chunks_exact_count(audio, chunk_count, chunk_duration, fade_duration, return_timings)

    elif mode == 'min_overlap':
        chunk_duration = kwargs.get('chunk_duration', 4000)
        min_overlap = kwargs.get('min_overlap', 0)

        result = _break_into_chunks_min_overlap(audio, chunk_duration, min_overlap, fade_duration, return_timings)

    elif mode == 'max_overlap':
        chunk_duration = kwargs.get('chunk_duration', 4000)
        max_overlap = kwargs.get('max_overlap', 0)

        result = _break_into_chunks_max_overlap(audio, chunk_duration, max_overlap, fade_duration, return_timings)

    elif mode == 'split_by_time':
        split_points = kwargs.get('split_points')
        if split_points is None:
            raise ValueError("mode='split_by_time' requires 'split_points' parameter")

        overlap = kwargs.get('overlap', 0)

        result = _break_into_chunks_split_by_time(audio, split_points, overlap, fade_duration, return_timings)

    elif mode == 'extract_intervals':
        intervals = kwargs.get('intervals')
        if intervals is None:
            raise ValueError("mode='extract_intervals' requires 'intervals' parameter")

        result = _break_into_chunks_extract_intervals(audio, intervals, fade_duration, return_timings)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'exact_count', 'min_overlap', 'max_overlap', 'split_by_time', 'extract_intervals'")

    return result


def normalize_audio_rms(audio: AudioSamples, target_rms=-15, inplace: bool = True) -> AudioSamples:
    """
    Normalizes the audio using the RMS method.

    Parameters:
    audio (Audio): The audio to normalize.
    target_rms (float): The target RMS level in dB (default: -15 dB).
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.

    Returns:
    Audio: The normalized audio.
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(audio.audio_data ** 2))
    
    # Calculate desired RMS
    target_rms_linear = 10 ** (target_rms / 20)
    
    # Calculate gain
    gain = target_rms_linear / rms
    
    # Apply gain
    normalized_audio = audio.audio_data * gain
    
    if inplace:
        audio.audio_data = normalized_audio
        return audio
    else:
        return AudioSamples(normalized_audio, audio.sample_rate)

def trim_audio(audio: AudioSamples, silence_thresh=-30, mode='both', inplace: bool = True) -> AudioSamples:
    """
    Trims silence from the beginning and/or end of audio.

    Parameters:
    audio (AudioSamples): The audio to trim.
    silence_thresh (int): The threshold for considering silence (in dB).
    mode (str): Trimming mode - 'both' (default), 'start', or 'end'.
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.

    Returns:
    AudioSamples: The trimmed audio.
    
    Raises:
    ValueError: If mode is not 'both', 'start', or 'end'.
    """
    if mode not in ['both', 'start', 'end']:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'both', 'start', or 'end'.")
    
    # Convert silence threshold from dB to amplitude (same as remove_silence)
    silence_thresh_linear = 10 ** (silence_thresh / 20)
    
    # Detect silent samples (same approach as remove_silence)
    is_silent = torch.lt(torch.abs(audio.audio_data), silence_thresh_linear)
    # For multi-channel audio, a sample is silent if ALL channels are silent
    if audio.audio_data.shape[0] > 1:
        is_silent = torch.all(is_silent, dim=0)
    else:
        is_silent = is_silent.squeeze(0)
    
    # Find non-silent samples
    non_silent_indices = torch.where(~is_silent)[0]
    
    if len(non_silent_indices) == 0:
        # All audio is silent - return minimal audio (1 sample to avoid empty tensor)
        trimmed_audio = audio.audio_data[:, :1]
    else:
        start_idx = 0
        end_idx = audio.audio_data.shape[1]
        
        if mode in ['both', 'start']:
            start_idx = non_silent_indices[0].item()
        
        if mode in ['both', 'end']:
            end_idx = non_silent_indices[-1].item() + 1
        
        trimmed_audio = audio.audio_data[:, start_idx:end_idx]
    
    if inplace:
        audio.audio_data = trimmed_audio
        return audio
    else:
        return AudioSamples(trimmed_audio, audio.sample_rate)

# You can add more preprocessing functions here as needed