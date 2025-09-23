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

def break_into_chunks(audio: AudioSamples, chunk_size=5000, fade_duration=50, inplace: bool = False) -> list[AudioSamples]:
    """
    Breaks the audio into n_chunks equal parts, and from each part,
    extracts a segment of duration chunk_size milliseconds that is
    centered within that part. Applies fade-in and fade-out to each chunk,
    and returns a list of these chunks.

    Parameters:
    audio (Audio): Audio object to process.
    chunk_size (int): Size of each chunk in milliseconds.
    fade_duration (int): Duration of fade in and fade out in milliseconds.
    inplace (bool): Parameter ignored - function always returns a list of new AudioSamples instances.

    Returns:
    list[Audio]: List of Audio chunks with fade-in and fade-out applied.
    """
    chunks = []
    audio_length = audio.audio_data.shape[1]
    chunk_length = int(chunk_size * audio.sample_rate / 1000)
    fade_length = int(fade_duration * audio.sample_rate / 1000)

    n_chunks = audio_length // chunk_length
    if n_chunks == 0:
        return chunks

    part_length = audio_length // n_chunks

    for i in range(n_chunks):
        part_start = i * part_length
        # part_end = (i + 1) * part_length  # Not used, commenting out

        chunk_start = part_start + (part_length - chunk_length) // 2
        chunk_end = chunk_start + chunk_length

        if chunk_end > audio_length:
            chunk_end = audio_length
            chunk_start = chunk_end - chunk_length

        chunk = audio.audio_data[:, chunk_start:chunk_end]

        # Apply fade in and fade out (create tensors on same device)
        fade_in = torch.linspace(0, 1, steps=fade_length, device=audio.device)
        fade_out = torch.linspace(1, 0, steps=fade_length, device=audio.device)
        chunk[:, :fade_length] *= fade_in
        chunk[:, -fade_length:] *= fade_out

        chunks.append(AudioSamples(chunk, audio.sample_rate))

    return chunks

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