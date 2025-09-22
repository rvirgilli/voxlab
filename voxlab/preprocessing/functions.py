import numpy as np
from scipy.signal import resample
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

def remove_silence(audio: AudioSamples, silence_thresh=-30, min_silence_len=1000, min_segment_len=1000, fade_duration=50, silence_duration=50, inplace: bool = True) -> AudioSamples:
    """
    Removes silence from an audio, joins selected segments with fade-in, fade-out, and silence between them.

    Parameters:
    audio (Audio): The audio to process.
    silence_thresh (int): The threshold for considering silence (in dB).
    min_silence_len (int): The minimum length of silence to be removed (in ms).
    min_segment_len (int): The minimum length of a segment to be kept (in ms).
    fade_duration (int): The duration of fade-in and fade-out (in ms).
    silence_duration (int): The duration of silence to insert between segments (in ms).
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.

    Returns:
    Audio: The audio with silence removed.
    """
    # Convert parameters from ms to samples
    min_silence_samples = int(min_silence_len * audio.sample_rate / 1000)
    min_segment_samples = int(min_segment_len * audio.sample_rate / 1000)
    fade_samples = int(fade_duration * audio.sample_rate / 1000)
    silence_samples = int(silence_duration * audio.sample_rate / 1000)

    # Convert silence threshold from dB to amplitude
    silence_thresh = 10 ** (silence_thresh / 20)

    # Detect non-silent segments
    is_silent = torch.lt(torch.abs(audio.audio_data), silence_thresh)
    mask = torch.nn.functional.max_pool1d(is_silent.float(), kernel_size=min_silence_samples, stride=1) < 0.5
    segments = torch.nonzero(mask.squeeze()).squeeze()
    
    # Filter out segments that are shorter than the minimum segment length
    segment_lengths = segments[1:] - segments[:-1]
    valid_segments = torch.nonzero(segment_lengths >= min_segment_samples).squeeze()
    
    # Join segments with fade-in, fade-out, and silence between them
    output_audio = torch.zeros_like(audio.audio_data)
    current_position = 0
    
    for i in range(len(valid_segments)):
        start = segments[valid_segments[i]]
        if i < len(valid_segments) - 1:
            end = segments[valid_segments[i] + 1]
        else:
            end = len(audio.audio_data)
        
        segment = audio.audio_data[:, start:end]
        
        # Apply fade-in and fade-out (create tensors on same device)
        fade_in = torch.linspace(0, 1, steps=fade_samples, device=audio.device)
        fade_out = torch.linspace(1, 0, steps=fade_samples, device=audio.device)
        segment[:, :fade_samples] *= fade_in
        segment[:, -fade_samples:] *= fade_out
        
        # Add segment to output
        output_audio[:, current_position:current_position + segment.shape[1]] = segment
        current_position += segment.shape[1] + silence_samples

    # Remove the trailing silence
    if silence_samples > 0:
        output_audio = output_audio[:, :-silence_samples]

    if inplace:
        audio.audio_data = output_audio
        return audio
    else:
        return AudioSamples(output_audio, audio.sample_rate)

def break_into_chunks(audio: AudioSamples, chunk_size=5000, fade_duration=50, inplace: bool = True) -> list[AudioSamples]:
    """
    Breaks the audio into n_chunks equal parts, and from each part,
    extracts a segment of duration chunk_size milliseconds that is
    centered within that part. Applies fade-in and fade-out to each chunk,
    and returns a list of these chunks.

    Parameters:
    audio (Audio): Audio object to process.
    chunk_size (int): Size of each chunk in milliseconds.
    fade_duration (int): Duration of fade in and fade out in milliseconds.
    inplace (bool): If True, modifies the audio in-place. If False, returns a new AudioSamples instance.
                   Note: This function always returns a list, so inplace only affects tensor device placement.

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
        part_end = (i + 1) * part_length

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