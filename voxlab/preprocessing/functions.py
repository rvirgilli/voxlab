import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.signal import resample


def resample_audio(audio, new_sample_rate):
    """
    Resamples an AudioSegment to the new sample rate.

    Parameters:
    audio (AudioSegment): The audio segment to resample.
    new_sample_rate (int): The target sample rate.

    Returns:
    AudioSegment: The resampled audio segment.
    """
    # Get raw audio data and convert it to a numpy array
    raw_audio_data = np.array(audio.get_array_of_samples())

    # Calculate new length of the sample
    new_length = int(len(raw_audio_data) * new_sample_rate / audio.frame_rate)

    # Resample the audio to a new sample rate
    resampled_audio_data = resample(raw_audio_data, new_length)

    # Ensure the data type is consistent with the original audio
    if audio.sample_width == 1:  # 8-bit
        resampled_audio_data = np.int8(np.clip(resampled_audio_data, -128, 127))
    elif audio.sample_width == 2:  # 16-bit
        resampled_audio_data = np.int16(np.clip(resampled_audio_data, -32768, 32767))
    elif audio.sample_width == 3:  # 24-bit
        resampled_audio_data = (resampled_audio_data / resampled_audio_data.max()) * 8388607
        resampled_audio_data = np.int32(np.clip(resampled_audio_data, -8388608, 8388607))
    else:  # 32-bit
        resampled_audio_data = np.int32(resampled_audio_data)

    # Convert the resampled numpy array back into an AudioSegment
    resampled_audio = AudioSegment(
        resampled_audio_data.tobytes(),
        frame_rate=new_sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    return resampled_audio


# Define other preprocessing functions similarly
def convert_to_mono(audio, method='left'):
    """
    Converts stereo audio to mono.

    Parameters:
    audio (AudioSegment): The audio segment to convert.
    method (str): The method to convert to mono ('left' or 'right').

    Returns:
    AudioSegment: The mono audio segment.

    Raises:
    ValueError: If the method is unsupported or if the audio has more than two channels.
    """
    if audio.channels == 1:
        return audio  # Already mono

    if audio.channels > 2:
        raise ValueError("Audio has more than two channels.")

    if method == 'left':
        mono_audio = audio.split_to_mono()[0]
    elif method == 'right':
        mono_audio = audio.split_to_mono()[1]
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'left' or 'right'.")

    return mono_audio


def remove_silence(audio, silence_thresh=-30, min_silence_len=1000, min_segment_len=1000, fade_duration=50,
                   silence_duration=50):
    """
    Removes silence from an audio segment, joins selected segments with fade-in, fade-out, and silence between them.

    Parameters:
    audio (AudioSegment): The audio segment to process.
    silence_thresh (int): The threshold for considering silence (in dBFS).
    min_silence_len (int): The minimum length of silence to be removed (in ms).
    min_segment_len (int): The minimum length of a segment to be kept (in ms).
    fade_duration (int): The duration of fade-in and fade-out (in ms).
    silence_duration (int): The duration of silence to insert between segments (in ms).

    Returns:
    AudioSegment: The audio segment with silence removed.
    """
    # Detect non-silent segments
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Filter out segments that are shorter than the minimum segment length
    non_silent_segments = [audio[start:end] for start, end in non_silent_ranges if end - start >= min_segment_len]

    # Join segments with fade-in, fade-out, and silence between them
    output_audio = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    for segment in non_silent_segments:
        segment = segment.fade_in(fade_duration).fade_out(fade_duration)
        output_audio += segment + AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate)

    # Remove the trailing silence
    if silence_duration > 0:
        output_audio = output_audio[:-silence_duration]

    return output_audio


def break_into_chunks(audio, chunk_size=5000, fade_duration=50):
    """
    Breaks the audio into n_chunks equal parts, and from each part,
    extracts a segment of duration chunk_size milliseconds that is
    centered within that part. Applies fade-in and fade-out to each chunk,
    and returns a list of these chunks.

    :param audio: AudioSegment object representing the audio
    :param chunk_size: Size of each chunk in milliseconds
    :param fade_duration: Duration of fade in and fade out in milliseconds
    :return: List of AudioSegment chunks with fade-in and fade-out applied
    """
    chunks = []
    n_chunks = len(audio) // chunk_size
    if n_chunks == 0:
        return chunks

    part_duration = len(audio) / n_chunks

    for i in range(n_chunks):
        # Calculate the start and end times for each part
        part_start_time = int(i * part_duration)
        part_end_time = int((i + 1) * part_duration)

        # Determine the start and end times for the chunk centered within the part
        chunk_start_time = part_start_time + int((part_duration - chunk_size) / 2)
        chunk_end_time = chunk_start_time + chunk_size

        # Ensure chunk_end_time does not exceed audio length
        if chunk_end_time > len(audio):
            chunk_end_time = len(audio)
            chunk_start_time = chunk_end_time - chunk_size

        # Extract chunk from the audio
        chunk = audio[chunk_start_time:chunk_end_time]

        # Apply fade-in and fade-out to the chunk
        faded_chunk = chunk.fade_in(fade_duration).fade_out(fade_duration)

        # Add the processed chunk to the list
        chunks.append(faded_chunk)

    return chunks


def normalize_audio_rms(audio, target_rms=-15):
    """
    Normalizes the audio using the RMS method.

    Parameters:
    audio (AudioSegment): The audio segment to normalize.
    target_rms (float): The target RMS level in dB (default: -20 dB).

    Returns:
    AudioSegment: The normalized audio segment.
    """

    #TODO: IMPLEMENT THIS FUNCTION

    return audio