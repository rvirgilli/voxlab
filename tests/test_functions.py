import pytest
from pydub import AudioSegment
import numpy as np
from voxlab.preprocessing.functions import resample_audio, convert_to_mono, remove_silence, break_into_chunks


def generate_sine_wave(duration_ms, freq=440, sample_rate=44100, amplitude=0.5):
    """
    Generates a sine wave AudioSegment.

    Parameters:
    duration_ms (int): Duration of the sine wave in milliseconds.
    freq (int): Frequency of the sine wave.
    sample_rate (int): Sample rate of the sine wave.
    amplitude (float): Amplitude of the sine wave (0.0 to 1.0).

    Returns:
    AudioSegment: Generated sine wave audio segment.
    """
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t) * 32767
    sine_wave = sine_wave.astype(np.int16)  # Convert to 16-bit PCM format
    return AudioSegment(
        sine_wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit PCM
        channels=1
    )


@pytest.fixture
def sample_audio():
    # Create a 10-second sine wave audio segment for testing
    return generate_sine_wave(duration_ms=10000)


@pytest.fixture
def stereo_audio(sample_audio):
    # Create a stereo audio segment by duplicating the mono sine wave
    return sample_audio.set_channels(2)


def test_resample_audio(sample_audio):
    # Test resampling from 44.1 kHz to 22.05 kHz
    sample_audio = sample_audio.set_frame_rate(44100)
    resampled_audio = resample_audio(sample_audio, new_sample_rate=22050)
    assert resampled_audio.frame_rate == 22050

    # Test resampling from 44.1 kHz to 48 kHz
    resampled_audio = resample_audio(sample_audio, new_sample_rate=48000)
    assert resampled_audio.frame_rate == 48000

    # Test resampling from 48 kHz to 44.1 kHz
    sample_audio = sample_audio.set_frame_rate(48000)
    resampled_audio = resample_audio(sample_audio, new_sample_rate=44100)
    assert resampled_audio.frame_rate == 44100

    # Test resampling from 22.05 kHz to 44.1 kHz
    sample_audio = sample_audio.set_frame_rate(22050)
    resampled_audio = resample_audio(sample_audio, new_sample_rate=44100)
    assert resampled_audio.frame_rate == 44100


def test_convert_to_mono(stereo_audio):
    # Test conversion to mono using the left channel
    mono_audio_left = convert_to_mono(stereo_audio, method='left')
    assert mono_audio_left.channels == 1

    # Test conversion to mono using the right channel
    mono_audio_right = convert_to_mono(stereo_audio, method='right')
    assert mono_audio_right.channels == 1

    # Ensure error is raised for unsupported method
    with pytest.raises(ValueError):
        convert_to_mono(stereo_audio, method='mean')


def test_remove_silence():
    # Generate a 5-second sine wave followed by 5 seconds of silence
    sine_wave_audio = generate_sine_wave(duration_ms=5000)
    silent_audio = AudioSegment.silent(duration=5000)
    combined_audio = sine_wave_audio + silent_audio

    # Test removing silence
    processed_audio = remove_silence(combined_audio, silence_thresh=-30, min_silence_len=500,
                                     min_segment_len=1000, fade_duration=50, silence_duration=50)

    # Calculate expected length: sine_wave_audio length + silence between segments + fade_duration adjustment
    fade_duration = 50
    silence_duration = 50
    expected_length = len(sine_wave_audio) + silence_duration

    assert len(
        processed_audio) == expected_length, f"Expected {expected_length}, got {len(processed_audio)}"  # Remaining length with added silences


def test_break_into_chunks(sample_audio):
    # Test breaking audio into chunks
    chunks = break_into_chunks(sample_audio, chunk_size=2000, fade_duration=50)
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk) == 2000

    # Test breaking audio into larger chunks
    chunks = break_into_chunks(sample_audio, chunk_size=5000, fade_duration=50)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk) == 5000

    # Test breaking audio into smaller chunks
    chunks = break_into_chunks(sample_audio, chunk_size=1000, fade_duration=50)
    assert len(chunks) == 10
    for chunk in chunks:
        assert len(chunk) == 1000


if __name__ == "__main__":
    pytest.main()
