"""Preprocessing functions for Whisper audio data."""

import wyoming_hailo_whisper.common.audio_utils as audio_utils
import numpy as np
import logging
from scipy.signal import butter, sosfilt, stft, istft

_LOGGER = logging.getLogger(__name__)

def preprocess(audio, is_nhwc=False, chunk_length = 10, chunk_offset=0, max_duration = 60, overlap=0.0):
    """
    Generate the mel spectrograms

    Parameters:
    - audio: The audio sample.
    - chunk_length: Length in seconds of each audio chunk to process. This must match the input length of the model.
    - chunk_offset: Position - in seconds - to start processing the audio. This is useful for skipping silence at the beginning of the audio.
    - max_duration: Max duration of the audio sample to process.
    - overlap: Overlap between chunks. This is useful for continuous audio processing. Add some overlap (e.g. 0.2) when processing an audio longer than 10 seonds.
    """
    # Limit the audio duration
    sample_rate = audio_utils.SAMPLE_RATE
    max_samples = max_duration * sample_rate
    offset = int(chunk_offset * sample_rate)

    # Define parameters for chunking
    segment_duration = chunk_length  # in seconds
    segment_samples = segment_duration * sample_rate
    step = int(segment_samples * (1 - overlap))

    audio = audio[offset:max_samples]
    mel_spectrograms = []

    for start in range(0, len(audio), step):
        end = int(start + segment_samples)
        if start >= len(audio):
            break
        chunk = audio[start:end]

        # Ensure the chunk is 10s long (Whisper requires this)
        chunk = audio_utils.pad_or_trim(chunk, int(segment_duration * sample_rate))

        # Convert to Mel spectrogram
        mel = audio_utils.log_mel_spectrogram(chunk).to("cpu")
        # Run the encoder

        mel = np.expand_dims(mel, axis=0)  # Add new axis to match shape (1, 80, 1, 1000)
        #print(mel.shape)
        mel = np.expand_dims(mel, axis=2)
        #print(mel.shape)

        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])

        mel_spectrograms.append(mel)

    return mel_spectrograms


def apply_gain(audio, gain_db):
    """
    Apply gain to the audio signal.
    Parameters:
    - audio: The audio sample.
    - gain_db: Gain in decibels (dB).
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def highpass_filter(audio, sample_rate, cutoff=80, order=5):
    """
    Remove frequencies below cutoff Hz (DC offset, rumble, mains hum).
    """
    sos = butter(order, cutoff, btype='highpass', fs=sample_rate, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def rms_normalize(audio, target_rms=0.1, max_gain_db=30):
    """
    Normalize audio to a target RMS level for consistent input volume.
    Limits gain to max_gain_db to avoid over-amplifying noise.
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-8:
        return audio  # silence, don't amplify
    gain = target_rms / current_rms
    max_gain_linear = 10 ** (max_gain_db / 20)
    gain = min(gain, max_gain_linear)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def spectral_noise_reduce(audio, sample_rate, spectral_floor=0.08):
    """
    Reduce stationary background noise using Wiener-like spectral masking.

    Estimates the noise spectrum from the quietest 20% of STFT frames,
    then applies a soft gain mask based on per-frame SNR. This is
    conservative to avoid removing speech or introducing artifacts.
    """
    nperseg = 512
    noverlap = nperseg // 2

    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise from the quietest 20% of frames
    frame_energy = np.sum(magnitude ** 2, axis=0)
    n_noise_frames = max(1, int(0.2 * len(frame_energy)))
    noise_frame_indices = np.argsort(frame_energy)[:n_noise_frames]
    noise_spectrum = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

    # Wiener-like soft mask: gain = SNR / (SNR + 1)
    noise_power = noise_spectrum ** 2 + 1e-10
    signal_power = magnitude ** 2
    snr = signal_power / noise_power
    gain = snr / (snr + 1.0)
    gain = np.maximum(gain, spectral_floor)

    clean_Zxx = magnitude * gain * np.exp(1j * phase)
    _, clean_audio = istft(clean_Zxx, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Match original length
    if len(clean_audio) > len(audio):
        clean_audio = clean_audio[:len(audio)]
    elif len(clean_audio) < len(audio):
        clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))

    return clean_audio.astype(np.float32)


def improve_input_audio(audio, vad=True, enhance=False, low_audio_gain=None):
    """
    Improve the input audio quality before transcription.

    Processing pipeline:
    1. High-pass filter (remove DC offset, rumble, mains hum)
    2. Spectral noise reduction (reduce stationary background noise)
    3. RMS normalization (consistent input level)
    4. Voice activity detection (find speech onset)
    """
    sample_rate = audio_utils.SAMPLE_RATE

    if np.size(audio) == 0:
        return audio, 0.0

    # Backward-compatible alias used by the original fork. When supplied, keep
    # its simple 20 dB boost semantics instead of enabling the new DSP chain.
    if low_audio_gain is not None:
        if low_audio_gain and audio.size and np.max(np.abs(audio)) < 0.1:
            audio = apply_gain(audio, gain_db=20)
    elif enhance:
        peak_before = np.max(np.abs(audio))
        rms_before = np.sqrt(np.mean(audio ** 2))
        _LOGGER.info("Audio before enhancement: peak=%.4f, rms=%.4f, samples=%d",
                     peak_before, rms_before, len(audio))

        # 1. High-pass filter: remove DC offset and low-frequency noise
        audio = highpass_filter(audio, sample_rate, cutoff=80)

        # 2. Spectral noise reduction: reduce stationary background noise
        audio = spectral_noise_reduce(audio, sample_rate)

        # 3. RMS normalization: ensure consistent input level
        audio = rms_normalize(audio, target_rms=0.1)

        peak_after = np.max(np.abs(audio))
        rms_after = np.sqrt(np.mean(audio ** 2))
        _LOGGER.info("Audio after enhancement: peak=%.4f, rms=%.4f",
                     peak_after, rms_after)

    start_time = 0
    if vad:
        start_time = detect_first_speech(audio, sample_rate, threshold=0.2, frame_duration=0.2)
        if start_time is not None:
            _LOGGER.info(f"Speech detected at {start_time:.2f} seconds.")
        else:
            _LOGGER.info("No speech detected.")
    return audio, start_time


def detect_first_speech(audio_data, sample_rate, threshold=0.2, frame_duration=0.02):
    """
    Detect the first time when human speech occurs in preloaded audio data.

    Parameters:
    - audio_data: NumPy array containing the audio samples.
    - sample_rate: Sample rate of the audio data.
    - threshold: Energy threshold for detecting speech (default: 0.02).
    - frame_duration: Duration of each frame in seconds (default: 0.02).

    Returns:
    - start_time: The time (in seconds) when speech is first detected.
    """
    # Convert stereo to mono if necessary
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Calculate frame size in samples
    frame_size = int(frame_duration * sample_rate)
    if frame_size <= 0 or len(audio_data) == 0:
        return None

    # Split the audio into frames
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]
    if not frames:
        return None

    # Calculate the energy of each frame
    energy = np.asarray(
        [np.sum(np.abs(frame) ** 2) / len(frame) for frame in frames]
    )

    # Estimate the background level and require speech to rise above it. This
    # keeps quiet-but-distinct speech detectable without treating flat,
    # low-level microphone noise as speech.
    max_energy = float(np.max(energy))
    if max_energy <= np.finfo(np.float64).eps:
        return None

    noise_floor = float(np.percentile(energy, 10, method="lower"))
    minimum_contrast = max(noise_floor, np.finfo(np.float64).eps) * 2.0
    if max_energy <= minimum_contrast:
        return None

    threshold_energy = max(
        noise_floor + threshold * (max_energy - noise_floor),
        minimum_contrast,
    )

    # Detect the first frame above both the relative and estimated-noise gates.
    for i, e in enumerate(energy):
        if e > threshold_energy:
            start_time = i * frame_duration
            #start_time_rounded = math.floor(start_time)
            start_time_rounded = round(start_time, 1)
            return start_time_rounded

    return None  # No speech detected
