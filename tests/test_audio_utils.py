"""
Tests for audio utility functions.

These tests demonstrate how mel spectrogram generation works:
1. Audio loading and conversion
2. Padding/trimming to fixed length
3. STFT (Short-Time Fourier Transform)
4. Mel filterbank application
5. Log scaling and normalization
"""

import numpy as np
import torch
import pytest
from wyoming_hailo_whisper.common import audio_utils


class TestPadOrTrim:
    """Tests for pad_or_trim - demonstrates audio length normalization."""

    def test_pads_short_audio(self):
        """
        Shows how short audio is padded with zeros to reach target length.

        Whisper requires exactly 30s (480,000 samples at 16kHz).
        """
        # Create 1 second of audio (16,000 samples)
        audio = np.random.randn(16000).astype(np.float32)

        # Pad to 30 seconds (480,000 samples)
        target_length = audio_utils.N_SAMPLES  # 480,000
        padded = audio_utils.pad_or_trim(audio, length=target_length)

        assert len(padded) == target_length
        # Original audio should be preserved
        np.testing.assert_array_equal(padded[:16000], audio)
        # Rest should be zeros
        assert np.all(padded[16000:] == 0)

    def test_trims_long_audio(self):
        """Shows how audio longer than target is trimmed."""
        # Create 35 seconds of audio (560,000 samples)
        audio = np.random.randn(560000).astype(np.float32)

        # Trim to 30 seconds
        target_length = audio_utils.N_SAMPLES  # 480,000
        trimmed = audio_utils.pad_or_trim(audio, length=target_length)

        assert len(trimmed) == target_length
        # Should keep the first 480,000 samples
        np.testing.assert_array_equal(trimmed, audio[:target_length])

    def test_leaves_exact_length_unchanged(self):
        """Shows that audio of exact target length is unchanged."""
        # Create exactly 30 seconds
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)

        result = audio_utils.pad_or_trim(audio, length=audio_utils.N_SAMPLES)

        np.testing.assert_array_equal(result, audio)

    def test_works_with_torch_tensors(self):
        """Shows that pad_or_trim handles both NumPy arrays and PyTorch tensors."""
        # Create torch tensor
        audio = torch.randn(16000)

        padded = audio_utils.pad_or_trim(audio, length=audio_utils.N_SAMPLES)

        assert isinstance(padded, torch.Tensor)
        assert len(padded) == audio_utils.N_SAMPLES

    def test_custom_lengths(self):
        """Shows how to use custom target lengths for different chunk sizes."""
        audio = np.random.randn(50000).astype(np.float32)

        # Pad to 5 seconds (80,000 samples at 16kHz)
        target = 5 * audio_utils.SAMPLE_RATE  # 80,000
        result = audio_utils.pad_or_trim(audio, length=target)

        assert len(result) == target


class TestLogMelSpectrogram:
    """
    Tests for log_mel_spectrogram - demonstrates the core feature extraction.

    This function converts raw audio waveforms into mel spectrograms,
    the input format expected by Whisper models.
    """

    def test_generates_correct_shape(self):
        """
        Shows the output shape of mel spectrograms.

        For 30s audio at 16kHz:
        - Input: 480,000 samples
        - Output: (80, 3000) = 80 mel bins × 3000 time frames

        Time resolution: 10ms per frame (HOP_LENGTH=160 samples)
        """
        # Create 30 seconds of audio
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)

        mel = audio_utils.log_mel_spectrogram(audio, n_mels=80)

        # Should produce (80, 3000) shape
        assert mel.shape == (80, audio_utils.N_FRAMES)
        assert audio_utils.N_FRAMES == 3000
        assert isinstance(mel, torch.Tensor)

    def test_accepts_numpy_array(self):
        """Shows that the function accepts NumPy arrays."""
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)
        mel = audio_utils.log_mel_spectrogram(audio)

        assert mel.shape == (80, 3000)

    def test_accepts_torch_tensor(self):
        """Shows that the function accepts PyTorch tensors."""
        audio = torch.randn(audio_utils.N_SAMPLES)
        mel = audio_utils.log_mel_spectrogram(audio)

        assert mel.shape == (80, 3000)

    def test_output_range(self):
        """
        Shows the value range of mel spectrograms.

        After log scaling and normalization:
        - Range is approximately [0, 1]
        - log_spec = (log_spec + 4.0) / 4.0
        """
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)
        mel = audio_utils.log_mel_spectrogram(audio)

        # Normalization is relative to the signal maximum, so loud synthetic
        # white noise can exceed 1 while remaining in the expected range.
        assert torch.isfinite(mel).all()
        assert mel.min() >= -1.0
        assert mel.max() <= 2.0

    def test_different_mel_bins(self):
        """
        Shows support for 80 or 128 mel bins.

        - 80 mel bins: Used by Whisper tiny/base models
        - 128 mel bins: Used by larger Whisper models
        """
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)

        mel_80 = audio_utils.log_mel_spectrogram(audio, n_mels=80)
        mel_128 = audio_utils.log_mel_spectrogram(audio, n_mels=128)

        assert mel_80.shape == (80, 3000)
        assert mel_128.shape == (128, 3000)

    def test_padding_parameter(self):
        """Shows how padding adds zeros to the audio before processing."""
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)

        # With padding, more samples are processed
        mel_no_pad = audio_utils.log_mel_spectrogram(audio, padding=0)
        mel_with_pad = audio_utils.log_mel_spectrogram(audio, padding=1000)

        assert mel_with_pad.shape[0] == mel_no_pad.shape[0]
        assert mel_with_pad.shape[1] > mel_no_pad.shape[1]

    def test_consistent_output_for_same_input(self):
        """Shows that the same audio always produces the same mel spectrogram."""
        audio = np.random.randn(audio_utils.N_SAMPLES).astype(np.float32)

        mel_1 = audio_utils.log_mel_spectrogram(audio)
        mel_2 = audio_utils.log_mel_spectrogram(audio)

        torch.testing.assert_close(mel_1, mel_2)


class TestAudioConstants:
    """Tests demonstrating the audio hyperparameters used by Whisper."""

    def test_sample_rate(self):
        """
        Shows that Whisper uses 16kHz sampling rate.

        This is a standard rate for speech recognition:
        - High enough for speech frequencies (up to 8kHz)
        - Low enough for efficient processing
        """
        assert audio_utils.SAMPLE_RATE == 16000

    def test_stft_parameters(self):
        """
        Shows the Short-Time Fourier Transform (STFT) parameters.

        - N_FFT = 400: FFT window size (25ms at 16kHz)
        - HOP_LENGTH = 160: Hop between windows (10ms at 16kHz)

        This gives 100 frames per second.
        """
        assert audio_utils.N_FFT == 400
        assert audio_utils.HOP_LENGTH == 160
        assert audio_utils.FRAMES_PER_SECOND == 100

    def test_chunk_parameters(self):
        """
        Shows the chunk size expected by Whisper.

        - CHUNK_LENGTH = 30 seconds
        - N_SAMPLES = 480,000 samples (30s × 16kHz)
        - N_FRAMES = 3,000 frames (30s × 100 frames/s)
        """
        assert audio_utils.CHUNK_LENGTH == 30
        assert audio_utils.N_SAMPLES == 480000
        assert audio_utils.N_FRAMES == 3000

    def test_token_timing(self):
        """
        Shows how audio frames map to tokens.

        - Each token represents 2 frames (stride=2 in Whisper's convolution)
        - N_SAMPLES_PER_TOKEN = 320 samples (20ms)
        - TOKENS_PER_SECOND = 50 tokens/second
        """
        assert audio_utils.N_SAMPLES_PER_TOKEN == 320
        assert audio_utils.TOKENS_PER_SECOND == 50


class TestMelSpectrogramVisual:
    """Integration test showing mel spectrogram properties."""

    def test_mel_spectrogram_for_tone(self):
        """
        Shows how a pure tone appears in the mel spectrogram.

        This demonstrates the frequency analysis:
        - Low frequency tone → energy in lower mel bins
        - High frequency tone → energy in higher mel bins
        """
        sample_rate = audio_utils.SAMPLE_RATE
        duration = 30  # seconds
        n_samples = duration * sample_rate

        # Create a 1000 Hz tone (middle frequency for speech)
        t = np.linspace(0, duration, n_samples)
        frequency = 1000  # Hz
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

        mel = audio_utils.log_mel_spectrogram(audio, n_mels=80)

        # For a pure tone, energy should be concentrated in specific mel bins
        # Find which mel bin has the most energy
        energy_per_bin = mel.mean(dim=1)  # Average across time
        max_bin = torch.argmax(energy_per_bin).item()

        print(f"\n=== Mel Spectrogram for {frequency}Hz tone ===")
        print(f"Output shape: {mel.shape}")
        print(f"Peak energy at mel bin: {max_bin}/80")
        print(f"Value range: [{mel.min():.3f}, {mel.max():.3f}]")

        # 1000 Hz should appear in middle-range mel bins (roughly 20-40)
        assert 15 <= max_bin <= 50

    def test_mel_spectrogram_for_speech_like_signal(self):
        """
        Shows how complex audio (like speech) creates a full mel spectrogram.

        Speech has energy across many frequencies, so the mel spectrogram
        should have non-zero values across many mel bins.
        """
        sample_rate = audio_utils.SAMPLE_RATE
        duration = 5  # seconds
        n_samples = duration * sample_rate

        # Simulate speech-like signal: mix of frequencies with varying amplitude
        t = np.linspace(0, duration, n_samples)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Add multiple frequency components (typical speech range: 100-4000 Hz)
        for freq in [200, 500, 1000, 1500, 2000, 3000]:
            amplitude = np.random.uniform(0.1, 0.3)
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Pad to 30s for Whisper
        audio = audio_utils.pad_or_trim(audio, audio_utils.N_SAMPLES)
        mel = audio_utils.log_mel_spectrogram(audio, n_mels=80)

        # The signal occupies the first 5 seconds of a 30-second padded clip;
        # validate structure and dynamic range instead of averaging in silence.
        energy_per_bin = mel.max(dim=1).values
        active_bins = (energy_per_bin > 0.3).sum().item()

        print(f"\n=== Mel Spectrogram for speech-like signal ===")
        print(f"Active mel bins (>0.3 energy): {active_bins}/80")
        print(f"Mean energy: {energy_per_bin.mean():.3f}")

        assert active_bins >= 20
        assert torch.isfinite(mel).all()


class TestExactDiv:
    """Tests for exact_div utility function."""

    def test_exact_division(self):
        """Shows that exact_div enforces integer division."""
        result = audio_utils.exact_div(100, 10)
        assert result == 10

    def test_raises_on_inexact_division(self):
        """Shows that exact_div fails if division is not exact."""
        with pytest.raises(AssertionError):
            audio_utils.exact_div(100, 7)  # 100/7 is not an integer
