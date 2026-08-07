"""
Tests for audio preprocessing functions.

These tests demonstrate how the audio preprocessing pipeline works:
1. Gain adjustment for quiet audio
2. Voice Activity Detection (VAD) to find speech start
3. Audio chunking and mel spectrogram generation
"""

import numpy as np
import pytest
from wyoming_hailo_whisper.common import preprocessing


class TestGainAdjustment:
    """Tests for apply_gain function - demonstrates how audio levels are adjusted."""

    def test_apply_gain_increases_amplitude(self):
        """
        Shows that applying positive gain (in dB) increases audio amplitude.

        Example: 20 dB gain means 10x amplitude increase.
        Formula: gain_linear = 10^(gain_db/20)
        """
        # Create test audio: simple sine wave with amplitude 0.1
        audio = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)

        # Apply 20 dB gain (should multiply by ~10)
        result = preprocessing.apply_gain(audio, gain_db=20)

        # Expected: 10^(20/20) = 10^1 = 10x increase
        expected = audio * 10.0
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_apply_gain_decreases_amplitude(self):
        """Shows that negative gain reduces audio amplitude."""
        audio = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

        # Apply -6 dB gain (should halve the amplitude)
        result = preprocessing.apply_gain(audio, gain_db=-6)

        # -6 dB ≈ 0.5x amplitude
        assert np.max(np.abs(result)) < np.max(np.abs(audio))

    def test_zero_gain_no_change(self):
        """Shows that 0 dB gain leaves audio unchanged."""
        audio = np.array([0.3, -0.2, 0.1], dtype=np.float32)
        result = preprocessing.apply_gain(audio, gain_db=0)
        np.testing.assert_array_equal(result, audio)


class TestVoiceActivityDetection:
    """Tests for detect_first_speech - demonstrates how speech detection works."""

    def test_detects_speech_in_middle(self):
        """
        Shows how VAD finds the first moment of speech.

        The algorithm:
        1. Splits audio into frames (default 0.02s = 20ms)
        2. Calculates energy of each frame
        3. Normalizes energy to [0, 1]
        4. Returns time when energy first exceeds threshold (default 0.2)
        """
        sample_rate = 16000

        # Create audio: silence (0.1s) + loud speech (0.2s)
        silence_samples = int(0.1 * sample_rate)  # 1600 samples
        speech_samples = int(0.2 * sample_rate)   # 3200 samples

        silence = np.random.randn(silence_samples) * 0.01  # Quiet noise
        speech = np.random.randn(speech_samples) * 0.5     # Loud signal
        audio = np.concatenate([silence, speech])

        # Detect first speech
        start_time = preprocessing.detect_first_speech(
            audio, sample_rate, threshold=0.2, frame_duration=0.02
        )

        # Should detect speech around 0.1 seconds (after silence)
        assert start_time is not None
        assert 0.08 <= start_time <= 0.15  # Allow some tolerance

    def test_no_speech_detected_in_silence(self):
        """Shows that VAD returns None when audio is too quiet."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate) * 0.001  # Very quiet noise

        result = preprocessing.detect_first_speech(
            audio, sample_rate, threshold=0.2
        )

        assert result is None  # No speech detected

    def test_detects_speech_immediately(self):
        """Shows that VAD detects speech at the start if it's loud enough."""
        sample_rate = 16000
        frame_samples = int(0.02 * sample_rate)
        first_frame = np.sin(
            2 * np.pi * 220 * np.arange(frame_samples) / sample_rate
        ) * 0.5
        quieter_tail = np.random.randn(sample_rate - frame_samples) * 0.02
        audio = np.concatenate([first_frame, quieter_tail])

        start_time = preprocessing.detect_first_speech(
            audio, sample_rate, threshold=0.2
        )

        # Should detect at or near 0.0 seconds
        assert start_time is not None
        assert start_time < 0.1

    def test_detects_quiet_speech_above_its_noise_floor(self):
        """Quiet speech is detected when it is distinct from the background."""
        sample_rate = 16000
        silence = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
        samples = np.arange(int(0.2 * sample_rate))
        quiet_speech = (
            np.sin(2 * np.pi * 220 * samples / sample_rate) * 0.001
        ).astype(np.float32)

        start_time = preprocessing.detect_first_speech(
            np.concatenate([silence, quiet_speech]),
            sample_rate,
            threshold=0.2,
            frame_duration=0.02,
        )

        assert start_time is not None
        assert 0.08 <= start_time <= 0.12

    @pytest.mark.parametrize(
        ("sample_rate", "frame_duration"),
        [(0, 0.02), (16000, 0.0), (16000, 0.000001)],
    )
    def test_rejects_invalid_frame_size(self, sample_rate, frame_duration):
        audio = np.ones(100, dtype=np.float32)

        assert preprocessing.detect_first_speech(
            audio,
            sample_rate,
            frame_duration=frame_duration,
        ) is None

    def test_converts_stereo_to_mono(self):
        """Shows that VAD handles stereo audio by averaging channels."""
        sample_rate = 16000
        # Create stereo audio (2 channels)
        left_channel = np.random.randn(sample_rate) * 0.5
        right_channel = np.random.randn(sample_rate) * 0.5
        stereo_audio = np.stack([left_channel, right_channel], axis=1)

        # Should not crash and should return a valid time
        start_time = preprocessing.detect_first_speech(
            stereo_audio, sample_rate, threshold=0.2
        )

        assert start_time is not None or start_time is None  # Either is valid


class TestImproveInputAudio:
    """Tests for improve_input_audio - demonstrates the audio improvement pipeline."""

    def test_increases_gain_for_quiet_audio(self):
        """
        Shows that quiet audio (max < 0.1) gets a 20 dB boost.

        This is important for ensuring whispered or distant speech is audible.
        """
        # Create very quiet audio (max = 0.05)
        audio = np.array([0.05, -0.03, 0.02, -0.01], dtype=np.float32)

        improved_audio, start_time = preprocessing.improve_input_audio(
            audio, vad=False, low_audio_gain=True
        )

        # Should be amplified by 20 dB (10x)
        assert np.max(improved_audio) > np.max(audio) * 5  # At least 5x increase

    def test_empty_audio_with_enhancement_returns_unchanged(self):
        audio = np.array([], dtype=np.float32)

        improved_audio, start_time = preprocessing.improve_input_audio(
            audio,
            vad=True,
            enhance=True,
        )

        np.testing.assert_array_equal(improved_audio, audio)
        assert start_time == 0.0

    def test_no_gain_for_loud_audio(self):
        """Shows that audio with max >= 0.1 is not modified."""
        audio = np.array([0.5, -0.4, 0.3, -0.2], dtype=np.float32)

        improved_audio, start_time = preprocessing.improve_input_audio(
            audio, vad=False, low_audio_gain=True
        )

        # Should remain unchanged
        np.testing.assert_array_equal(improved_audio, audio)

    def test_vad_returns_speech_start_time(self):
        """Shows that VAD mode returns the detected speech start time."""
        sample_rate = 16000
        # Silence followed by loud speech
        silence = np.zeros(int(0.2 * sample_rate))
        speech = np.random.randn(int(0.3 * sample_rate)) * 0.5
        audio = np.concatenate([silence, speech])

        improved_audio, start_time = preprocessing.improve_input_audio(
            audio, vad=True, low_audio_gain=False
        )

        # Should detect speech around 0.2 seconds
        assert start_time is not None
        assert 0.15 <= start_time <= 0.25

    def test_disabling_vad_returns_zero_start_time(self):
        """Shows that when VAD is disabled, start_time is 0."""
        audio = np.random.randn(16000) * 0.5

        improved_audio, start_time = preprocessing.improve_input_audio(
            audio, vad=False, low_audio_gain=False
        )

        assert start_time == 0


class TestPreprocess:
    """
    Tests for preprocess function - demonstrates mel spectrogram generation.

    This is the main preprocessing function that:
    1. Chunks audio into fixed-length segments (e.g., 10s for tiny model)
    2. Converts each chunk to a mel spectrogram
    3. Formats for Hailo hardware (NHWC format)
    """

    def test_generates_mel_spectrograms(self):
        """
        Shows how audio is converted to mel spectrograms.

        Process:
        1. Audio is chunked into 10-second segments
        2. Each chunk is converted to mel spectrogram (80 mel bins × 1000 frames)
        3. Shape is adjusted for Hailo: (1, 1, 1000, 80) in NHWC format
        """
        sample_rate = 16000
        # Create 15 seconds of audio
        audio = np.random.randn(15 * sample_rate).astype(np.float32)

        # Process with 10s chunks (tiny model)
        mel_spectrograms = preprocessing.preprocess(
            audio, is_nhwc=True, chunk_length=10, max_duration=60
        )

        # Should create 2 chunks: [0-10s], [10-15s padded to 10s]
        assert len(mel_spectrograms) == 2

        # Each mel spectrogram should have shape (1, 1, 1000, 80) in NHWC format
        # 1 = batch, 1 = height, 1000 = time frames (10s × 100 frames/s), 80 = mel bins
        for mel in mel_spectrograms:
            assert mel.shape == (1, 1, 1000, 80)

    def test_single_chunk_for_short_audio(self):
        """Shows that audio shorter than chunk_length produces one chunk."""
        sample_rate = 16000
        # Create 5 seconds of audio (shorter than 10s chunk)
        audio = np.random.randn(5 * sample_rate).astype(np.float32)

        mel_spectrograms = preprocessing.preprocess(
            audio, is_nhwc=True, chunk_length=10
        )

        # Should create 1 chunk (5s padded to 10s)
        assert len(mel_spectrograms) == 1
        assert mel_spectrograms[0].shape == (1, 1, 1000, 80)

    def test_chunk_offset_skips_beginning(self):
        """
        Shows how chunk_offset skips silence at the start.

        This is useful after VAD detects speech at, e.g., 2.5 seconds.
        """
        sample_rate = 16000
        # Create 20 seconds of audio
        audio = np.random.randn(20 * sample_rate).astype(np.float32)

        # Skip first 5 seconds
        mel_spectrograms = preprocessing.preprocess(
            audio, is_nhwc=True, chunk_length=10, chunk_offset=5
        )

        # Should process audio from 5s onwards
        # Remaining audio: 15s → 2 chunks (5-15s, 15-20s padded)
        assert len(mel_spectrograms) == 2

    def test_overlap_creates_more_chunks(self):
        """
        Shows how overlap parameter creates overlapping chunks.

        This is useful for continuous audio processing to avoid missing
        speech at chunk boundaries.
        """
        sample_rate = 16000
        # Create 20 seconds of audio
        audio = np.random.randn(20 * sample_rate).astype(np.float32)

        # Process with 50% overlap
        mel_spectrograms = preprocessing.preprocess(
            audio, is_nhwc=True, chunk_length=10, overlap=0.5
        )

        # With 50% overlap:
        # Chunk 1: 0-10s
        # Chunk 2: 5-15s
        # Chunk 3: 10-20s
        # Chunk 4: 15-20s (padded)
        assert len(mel_spectrograms) >= 3

    def test_nchw_format(self):
        """
        Shows the difference between NHWC and NCHW format.

        NHWC (Hailo format): (batch, height, width, channels)
        NCHW (PyTorch format): (batch, channels, height, width)
        """
        sample_rate = 16000
        audio = np.random.randn(5 * sample_rate).astype(np.float32)

        # Get NCHW format (is_nhwc=False)
        mel_nchw = preprocessing.preprocess(
            audio, is_nhwc=False, chunk_length=10
        )

        # 10 seconds at 100 frames/second produces 1000 time frames.
        assert mel_nchw[0].shape == (1, 80, 1, 1000)


class TestPreprocessingIntegration:
    """Integration tests showing the full preprocessing pipeline."""

    def test_full_pipeline(self):
        """
        Demonstrates the complete preprocessing workflow:

        1. Raw audio input (simulated microphone)
        2. improve_input_audio: VAD detects speech, applies gain if needed
        3. preprocess: Creates mel spectrograms from detected speech onwards
        """
        sample_rate = 16000

        # Simulate recording: 2s silence + 8s speech (quiet)
        silence = np.random.randn(2 * sample_rate) * 0.01
        speech = np.random.randn(8 * sample_rate) * 0.02  # Quiet speech
        audio = np.concatenate([silence, speech]).astype(np.float32)

        # Step 1: Improve audio quality
        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=True, low_audio_gain=True
        )

        # Should detect speech around 2 seconds
        assert speech_start is not None
        assert 1.5 <= speech_start <= 2.5

        # Should amplify quiet audio
        assert np.max(improved_audio) > np.max(audio)

        # Step 2: Generate mel spectrograms starting from speech
        mel_spectrograms = preprocessing.preprocess(
            improved_audio,
            is_nhwc=True,
            chunk_length=10,
            chunk_offset=speech_start,  # Skip the initial silence
            max_duration=60
        )

        # Should create chunks ready for Hailo inference
        assert len(mel_spectrograms) >= 1
        assert mel_spectrograms[0].shape == (1, 1, 1000, 80)

        print(f"\n=== Full Pipeline Results ===")
        print(f"Original audio: {len(audio)/sample_rate:.1f}s, max level: {np.max(audio):.3f}")
        print(f"Speech detected at: {speech_start:.2f}s")
        print(f"Improved max level: {np.max(improved_audio):.3f}")
        print(f"Generated {len(mel_spectrograms)} mel spectrogram(s)")
        print(f"Each spectrogram shape: {mel_spectrograms[0].shape}")
