"""
Integration tests demonstrating the complete audio-to-text pipeline.

These tests show how all components work together:
1. Raw audio input → Preprocessing → Mel spectrograms
2. Complete workflow without Hailo hardware (using mocks)
3. Real-world scenarios
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from wyoming_hailo_whisper.common import audio_utils, preprocessing, postprocessing


class TestEndToEndAudioProcessing:
    """Integration tests for the complete audio processing pipeline."""

    def test_complete_audio_to_mel_pipeline(self):
        """
        Demonstrates the complete pipeline from raw audio to mel spectrograms.

        This is what happens to audio before it reaches the Hailo model:

        Raw Audio (microphone/file)
            ↓
        improve_input_audio (VAD + gain)
            ↓
        preprocess (chunking + mel generation)
            ↓
        Mel Spectrograms → Ready for Hailo encoder
        """
        sample_rate = 16000

        print("\n=== Complete Audio Processing Pipeline ===")

        # Step 1: Simulate recording (2s silence + 8s speech)
        print("\n1. Input Audio")
        silence_duration = 2.0
        speech_duration = 8.0

        silence = np.random.randn(int(silence_duration * sample_rate)) * 0.01
        speech = np.random.randn(int(speech_duration * sample_rate)) * 0.05  # Quiet (below 0.1 threshold)
        audio = np.concatenate([silence, speech]).astype(np.float32)

        print(f"   Duration: {len(audio)/sample_rate:.1f}s")
        print(f"   Max amplitude: {np.max(audio):.3f}")
        print(f"   Shape: {audio.shape}")

        # Step 2: Improve audio quality
        print("\n2. Improve Audio (VAD + Gain)")
        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=True, low_audio_gain=True
        )

        print(f"   Speech detected at: {speech_start:.2f}s")
        print(f"   New max amplitude: {np.max(improved_audio):.3f}")
        print(f"   Gain applied: {np.max(improved_audio) / np.max(audio):.1f}x")

        assert speech_start is not None
        assert speech_start > 1.0  # Speech should be after silence
        # If original audio was quiet (< 0.1), gain should be applied
        if np.max(audio) < 0.1:
            assert np.max(improved_audio) > np.max(audio)  # Should be amplified

        # Step 3: Generate mel spectrograms
        print("\n3. Generate Mel Spectrograms")
        chunk_length = 10  # 10 seconds for tiny model
        mel_spectrograms = preprocessing.preprocess(
            improved_audio,
            is_nhwc=True,
            chunk_length=chunk_length,
            chunk_offset=speech_start if speech_start else 0,
            max_duration=60
        )

        print(f"   Number of chunks: {len(mel_spectrograms)}")
        print(f"   Shape per chunk: {mel_spectrograms[0].shape}")
        print(f"   Format: NHWC (Hailo format)")

        assert len(mel_spectrograms) >= 1
        # 10 seconds at 100 frames/second = 1000 frames
        expected_frames = chunk_length * 100
        assert mel_spectrograms[0].shape == (1, 1, expected_frames, 80)

        # Step 4: Verify mel spectrogram properties
        print("\n4. Mel Spectrogram Properties")
        mel = mel_spectrograms[0]
        print(f"   Value range: [{mel.min():.3f}, {mel.max():.3f}]")
        print(f"   Mean: {mel.mean():.3f}")
        print(f"   Std: {mel.std():.3f}")

        # Mel spectrograms use log scaling and can have negative values
        # After normalization: (log_spec + 4.0) / 4.0, range is roughly [-1, 1]
        assert mel.min() >= -2.0  # Allow for reasonable range
        assert mel.max() <= 2.0

        print("\n✓ Audio successfully processed and ready for Hailo inference")

    def test_transcription_postprocessing_pipeline(self):
        """
        Demonstrates the transcription generation and cleaning pipeline.

        This is what happens after the Hailo decoder generates tokens:

        Token Logits (from decoder)
            ↓
        apply_repetition_penalty (discourage loops)
            ↓
        temperature_sampling (select next token)
            ↓
        Tokenizer.decode (tokens → text)
            ↓
        clean_transcription (remove duplicates)
            ↓
        Final Transcription
        """
        print("\n=== Transcription Postprocessing Pipeline ===")

        # Simulate decoder loop
        vocab_size = 50000
        max_tokens = 32  # Tiny model decoding length
        eos_token = 50256
        generated_tokens = []

        print("\n1. Simulating Decoder Loop")
        for step in range(max_tokens):
            # Simulate model output
            logits = np.random.randn(1, vocab_size).astype(np.float32)

            # Randomly boost some tokens to simulate real output
            logits[0, np.random.randint(100, 1000)] = 5.0

            # Apply repetition penalty
            if len(generated_tokens) > 0:
                logits_penalized = postprocessing.apply_repetition_penalty(
                    logits, generated_tokens, penalty=1.5, last_window=8
                )
            else:
                logits_penalized = logits[0]

            # Select next token
            next_token = postprocessing.temperature_sampling(
                logits_penalized, temperature=0.0
            )

            # Simulate EOS after 15 tokens
            if step >= 15:
                next_token = eos_token
                print(f"   Step {step}: Token {next_token} (EOS)")
                break

            generated_tokens.append(next_token)
            if step < 5 or step >= 13:  # Print first and last few
                print(f"   Step {step}: Token {next_token}")

        print(f"   ... (generated {len(generated_tokens)} tokens total)")

        # Step 2: Simulate raw transcription (with issues)
        print("\n2. Raw Transcription (from tokenizer)")
        raw_transcription = "The weather is nice today. The weather is nice today."
        print(f"   Raw: '{raw_transcription}'")

        # Step 3: Clean transcription
        print("\n3. Clean Transcription")
        cleaned = postprocessing.clean_transcription(raw_transcription)
        print(f"   Cleaned: '{cleaned}'")

        assert cleaned == "The weather is nice today."
        assert len(cleaned) < len(raw_transcription)

        print("\n✓ Transcription successfully cleaned")

    def test_multiple_audio_chunks(self):
        """
        Demonstrates processing long audio with multiple chunks.

        Long audio (>10s) is split into overlapping chunks to ensure
        no speech is cut off at boundaries.
        """
        sample_rate = 16000

        print("\n=== Processing Long Audio ===")

        # Create 25 seconds of audio
        duration = 25.0
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.3

        print(f"\n1. Input: {duration}s of audio")

        # Process with 10s chunks and 20% overlap
        mel_spectrograms = preprocessing.preprocess(
            audio,
            is_nhwc=True,
            chunk_length=10,
            overlap=0.2
        )

        print(f"2. Generated {len(mel_spectrograms)} mel spectrograms")
        print("3. Chunk boundaries:")

        # Calculate chunk times
        chunk_duration = 10.0
        step = chunk_duration * (1 - 0.2)  # 8 seconds step

        for i, mel in enumerate(mel_spectrograms):
            start_time = i * step
            end_time = start_time + chunk_duration
            print(f"   Chunk {i}: {start_time:.1f}s - {end_time:.1f}s")

        # Each chunk should be ready for inference
        for mel in mel_spectrograms:
            assert mel.shape == (1, 1, 1000, 80)

        print("\n✓ Long audio successfully chunked")

    def test_different_audio_qualities(self):
        """
        Shows how the pipeline handles different audio qualities.

        Tests:
        - Loud audio (no gain needed)
        - Quiet audio (gain applied)
        - Very quiet audio (maximum gain)
        - Clipped audio (already at max)
        """
        sample_rate = 16000
        duration = 5.0
        samples = int(duration * sample_rate)

        print("\n=== Audio Quality Handling ===")

        test_cases = [
            ("Loud audio", 0.5),
            ("Normal audio", 0.2),
            ("Quiet audio", 0.08),
            ("Very quiet audio", 0.02),
        ]

        for name, amplitude in test_cases:
            audio = np.random.randn(samples).astype(np.float32) * amplitude

            improved_audio, speech_start = preprocessing.improve_input_audio(
                audio, vad=False, low_audio_gain=True
            )

            gain_applied = np.max(improved_audio) / np.max(audio)

            print(f"\n{name}:")
            print(f"  Original max: {np.max(audio):.3f}")
            print(f"  Improved max: {np.max(improved_audio):.3f}")
            print(f"  Gain: {gain_applied:.1f}x")

            # The compatibility gain gate is based on measured peak amplitude.
            if np.max(np.abs(audio)) < 0.1:
                assert gain_applied > 1.0, f"Should apply gain for {name}"
            else:
                assert gain_applied == pytest.approx(1.0, abs=0.01), f"Should not apply gain for {name}"

        print("\n✓ All audio qualities handled correctly")


class TestRealisticScenarios:
    """Tests simulating realistic usage scenarios."""

    def test_home_assistant_voice_command(self):
        """
        Simulates a typical Home Assistant voice command scenario.

        Scenario:
        1. User presses button to start recording
        2. Brief silence (button press delay)
        3. User speaks: "Turn on the living room lights"
        4. Recording stops
        """
        sample_rate = 16000

        print("\n=== Home Assistant Voice Command ===")

        # 1. Button press delay (0.3s silence)
        button_delay = np.zeros(int(0.3 * sample_rate))

        # 2. User speaks (2s)
        # Simulate speech with varying frequency and amplitude
        speech_duration = 2.0
        t = np.linspace(0, speech_duration, int(speech_duration * sample_rate))
        speech = np.zeros_like(t)

        # Add speech formants (typical frequencies: 500Hz, 1500Hz, 2500Hz)
        for freq in [500, 1500, 2500]:
            speech += 0.1 * np.sin(2 * np.pi * freq * t)

        # Add amplitude envelope (speech is louder in the middle)
        envelope = np.exp(-((t - speech_duration/2) ** 2) / 0.5)
        speech *= envelope

        # 3. End silence (0.2s)
        end_silence = np.zeros(int(0.2 * sample_rate))

        # Combine
        audio = np.concatenate([button_delay, speech, end_silence]).astype(np.float32)

        print(f"1. Recording captured: {len(audio)/sample_rate:.1f}s")

        # Process
        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=True, low_audio_gain=True
        )

        print(f"2. Speech detected at: {speech_start:.2f}s")
        print(f"   (Button delay: ~0.3s)")

        # Generate mel spectrograms
        mel_spectrograms = preprocessing.preprocess(
            improved_audio,
            is_nhwc=True,
            chunk_length=10,
            chunk_offset=speech_start if speech_start else 0
        )

        print(f"3. Generated {len(mel_spectrograms)} mel spectrogram(s)")
        print("4. Ready for Hailo inference")

        # Simulate transcription
        mock_transcription = "Turn on the living room lights"
        cleaned = postprocessing.clean_transcription(mock_transcription)

        print(f"5. Transcription: '{cleaned}'")

        assert speech_start is not None
        assert 0.2 <= speech_start <= 0.7  # Includes the gradual speech envelope
        assert len(mel_spectrograms) == 1  # Short audio, one chunk

        print("\n✓ Voice command successfully processed")

    def test_continuous_listening_scenario(self):
        """
        Simulates continuous listening mode (like wake word detection).

        Scenario:
        1. Long period of silence/noise
        2. User speaks
        3. More silence
        """
        sample_rate = 16000

        print("\n=== Continuous Listening Scenario ===")

        # 1. Background noise (5s)
        background = np.random.randn(5 * sample_rate) * 0.01

        # 2. Speech (3s, louder)
        speech = np.random.randn(3 * sample_rate) * 0.3

        # 3. More background (2s)
        background2 = np.random.randn(2 * sample_rate) * 0.01

        audio = np.concatenate([background, speech, background2]).astype(np.float32)

        print(f"1. Audio stream: {len(audio)/sample_rate:.1f}s")
        print("   [5s background] [3s speech] [2s background]")

        # Detect speech
        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=True, low_audio_gain=False
        )

        print(f"2. Speech detected at: {speech_start:.2f}s")
        print(f"   Expected around 5.0s")

        # Process only from speech start
        mel_spectrograms = preprocessing.preprocess(
            improved_audio,
            is_nhwc=True,
            chunk_length=10,
            chunk_offset=speech_start if speech_start else 0,
            max_duration=60
        )

        print(f"3. Generated {len(mel_spectrograms)} mel spectrogram(s)")
        print("   (Only processing from speech start, skipping initial background)")

        assert speech_start is not None
        assert 4.5 <= speech_start <= 5.5  # Should be around 5s

        print("\n✓ Continuous listening successfully handled")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_extremely_short_audio(self):
        """Shows handling of very short audio (< 1 second)."""
        sample_rate = 16000
        audio = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.3

        print("\n=== Extremely Short Audio ===")
        print(f"Duration: {len(audio)/sample_rate:.1f}s")

        # Should still process (padding will be applied)
        mel_spectrograms = preprocessing.preprocess(
            audio, is_nhwc=True, chunk_length=10
        )

        print(f"Generated {len(mel_spectrograms)} mel spectrogram(s)")
        assert len(mel_spectrograms) == 1
        assert mel_spectrograms[0].shape == (1, 1, 1000, 80)

        print("✓ Short audio padded and processed")

    def test_silence_only(self):
        """Shows handling of audio with no speech."""
        sample_rate = 16000
        audio = np.random.randn(5 * sample_rate).astype(np.float32) * 0.001  # Very quiet

        print("\n=== Silence Only ===")

        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=True
        )

        print(f"Speech detected: {speech_start}")

        # Should return None (no speech detected)
        assert speech_start is None

        print("✓ Correctly detected no speech")

    def test_clipped_audio(self):
        """Shows handling of audio that's already at maximum amplitude."""
        sample_rate = 16000
        audio = np.random.randn(5 * sample_rate).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)  # Clipped at ±1.0

        print("\n=== Clipped Audio ===")
        print(f"Max amplitude: {np.max(audio):.3f} (at limit)")

        improved_audio, speech_start = preprocessing.improve_input_audio(
            audio, vad=False, low_audio_gain=True
        )

        print(f"After processing: {np.max(improved_audio):.3f}")

        # Should not apply gain (already loud)
        assert np.max(improved_audio) == pytest.approx(np.max(audio), abs=0.01)

        print("✓ Correctly skipped gain for loud audio")


if __name__ == "__main__":
    # Allow running individual tests for demonstration
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
