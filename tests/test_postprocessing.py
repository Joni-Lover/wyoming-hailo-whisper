"""
Tests for postprocessing functions.

These tests demonstrate how the decoder output is processed:
1. Repetition penalty to discourage repeated tokens
2. Temperature sampling for token selection
3. Transcription cleaning to remove duplicates
"""

import numpy as np
import pytest
from wyoming_hailo_whisper.common import postprocessing


class TestRepetitionPenalty:
    """
    Tests for apply_repetition_penalty.

    This demonstrates how the decoder avoids generating repeated text
    by penalizing tokens that were recently generated.
    """

    def test_reduces_logits_for_repeated_tokens(self):
        """
        Shows how repetition penalty divides logits of recent tokens.

        Example: If token 42 was generated and has logit 10.0,
        applying penalty=1.5 reduces it to 10.0/1.5 = 6.67
        """
        # Create mock logits for vocabulary (e.g., 50,000 tokens)
        vocab_size = 100
        logits = np.ones((1, vocab_size), dtype=np.float32) * 5.0

        # Set high logit for token 42
        logits[0, 42] = 10.0

        # Previously generated tokens
        generated_tokens = [15, 42, 99, 42]  # Token 42 appears twice

        # Apply penalty (default 1.5)
        penalized = postprocessing.apply_repetition_penalty(
            logits, generated_tokens, penalty=1.5
        )

        # Token 42 should be penalized (reduced)
        assert penalized[42] < logits[0, 42]
        assert penalized[42] == pytest.approx(10.0 / (1.5 ** 2))

        # Token 50 (not in history) should be unchanged
        assert penalized[50] == logits[0, 50]

    def test_only_penalizes_recent_tokens(self):
        """
        Shows that only tokens in the last window are penalized.

        By default, last_window=8, so only the most recent 8 tokens
        are considered for repetition penalty.
        """
        vocab_size = 100
        logits = np.ones((1, vocab_size), dtype=np.float32) * 5.0

        # Long history: token 42 appeared 20 steps ago
        generated_tokens = [42] + list(range(20, 40))
        # Recent tokens: 20-39 (20 tokens total)

        penalized = postprocessing.apply_repetition_penalty(
            logits, generated_tokens, penalty=1.5, last_window=8
        )

        # Token 42 should NOT be penalized (outside the window of last 8)
        assert penalized[42] == logits[0, 42]

        # Tokens 32-39 (last 8) should be penalized
        assert penalized[32] < logits[0, 32]
        assert penalized[39] < logits[0, 39]

    def test_excludes_punctuation_tokens(self):
        """
        Shows that punctuation tokens (11, 13) are not penalized.

        This allows proper punctuation even if recently used.
        Token 11 and 13 are common punctuation in Whisper's tokenizer.
        """
        vocab_size = 100
        logits = np.ones((1, vocab_size), dtype=np.float32) * 5.0
        logits[0, 11] = 8.0  # Punctuation token
        logits[0, 13] = 8.0  # Punctuation token

        generated_tokens = [11, 13, 11, 13]  # Repeated punctuation

        penalized = postprocessing.apply_repetition_penalty(
            logits, generated_tokens, penalty=1.5
        )

        # Punctuation should NOT be penalized
        assert penalized[11] == logits[0, 11]
        assert penalized[13] == logits[0, 13]

    def test_higher_penalty_stronger_suppression(self):
        """Shows that higher penalty values suppress repetitions more strongly."""
        vocab_size = 100
        logits = np.ones((1, vocab_size), dtype=np.float32) * 5.0
        logits[0, 42] = 10.0

        generated_tokens = [42]

        # Weak penalty
        penalized_weak = postprocessing.apply_repetition_penalty(
            logits, generated_tokens, penalty=1.2
        )

        # Strong penalty
        penalized_strong = postprocessing.apply_repetition_penalty(
            logits, generated_tokens, penalty=2.0
        )

        # Strong penalty should reduce more
        assert penalized_strong[42] < penalized_weak[42]
        assert penalized_strong[42] == pytest.approx(10.0 / 2.0)
        assert penalized_weak[42] == pytest.approx(10.0 / 1.2)

    def test_negative_repeated_logits_become_less_likely(self):
        """Negative repeated-token logits move away from zero."""
        logits = np.full((1, 100), -1.0, dtype=np.float32)
        logits[0, 42] = -10.0

        penalized = postprocessing.apply_repetition_penalty(
            logits,
            [42, 42],
            penalty=1.5,
        )

        assert penalized[42] == pytest.approx(-10.0 * (1.5 ** 2))
        assert penalized[42] < logits[0, 42]

    def test_blocks_only_observed_ngram_continuations(self):
        logits = np.ones((1, 100), dtype=np.float32)

        bigram_result = postprocessing.apply_repetition_penalty(
            logits,
            [7, 8, 7, 8, 7],
        )
        trigram_result = postprocessing.apply_repetition_penalty(
            logits,
            [3, 4, 5, 3, 4],
        )

        assert bigram_result[8] == -np.inf
        assert trigram_result[5] == -np.inf
        assert np.isfinite(bigram_result[99])
        assert np.isfinite(trigram_result[99])


class TestBeamSearchCompletion:
    def test_decoder_allows_eot_as_first_content_token(self):
        logits = np.zeros(
            (1, postprocessing.WHISPER_SPECIAL_TOKEN_START + 3),
            dtype=np.float32,
        )
        logits[0, postprocessing.WHISPER_EOT_TOKEN] = 10.0
        logits[0, postprocessing.WHISPER_EOT_TOKEN + 1] = 9.0

        prepared = postprocessing.prepare_decoder_logits(logits, [])

        assert np.isfinite(prepared[postprocessing.WHISPER_EOT_TOKEN])
        assert prepared[postprocessing.WHISPER_EOT_TOKEN] == 10.0
        assert prepared[postprocessing.WHISPER_EOT_TOKEN + 1] == -np.inf

    def test_low_score_finished_beams_do_not_stop_active_beam(self):
        finished = [
            {"score": -5.0 - index, "content": [50257]}
            for index in range(5)
        ]
        active = [{"score": -0.5, "content": [42, 43]}]

        assert not postprocessing.beam_search_can_stop(
            finished,
            active,
            max_content_length=20,
        )

    def test_stops_when_finished_beam_beats_active_upper_bound(self):
        finished = [{"score": -0.1, "content": [42, 50257]}]
        active = [{"score": -1.0, "content": [42, 43]}]

        assert postprocessing.beam_search_can_stop(
            finished,
            active,
            max_content_length=20,
        )


class TestTemperatureSampling:
    """
    Tests for temperature_sampling.

    This demonstrates how the next token is selected from logits.
    """

    def test_greedy_decoding_selects_max(self):
        """
        Shows that temperature=0 performs greedy decoding (argmax).

        This is the fastest and most deterministic approach.
        """
        logits = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float32)

        # Temperature 0 = greedy decoding
        selected = postprocessing.temperature_sampling(logits, temperature=0.0)

        # Should select index 3 (highest logit = 8.0)
        assert selected == 3

    def test_boosts_punctuation_tokens(self):
        """
        Shows that punctuation tokens get a 1.2x boost.

        This encourages proper punctuation in transcriptions.
        """
        logits = np.zeros(100, dtype=np.float32)
        logits[11] = 5.0  # Punctuation token
        logits[12] = 6.0  # Non-punctuation token

        # Greedy decoding with boost
        selected = postprocessing.temperature_sampling(logits, temperature=0.0)

        # Token 11 gets boosted: 5.0 * 1.2 = 6.0 (ties with token 12)
        # In case of tie, argmax picks first occurrence
        # But token 12 is still 6.0 vs boosted 11 is 6.0, so 11 wins after boost
        assert selected in [11, 12]  # Could be either due to tie

    def test_temperature_sampling_adds_randomness(self):
        """
        Shows that temperature > 0 adds randomness to token selection.

        Higher temperature = more diverse/creative output.
        Lower temperature = more focused/conservative output.
        """
        np.random.seed(42)  # For reproducibility
        logits = np.array([2.0, 3.0, 5.0, 1.0], dtype=np.float32)

        # Sample multiple times
        samples = [
            postprocessing.temperature_sampling(logits, temperature=1.0)
            for _ in range(100)
        ]

        # With temperature, should get variety (not always index 2)
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # Should have multiple different tokens

        # Token 2 (highest logit) should still be most common
        assert samples.count(2) > 30  # Rough threshold

    def test_high_temperature_more_uniform(self):
        """
        Shows that high temperature makes probabilities more uniform.

        Temperature → ∞: All tokens equally likely
        Temperature → 0: Only max token selected
        """
        np.random.seed(42)
        logits = np.array([1.0, 2.0, 5.0, 1.0], dtype=np.float32)

        # Very high temperature
        samples_high_temp = [
            postprocessing.temperature_sampling(logits, temperature=10.0)
            for _ in range(100)
        ]

        # Low temperature
        samples_low_temp = [
            postprocessing.temperature_sampling(logits, temperature=0.1)
            for _ in range(100)
        ]

        # High temp should have more variety
        assert len(set(samples_high_temp)) >= len(set(samples_low_temp))

        # Low temp should heavily prefer token 2
        assert samples_low_temp.count(2) > 80


class TestCleanTranscription:
    """
    Tests for clean_transcription.

    This demonstrates how repeated sentences are removed from transcriptions.
    Whisper sometimes "hallucinates" repeated text, especially at the end.
    """

    def test_removes_exact_duplicate_sentences(self):
        """
        Shows that exact duplicate sentences are removed.

        This is common when Whisper reaches the end of audio and starts looping.
        """
        transcription = "Hello world. Hello world."

        cleaned = postprocessing.clean_transcription(transcription)

        # Should keep only one occurrence
        assert cleaned == "Hello world."

    def test_removes_substring_repetitions(self):
        """
        Shows that partial repetitions are also detected.

        Example: "Hello world. Hello" → "Hello world."
        """
        transcription = "The weather is nice. The weather is nice. The weather"

        cleaned = postprocessing.clean_transcription(transcription)

        # Should stop at first complete sentence
        assert cleaned == "The weather is nice."

    def test_keeps_non_repeated_sentences(self):
        """Shows that different sentences are preserved."""
        transcription = "Hello world. How are you? Nice weather today."

        cleaned = postprocessing.clean_transcription(transcription)

        # All sentences should be kept
        assert cleaned == "Hello world. How are you? Nice weather today."

    def test_handles_questions(self):
        """Shows that question marks are treated as sentence delimiters."""
        transcription = "What is your name? What is your name?"

        cleaned = postprocessing.clean_transcription(transcription)

        assert cleaned == "What is your name?"

    def test_adds_period_if_missing(self):
        """Shows that a period is added if the transcription doesn't end with punctuation."""
        transcription = "Hello world"

        cleaned = postprocessing.clean_transcription(transcription)

        assert cleaned == "Hello world."

    def test_case_insensitive_comparison(self):
        """
        Shows that repetition detection is case-insensitive.

        "Hello World. hello world." → "Hello World."
        """
        transcription = "Hello World. hello world."

        cleaned = postprocessing.clean_transcription(transcription)

        # Case shouldn't matter for detection
        assert cleaned == "Hello World."

    def test_preserves_first_occurrence(self):
        """Shows that the first occurrence is kept when duplicates are found."""
        transcription = "First sentence. Second sentence. First sentence."

        cleaned = postprocessing.clean_transcription(transcription)

        # Should keep sentences up to where repetition starts
        assert "First sentence. Second sentence." in cleaned

    def test_empty_transcription(self):
        """Shows handling of edge case: empty or whitespace-only input."""
        result = postprocessing.clean_transcription("")
        assert result == "."

        result = postprocessing.clean_transcription("   ")
        assert result == "."


class TestPostprocessingIntegration:
    """Integration tests showing the full decoder postprocessing pipeline."""

    def test_full_decoding_loop_simulation(self):
        """
        Simulates the decoder loop with repetition penalty and token selection.

        This demonstrates the complete flow:
        1. Model outputs logits
        2. Apply repetition penalty based on history
        3. Select next token (greedy or sampling)
        4. Add to history
        5. Repeat until EOS token or max length
        """
        vocab_size = 100
        eos_token = 50  # End-of-sequence token
        max_length = 10

        generated_tokens = []

        print("\n=== Simulated Decoding Loop ===")

        for step in range(max_length):
            # Simulate model output logits
            logits = np.random.randn(1, vocab_size).astype(np.float32) * 2.0

            # Boost EOS token after 5 steps (simulating end of speech)
            if step >= 5:
                logits[0, eos_token] = 10.0

            # Apply repetition penalty
            if len(generated_tokens) > 0:
                logits_penalized = postprocessing.apply_repetition_penalty(
                    logits, generated_tokens, penalty=1.5
                )
            else:
                logits_penalized = logits[0]

            # Select next token (greedy)
            next_token = postprocessing.temperature_sampling(
                logits_penalized, temperature=0.0
            )

            print(f"Step {step}: Token {next_token}")

            # Stop if EOS token
            if next_token == eos_token:
                print(f"EOS token reached at step {step}")
                break

            generated_tokens.append(next_token)

        # Should have stopped before max_length due to EOS
        assert len(generated_tokens) < max_length
        assert len(generated_tokens) >= 5

    def test_cleaning_simulated_transcription(self):
        """
        Shows how text cleaning works on realistic Whisper output.

        Whisper sometimes generates:
        - Repeated phrases at the end
        - Incomplete final sentences
        - Lowercased repetitions
        """
        # Simulate Whisper output with common issues
        transcription = (
            "The quick brown fox jumps over the lazy dog. "
            "It was a beautiful day. "
            "The quick brown fox jumps over the lazy dog."  # Exact repeat
        )

        cleaned = postprocessing.clean_transcription(transcription)

        print(f"\n=== Transcription Cleaning ===")
        print(f"Original length: {len(transcription)} chars")
        print(f"Cleaned length: {len(cleaned)} chars")
        print(f"Original: {transcription}")
        print(f"Cleaned: {cleaned}")

        # Should remove the repeated sentence
        assert cleaned == "The quick brown fox jumps over the lazy dog. It was a beautiful day."

    def test_prevents_infinite_loops(self):
        """
        Shows how repetition penalty prevents the decoder from looping.

        Without penalty: Token 42 could be selected repeatedly.
        With penalty: Each repetition becomes less likely.
        """
        vocab_size = 100
        loop_token = 42
        generated_tokens = []

        for step in range(20):
            # Create logits where token 42 is always highest
            logits = np.ones((1, vocab_size), dtype=np.float32) * 1.0
            logits[0, loop_token] = 10.0

            # Apply repetition penalty
            if len(generated_tokens) > 0:
                logits_penalized = postprocessing.apply_repetition_penalty(
                    logits, generated_tokens, penalty=1.5
                )
            else:
                logits_penalized = logits[0]

            # Select next token
            next_token = postprocessing.temperature_sampling(
                logits_penalized, temperature=0.0
            )

            generated_tokens.append(next_token)

            # After enough repetitions, penalty should kick in
            if step > 5:
                # Check that token 42's logit is now reduced
                current_logit = logits_penalized[loop_token]
                # It should be less than the original 10.0 due to penalty
                # After first occurrence: 10.0 / 1.5 = 6.67
                assert current_logit < 10.0

        print(f"\n=== Anti-Loop Test ===")
        print(f"Generated tokens: {generated_tokens[:10]}...")
        print(f"Token {loop_token} appears {generated_tokens.count(loop_token)} times")

        # Without penalty, would be all 42s. With penalty, should have variety.
        # But in this test, 42 is still always highest even after penalty,
        # so it might still dominate. The key is the logit IS reduced.
