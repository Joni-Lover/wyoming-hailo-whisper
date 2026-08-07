"""Postprocessing functions for Whisper-generated transcriptions."""

import numpy as np
import re


excluded_tokens = [11, 13]  # Punctuation tokens to exclude from repetition penalty

# All Whisper special tokens start at this ID
WHISPER_SPECIAL_TOKEN_START = 50257
WHISPER_EOT_TOKEN = 50257


def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=16):
    """
    Apply frequency-scaled repetition penalty to the logits.

    Tokens that appear multiple times in the recent window get exponentially
    stronger, sign-aware penalties (penalty^count). Positive logits are
    divided and negative logits are multiplied so both become less likely.
    Tokens repeated 3+ consecutive times and tokens forming repeated bigrams
    are suppressed entirely.
    """
    from collections import Counter

    # Decoding helpers must not mutate the model output or another beam's view.
    logits = np.squeeze(logits, axis=0).copy()
    recent_tokens = generated_tokens[-last_window:] if len(generated_tokens) >= last_window else generated_tokens

    # Count occurrences for frequency-scaled penalty
    token_counts = Counter(recent_tokens)

    for token, count in token_counts.items():
        if token not in excluded_tokens and token < WHISPER_SPECIAL_TOKEN_START:
            scaled_penalty = penalty ** count
            if logits[token] < 0:
                logits[token] *= scaled_penalty
            else:
                logits[token] /= scaled_penalty

    # Suppress tokens repeated more than 3 consecutive times
    if len(generated_tokens) >= 3:
        last_three = generated_tokens[-3:]
        if (
            last_three[0] == last_three[1] == last_three[2]
            and last_three[0] not in excluded_tokens
        ):
            logits[last_three[0]] = -np.inf

    # N-gram blocking: suppress tokens that would form a repeated bigram or trigram
    if len(generated_tokens) >= 2:
        # Bigram blocking: if (prev_token, candidate) already appeared 2+ times, suppress candidate
        prev_token = generated_tokens[-1]
        bigram_counts = Counter()
        for j in range(len(generated_tokens) - 1):
            bigram_counts[(generated_tokens[j], generated_tokens[j + 1])] += 1
        blocked_tokens = {
            token_id
            for (first_token, token_id), count in bigram_counts.items()
            if first_token == prev_token and count >= 2
        }
        for token_id in blocked_tokens:
            if (
                0 <= token_id < min(len(logits), WHISPER_SPECIAL_TOKEN_START)
                and token_id not in excluded_tokens
            ):
                logits[token_id] = -np.inf

    if len(generated_tokens) >= 4:
        # Trigram blocking: if (t-2, t-1, candidate) already appeared, suppress candidate
        t_minus_2 = generated_tokens[-2]
        t_minus_1 = generated_tokens[-1]
        blocked_tokens = {
            generated_tokens[j + 2]
            for j in range(len(generated_tokens) - 2)
            if (
                generated_tokens[j] == t_minus_2
                and generated_tokens[j + 1] == t_minus_1
            )
        }
        for token_id in blocked_tokens:
            if (
                0 <= token_id < min(len(logits), WHISPER_SPECIAL_TOKEN_START)
                and token_id not in excluded_tokens
            ):
                logits[token_id] = -np.inf

    return logits


def length_normalized_score(score, length, alpha=0.6):
    """Normalize a beam score while avoiding division by zero."""
    return score / (max(length, 1) ** alpha)


def beam_search_can_stop(finished_beams, active_beams, max_content_length, alpha=0.6):
    """Return whether no active beam can beat the best finished beam.

    Log-probabilities are non-positive, so an active beam's optimistic bound
    assumes all remaining tokens have log-probability zero and uses the
    maximum possible output length for normalization.
    """
    if not finished_beams:
        return False
    if not active_beams:
        return True

    best_finished = max(
        length_normalized_score(beam["score"], len(beam["content"]), alpha)
        for beam in finished_beams
    )
    best_active_bound = max(
        length_normalized_score(beam["score"], max_content_length, alpha)
        for beam in active_beams
    )
    return best_finished >= best_active_bound


def suppress_special_tokens(logits, allow_eot=True):
    """
    Suppress all special tokens during content generation.
    Optionally allows EOT to remain unsuppressed.
    """
    start = WHISPER_EOT_TOKEN + 1 if allow_eot else WHISPER_EOT_TOKEN
    logits[start:] = -np.inf
    return logits


def prepare_decoder_logits(logits, generated_tokens, penalty=1.5):
    """Apply Hailo decoding constraints while always allowing EOT.

    EOT is a valid first prediction for a silent chunk. Suppressing it until a
    content token exists forces Whisper to emit a lexical token and can create
    hallucinated text.
    """
    logits = apply_repetition_penalty(logits, generated_tokens, penalty=penalty)
    return suppress_special_tokens(logits, allow_eot=True)


def temperature_sampling(logits, temperature=0.0):
    """
    Apply temperature sampling to the logits.
    """
    # Boost the logits for punctuation tokens
    for punct_idx in excluded_tokens:
        if punct_idx < len(logits):
            logits[punct_idx] *= 1.2

    if temperature == 0.0:
        return np.argmax(logits)  # Greedy decoding
    # Subtract max for numerical stability
    logits = logits - np.max(logits)
    logits = logits / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    if np.isnan(probs).any():
        print("Warning: Probabilities contain NaN values. Falling back to greedy decoding.")
        return np.argmax(logits)  # Fall back to greedy decoding
    # Ensure probabilities sum to 1
    probs = probs / np.sum(probs)
    next_token = np.random.choice(len(probs), p=probs)  # Sample from the distribution
    return next_token


def clean_transcription(transcription):
    # An empty decoder result is valid for silence (immediate EOT). Preserve it
    # instead of turning it into punctuation that looks like a transcription.
    if not transcription or not transcription.strip():
        return ""

    # Split the transcription into sentences using both '.' and '?' as delimiters
    sentences = re.split(r'(?<=[.?])\s+', transcription)
    
    # Initialize a list to store unique sentences
    unique_sentences = []
    
    # Iterate through the sentences
    for sentence in sentences:
        # Check if any part of the current sentence has already appeared in the unique sentences
        for unique_sentence in unique_sentences:
            # Normalize both sentences for comparison
            normalized_current = sentence.lower().strip()
            normalized_unique = unique_sentence.lower().strip()
            
            # Check if the current sentence is a substring of the unique sentence or vice versa
            if normalized_current in normalized_unique or normalized_unique in normalized_current:
                # If a repetition is found, stop processing and return the cleaned transcription
                cleaned_transcription = ' '.join(unique_sentences)
                #cleaned_transcription = '. '.join(unique_sentences)
                # Ensure the last character is a proper delimiter (e.g., '.' or '?')
                if not cleaned_transcription.endswith(('.', '?')):
                    cleaned_transcription += '.'
                return cleaned_transcription
        
        # If no repetition is found, add the current sentence to the unique list
        unique_sentences.append(sentence.strip())
    
    # If no repetition is found, join all sentences and return
    #cleaned_transcription = '. '.join(unique_sentences)
    cleaned_transcription = ' '.join(unique_sentences)
    # Ensure the last character is a proper delimiter (e.g., '.' or '?')
    if not cleaned_transcription.endswith(('.', '?')):
        cleaned_transcription += '.'
    return cleaned_transcription
