# Quick Start Guide - Unit Tests

This guide helps you quickly get started with the unit tests to understand the codebase.

## Installation

```bash
# Install test dependencies
pip install -r requirements-test.txt

# If you get import errors, install the package in development mode
pip install -e .
```

## Quick Start - 5 Minute Tour

### 1. Run All Tests
```bash
pytest -v
```

This runs all 70+ tests and shows you which components work.

### 2. See How Audio is Processed
```bash
pytest tests/test_integration.py::TestEndToEndAudioProcessing::test_complete_audio_to_mel_pipeline -v -s
```

**What you'll see:**
- Raw audio stats (duration, amplitude)
- Speech detection results
- Mel spectrogram generation
- Complete pipeline flow

**Output:**
```
=== Complete Audio Processing Pipeline ===

1. Input Audio
   Duration: 10.0s
   Max amplitude: 0.080
   Shape: (160000,)

2. Improve Audio (VAD + Gain)
   Speech detected at: 2.00s
   New max amplitude: 0.800
   Gain applied: 10.0x

3. Generate Mel Spectrograms
   Number of chunks: 1
   Shape per chunk: (1, 1, 1000, 80)
   Format: NHWC (Hailo format)

4. Mel Spectrogram Properties
   Value range: [-0.234, 1.000]
   Mean: 0.456
   Std: 0.123

✓ Audio successfully processed and ready for Hailo inference
```

### 3. Understand Voice Activity Detection
```bash
pytest tests/test_preprocessing.py::TestVoiceActivityDetection::test_detects_speech_in_middle -v -s
```

**What you'll learn:**
- How VAD finds speech in audio
- Energy-based detection algorithm
- Frame-by-frame analysis

### 4. See How Mel Spectrograms Work
```bash
pytest tests/test_audio_utils.py::TestMelSpectrogramVisual::test_mel_spectrogram_for_tone -v -s
```

**What you'll learn:**
- How frequencies map to mel bins
- STFT → Mel filterbank → Log scaling
- Visual representation of audio

**Output:**
```
=== Mel Spectrogram for 1000Hz tone ===
Output shape: (80, 3000)
Peak energy at mel bin: 32/80
Value range: [0.123, 0.987]
```

### 5. Understand Text Cleaning
```bash
pytest tests/test_postprocessing.py::TestCleanTranscription -v
```

**What you'll learn:**
- How repeated sentences are removed
- Case-insensitive duplicate detection
- Automatic punctuation addition

### 6. See Wyoming Protocol in Action
```bash
pytest tests/test_handler.py::TestHandlerIntegration::test_complete_transcription_workflow -v -s
```

**What you'll learn:**
- Event-based communication flow
- AudioChunk accumulation
- AudioStop triggers transcription
- Complete client-server interaction

**Output:**
```
=== Complete Transcription Workflow ===
1. Client sends Describe event
   → Server responds with info
2. Client sends Transcribe event (language: en)
3. Client streams audio chunks
   → Sent 20 audio chunks (2.0s)
4. Client sends AudioStop event
5. Server sends Transcript event
   → Transcript: 'This is a test transcription'
```

## Understanding the Codebase Through Tests

### Audio Processing Flow

```python
# 1. Raw audio → NumPy array
audio = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0

# 2. Improve quality
improved_audio, speech_start = preprocessing.improve_input_audio(
    audio, vad=True, enhance=True
)

# 3. Generate mel spectrograms
mel_spectrograms = preprocessing.preprocess(
    improved_audio,
    is_nhwc=True,
    chunk_length=10,
    chunk_offset=speech_start
)

# 4. Each mel is ready for Hailo
# Shape: (1, 1, 1000, 80) for a 10-second tiny-model chunk
```

### Transcription Generation Flow

```python
# 1. Model outputs logits
logits = decoder.run(encoded_features, token_embeddings)

# 2. Apply repetition penalty
penalized_logits = apply_repetition_penalty(logits, generated_tokens, penalty=1.5)

# 3. Select next token
next_token = temperature_sampling(penalized_logits, temperature=0.0)

# 4. Generate text
text = tokenizer.decode(generated_tokens)

# 5. Clean transcription
cleaned_text = clean_transcription(text)
```

## Test Categories

### Unit Tests (Fast, Focused)
- `test_preprocessing.py` - Audio improvement functions
- `test_audio_utils.py` - Mel spectrogram generation
- `test_postprocessing.py` - Text cleaning and token selection

### Integration Tests (Slower, Comprehensive)
- `test_handler.py` - Wyoming protocol event handling
- `test_integration.py` - End-to-end pipeline

## Interactive Exploration

Want to experiment? Run Python in the project directory:

```python
import numpy as np
from wyoming_hailo_whisper.common import preprocessing, audio_utils

# Create test audio
sample_rate = 16000
duration = 5  # seconds
audio = np.random.randn(duration * sample_rate).astype(np.float32) * 0.3

# Try preprocessing
improved_audio, speech_start = preprocessing.improve_input_audio(audio, vad=True)
print(f"Speech at: {speech_start}s")

# Generate mel spectrogram
mel_spectrograms = preprocessing.preprocess(improved_audio, is_nhwc=True, chunk_length=10)
print(f"Mel shape: {mel_spectrograms[0].shape}")
print(f"Values: [{mel_spectrograms[0].min():.3f}, {mel_spectrograms[0].max():.3f}]")
```

## Common Questions Answered by Tests

### Q: Why 16kHz sample rate?
**A:** See `test_audio_utils.py::TestAudioConstants::test_sample_rate`
- Standard for speech recognition
- Captures frequencies up to 8kHz (sufficient for human speech)
- Efficient processing

### Q: What is a mel spectrogram?
**A:** See `test_audio_utils.py::TestLogMelSpectrogram`
- Converts audio waveform to frequency representation
- Uses mel scale (mimics human hearing)
- Output: (80, 3000) = 80 mel bins × 3000 time frames

### Q: How does VAD work?
**A:** See `test_preprocessing.py::TestVoiceActivityDetection`
- Splits audio into frames (20ms)
- Calculates energy per frame
- Detects when energy exceeds threshold (0.2)

### Q: Why repetition penalty?
**A:** See `test_postprocessing.py::TestRepetitionPenalty`
- Prevents decoder from looping
- Reduces logits of recently generated tokens
- Default penalty: 1.5x

### Q: What is NHWC format?
**A:** See `test_preprocessing.py::TestPreprocess::test_nchw_format`
- NHWC = (batch, height, width, channels) = Hailo format
- NCHW = (batch, channels, height, width) = PyTorch format
- Example: (1, 1, 1000, 80) NHWC for a 10-second tiny-model chunk

### Q: How are long audio files handled?
**A:** See `test_integration.py::TestEndToEndAudioProcessing::test_multiple_audio_chunks`
- Split into 10s (tiny) or 5s (base) chunks
- Optional overlap (e.g., 20%) to avoid cutting speech
- Each chunk processed independently

## Next Steps

1. **Read the tests** - They're heavily documented with explanations
2. **Run with -v -s** - See verbose output and print statements
3. **Modify test parameters** - Experiment with different values
4. **Add your own tests** - Test new features or edge cases

## Full Documentation

See `tests/README.md` for comprehensive documentation including:
- Detailed test descriptions
- Parameter explanations
- Learning path recommendations
- Contributing guidelines

## Quick Reference

```bash
# All tests
pytest

# Specific category
pytest tests/test_preprocessing.py
pytest tests/test_audio_utils.py
pytest tests/test_postprocessing.py
pytest tests/test_handler.py
pytest tests/test_integration.py

# Verbose with output
pytest -v -s

# Specific test
pytest tests/test_preprocessing.py::TestGainAdjustment::test_apply_gain_increases_amplitude

# Integration tests only
pytest tests/test_integration.py -v -s

# Run and stop on first failure
pytest -x

# Run tests matching pattern
pytest -k "audio" -v
```

## Help

- Tests not running? Check `tests/README.md` → Troubleshooting
- Want to understand a component? See "Learning Path" in `tests/README.md`
- Questions about the code? The test docstrings explain each function's purpose
