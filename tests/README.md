# Wyoming Hailo Whisper - Unit Tests

This directory contains comprehensive unit tests that help you understand how the audio transcription system works.

## Test Structure

```
tests/
├── README.md                    # This file
├── __init__.py                  # Package marker
├── test_preprocessing.py        # Audio preprocessing tests (VAD, gain, chunking)
├── test_audio_utils.py          # Mel spectrogram generation tests
├── test_postprocessing.py       # Transcription cleaning tests
├── test_handler.py              # Wyoming protocol event handler tests
└── test_integration.py          # End-to-end integration tests
```

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_preprocessing.py
pytest tests/test_audio_utils.py
pytest tests/test_postprocessing.py
pytest tests/test_handler.py
pytest tests/test_integration.py
```

### Run specific test class
```bash
pytest tests/test_preprocessing.py::TestGainAdjustment
pytest tests/test_audio_utils.py::TestLogMelSpectrogram
```

### Run specific test function
```bash
pytest tests/test_preprocessing.py::TestGainAdjustment::test_apply_gain_increases_amplitude
```

### Run with verbose output (show print statements)
```bash
pytest -v -s
```

### Run integration tests only
```bash
pytest tests/test_integration.py -v -s
```

## Test Categories

### 1. Audio Preprocessing Tests (`test_preprocessing.py`)

**Purpose:** Understand how raw audio is prepared for transcription.

**Key Topics:**
- **Gain Adjustment** - How quiet audio is amplified
- **Voice Activity Detection (VAD)** - How speech is detected in audio
- **Audio Improvement** - Complete preprocessing pipeline
- **Chunking** - How long audio is split into segments
- **Mel Spectrogram Generation** - Converting audio to model input format

**Example Test:**
```bash
# See how VAD detects speech in audio
pytest tests/test_preprocessing.py::TestVoiceActivityDetection::test_detects_speech_in_middle -v -s
```

**What You'll Learn:**
- Audio is normalized to 16kHz, mono, float32 [-1.0, 1.0]
- VAD uses energy-based detection (frame energy > threshold)
- Quiet audio (max < 0.1) gets 20 dB gain boost
- Audio is chunked into 10s (tiny) or 5s (base) segments
- Each 10-second chunk becomes a `(1, 1, 1000, 80)` NHWC mel spectrogram

### 2. Audio Utils Tests (`test_audio_utils.py`)

**Purpose:** Understand mel spectrogram generation (Whisper's input format).

**Key Topics:**
- **Padding/Trimming** - Normalizing audio to fixed length (30s chunks)
- **Mel Spectrogram Generation** - STFT → Mel filterbank → Log scaling
- **Audio Constants** - Whisper's hyperparameters (16kHz, 400 FFT window, etc.)
- **Different Mel Bins** - 80 vs 128 mel frequencies

**Example Test:**
```bash
# See how audio becomes a mel spectrogram
pytest tests/test_audio_utils.py::TestLogMelSpectrogram::test_generates_correct_shape -v -s
```

**What You'll Learn:**
- Whisper uses 16kHz sample rate, 400 FFT window, 160 hop length
- 30 seconds of audio → (80, 3000) mel spectrogram
- Each frame represents 10ms of audio
- Log-mel spectrograms are normalized to roughly [0, 1] range
- Pure tones appear as energy in specific mel bins

### 3. Postprocessing Tests (`test_postprocessing.py`)

**Purpose:** Understand how decoder output becomes clean text.

**Key Topics:**
- **Repetition Penalty** - Preventing token loops
- **Temperature Sampling** - Token selection strategies
- **Transcription Cleaning** - Removing duplicate sentences

**Example Test:**
```bash
# See how repetition penalty works
pytest tests/test_postprocessing.py::TestRepetitionPenalty::test_reduces_logits_for_repeated_tokens -v -s
```

**What You'll Learn:**
- Recent tokens get a sign-aware penalty: positive logits are divided and
  negative logits are multiplied by the penalty factor
- Punctuation tokens (11, 13) are excluded from penalty
- Temperature = 0 → greedy decoding (argmax)
- Temperature > 0 → sampling with randomness
- Duplicate sentences are detected case-insensitively and removed

### 4. Handler Tests (`test_handler.py`)

**Purpose:** Understand the Wyoming protocol event handling.

**Key Topics:**
- **Event Types** - AudioChunk, AudioStop, Transcribe, Describe
- **Audio Accumulation** - Buffering chunks until AudioStop
- **Transcription Triggering** - Complete pipeline from AudioStop
- **Model Locking** - Thread-safe Hailo access

**Example Test:**
```bash
# See the complete transcription workflow
pytest tests/test_handler.py::TestHandlerIntegration::test_complete_transcription_workflow -v -s
```

**What You'll Learn:**
- Wyoming protocol uses async event handling
- AudioChunk events accumulate audio in memory
- AudioStop triggers: WAV conversion → preprocessing → inference → transcript
- Model inference uses async lock for thread safety
- Chunk length varies by model: 10s (tiny) vs 5s (base)

### 5. Integration Tests (`test_integration.py`)

**Purpose:** See how all components work together in realistic scenarios.

**Key Topics:**
- **Complete Pipeline** - Raw audio → mel spectrograms → transcription
- **Long Audio** - Multiple overlapping chunks
- **Audio Quality** - Different amplitude levels
- **Realistic Scenarios** - Voice commands, continuous listening
- **Edge Cases** - Short audio, silence, clipped audio

**Example Test:**
```bash
# See the complete end-to-end pipeline
pytest tests/test_integration.py::TestEndToEndAudioProcessing::test_complete_audio_to_mel_pipeline -v -s
```

**What You'll Learn:**
- Complete flow: audio → VAD → gain → chunking → mel → inference → cleaning
- Long audio (>10s) is split with overlap to avoid cutting speech
- Different audio qualities are handled appropriately
- Voice commands: button delay → speech detection → processing
- Edge cases: short audio is padded, silence returns None for speech_start

## Understanding the Pipeline

### Audio Input → Transcription Flow

```
1. Raw Audio (microphone/file)
   ├─ Format: Any sample rate, channels, bit depth
   └─ Example: 44.1kHz stereo, 5 seconds

2. Audio Conversion (handler.py)
   ├─ Convert to 16kHz, mono, 16-bit PCM
   └─ Output: NumPy array, float32, normalized to [-1, 1]

3. Audio Improvement (preprocessing.py)
   ├─ Voice Activity Detection (detect_first_speech)
   │  └─ Frame-based energy analysis → speech start time
   ├─ Gain Adjustment (apply_gain)
   │  └─ If max < 0.1: +20 dB boost
   └─ Output: Improved audio + speech_start_time

4. Mel Spectrogram Generation (preprocessing.py + audio_utils.py)
   ├─ Chunking: Split into 10s (tiny) or 5s (base) segments
   ├─ For each chunk:
   │  ├─ Pad/trim to exact length
   │  ├─ STFT (Short-Time Fourier Transform)
   │  ├─ Apply mel filterbank (80 bins)
   │  ├─ Log scaling + normalization
   │  └─ Format: (1, 1, chunk_seconds × 100, 80) NHWC for Hailo
   └─ Output: List of mel spectrograms

5. Hailo Inference (hailo_whisper_pipeline.py)
   ├─ Encoder: mel → encoded features
   ├─ Decoder: iterative token generation
   │  ├─ Apply repetition penalty
   │  ├─ Temperature sampling
   │  └─ Generate until EOS or max length
   └─ Output: Token IDs

6. Text Generation (postprocessing.py)
   ├─ Tokenizer.decode: tokens → raw text
   ├─ clean_transcription: remove duplicates
   ├─ Remove [BLANK_AUDIO] markers
   └─ Output: Final transcription text

7. Wyoming Response (handler.py)
   └─ Send Transcript event to client
```

## Key Parameters

### Audio Constants
```python
SAMPLE_RATE = 16000        # 16kHz
N_FFT = 400                # FFT window (25ms)
HOP_LENGTH = 160           # Hop size (10ms)
N_MELS = 80                # Mel frequency bins
CHUNK_LENGTH = 30          # Whisper chunk size (seconds)
```

### Model-Specific
```python
# Tiny model
chunk_length = 10          # 10-second chunks
decoding_length = 32       # Max tokens

# Base model
chunk_length = 5           # 5-second chunks
decoding_length = 24       # Max tokens
```

### VAD Parameters
```python
threshold = 0.2            # Energy threshold (normalized)
frame_duration = 0.02      # 20ms frames
```

### Postprocessing
```python
repetition_penalty = 1.5   # Move repeated-token logits away from zero
last_window = 8            # Penalize last 8 tokens
excluded_tokens = [11, 13] # Punctuation (no penalty)
temperature = 0.0          # Greedy decoding
```

## Example Usage

### Example 1: Test Audio Preprocessing
```python
import numpy as np
from wyoming_hailo_whisper.common import preprocessing

# Create test audio
sample_rate = 16000
audio = np.random.randn(5 * sample_rate).astype(np.float32) * 0.3

# Improve quality
improved_audio, speech_start = preprocessing.improve_input_audio(
    audio, vad=True, low_audio_gain=True
)

print(f"Speech detected at: {speech_start:.2f}s")

# Generate mel spectrograms
mel_spectrograms = preprocessing.preprocess(
    improved_audio,
    is_nhwc=True,
    chunk_length=10,
    chunk_offset=speech_start or 0
)

print(f"Generated {len(mel_spectrograms)} mel spectrogram(s)")
print(f"Shape: {mel_spectrograms[0].shape}")
```

### Example 2: Test Transcription Cleaning
```python
from wyoming_hailo_whisper.common import postprocessing

# Raw transcription with issues
raw = "Hello world. Hello world. Hello"

# Clean it
cleaned = postprocessing.clean_transcription(raw)

print(f"Raw: '{raw}'")
print(f"Cleaned: '{cleaned}'")
# Output: "Hello world."
```

### Example 3: Test Repetition Penalty
```python
import numpy as np
from wyoming_hailo_whisper.common import postprocessing

# Mock logits
logits = np.ones((1, 100), dtype=np.float32) * 5.0
logits[0, 42] = 10.0  # Token 42 is likely

# Previous tokens include 42
generated_tokens = [15, 20, 42, 25]

# Apply penalty
penalized = postprocessing.apply_repetition_penalty(
    logits, generated_tokens, penalty=1.5
)

print(f"Original logit for token 42: {logits[0, 42]}")
print(f"Penalized logit for token 42: {penalized[42]}")
# Output: 10.0 → 6.67
```

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Install package in development mode
pip install -e .
```

### Tests Fail with Missing Dependencies
```bash
# Install test requirements
pip install -r requirements-test.txt
```

### Tests Timeout or Hang
Some tests process audio and may take a few seconds. Use `-v -s` for verbose output to see progress.

### Mock Objects Not Working
The handler tests use mocks to simulate the Hailo hardware. If you want to test with real hardware, you'll need to:
1. Have Hailo device connected
2. Install hailo_platform
3. Modify tests to use real HailoWhisperPipeline

## Learning Path

Recommended order for understanding the codebase:

1. **Start Here:** `test_audio_utils.py`
   - Learn about Whisper's audio format
   - Understand mel spectrograms

2. **Next:** `test_preprocessing.py`
   - See how audio is cleaned and prepared
   - Understand VAD and gain adjustment

3. **Then:** `test_postprocessing.py`
   - Learn how text is generated from tokens
   - Understand repetition penalty and cleaning

4. **After That:** `test_handler.py`
   - See how Wyoming protocol works
   - Understand event handling

5. **Finally:** `test_integration.py`
   - See the complete pipeline
   - Realistic scenarios and edge cases

## Contributing

When adding new features, please add corresponding tests:
- Unit tests for individual functions
- Integration tests for component interactions
- Document what the test demonstrates (docstrings)

## Resources

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Hailo AI](https://hailo.ai/)
- [pytest Documentation](https://docs.pytest.org/)
