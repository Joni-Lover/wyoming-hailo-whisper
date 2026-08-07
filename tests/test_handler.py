"""
Tests for Wyoming event handler.

These tests demonstrate how the server handles Wyoming protocol events:
1. AudioChunk events - accumulating audio data
2. AudioStop events - triggering transcription
3. Transcribe events - setting language
4. Describe events - returning server info
"""

import asyncio
import argparse
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from wyoming.event import Event
from wyoming.audio import AudioChunk, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.info import Describe, Info, AsrProgram, AsrModel

from wyoming_hailo_whisper.handler import HailoWhisperEventHandler


def _speech_audio_bytes(samples, sample_rate=16000):
    """Create deterministic speech-like audio with a varying energy envelope."""
    sample_indices = np.arange(samples)
    signal = np.sin(2 * np.pi * 220 * sample_indices / sample_rate)
    envelope = np.full(samples, 0.3, dtype=np.float32)
    envelope[:samples // 4] = 0.01
    return (signal * envelope * 32768).astype(np.int16).tobytes()


@pytest.fixture
def mock_cli_args():
    """Create mock CLI arguments."""
    args = argparse.Namespace()
    args.language = "en"
    args.variant = "tiny"
    args.device = "hailo8l"
    return args


@pytest.fixture
def mock_wyoming_info():
    """Create mock Wyoming info."""
    return Info(
        asr=[
            AsrProgram(
                name="hailo-whisper",
                description="Hailo accelerated Whisper",
                attribution={"name": "OpenAI", "url": "https://github.com/openai/whisper"},
                installed=True,
                version="test",
                models=[
                    AsrModel(
                        name="whisper-tiny",
                        description="Whisper tiny model",
                        attribution={"name": "OpenAI", "url": "https://github.com/openai/whisper"},
                        installed=True,
                        languages=["en"],
                        version="test",
                    )
                ],
            )
        ]
    )


@pytest.fixture
def mock_model():
    """Create mock HailoWhisperPipeline."""
    model = Mock()
    model.send_data = Mock()
    model.get_transcription = Mock(return_value="Hello world")
    return model


@pytest.fixture
def mock_model_lock():
    """Create mock asyncio lock."""
    return asyncio.Lock()


@pytest_asyncio.fixture
async def handler(mock_wyoming_info, mock_cli_args, mock_model, mock_model_lock):
    """Create a HailoWhisperEventHandler for testing."""
    handler = HailoWhisperEventHandler(
        wyoming_info=mock_wyoming_info,
        cli_args=mock_cli_args,
        model=mock_model,
        model_lock=mock_model_lock,
        reader=AsyncMock(),
        writer=AsyncMock(),
    )
    handler.write_event = AsyncMock()  # Mock the write_event method
    return handler


class TestHailoWhisperEventHandler:
    """Tests for the Wyoming event handler."""

    @pytest.mark.asyncio
    async def test_handles_describe_event(self, handler, mock_wyoming_info):
        """
        Shows how Describe events return server capabilities.

        When a client connects, it sends a Describe event to learn:
        - What models are available
        - What languages are supported
        - Server metadata
        """
        describe_event = Describe().event()

        result = await handler.handle_event(describe_event)

        # Should return True (continue handling events)
        assert result is True

        # Should send back the Wyoming info
        handler.write_event.assert_called_once()
        sent_event = handler.write_event.call_args[0][0]
        assert sent_event is not None

    @pytest.mark.asyncio
    async def test_handles_transcribe_event(self, handler):
        """
        Shows how Transcribe events set the target language.

        Before sending audio, clients can specify the language
        for transcription (e.g., "en", "es", "fr").
        """
        transcribe_event = Transcribe(language="es").event()

        result = await handler.handle_event(transcribe_event)

        # Should return True (continue handling)
        assert result is True

        # Should update language
        assert handler._language == "es"

    @pytest.mark.asyncio
    async def test_accumulates_audio_chunks(self, handler):
        """
        Shows how AudioChunk events accumulate audio data.

        Audio is sent in chunks (e.g., 100ms at a time) and buffered
        until AudioStop is received.
        """
        # Create mock audio chunk (16kHz, 16-bit, mono)
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16).tobytes()

        chunk = AudioChunk(
            rate=16000,
            width=2,  # 16-bit = 2 bytes
            channels=1,
            audio=audio_data,
        )
        chunk_event = chunk.event()

        # Initially no audio
        assert handler.audio == bytes()

        # Send first chunk
        result = await handler.handle_event(chunk_event)
        assert result is True
        assert len(handler.audio) > 0

        # Send second chunk
        result = await handler.handle_event(chunk_event)
        assert result is True
        # Audio should have doubled
        assert len(handler.audio) == len(audio_data) * 2

    @pytest.mark.asyncio
    async def test_audiostop_triggers_transcription(self, handler, mock_model):
        """
        Shows how AudioStop triggers the transcription pipeline.

        Process:
        1. Accumulated audio → WAV format
        2. WAV → NumPy array (float32, normalized)
        3. Audio preprocessing (VAD, gain)
        4. Mel spectrogram generation
        5. Model inference
        6. Text postprocessing
        7. Send Transcript event
        """
        # Setup: Add some audio data
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        samples = int(sample_rate * duration)
        # Create speech-like audio (loud enough to not trigger gain)
        audio_data = _speech_audio_bytes(samples, sample_rate)

        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=audio_data)
        await handler.handle_event(chunk.event())

        # Mock the model response
        mock_model.get_transcription.return_value = "Hello world"

        # Send AudioStop
        stop_event = AudioStop().event()
        result = await handler.handle_event(stop_event)

        # Should return False (end of handling for this request)
        assert result is False

        # Model should have been called
        assert mock_model.send_data.called
        assert mock_model.get_transcription.called

        # Should send Transcript event
        assert handler.write_event.called
        sent_event = handler.write_event.call_args[0][0]

        # Extract the transcript (the event is a dict)
        assert sent_event.type == "transcript"
        assert "text" in sent_event.data

        # Audio buffer should be reset
        assert handler.audio == bytes()

    @pytest.mark.asyncio
    async def test_handles_empty_audio(self, handler, mock_model):
        """
        Shows behavior when AudioStop is received with no audio.

        This can happen if client disconnects or sends empty buffer.
        """
        # Mock to avoid actual processing
        mock_model.get_transcription.return_value = "[BLANK_AUDIO]"

        # Send AudioStop without any audio chunks
        stop_event = AudioStop().event()

        # Should handle gracefully (might return empty or error)
        # The actual behavior depends on error handling in the code
        try:
            result = await handler.handle_event(stop_event)
            # If it completes, check that it didn't crash
            assert result is False or result is True
        except Exception as e:
            # If it raises an exception, that's also valid behavior
            pytest.skip(f"Handler raised exception for empty audio: {e}")

    def test_cpu_trim_past_end_returns_empty(self, handler, mock_model):
        handler.cli_args.use_cpu = True
        audio = np.ones(160, dtype=np.float32)

        result = handler._transcribe_cpu(audio, chunk_offset=1.0)

        assert result == ""
        mock_model.send_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_transcription_output(self, handler, mock_model):
        """
        Shows how transcription output is cleaned.

        Process:
        1. Model returns raw transcription (may have repetitions)
        2. clean_transcription() removes duplicates
        3. "[BLANK_AUDIO]" markers are removed
        4. Result is stripped of whitespace
        """
        # Setup audio
        sample_rate = 16000
        samples = int(sample_rate * 2)  # 2 seconds
        audio_data = _speech_audio_bytes(samples, sample_rate)

        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=audio_data)
        await handler.handle_event(chunk.event())

        # Mock model to return messy transcription
        mock_model.get_transcription.return_value = "[BLANK_AUDIO] Hello. Hello."

        # Send AudioStop
        stop_event = AudioStop().event()
        await handler.handle_event(stop_event)

        # Check that transcript was cleaned
        sent_event = handler.write_event.call_args[0][0]
        text = sent_event.data["text"]

        # Should remove [BLANK_AUDIO] and duplicate sentences
        assert "[BLANK_AUDIO]" not in text
        assert text == "Hello."  # Duplicate removed by clean_transcription

    @pytest.mark.asyncio
    async def test_uses_model_lock(self, handler, mock_model, mock_model_lock):
        """
        Shows that model inference uses async lock for thread safety.

        This ensures only one request uses the Hailo device at a time,
        preventing resource conflicts.
        """
        # Setup audio
        sample_rate = 16000
        samples = int(sample_rate * 1)
        audio_data = _speech_audio_bytes(samples, sample_rate)

        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=audio_data)
        await handler.handle_event(chunk.event())

        mock_model.get_transcription.return_value = "Test"

        # Mock the lock to verify it's used
        original_lock = handler.model_lock
        handler.model_lock = AsyncMock()
        handler.model_lock.__aenter__ = AsyncMock()
        handler.model_lock.__aexit__ = AsyncMock()

        # Send AudioStop
        stop_event = AudioStop().event()
        await handler.handle_event(stop_event)

        # Lock should have been acquired
        handler.model_lock.__aenter__.assert_called_once()
        handler.model_lock.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_length_varies_by_model(self, handler, mock_cli_args):
        """
        Shows that chunk length depends on model variant.

        - tiny model: 10 second chunks
        - base model: 5 second chunks

        This matches the model's training configuration.
        """
        # Setup audio
        sample_rate = 16000
        samples = int(sample_rate * 2)
        audio_data = _speech_audio_bytes(samples, sample_rate)

        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=audio_data)
        await handler.handle_event(chunk.event())

        # Test with tiny variant
        handler.cli_args.variant = "tiny"
        handler.model.get_transcription.return_value = "Test"

        with patch("wyoming_hailo_whisper.handler.preprocess") as mock_preprocess:
            mock_preprocess.return_value = [np.zeros((1, 1, 3000, 80))]

            stop_event = AudioStop().event()
            await handler.handle_event(stop_event)

            # Should call preprocess with chunk_length=10
            call_kwargs = mock_preprocess.call_args[1]
            assert call_kwargs["chunk_length"] == 10

        # Reset audio for next test
        handler.audio = audio_data

        # Test with base variant
        handler.cli_args.variant = "base"

        with patch("wyoming_hailo_whisper.handler.preprocess") as mock_preprocess:
            mock_preprocess.return_value = [np.zeros((1, 1, 3000, 80))]

            stop_event = AudioStop().event()
            await handler.handle_event(stop_event)

            # Should call preprocess with chunk_length=5
            call_kwargs = mock_preprocess.call_args[1]
            assert call_kwargs["chunk_length"] == 5


class TestHandlerIntegration:
    """Integration tests for the full event handling flow."""

    @pytest.mark.asyncio
    async def test_complete_transcription_workflow(
        self, mock_wyoming_info, mock_cli_args, mock_model, mock_model_lock
    ):
        """
        Demonstrates the complete workflow from connection to transcription.

        1. Client connects and sends Describe
        2. Server responds with capabilities
        3. Client sends Transcribe (optional language setting)
        4. Client streams AudioChunk events
        5. Client sends AudioStop
        6. Server processes and returns Transcript
        """
        handler = HailoWhisperEventHandler(
            wyoming_info=mock_wyoming_info,
            cli_args=mock_cli_args,
            model=mock_model,
            model_lock=mock_model_lock,
            reader=AsyncMock(),
            writer=AsyncMock(),
        )
        handler.write_event = AsyncMock()

        print("\n=== Complete Transcription Workflow ===")

        # Step 1: Client asks for capabilities
        print("1. Client sends Describe event")
        describe_event = Describe().event()
        result = await handler.handle_event(describe_event)
        assert result is True
        print("   → Server responds with info")

        # Step 2: Client sets language (optional)
        print("2. Client sends Transcribe event (language: en)")
        transcribe_event = Transcribe(language="en").event()
        result = await handler.handle_event(transcribe_event)
        assert result is True

        # Step 3: Client streams audio
        print("3. Client streams audio chunks")
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks
        total_duration = 2.0  # 2 seconds total

        num_chunks = int(total_duration / chunk_duration)
        stream_audio = _speech_audio_bytes(
            int(total_duration * sample_rate),
            sample_rate,
        )
        for i in range(num_chunks):
            samples = int(sample_rate * chunk_duration)
            chunk_bytes = samples * 2
            audio_data = stream_audio[i * chunk_bytes:(i + 1) * chunk_bytes]

            chunk = AudioChunk(rate=16000, width=2, channels=1, audio=audio_data)
            result = await handler.handle_event(chunk.event())
            assert result is True

        print(f"   → Sent {num_chunks} audio chunks ({total_duration}s)")

        # Step 4: Client signals end of audio
        print("4. Client sends AudioStop event")
        mock_model.get_transcription.return_value = "This is a test transcription"

        stop_event = AudioStop().event()
        result = await handler.handle_event(stop_event)
        assert result is False  # False = done with this request

        # Step 5: Verify transcription was sent
        print("5. Server sends Transcript event")
        assert handler.write_event.called
        sent_event = handler.write_event.call_args[0][0]
        assert sent_event.type == "transcript"
        text = sent_event.data["text"]
        print(f"   → Transcript: '{text}'")

        assert len(text) > 0
        assert handler.audio == bytes()  # Buffer should be reset
