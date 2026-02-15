"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
from typing import Optional

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_hailo_whisper.app.hailo_whisper_pipeline import HailoWhisperPipeline
from wyoming_hailo_whisper.common.postprocessing import clean_transcription
from wyoming_hailo_whisper.common.preprocessing import improve_input_audio, preprocess
from wyoming_hailo_whisper.const import DEFAULT_LANGUAGE

_LOGGER = logging.getLogger(__name__)


class HailoWhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: HailoWhisperPipeline,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _LOGGER.info(cli_args)
        _LOGGER.info(model)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )

        # Language management
        self._language: Optional[str] = None
        self._default_language = cli_args.language or DEFAULT_LANGUAGE

        # Model loading state
        self._model_load_task: Optional[asyncio.Task] = None
        self._is_audio_receiving = False

    async def handle_event(self, event: Event) -> bool:
        if Transcribe.is_type(event.type):
            # Capture language BEFORE audio starts
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._default_language
            _LOGGER.debug("Language set to %s", self._language)

            # Start loading model in background with the specified language
            if self._model_load_task is None or self._model_load_task.done():
                self._model_load_task = asyncio.create_task(
                    self._load_model_for_language(self._language)
                )
                _LOGGER.debug("Background model loading started for language: %s", self._language)

            return True

        if AudioChunk.is_type(event.type):
            # Wait for model to be ready if it's still loading
            if self._model_load_task is not None and not self._model_load_task.done():
                if not self._is_audio_receiving:
                    _LOGGER.debug("Waiting for model to load before processing audio...")
                    await self._model_load_task

            if not self._is_audio_receiving:
                _LOGGER.debug("Receiving audio")
                self._is_audio_receiving = True

            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)
            self.audio += chunk.audio

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            self._is_audio_receiving = False

            if not self.audio:
                await self.write_event(Transcript(text="").event())
                _LOGGER.debug("Completed empty request")
                return False

            sampled_audio = np.frombuffer(self.audio, dtype=np.int16).flatten().astype(np.float32) / 32768.0
            sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True)

            chunk_offset = max((start_time or 0.0) - 0.2, 0.0)
            chunk_length = self.model.get_model_input_audio_length()

            mel_spectrograms = preprocess(
                sampled_audio,
                True,
                chunk_length=chunk_length,
                chunk_offset=chunk_offset,
            )

            request_language = self._language or self._default_language
            transcription = ""
            async with self.model_lock:
                _LOGGER.info("Processing mel spectrograms: %s", len(mel_spectrograms))
                for mel in mel_spectrograms:
                    _LOGGER.debug("Processing mel spectrogram shape: %s", mel.shape)
                    raw_transcription = await asyncio.to_thread(
                        self.model.transcribe_mel,
                        mel,
                        request_language,
                    )
                    _LOGGER.debug("Raw transcription: %s", raw_transcription)
                    transcription += clean_transcription(raw_transcription)

            text = transcription.replace("[BLANK_AUDIO]", "").strip()
            _LOGGER.info("Completed transcription (len=%s, language=%s)", len(text), request_language)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self.audio = bytes()
            self._language = None
            self._model_load_task = None

            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    async def _load_model_for_language(self, language: str) -> None:
        """Load/configure model for the specified language."""
        _LOGGER.debug("Loading model for language: %s", language)
        await asyncio.to_thread(self.model.prepare_language, language)
        _LOGGER.debug("Model ready for language: %s", language)
