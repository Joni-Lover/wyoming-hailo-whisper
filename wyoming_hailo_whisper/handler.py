"""Wyoming event handler for Whisper transcription."""

import argparse
import asyncio
import logging
import time
from numbers import Real

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_hailo_whisper.common.postprocessing import clean_transcription
from wyoming_hailo_whisper.common.preprocessing import improve_input_audio, preprocess
from wyoming_hailo_whisper.const import DEFAULT_LANGUAGE

_LOGGER = logging.getLogger(__name__)


class HailoWhisperEventHandler(AsyncEventHandler):
    """Handle one Wyoming client stream using the selected inference backend."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._default_language = cli_args.language or DEFAULT_LANGUAGE
        self._language = self._default_language

    def _model_chunk_length(self) -> float:
        """Read the HEF input length, with legacy variant defaults as fallback."""
        get_length = getattr(self.model, "get_model_input_audio_length", None)
        if callable(get_length):
            chunk_length = get_length()
            if isinstance(chunk_length, Real) and chunk_length > 0:
                return float(chunk_length)

        return 10.0 if self.cli_args.variant == "tiny" else 5.0

    def _transcribe_hailo(self, sampled_audio: np.ndarray, chunk_offset: float) -> str:
        mel_spectrograms = preprocess(
            sampled_audio,
            True,
            chunk_length=self._model_chunk_length(),
            chunk_offset=chunk_offset,
        )
        prompt = getattr(self.cli_args, "hailo_initial_prompt", "")
        parts = []
        _LOGGER.info("Hailo: processing %d mel spectrogram(s)", len(mel_spectrograms))
        for mel in mel_spectrograms:
            _LOGGER.debug("Hailo mel shape: %s", mel.shape)
            self.model.send_data(
                mel,
                language=self._language,
                initial_prompt=prompt,
            )
            parts.append(clean_transcription(self.model.get_transcription()))

        return " ".join(part for part in parts if part).strip()

    def _transcribe_cpu(self, sampled_audio: np.ndarray, chunk_offset: float) -> str:
        offset_samples = min(
            max(int(chunk_offset * 16000), 0),
            len(sampled_audio),
        )
        trimmed_audio = sampled_audio[offset_samples:]
        if not trimmed_audio.size:
            _LOGGER.info("CPU: no audio remains after trimming")
            return ""

        _LOGGER.info("CPU: processing %.2fs of audio", len(trimmed_audio) / 16000)
        self.model.send_data(
            trimmed_audio,
            language=self._language,
            initial_prompt=getattr(self.cli_args, "initial_prompt", ""),
        )
        return clean_transcription(self.model.get_transcription())

    def _transcribe(self, sampled_audio: np.ndarray, chunk_offset: float) -> str:
        if getattr(self.cli_args, "use_cpu", False):
            return self._transcribe_cpu(sampled_audio, chunk_offset)
        return self._transcribe_hailo(sampled_audio, chunk_offset)

    async def handle_event(self, event: Event) -> bool:
        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._default_language
            _LOGGER.debug("Language set to %s", self._language)
            return True

        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")

            chunk = self.audio_converter.convert(AudioChunk.from_event(event))
            self.audio += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            if not self.audio:
                await self.write_event(Transcript(text="").event())
                self._language = self._default_language
                return False

            sampled_audio = (
                np.frombuffer(self.audio, dtype=np.int16)
                .flatten()
                .astype(np.float32)
                / 32768.0
            )
            enhance = getattr(self.cli_args, "enhance_audio", False)
            sampled_audio, start_time = improve_input_audio(
                sampled_audio,
                vad=True,
                enhance=enhance,
            )

            if start_time is None:
                _LOGGER.info("No speech detected in audio")
                text = ""
            else:
                chunk_offset = max(start_time - 0.2, 0.0)
                started = time.monotonic()
                async with self.model_lock:
                    text = await asyncio.to_thread(
                        self._transcribe,
                        sampled_audio,
                        chunk_offset,
                    )
                _LOGGER.info(
                    "%s transcription completed in %.2fs",
                    "CPU" if getattr(self.cli_args, "use_cpu", False) else "Hailo",
                    time.monotonic() - started,
                )

            text = text.replace("[BLANK_AUDIO]", "").strip()
            _LOGGER.info(
                "Completed transcription (len=%d, language=%s)",
                len(text),
                self._language,
            )
            await self.write_event(Transcript(text=text).event())

            self.audio = bytes()
            self._language = self._default_language
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
