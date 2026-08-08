"""CPU-based Whisper pipeline using transformers for higher accuracy decoding."""

import logging
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from wyoming_hailo_whisper.app.request_queue import InferenceRequestQueue

_LOGGER = logging.getLogger(__name__)

# CPU decoding can take substantially longer than Hailo inference, especially
# for the larger model variants on a Raspberry Pi. Keep the default generous,
# but finite, so a wedged worker cannot block a Wyoming request forever.
DEFAULT_TRANSCRIPTION_TIMEOUT_SEC = 300.0
DEFAULT_MAX_NEW_TOKENS = 224
DEFAULT_DECODER_CONTEXT_LENGTH = 448


class CpuWhisperPipeline:
    """
    A pipeline that runs Whisper entirely on CPU using the transformers library.
    Trades speed for accuracy: full float32 precision with beam search.
    """

    def __init__(self, variant="base", beam_size=5):
        self.variant = variant
        self.beam_size = beam_size

        model_name = f"openai/whisper-{variant}"
        _LOGGER.info("Loading CPU Whisper model: %s (beam_size=%d)", model_name, beam_size)

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        _LOGGER.info("CPU Whisper model loaded (%.0fM parameters)",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)

        self.data_queue = Queue()
        self._requests = InferenceRequestQueue("CPU transcription")
        self._error = None
        self.running = True
        self.thread = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        try:
            while self.running:
                try:
                    audio, language, initial_prompt, future = self.data_queue.get(
                        timeout=1
                    )
                except Empty:
                    continue

                try:
                    _LOGGER.info(
                        "CPU decode: audio length=%.2fs, language=%s, prompt='%s'",
                        len(audio) / 16000,
                        language,
                        initial_prompt or "",
                    )

                    transcription = self._transcribe_audio(
                        audio,
                        language=language,
                        initial_prompt=initial_prompt,
                    )

                    _LOGGER.info("CPU transcription: '%s'", transcription)
                    self._requests.set_result(future, transcription)
                except Exception as err:
                    _LOGGER.exception("Error during CPU inference")
                    # Keep request failures distinct from valid empty
                    # transcriptions produced for silence. The worker remains
                    # available for subsequent requests while this caller gets
                    # the original generation error.
                    self._requests.set_exception(future, err)
        except Exception as err:
            self._error = err
            self.running = False
            worker_error = RuntimeError("CPU inference worker failed")
            worker_error.__cause__ = err
            self._requests.fail_all(worker_error)
            _LOGGER.exception("CPU inference loop crashed")

    def _transcribe_audio(self, audio, language, initial_prompt=""):
        """Transcribe arbitrary-length audio in Whisper's 30-second windows."""
        audio = np.asarray(audio, dtype=np.float32)
        if not audio.size:
            return ""

        chunk_samples = int(self.get_model_input_audio_length() * 16000)
        transcriptions = []
        for chunk_index, start in enumerate(range(0, len(audio), chunk_samples), start=1):
            chunk = audio[start:start + chunk_samples]
            _LOGGER.debug(
                "CPU decode chunk %d: %.2fs",
                chunk_index,
                len(chunk) / 16000,
            )

            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask

            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

            context_length = self._decoder_context_length()
            control_token_count = len(forced_decoder_ids) + 1
            prompt_token_count = 0

            generate_kwargs = {
                "attention_mask": attention_mask,
                "forced_decoder_ids": forced_decoder_ids,
                "num_beams": self.beam_size,
                "no_repeat_ngram_size": 3,
            }

            if initial_prompt:
                prompt_token_ids = self.processor.get_prompt_ids(
                    initial_prompt,
                    return_tensors="pt",
                )
                max_prompt_ids = max(context_length // 2, 1)
                if len(prompt_token_ids) > max_prompt_ids:
                    # Preserve <|startofprev|> and the most recent prompt
                    # tokens, matching Whisper's half-context prompt policy.
                    if max_prompt_ids == 1:
                        prompt_token_ids = prompt_token_ids[:1]
                    else:
                        prompt_token_ids = torch.cat(
                            (
                                prompt_token_ids[:1],
                                prompt_token_ids[-(max_prompt_ids - 1):],
                            )
                        )
                generate_kwargs["prompt_ids"] = prompt_token_ids
                prompt_token_count = len(prompt_token_ids)
                _LOGGER.debug(
                    "CPU prompt_ids: %d tokens (context=%d)",
                    prompt_token_count,
                    context_length,
                )

            available_new_tokens = (
                context_length - control_token_count - prompt_token_count
            )
            if available_new_tokens < 1:
                raise ValueError(
                    "Whisper decoder context is too short for the configured prompt"
                )
            generate_kwargs["max_new_tokens"] = min(
                DEFAULT_MAX_NEW_TOKENS,
                available_new_tokens,
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    **generate_kwargs,
                )

            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            if transcription:
                transcriptions.append(transcription)

        return " ".join(transcriptions)

    def get_model_input_audio_length(self):
        return 30  # transformers handles up to 30s natively

    def _decoder_context_length(self):
        value = getattr(
            getattr(self.model, "config", None),
            "max_target_positions",
            DEFAULT_DECODER_CONTEXT_LENGTH,
        )
        if not isinstance(value, int) or value <= 0:
            return DEFAULT_DECODER_CONTEXT_LENGTH
        return value

    def send_data(self, data, language="en", initial_prompt=""):
        self._raise_if_failed()
        if not self.running:
            raise RuntimeError("CPU inference worker is not running")
        future = self._requests.submit()
        self.data_queue.put((data, language, initial_prompt, future))

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("CPU inference worker failed") from self._error

    def get_transcription(self, timeout_sec: Optional[float] = None):
        wait_timeout = (
            DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
            if timeout_sec is None
            else timeout_sec
        )
        self._raise_if_failed()
        return self._requests.get_result(wait_timeout)

    def stop(self):
        self.running = False
        self._requests.fail_all(RuntimeError("CPU inference worker stopped"))
        self.thread.join(timeout=5)
