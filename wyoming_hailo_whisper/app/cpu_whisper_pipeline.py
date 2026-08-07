"""CPU-based Whisper pipeline using transformers for higher accuracy decoding."""

import logging
import numpy as np
import torch
from queue import Queue, Empty
from threading import Thread
from typing import Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_LOGGER = logging.getLogger(__name__)


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
        self.results_queue = Queue()
        self.running = True
        self.thread = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        while self.running:
            try:
                audio, language, initial_prompt = self.data_queue.get(timeout=1)
            except Empty:
                continue

            try:
                _LOGGER.info("CPU decode: audio length=%.2fs, language=%s, prompt='%s'",
                             len(audio) / 16000, language, initial_prompt or "")

                transcription = self._transcribe_audio(
                    audio,
                    language=language,
                    initial_prompt=initial_prompt,
                )

                _LOGGER.info("CPU transcription: '%s'", transcription)
                self.results_queue.put(transcription)
            except Exception:
                _LOGGER.exception("Error during CPU inference")
                self.results_queue.put("")

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

            generate_kwargs = dict(
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=self.beam_size,
                max_new_tokens=224,
                no_repeat_ngram_size=3,
            )

            if initial_prompt:
                prompt_token_ids = self.processor.tokenizer.encode(
                    initial_prompt, add_special_tokens=False
                )
                generate_kwargs["prompt_ids"] = torch.tensor(
                    prompt_token_ids, dtype=torch.long
                )
                _LOGGER.debug("CPU prompt_ids: %d tokens", len(prompt_token_ids))

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

    def send_data(self, data, language="en", initial_prompt=""):
        self.data_queue.put((data, language, initial_prompt))

    def get_transcription(self, timeout_sec: Optional[float] = None):
        if timeout_sec is None:
            return self.results_queue.get()
        try:
            return self.results_queue.get(timeout=timeout_sec)
        except Empty as err:
            raise TimeoutError(
                f"Timed out waiting for transcription after {timeout_sec} seconds"
            ) from err

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)
