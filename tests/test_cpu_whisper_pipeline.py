"""Tests for CPU Whisper chunking without loading model weights."""

import logging
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from wyoming_hailo_whisper.app.cpu_whisper_pipeline import (
    DEFAULT_DECODER_CONTEXT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TRANSCRIPTION_TIMEOUT_SEC,
    CpuWhisperPipeline,
)
from wyoming_hailo_whisper.app.request_queue import InferenceRequestQueue


def _pipeline_without_model_load():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.beam_size = 5
    pipeline.processor = MagicMock()
    pipeline.model = MagicMock()
    pipeline.processor.side_effect = lambda *_args, **_kwargs: SimpleNamespace(
        input_features=MagicMock(),
        attention_mask=MagicMock(),
    )
    pipeline.processor.get_decoder_prompt_ids.return_value = [(1, 2)]
    pipeline.processor.batch_decode.side_effect = [["first"], ["second"]]
    return pipeline


def test_transcribe_audio_splits_audio_longer_than_30_seconds():
    pipeline = _pipeline_without_model_load()
    audio = np.zeros(31 * 16000, dtype=np.float32)

    result = pipeline._transcribe_audio(audio, language="pl")

    assert result == "first second"
    assert [len(call.args[0]) for call in pipeline.processor.call_args_list] == [
        30 * 16000,
        16000,
    ]
    assert pipeline.model.generate.call_count == 2
    pipeline.processor.get_decoder_prompt_ids.assert_called_with(
        language="pl",
        task="transcribe",
    )


def test_transcribe_audio_normalizes_wyoming_locale():
    pipeline = _pipeline_without_model_load()

    pipeline._transcribe_audio(
        np.zeros(16000, dtype=np.float32),
        language="ru_RU",
    )

    pipeline.processor.get_decoder_prompt_ids.assert_called_once_with(
        language="ru",
        task="transcribe",
    )


def test_transcribe_audio_accepts_empty_input():
    pipeline = _pipeline_without_model_load()

    assert pipeline._transcribe_audio([], language="en") == ""
    pipeline.processor.assert_not_called()


def test_transcribe_audio_builds_prompt_with_whisper_prompt_api():
    pipeline = _pipeline_without_model_load()
    prompt_ids = MagicMock(name="prompt_ids")
    prompt_ids.__len__.return_value = 4
    pipeline.processor.get_prompt_ids.return_value = prompt_ids

    pipeline._transcribe_audio(
        np.zeros(16000, dtype=np.float32),
        language="pl",
        initial_prompt="PandaDoc Wyoming",
    )

    pipeline.processor.get_prompt_ids.assert_called_once_with(
        "PandaDoc Wyoming",
        return_tensors="pt",
    )
    assert pipeline.model.generate.call_args.kwargs["prompt_ids"] is prompt_ids
    pipeline.processor.tokenizer.encode.assert_not_called()


def test_transcribe_audio_caps_long_prompt_and_preserves_output_budget():
    pipeline = _pipeline_without_model_load()
    pipeline.processor.get_decoder_prompt_ids.return_value = [
        (1, 10),
        (2, 20),
        (3, 30),
    ]
    pipeline.processor.get_prompt_ids.return_value = torch.arange(400)
    pipeline.model.config.max_target_positions = DEFAULT_DECODER_CONTEXT_LENGTH

    pipeline._transcribe_audio(
        np.zeros(16000, dtype=np.float32),
        language="en",
        initial_prompt="very long prompt",
    )

    generate_kwargs = pipeline.model.generate.call_args.kwargs
    prompt_ids = generate_kwargs["prompt_ids"]
    assert len(prompt_ids) == DEFAULT_DECODER_CONTEXT_LENGTH // 2
    assert prompt_ids[0].item() == 0  # Preserve <|startofprev|>.
    assert prompt_ids[-1].item() == 399  # Keep the most recent context.
    assert generate_kwargs["max_new_tokens"] == 220


def test_transcribe_audio_keeps_default_budget_without_prompt():
    pipeline = _pipeline_without_model_load()

    pipeline._transcribe_audio(
        np.zeros(16000, dtype=np.float32),
        language="en",
    )

    assert (
        pipeline.model.generate.call_args.kwargs["max_new_tokens"]
        == DEFAULT_MAX_NEW_TOKENS
    )


def test_inference_failure_is_raised_without_poisoning_worker():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.data_queue = Queue()
    pipeline._requests = InferenceRequestQueue("CPU transcription")
    pipeline._error = None
    pipeline.running = True
    failure = RuntimeError("CPU generation failed")

    def fail_request(*_args, **_kwargs):
        pipeline.running = False
        raise failure

    pipeline._transcribe_audio = MagicMock(side_effect=fail_request)
    pipeline.send_data(np.zeros(16000, dtype=np.float32), "en", "")

    pipeline._inference_loop()

    with pytest.raises(RuntimeError, match="CPU generation failed") as exc_info:
        pipeline.get_transcription(timeout_sec=0.01)

    assert exc_info.value is failure

    pipeline.running = True
    next_future = pipeline._requests.submit()
    pipeline._requests.set_result(next_future, "next request succeeded")
    assert pipeline.get_transcription(timeout_sec=0.01) == "next request succeeded"


def test_normal_logs_do_not_include_prompt_or_transcription(caplog):
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.data_queue = Queue()
    pipeline._requests = InferenceRequestQueue("CPU transcription")
    pipeline._error = None
    pipeline.running = True

    def transcribe_once(*_args, **_kwargs):
        pipeline.running = False
        return "private recognized speech"

    pipeline._transcribe_audio = transcribe_once
    pipeline.send_data(
        np.zeros(16000, dtype=np.float32),
        "en",
        "private prompt",
    )

    with caplog.at_level(logging.INFO):
        pipeline._inference_loop()

    assert "private prompt" not in caplog.text
    assert "private recognized speech" not in caplog.text


def test_default_transcription_timeout_is_finite():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline._requests = MagicMock()
    pipeline._requests.get_result.side_effect = TimeoutError(
        "Timed out waiting for CPU transcription after 300.0 seconds"
    )
    pipeline._error = None

    with pytest.raises(TimeoutError, match="300.0 seconds"):
        pipeline.get_transcription()

    pipeline._requests.get_result.assert_called_once_with(
        DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
    )
