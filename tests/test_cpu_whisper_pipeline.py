"""Tests for CPU Whisper chunking without loading model weights."""

from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from wyoming_hailo_whisper.app.cpu_whisper_pipeline import (
    DEFAULT_TRANSCRIPTION_TIMEOUT_SEC,
    CpuWhisperPipeline,
)


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


def test_inference_failure_is_raised_without_poisoning_worker():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.data_queue = Queue()
    pipeline.results_queue = Queue()
    pipeline.running = True
    pipeline.data_queue.put((np.zeros(16000, dtype=np.float32), "en", ""))
    failure = RuntimeError("CPU generation failed")

    def fail_request(*_args, **_kwargs):
        pipeline.running = False
        raise failure

    pipeline._transcribe_audio = MagicMock(side_effect=fail_request)

    pipeline._inference_loop()

    with pytest.raises(RuntimeError, match="CPU generation failed") as exc_info:
        pipeline.get_transcription(timeout_sec=0.01)

    assert exc_info.value is failure

    pipeline.results_queue.put("next request succeeded")
    assert pipeline.get_transcription(timeout_sec=0.01) == "next request succeeded"


def test_default_transcription_timeout_is_finite():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.results_queue = MagicMock()
    pipeline.results_queue.get.side_effect = Empty

    with pytest.raises(TimeoutError, match="300.0 seconds"):
        pipeline.get_transcription()

    pipeline.results_queue.get.assert_called_once_with(
        timeout=DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
    )
