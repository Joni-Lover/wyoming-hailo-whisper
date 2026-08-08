"""Tests for Hailo worker failure handling without requiring Hailo hardware."""

import sys
from queue import Queue
from types import ModuleType
from unittest.mock import MagicMock

import pytest

try:
    import hailo_platform
except ImportError:
    hailo_platform = ModuleType("hailo_platform")
    for name in (
        "HEF",
        "VDevice",
        "HailoSchedulingAlgorithm",
        "FormatType",
    ):
        setattr(hailo_platform, name, MagicMock())
    sys.modules["hailo_platform"] = hailo_platform

from wyoming_hailo_whisper.app.hailo_whisper_pipeline import (
    DEFAULT_TRANSCRIPTION_TIMEOUT_SEC,
    HailoWhisperPipeline,
    max_initial_prompt_tokens,
)
from wyoming_hailo_whisper.app.request_queue import InferenceRequestQueue


def _pipeline_without_hardware():
    pipeline = HailoWhisperPipeline.__new__(HailoWhisperPipeline)
    pipeline.data_queue = Queue()
    pipeline._requests = InferenceRequestQueue("Hailo transcription")
    pipeline._error = None
    pipeline.running = True
    return pipeline


def test_worker_crash_is_exposed_to_callers():
    pipeline = _pipeline_without_hardware()
    failure = RuntimeError("device initialization failed")
    pipeline._run_inference = MagicMock(side_effect=failure)

    pipeline._inference_loop()

    assert pipeline._error is failure
    assert pipeline.running is False
    with pytest.raises(RuntimeError, match="worker failed") as exc_info:
        pipeline.get_transcription(timeout_sec=0.01)
    assert exc_info.value.__cause__ is failure


def test_worker_failure_unblocks_waiting_caller():
    pipeline = _pipeline_without_hardware()
    failure = RuntimeError("worker crashed")
    pipeline._requests.submit()
    pipeline._error = failure
    worker_error = RuntimeError("Hailo inference worker failed")
    worker_error.__cause__ = failure
    pipeline._requests.fail_all(worker_error)

    with pytest.raises(RuntimeError, match="worker failed"):
        pipeline._requests.get_result(0.01)


def test_request_failure_is_raised_without_poisoning_worker():
    pipeline = _pipeline_without_hardware()
    failure = ValueError("decoder buffer rejected")
    failed_future = pipeline._requests.submit()
    pipeline._requests.set_exception(failed_future, failure)

    with pytest.raises(ValueError, match="decoder buffer rejected") as exc_info:
        pipeline.get_transcription(timeout_sec=0.01)

    assert exc_info.value is failure
    assert pipeline._error is None

    next_future = pipeline._requests.submit()
    pipeline._requests.set_result(next_future, "next request succeeded")
    assert pipeline.get_transcription(timeout_sec=0.01) == "next request succeeded"


def test_late_hailo_result_cannot_leak_into_next_request():
    pipeline = _pipeline_without_hardware()
    first = pipeline._requests.submit()

    with pytest.raises(TimeoutError):
        pipeline.get_transcription(timeout_sec=0.0)

    second = pipeline._requests.submit()
    pipeline._requests.set_result(first, "late first result")
    pipeline._requests.set_result(second, "second result")

    assert pipeline.get_transcription(timeout_sec=0.01) == "second result"


def test_prompt_budget_reserves_half_of_decoder_context_for_output():
    sequence_length = 448
    prompt_tokens = max_initial_prompt_tokens(sequence_length)
    prefix_length = 1 + prompt_tokens + 4

    assert prompt_tokens == 223
    assert sequence_length - prefix_length == 220


def test_default_transcription_timeout_is_finite():
    pipeline = _pipeline_without_hardware()
    pipeline._requests = MagicMock()
    pipeline._requests.get_result.side_effect = TimeoutError(
        "Timed out waiting for Hailo transcription after 15.0 seconds"
    )

    with pytest.raises(TimeoutError, match="15.0 seconds"):
        pipeline.get_transcription()

    pipeline._requests.get_result.assert_called_once_with(
        DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
    )
