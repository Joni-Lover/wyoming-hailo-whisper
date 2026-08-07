"""Tests for Hailo worker failure handling without requiring Hailo hardware."""

import sys
from queue import Empty, Queue
from types import ModuleType
from unittest.mock import MagicMock

import pytest

try:
    import hailo_platform  # noqa: F401
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

from wyoming_hailo_whisper.app.hailo_whisper_pipeline import (  # noqa: E402
    DEFAULT_TRANSCRIPTION_TIMEOUT_SEC,
    HailoWhisperPipeline,
    _INFERENCE_FAILED,
)


def _pipeline_without_hardware():
    pipeline = HailoWhisperPipeline.__new__(HailoWhisperPipeline)
    pipeline.results_queue = Queue()
    pipeline.data_queue = Queue()
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


def test_failure_sentinel_unblocks_waiting_caller():
    pipeline = _pipeline_without_hardware()
    failure = RuntimeError("worker crashed")
    results_queue = MagicMock()

    def receive_failure(*, timeout):
        pipeline._error = failure
        return _INFERENCE_FAILED

    results_queue.get.side_effect = receive_failure
    pipeline.results_queue = results_queue

    with pytest.raises(RuntimeError, match="worker failed"):
        pipeline.get_transcription(timeout_sec=0.01)


def test_request_failure_is_raised_without_poisoning_worker():
    pipeline = _pipeline_without_hardware()
    failure = ValueError("decoder buffer rejected")
    pipeline.results_queue.put(failure)

    with pytest.raises(ValueError, match="decoder buffer rejected") as exc_info:
        pipeline.get_transcription(timeout_sec=0.01)

    assert exc_info.value is failure
    assert pipeline._error is None

    pipeline.results_queue.put("next request succeeded")
    assert pipeline.get_transcription(timeout_sec=0.01) == "next request succeeded"


def test_default_transcription_timeout_is_finite():
    pipeline = _pipeline_without_hardware()
    pipeline.results_queue = MagicMock()
    pipeline.results_queue.get.side_effect = Empty

    with pytest.raises(TimeoutError, match="15.0 seconds"):
        pipeline.get_transcription()

    pipeline.results_queue.get.assert_called_once_with(
        timeout=DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
    )
