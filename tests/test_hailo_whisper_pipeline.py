"""Tests for Hailo worker failure handling without requiring Hailo hardware."""

import inspect
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

from wyoming_hailo_whisper.app import hailo_whisper_pipeline as pipeline_module
from wyoming_hailo_whisper.app.hailo_whisper_pipeline import (
    DEFAULT_HAILO_RUN_TIMEOUT_MS,
    DEFAULT_TRANSCRIPTION_TIMEOUT_SEC,
    HailoWhisperPipeline,
    build_decode_metrics,
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


@pytest.mark.parametrize(
    ("sequence_length", "expected_prompt_tokens"),
    [(24, 3), (32, 11), (448, 223)],
)
def test_prompt_budget_reserves_result_tokens_for_short_hailo_contexts(
    sequence_length,
    expected_prompt_tokens,
):
    assert max_initial_prompt_tokens(sequence_length) == expected_prompt_tokens


def test_prompt_tokenization_uses_whisper_leading_space():
    pipeline = _pipeline_without_hardware()
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.encode.return_value = [1, 2]

    assert pipeline._encode_initial_prompt("  гостиная ") == [1, 2]
    pipeline.tokenizer.encode.assert_called_once_with(
        " гостиная",
        add_special_tokens=False,
    )


def test_hailo_language_token_accepts_wyoming_locale():
    pipeline = _pipeline_without_hardware()
    pipeline.language = "en"
    pipeline._language_token_cache = {}
    pipeline.tokenizer = MagicMock(unk_token_id=-1)
    pipeline.tokenizer.convert_tokens_to_ids.return_value = 50263

    assert pipeline._get_language_token("ru-RU") == 50263
    pipeline.tokenizer.convert_tokens_to_ids.assert_called_once_with("<|ru|>")


def test_hailo_language_token_falls_back_for_unknown_code():
    pipeline = _pipeline_without_hardware()
    pipeline.language = "en"
    pipeline._language_token_cache = {}
    pipeline.tokenizer = MagicMock(unk_token_id=-1)
    pipeline.tokenizer.convert_tokens_to_ids.return_value = 50259

    assert pipeline._get_language_token("xx-ZZ") == 50259
    pipeline.tokenizer.convert_tokens_to_ids.assert_called_once_with("<|en|>")


def test_hailo_tokenizer_load_is_offline_only(monkeypatch):
    pipeline = _pipeline_without_hardware()
    pipeline.variant = "base"
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = 1
    loader = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(
        pipeline_module.AutoTokenizer,
        "from_pretrained",
        loader,
    )

    pipeline._load_tokenizer()

    tokenizer_path = loader.call_args.args[0]
    assert tokenizer_path.endswith("decoder_assets/base/tokenizer")
    assert loader.call_args.kwargs == {"local_files_only": True}


def test_decode_metrics_report_finished_and_truncated_results():
    finished = build_decode_metrics(
        transcription="включи свет",
        content_tokens=[10, 11, 50257],
        score=-1.5,
        finished=True,
        max_content_length=3,
    )
    truncated = build_decode_metrics(
        transcription="включи свет",
        content_tokens=[10, 11, 12],
        score=-3.0,
        finished=False,
        max_content_length=3,
    )

    assert finished["token_count"] == 2
    assert finished["avg_logprob"] == pytest.approx(-0.5)
    assert finished["compression_ratio"] > 0
    assert finished["truncated"] is False
    assert truncated["token_count"] == 3
    assert truncated["avg_logprob"] == pytest.approx(-1.0)
    assert truncated["truncated"] is True


def test_hailo_runtime_calls_have_a_finite_timeout():
    assert DEFAULT_HAILO_RUN_TIMEOUT_MS == 15_000


def test_constructor_keeps_legacy_language_positional_slot():
    parameters = list(inspect.signature(HailoWhisperPipeline).parameters)

    assert parameters[:7] == [
        "encoder_model_path",
        "decoder_model_path",
        "variant",
        "host",
        "multi_process_service",
        "language",
        "beam_size",
    ]


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
