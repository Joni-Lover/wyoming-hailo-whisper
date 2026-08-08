"""Tests for request/result correlation around inference timeouts."""

import threading

import pytest

from wyoming_hailo_whisper.app.request_queue import InferenceRequestQueue


def test_late_result_cannot_leak_into_next_request():
    requests = InferenceRequestQueue("test inference")
    first = requests.submit()

    with pytest.raises(TimeoutError, match="0.0 seconds"):
        requests.get_result(0.0)

    second = requests.submit()
    requests.set_result(first, "late first result")
    requests.set_result(second, "second result")

    assert requests.get_result(0.01) == "second result"


def test_worker_timeout_error_is_not_misreported_as_wait_timeout():
    requests = InferenceRequestQueue("test inference")
    future = requests.submit()
    failure = TimeoutError("model generation timed out")
    requests.set_exception(future, failure)

    with pytest.raises(TimeoutError, match="model generation timed out") as exc_info:
        requests.get_result(1.0)

    assert exc_info.value is failure


def test_fail_all_unblocks_waiting_request():
    requests = InferenceRequestQueue("test inference")
    requests.submit()
    failure = RuntimeError("worker failed")
    timer = threading.Timer(0.01, requests.fail_all, args=(failure,))
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="worker failed") as exc_info:
            requests.get_result(1.0)
    finally:
        timer.cancel()

    assert exc_info.value is failure
