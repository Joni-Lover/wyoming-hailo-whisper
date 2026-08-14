"""Correlate queued inference jobs with their individual results."""

from concurrent.futures import Future, InvalidStateError
from queue import Empty, Queue
from threading import Lock
from time import monotonic
from typing import Any


class InferenceRequestQueue:
    """Track FIFO requests while keeping every result in a private future.

    A shared result queue is unsafe after a caller times out: the worker can
    publish that request's late result and the next caller can consume it.
    Pairing every submitted job with its own future makes late completion
    harmless while retaining the existing ``send_data``/``get_transcription``
    API ordering.
    """

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self._pending: Queue[Future] = Queue()
        self._active: set[Future] = set()
        self._lock = Lock()

    def submit(self) -> Future:
        """Create and register the future for one inference job."""
        future: Future = Future()
        with self._lock:
            self._active.add(future)
        self._pending.put(future)
        return future

    def set_result(self, future: Future, result: Any) -> None:
        """Complete a request unless another terminal state won the race."""
        try:
            future.set_result(result)
        except InvalidStateError:
            pass

    def set_exception(self, future: Future, error: BaseException) -> None:
        """Fail a request unless another terminal state won the race."""
        try:
            future.set_exception(error)
        except InvalidStateError:
            pass

    def fail_all(self, error: BaseException) -> None:
        """Wake every currently outstanding caller with a worker failure."""
        with self._lock:
            active = tuple(self._active)
        for future in active:
            self.set_exception(future, error)

    def get_result(self, timeout_sec: float) -> Any:
        """Return the oldest submitted request's result within one deadline."""
        if timeout_sec < 0:
            raise ValueError("timeout_sec must be non-negative")

        deadline = monotonic() + timeout_sec
        try:
            future = self._pending.get(timeout=timeout_sec)
        except Empty as err:
            raise self._timeout_error(timeout_sec) from err

        try:
            remaining = max(0.0, deadline - monotonic())
            try:
                return future.result(timeout=remaining)
            except TimeoutError as err:
                # ``concurrent.futures.TimeoutError`` is an alias of the
                # built-in exception. If the worker itself raised TimeoutError,
                # the future is done and its original error must propagate.
                if future.done():
                    return future.result()
                raise self._timeout_error(timeout_sec) from err
        finally:
            with self._lock:
                self._active.discard(future)

    def _timeout_error(self, timeout_sec: float) -> TimeoutError:
        return TimeoutError(
            f"Timed out waiting for {self.operation_name} after "
            f"{timeout_sec} seconds"
        )
