"""Opt-in, value-free inference trace shared by Python parity checks."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

import numpy as np


TRACE_ENVIRONMENT_VARIABLE = "GENIE_TTS_TRACE"
TRACE_LOGGER_NAME = "GenieTtsTiming"


def _is_enabled() -> bool:
    return os.getenv(TRACE_ENVIRONMENT_VARIABLE, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class InferenceTrace:
    """Emit the same boundary fields as Android without logging user values."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = _is_enabled() if enabled is None else enabled
        self.logger = logger or logging.getLogger(TRACE_LOGGER_NAME)

    def _emit(self, line: str) -> None:
        if self.enabled:
            self.logger.info(line)

    def backend(self, value: str) -> None:
        self._emit(f"resolved_backend: {value}")

    def runtime(self, value: str) -> None:
        self._emit(f"runtime_label: {value}")

    def role(self, role: str, provider: str = "cpu") -> None:
        self._emit(f"role={role} provider={provider} cache=n/a")

    def cache(self, status: str) -> None:
        self._emit(f"reference_cache: {status}")

    def count(self, name: str, value: int) -> None:
        self._emit(f"{name}: {value}")

    def shape(self, name: str, value: Any) -> None:
        shape = tuple(int(dimension) for dimension in np.asarray(value).shape)
        self._emit(f"{name}_shape: [{','.join(str(dimension) for dimension in shape)}]")

    def tensor_hash(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.integer):
            normalized = np.ascontiguousarray(array, dtype="<i8")
            emit_statistics = False
        elif np.issubdtype(array.dtype, np.floating):
            normalized = np.ascontiguousarray(array, dtype="<f4")
            emit_statistics = True
        else:
            raise TypeError(f"Unsupported trace tensor dtype: {array.dtype}")
        digest = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
        self._emit(f"{name}_hash: {digest}")
        if emit_statistics:
            finite = np.isfinite(normalized)
            finite_values = normalized[finite]
            finite_count = int(finite_values.size)
            non_finite_count = int(normalized.size - finite_count)
            if finite_count:
                minimum = float(np.min(finite_values))
                maximum = float(np.max(finite_values))
                mean = float(np.mean(finite_values, dtype=np.float64))
            else:
                minimum = maximum = mean = None
            self._emit(
                f"{name}_stats: finite_count={finite_count} "
                f"non_finite_count={non_finite_count} "
                f"min={_format_statistic(minimum)} "
                f"max={_format_statistic(maximum)} "
                f"mean={_format_statistic(mean)}"
            )

    def stage(self, name: str, elapsed_seconds: float) -> None:
        self._emit(f"{name}={max(0.0, elapsed_seconds * 1000.0):.3f} ms")

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage(name, time.perf_counter() - started)


def trace_for_current_run() -> InferenceTrace:
    """Create a trace with the current environment flag evaluated once."""
    return InferenceTrace()


def _format_statistic(value: float | None) -> str:
    return "null" if value is None else format(value, ".9g")
