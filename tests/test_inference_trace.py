from __future__ import annotations

import hashlib
import importlib.util
import io
import logging
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src/genie_tts/Core/InferenceTrace.py"
SPEC = importlib.util.spec_from_file_location("genie_tts_inference_trace", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trace() -> tuple[object, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger("inference-trace-test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    return MODULE.InferenceTrace(enabled=True, logger=logger), stream


def test_tensor_hash_uses_android_compatible_little_endian_bytes() -> None:
    trace, stream = _trace()

    trace.tensor_hash("text_seq", np.array([1, 2], dtype=np.int64))

    expected = hashlib.sha256(np.array([1, 2], dtype="<i8").tobytes()).hexdigest()
    assert stream.getvalue().strip() == f"text_seq_hash: {expected}"


def test_float_tensor_hash_also_emits_finite_statistics() -> None:
    trace, stream = _trace()

    trace.tensor_hash("audio", np.array([-1.0, 1.0, np.nan, np.inf], dtype=np.float32))

    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("audio_hash: ")
    assert lines[1] == (
        "audio_stats: finite_count=2 non_finite_count=2 "
        "min=-1 max=1 mean=0"
    )


def test_trace_emits_boundary_shapes_roles_cache_and_timing() -> None:
    trace, stream = _trace()

    trace.backend("CPU")
    trace.runtime("Python ORT CPU EP")
    trace.role("prompt_encoder")
    trace.cache("hit")
    trace.count("semantic_tokens", 4)
    trace.shape("semantic", np.zeros((1, 1, 4), dtype=np.int64))
    trace.stage("vocoder_ms", 0.001)

    lines = stream.getvalue().splitlines()
    assert lines[:6] == [
        "resolved_backend: CPU",
        "runtime_label: Python ORT CPU EP",
        "role=prompt_encoder provider=cpu cache=n/a",
        "reference_cache: hit",
        "semantic_tokens: 4",
        "semantic_shape: [1,1,4]",
    ]
    assert lines[6].startswith("vocoder_ms=1.000 ms")


def test_trace_is_silent_when_disabled() -> None:
    trace, stream = _trace()
    trace.enabled = False

    trace.backend("CPU")

    assert stream.getvalue() == ""


def test_disabled_trace_skips_tensor_hash_work() -> None:
    trace, _ = _trace()
    trace.enabled = False

    # Object tensors are intentionally unsupported when tracing is enabled.
    # Disabled tracing must return before inspecting the dtype or data.
    trace.tensor_hash("audio", np.array([object()], dtype=object))
