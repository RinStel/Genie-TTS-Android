"""Versioned JSON contract for cross-runtime parity fixtures."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any

FIXTURE_SCHEMA_VERSION = 1

_SUPPORTED_FIXTURE_TYPES = {"chinese_frontend", "genie_tts_parity_contract"}
_REQUIRED_CASE_KEYS = {"id", "category", "input_text"}
_REQUIRED_FRONTEND_KEYS = {"normalized_text", "phones", "phone_ids", "word2ph"}
_OPTIONAL_EXPECTED_KEYS = {"tensors", "semantic_tokens", "timing", "trace"}
_FULL_CONTRACT_REQUIRED_KEYS = {"tensors", "semantic_tokens", "timing", "trace"}
_FULL_CONTRACT_TENSORS = {
    "reference_pcm",
    "hubert_features",
    "speaker_embedding",
    "prompt_conditioning",
    "t2s_output",
    "vocoder_output",
}
_FULL_CONTRACT_TRACE_ROLES = {
    "frontend",
    "hubert",
    "speaker_encoder",
    "prompt_encoder",
    "t2s",
    "vocoder",
}
_INTEGER_DTYPES = {"int32", "int64"}
_FLOAT_DTYPES = {"float32"}
_MODEL_ROLES = {
    "frontend",
    "roberta",
    "hubert",
    "speaker_encoder",
    "prompt_encoder",
    "t2s",
    "vocoder",
}
_PROVIDERS = {"cpu", "qnn", "xnnpack"}
_CACHE_STATUSES = {"hit", "miss", "n/a"}
_METADATA_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FixtureSchemaError(ValueError):
    """Raised when a fixture cannot be consumed by this schema version."""


def serialize_integer_tensor(*, shape: Sequence[int], values: Sequence[int]) -> dict[str, Any]:
    normalized_values = _validate_integer_values(values)
    normalized_shape = _validate_shape(shape, len(normalized_values))
    return {
        "dtype": "int64",
        "shape": normalized_shape,
        "values": normalized_values,
    }


def serialize_float_tensor(*, shape: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    if not _is_sequence(values):
        raise FixtureSchemaError("Float tensor values must be an array.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise FixtureSchemaError("Float tensor values must be numbers.")
    normalized_values = [float(value) for value in values]
    normalized_shape = _validate_shape(shape, len(normalized_values))
    payload = b"".join(struct.pack("<f", value) for value in normalized_values)
    finite_values = [value for value in normalized_values if math.isfinite(value)]
    statistics: dict[str, int | float | None] = {
        "finite_count": len(finite_values),
        "non_finite_count": len(normalized_values) - len(finite_values),
        "min": min(finite_values) if finite_values else None,
        "max": max(finite_values) if finite_values else None,
        "mean": sum(finite_values) / len(finite_values) if finite_values else None,
    }
    return {
        "dtype": "float32",
        "shape": normalized_shape,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "statistics": statistics,
    }


def decode_fixture(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise FixtureSchemaError("Fixture root must be an object.")

    version = document.get("schema_version")
    if type(version) is not int or version != FIXTURE_SCHEMA_VERSION:
        raise FixtureSchemaError(
            f"Unsupported schema_version {version!r}; expected integer {FIXTURE_SCHEMA_VERSION}."
        )

    fixture_type = document.get("fixture_type")
    if not isinstance(fixture_type, str) or fixture_type not in _SUPPORTED_FIXTURE_TYPES:
        raise FixtureSchemaError(
            f"Unsupported fixture_type {fixture_type!r}; expected one of {sorted(_SUPPORTED_FIXTURE_TYPES)}."
        )

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FixtureSchemaError("Fixture cases must be a non-empty array.")

    case_ids: set[str] = set()
    for case in cases:
        _validate_case(case, case_ids, fixture_type)
    return dict(document)


def _validate_case(case: Any, case_ids: set[str], fixture_type: str) -> None:
    if not isinstance(case, Mapping) or not _REQUIRED_CASE_KEYS.issubset(case):
        raise FixtureSchemaError(f"Each case must include {sorted(_REQUIRED_CASE_KEYS)}.")

    case_id = case["id"]
    if not isinstance(case_id, str) or not case_id or case_id in case_ids:
        raise FixtureSchemaError("Case ids must be unique non-empty strings.")
    if not isinstance(case["category"], str) or not case["category"]:
        raise FixtureSchemaError("Case category must be a non-empty string.")
    if not isinstance(case["input_text"], str):
        raise FixtureSchemaError("Case input_text must be a string.")

    expected = case.get("expected")
    if expected is not None:
        _validate_expected(expected, full_contract=fixture_type == "genie_tts_parity_contract")
    elif fixture_type == "genie_tts_parity_contract":
        raise FixtureSchemaError("Full parity contract cases must include expected data.")
    case_ids.add(case_id)


def _validate_expected(expected: Any, *, full_contract: bool = False) -> None:
    if not isinstance(expected, Mapping):
        raise FixtureSchemaError("Case expected data must be an object.")
    missing = _REQUIRED_FRONTEND_KEYS.difference(expected)
    if missing:
        raise FixtureSchemaError(f"Frontend expected data is missing {sorted(missing)}.")
    unknown = set(expected).difference(_REQUIRED_FRONTEND_KEYS | _OPTIONAL_EXPECTED_KEYS)
    if unknown:
        raise FixtureSchemaError(f"Frontend expected data has unknown fields {sorted(unknown)}.")

    if not isinstance(expected["normalized_text"], str):
        raise FixtureSchemaError("expected.normalized_text must be a string.")
    phones = expected["phones"]
    if not isinstance(phones, list) or any(not isinstance(phone, str) for phone in phones):
        raise FixtureSchemaError("expected.phones must be an array of strings.")
    _validate_integer_tensor(expected["phone_ids"], "expected.phone_ids")
    _validate_integer_tensor(expected["word2ph"], "expected.word2ph")

    tensors = expected.get("tensors")
    if tensors is not None:
        _validate_named_tensors(tensors)
    semantic_tokens = expected.get("semantic_tokens")
    if semantic_tokens is not None:
        _validate_integer_tensor(semantic_tokens, "expected.semantic_tokens")
    timing = expected.get("timing")
    if timing is not None:
        _validate_timing(timing)
    trace = expected.get("trace")
    if trace is not None:
        trace_roles = _validate_trace(trace)
    else:
        trace_roles = set()
    if full_contract:
        missing_contract = _FULL_CONTRACT_REQUIRED_KEYS.difference(expected)
        if missing_contract:
            raise FixtureSchemaError(
                f"Full parity contract expected data is missing {sorted(missing_contract)}."
            )
        if set(tensors) != _FULL_CONTRACT_TENSORS:
            raise FixtureSchemaError("Full parity contract tensors must record every required boundary.")
        if any(tensors[name].get("dtype") != "float32" for name in _FULL_CONTRACT_TENSORS):
            raise FixtureSchemaError("Full parity contract boundary tensors must be float32.")
        if trace_roles != _FULL_CONTRACT_TRACE_ROLES:
            raise FixtureSchemaError("Full parity contract trace must record every required model role.")


def _validate_named_tensors(tensors: Any) -> None:
    if not isinstance(tensors, Mapping) or not tensors:
        raise FixtureSchemaError("expected.tensors must be a non-empty object.")
    for name, record in tensors.items():
        if not isinstance(name, str) or not _METADATA_NAME.fullmatch(name):
            raise FixtureSchemaError("Tensor names must use lower_snake_case metadata names.")
        _validate_tensor(record, f"expected.tensors.{name}")


def _validate_tensor(record: Any, label: str) -> None:
    if not isinstance(record, Mapping):
        raise FixtureSchemaError(f"{label} must be an object.")
    dtype = record.get("dtype")
    if dtype in _INTEGER_DTYPES:
        _validate_integer_tensor(record, label)
    elif dtype in _FLOAT_DTYPES:
        _validate_float_tensor(record, label)
    else:
        raise FixtureSchemaError(f"{label}.dtype is unsupported: {dtype!r}.")


def _validate_integer_tensor(record: Any, label: str) -> None:
    if not isinstance(record, Mapping):
        raise FixtureSchemaError(f"{label} must be an object.")
    if set(record) != {"dtype", "shape", "values"}:
        raise FixtureSchemaError(f"{label} must contain only dtype, shape, and values.")
    if record["dtype"] not in _INTEGER_DTYPES:
        raise FixtureSchemaError(f"{label}.dtype must be an integer dtype.")
    values = _validate_integer_values(record["values"])
    _validate_shape(record["shape"], len(values))


def _validate_float_tensor(record: Mapping[str, Any], label: str) -> None:
    if set(record) != {"dtype", "shape", "sha256", "statistics"}:
        raise FixtureSchemaError(
            f"{label} must contain only dtype, shape, sha256, and statistics."
        )
    if record["dtype"] not in _FLOAT_DTYPES:
        raise FixtureSchemaError(f"{label}.dtype must be float32.")
    shape = _validate_shape(record["shape"])
    digest = record["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise FixtureSchemaError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    _validate_statistics(record["statistics"], _element_count(shape), label)


def _validate_statistics(statistics: Any, value_count: int, label: str) -> None:
    keys = {"finite_count", "non_finite_count", "min", "max", "mean"}
    if not isinstance(statistics, Mapping) or set(statistics) != keys:
        raise FixtureSchemaError(f"{label}.statistics must contain {sorted(keys)}.")

    finite_count = statistics["finite_count"]
    non_finite_count = statistics["non_finite_count"]
    if (
        type(finite_count) is not int
        or type(non_finite_count) is not int
        or finite_count < 0
        or non_finite_count < 0
        or finite_count + non_finite_count != value_count
    ):
        raise FixtureSchemaError(f"{label}.statistics counts must match tensor shape.")

    summary = [statistics["min"], statistics["max"], statistics["mean"]]
    if finite_count == 0:
        if any(value is not None for value in summary):
            raise FixtureSchemaError(f"{label}.statistics summaries must be null without finite values.")
        return
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in summary
    ):
        raise FixtureSchemaError(f"{label}.statistics summaries must be finite numbers.")
    minimum, maximum, mean = summary
    if minimum > maximum or not minimum <= mean <= maximum:
        raise FixtureSchemaError(f"{label}.statistics min/mean/max ordering is invalid.")


def _validate_timing(timing: Any) -> None:
    if not isinstance(timing, Mapping) or set(timing) != {"stages"}:
        raise FixtureSchemaError("expected.timing must contain only stages.")
    stages = timing["stages"]
    if not isinstance(stages, list) or not stages:
        raise FixtureSchemaError("expected.timing.stages must be a non-empty array.")
    for stage in stages:
        if not isinstance(stage, Mapping) or set(stage) != {"name", "elapsed_ms"}:
            raise FixtureSchemaError("Each timing stage must contain name and elapsed_ms.")
        if not isinstance(stage["name"], str) or not _METADATA_NAME.fullmatch(stage["name"]):
            raise FixtureSchemaError("Timing stage names must use lower_snake_case.")
        if type(stage["elapsed_ms"]) is not int or stage["elapsed_ms"] < 0:
            raise FixtureSchemaError("Timing elapsed_ms must be a non-negative integer.")


def _validate_trace(trace: Any) -> set[str]:
    allowed = {"model_role", "provider", "tensor_shape", "cache_status", "elapsed_ms"}
    required = {"model_role", "provider"}
    if not isinstance(trace, list) or not trace:
        raise FixtureSchemaError("expected.trace must be a non-empty array.")
    roles: set[str] = set()
    for record in trace:
        if not isinstance(record, Mapping) or not required.issubset(record):
            raise FixtureSchemaError("Each trace record must include model_role and provider.")
        unknown = set(record).difference(allowed)
        if unknown:
            raise FixtureSchemaError(f"Trace record has unknown fields {sorted(unknown)}.")
        if record["model_role"] not in _MODEL_ROLES:
            raise FixtureSchemaError("Trace model_role is unsupported.")
        if record["provider"] not in _PROVIDERS:
            raise FixtureSchemaError("Trace provider is unsupported.")
        if "tensor_shape" in record:
            _validate_shape(record["tensor_shape"])
        if "cache_status" in record and record["cache_status"] not in _CACHE_STATUSES:
            raise FixtureSchemaError("Trace cache_status is unsupported.")
        if "elapsed_ms" in record and (
            type(record["elapsed_ms"]) is not int or record["elapsed_ms"] < 0
        ):
            raise FixtureSchemaError("Trace elapsed_ms must be a non-negative integer.")
        roles.add(record["model_role"])
    return roles


def _validate_integer_values(values: Any) -> list[int]:
    if not _is_sequence(values):
        raise FixtureSchemaError("Integer tensor values must be an array.")
    if any(type(value) is not int for value in values):
        raise FixtureSchemaError("Integer tensor values must be integers.")
    return list(values)


def _validate_shape(shape: Any, value_count: int | None = None) -> list[int]:
    if not _is_sequence(shape) or any(type(dimension) is not int for dimension in shape):
        raise FixtureSchemaError("Tensor dimensions must be integers, not booleans or floats.")
    normalized_shape = list(shape)
    if any(dimension < 0 for dimension in normalized_shape):
        raise FixtureSchemaError("Tensor dimensions cannot be negative.")
    element_count = _element_count(normalized_shape)
    if value_count is not None and element_count != value_count:
        raise FixtureSchemaError(
            f"Tensor shape {normalized_shape} requires {element_count} values, got {value_count}."
        )
    return normalized_shape


def _element_count(shape: Sequence[int]) -> int:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return element_count


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
