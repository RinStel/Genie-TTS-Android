"""Versioned JSON contract for cross-runtime parity fixtures."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

FIXTURE_SCHEMA_VERSION = 1
_REQUIRED_CASE_KEYS = {"id", "category", "input_text"}


class FixtureSchemaError(ValueError):
    """Raised when a fixture cannot be consumed by this schema version."""


def serialize_integer_tensor(*, shape: Sequence[int], values: Sequence[int]) -> dict[str, Any]:
    normalized_shape = _validate_shape(shape, len(values))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise FixtureSchemaError("Integer tensor values must be integers.")
    return {
        "dtype": "int64",
        "shape": normalized_shape,
        "values": [int(value) for value in values],
    }


def serialize_float_tensor(*, shape: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    normalized_shape = _validate_shape(shape, len(values))
    normalized_values = [float(value) for value in values]
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
    if version != FIXTURE_SCHEMA_VERSION:
        raise FixtureSchemaError(
            f"Unsupported schema_version {version!r}; expected {FIXTURE_SCHEMA_VERSION}."
        )
    if document.get("fixture_type") != "chinese_frontend":
        raise FixtureSchemaError("Fixture type must be 'chinese_frontend'.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FixtureSchemaError("Fixture cases must be a non-empty array.")

    case_ids: set[str] = set()
    for case in cases:
        _validate_case(case, case_ids)
    return dict(document)


def _validate_case(case: Any, case_ids: set[str]) -> None:
    if not isinstance(case, Mapping) or not _REQUIRED_CASE_KEYS.issubset(case):
        raise FixtureSchemaError(f"Each case must include {sorted(_REQUIRED_CASE_KEYS)}.")
    case_id = case["id"]
    if not isinstance(case_id, str) or not case_id or case_id in case_ids:
        raise FixtureSchemaError("Case ids must be unique non-empty strings.")
    if not isinstance(case["category"], str) or not isinstance(case["input_text"], str):
        raise FixtureSchemaError("Case category and input_text must be strings.")
    case_ids.add(case_id)


def _validate_shape(shape: Sequence[int], value_count: int) -> list[int]:
    normalized_shape = [int(dimension) for dimension in shape]
    if any(dimension < 0 for dimension in normalized_shape):
        raise FixtureSchemaError("Tensor dimensions cannot be negative.")
    element_count = 1
    for dimension in normalized_shape:
        element_count *= dimension
    if element_count != value_count:
        raise FixtureSchemaError(
            f"Tensor shape {normalized_shape} requires {element_count} values, got {value_count}."
        )
    return normalized_shape
