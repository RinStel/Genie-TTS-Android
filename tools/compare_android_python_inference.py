"""Compare Android/Python inference traces without reading raw text or audio.

The Android app emits ``key=value ms`` timing fields in logcat.  Python or a
second Android run can use the same format.  Missing optional fields are
reported, while ``--strict`` turns them into a failed comparison.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIMING_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+)=(?P<value>[0-9]+(?:\.[0-9]+)?)\s*ms\s*$")
SHAPE_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+)_shape:\s*\[(?P<shape>[^]]*)\]")
HASH_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+)_hash:\s*(?P<hash>[0-9a-fA-F]{64})$")
VALUE_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.+)$")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STATS_RE = re.compile(
    rf"^(?P<key>[a-zA-Z0-9_]+)_stats:\s+"
    rf"finite_count=(?P<finite_count>\d+)\s+"
    rf"non_finite_count=(?P<non_finite_count>\d+)\s+"
    rf"min=(?P<min>null|{_NUMBER})\s+"
    rf"max=(?P<max>null|{_NUMBER})\s+"
    rf"mean=(?P<mean>null|{_NUMBER})$"
)


@dataclass(frozen=True)
class TensorStatistics:
    finite_count: int
    non_finite_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None


@dataclass
class Trace:
    name: str
    timings_ms: dict[str, float]
    values: dict[str, str]
    shapes: dict[str, tuple[int, ...]]
    hashes: dict[str, str]
    statistics: dict[str, TensorStatistics]
    model_roles: dict[str, str]
    reference_cache: str | None
    raw: str


def parse_trace(path: Path) -> Trace:
    timings: dict[str, float] = {}
    values: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    hashes: dict[str, str] = {}
    statistics: dict[str, TensorStatistics] = {}
    model_roles: dict[str, str] = {}
    reference_cache = None
    raw = path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        timing = TIMING_RE.match(line)
        if timing:
            timings[timing["key"]] = float(timing["value"])
            continue
        shape = SHAPE_RE.match(line)
        if shape:
            dimensions = tuple(
                int(value.strip())
                for value in shape["shape"].split(",")
                if value.strip()
            )
            shapes[shape["key"]] = dimensions
            continue
        hashed = HASH_RE.match(line)
        if hashed:
            hashes[hashed["key"]] = hashed["hash"].lower()
            continue
        stats = STATS_RE.match(line)
        if stats:
            statistics[stats["key"]] = TensorStatistics(
                finite_count=int(stats["finite_count"]),
                non_finite_count=int(stats["non_finite_count"]),
                minimum=_parse_statistic(stats["min"]),
                maximum=_parse_statistic(stats["max"]),
                mean=_parse_statistic(stats["mean"]),
            )
            continue
        if line.startswith("role="):
            fields = dict(
                field.split("=", 1)
                for field in line.split()
                if "=" in field
            )
            if fields.get("role") and fields.get("provider"):
                model_roles[fields["role"]] = fields["provider"]
            continue
        if line.startswith("reference_cache:"):
            reference_cache = line.split(":", 1)[1].strip().lower()
            continue
        value = VALUE_RE.match(line)
        if value:
            values[value["key"]] = value["value"].strip()
    return Trace(path.name, timings, values, shapes, hashes, statistics, model_roles, reference_cache, raw)


def read_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as audio:
        return {
            "channels": audio.getnchannels(),
            "sample_width": audio.getsampwidth(),
            "sample_rate": audio.getframerate(),
            "frames": audio.getnframes(),
        }


def compare_trace(
    left: Trace,
    right: Trace,
    *,
    strict: bool,
    float_atol: float = 1e-4,
    float_rtol: float = 1e-5,
) -> list[str]:
    messages: list[str] = []
    if left.values.get("resolved_backend") != right.values.get("resolved_backend"):
        messages.append(
            f"FAIL backend: {left.values.get('resolved_backend')} != "
            f"{right.values.get('resolved_backend')}"
        )
    if left.reference_cache and right.reference_cache and left.reference_cache != right.reference_cache:
        messages.append(f"WARN reference cache: {left.reference_cache} != {right.reference_cache}")

    for key in sorted(set(left.shapes) | set(right.shapes)):
        if key not in left.shapes or key not in right.shapes:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing shape: {key}")
        elif left.shapes[key] != right.shapes[key]:
            messages.append(f"FAIL shape {key}: {left.shapes[key]} != {right.shapes[key]}")

    for key in ("semantic_tokens", "audio_samples"):
        left_value = left.values.get(key)
        right_value = right.values.get(key)
        if left_value is None or right_value is None:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing value: {key}")
        elif left_value != right_value:
            messages.append(f"FAIL {key}: {left_value} != {right_value}")

    for key in sorted(set(left.statistics) | set(right.statistics)):
        if key not in left.statistics or key not in right.statistics:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing statistics: {key}")
            continue
        left_stats = left.statistics[key]
        right_stats = right.statistics[key]
        if (
            left_stats.finite_count != right_stats.finite_count
            or left_stats.non_finite_count != right_stats.non_finite_count
        ):
            messages.append(
                f"FAIL stats {key}: counts "
                f"({left_stats.finite_count},{left_stats.non_finite_count}) != "
                f"({right_stats.finite_count},{right_stats.non_finite_count})"
            )
        for field in ("minimum", "maximum", "mean"):
            left_value = getattr(left_stats, field)
            right_value = getattr(right_stats, field)
            if left_value is None or right_value is None:
                if left_value != right_value:
                    messages.append(f"FAIL stats {key}.{field}: null mismatch")
                continue
            tolerance = float_atol + float_rtol * max(abs(left_value), abs(right_value))
            if not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=tolerance):
                messages.append(
                    f"FAIL stats {key}.{field}: {left_value} != {right_value} "
                    f"(tolerance={tolerance})"
                )

    for key in sorted(set(left.hashes) | set(right.hashes)):
        # Float tensors are compared through explicit statistics tolerances;
        # integer tensors retain exact byte-level hash comparison.
        if key in left.statistics or key in right.statistics:
            continue
        if key not in left.hashes or key not in right.hashes:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing hash: {key}")
        elif left.hashes[key] != right.hashes[key]:
            messages.append(f"FAIL hash {key}: {left.hashes[key]} != {right.hashes[key]}")

    for role in sorted(set(left.model_roles) | set(right.model_roles)):
        if role not in left.model_roles or role not in right.model_roles:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing model role: {role}")
        elif left.model_roles[role] != right.model_roles[role]:
            messages.append(
                f"FAIL model role {role}: {left.model_roles[role]} != {right.model_roles[role]}"
            )

    for key in sorted(set(left.timings_ms) | set(right.timings_ms)):
        if key not in left.timings_ms or key not in right.timings_ms:
            messages.append(f"{'FAIL' if strict else 'WARN'} missing timing: {key}")
    return messages


def _parse_statistic(value: str) -> float | None:
    return None if value == "null" else float(value)


def print_timing(trace: Trace) -> None:
    total = sum(trace.timings_ms.values())
    print(f"{trace.name}: {total:.1f} ms across {len(trace.timings_ms)} stages")
    for key, value in sorted(trace.timings_ms.items(), key=lambda item: -item[1]):
        print(f"  {key}: {value:.1f} ms")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-log", type=Path, required=True)
    parser.add_argument("--python-log", type=Path)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--android-wav", type=Path)
    parser.add_argument("--python-wav", type=Path)
    parser.add_argument("--max-warm-regression", type=float, default=0.05)
    parser.add_argument("--float-atol", type=float, default=1e-4)
    parser.add_argument("--float-rtol", type=float, default=1e-5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    android = parse_trace(args.android_log)
    print_timing(android)

    failures = []
    if args.python_log:
        python = parse_trace(args.python_log)
        print_timing(python)
        failures.extend(
            compare_trace(
                android,
                python,
                strict=args.strict,
                float_atol=args.float_atol,
                float_rtol=args.float_rtol,
            )
        )

    if args.baseline_log:
        baseline = parse_trace(args.baseline_log)
        print_timing(baseline)
        current_total = android.timings_ms.get("generation_total_ms")
        baseline_total = baseline.timings_ms.get("generation_total_ms")
        if current_total is not None and baseline_total:
            regression = current_total / baseline_total - 1.0
            print(f"warm regression: {regression * 100:.2f}%")
            if regression > args.max_warm_regression:
                failures.append(
                    f"FAIL warm regression {regression * 100:.2f}% exceeds "
                    f"{args.max_warm_regression * 100:.2f}%"
                )

    if args.android_wav and args.python_wav:
        android_wav = read_wav(args.android_wav)
        python_wav = read_wav(args.python_wav)
        print(f"android wav: {android_wav}")
        print(f"python wav: {python_wav}")
        if android_wav != python_wav:
            failures.append(f"FAIL WAV metadata differs: {android_wav} != {python_wav}")

    for message in failures:
        print(message)
    return 1 if any(message.startswith("FAIL") for message in failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
