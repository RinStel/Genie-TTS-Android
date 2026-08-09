"""Generate compact soxr-HQ reference-resampler fixtures for Android parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soxr


def _cases() -> dict[str, np.ndarray]:
    impulse = np.zeros(257, dtype=np.float32)
    impulse[0] = 1.0
    sweep = np.sin(np.linspace(0.0, 40.0 * np.pi, 513, dtype=np.float64)).astype(np.float32)
    speech_like = (
        0.55 * np.sin(np.linspace(0.0, 13.0 * np.pi, 401))
        + 0.2 * np.sin(np.linspace(0.0, 71.0 * np.pi, 401))
    ).astype(np.float32)
    stereo = np.column_stack((sweep[:257], -sweep[:257])).astype(np.float32)
    return {
        "impulse": impulse,
        "sweep": sweep,
        "speech_like": speech_like,
        "stereo_mix": stereo.mean(axis=1),
        "edge_length": np.arange(1001, dtype=np.float32) / 1001.0,
    }


def generate(source_rate: int, target_rate: int) -> dict[str, object]:
    cases = {}
    for name, samples in _cases().items():
        output = soxr.resample(samples, source_rate, target_rate, quality="HQ").astype(np.float32)
        cases[name] = {
            "source_samples": samples.tolist(),
            "expected_samples": output.tolist(),
            "source_rate": source_rate,
            "target_rate": target_rate,
        }
    return {"schema_version": 1, "quality": "soxr_hq", "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-rate", type=int, default=32_000)
    parser.add_argument("--target-rate", type=int, default=16_000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generate(args.source_rate, args.target_rate), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
