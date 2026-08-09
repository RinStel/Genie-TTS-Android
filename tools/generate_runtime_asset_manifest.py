"""Generate a runtime manifest for an FP16 external-data HuBERT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


FLOAT32 = 1
EXTERNAL = 1


def build_manifest(model_path: Path, weights_path: Path) -> dict[str, object]:
    model = onnx.load_model(model_path, load_external_data=False)
    initializers: list[dict[str, object]] = []
    max_end = 0

    for initializer in model.graph.initializer:
        if initializer.data_location != EXTERNAL:
            raise ValueError(f"HuBERT initializer is not external: {initializer.name}")
        if initializer.data_type != FLOAT32:
            raise ValueError(f"HuBERT initializer is not float32: {initializer.name}")

        external = {item.key: item.value for item in initializer.external_data}
        if external.get("location") != "chinese-hubert-base_weights.bin":
            raise ValueError(
                f"Unexpected HuBERT external file for {initializer.name}: "
                f"{external.get('location')}"
            )
        offset = int(external["offset"])
        length = int(external["length"])
        if offset < 0 or length <= 0 or offset % 4 or length % 4:
            raise ValueError(f"Invalid HuBERT external range: {initializer.name}")
        shape = [int(dimension) for dimension in initializer.dims]
        element_count = 1
        for dimension in shape:
            if dimension < 0:
                raise ValueError(f"Invalid HuBERT shape: {initializer.name}")
            element_count *= dimension
        if element_count * 4 != length:
            raise ValueError(f"HuBERT shape/range mismatch: {initializer.name}")

        max_end = max(max_end, offset + length)
        initializers.append(
            {
                "name": initializer.name,
                "data_type": FLOAT32,
                "offset": offset,
                "length": length,
                "shape": shape,
            }
        )

    expected_fp16_size = max_end // 2
    actual_fp16_size = weights_path.stat().st_size
    if actual_fp16_size < expected_fp16_size:
        raise ValueError(
            f"FP16 HuBERT weights are truncated: expected at least "
            f"{expected_fp16_size}, got {actual_fp16_size}"
        )

    return {
        "version": 1,
        "logical_data_type": "float32",
        "weight_file": "chinese-hubert-base_weights_fp16.bin",
        "initializers": initializers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.model, args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
