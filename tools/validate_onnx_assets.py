"""Validate ONNX graphs and the external weight files beside them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import onnx


def validate_model(path: Path) -> list[str]:
    """Return diagnostics for one ONNX file; an empty list means valid."""
    model = onnx.load_model(path, load_external_data=False)
    locations = external_data_locations(model)
    missing = sorted(location for location in locations if not (path.parent / location).is_file())
    if missing:
        return [
            f"MISSING_EXTERNAL_DATA {path}: {', '.join(missing)}",
        ]

    onnx.checker.check_model(path, full_check=False)
    return []


def external_data_locations(model: onnx.ModelProto) -> set[str]:
    return {
        entry.value
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }


def model_paths(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.onnx")))
        else:
            paths.append(path)
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="ONNX files or directories to validate")
    args = parser.parse_args(list(argv) if argv is not None else None)

    failures = 0
    paths = model_paths(args.paths)
    if not paths:
        print("No ONNX files found.")
        return 1

    for path in paths:
        if not path.is_file():
            print(f"MISSING_FILE {path}")
            failures += 1
            continue
        try:
            diagnostics = validate_model(path)
        except Exception as error:
            diagnostics = [f"INVALID {path}: {error}"]
        if diagnostics:
            for diagnostic in diagnostics:
                print(diagnostic)
            failures += 1
        else:
            print(f"OK {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
