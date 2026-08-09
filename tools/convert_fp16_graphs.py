"""Convert T2S decoder and VITS shells to optional FP16-compute graphs."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/genie_tts/Converter/v2ProPlus/Fp16GraphConverter.py"
SPEC = importlib.util.spec_from_file_location("fp16_graph_converter", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DEFAULT_FP16_GRAPH_MODELS = MODULE.DEFAULT_FP16_GRAPH_MODELS
convert_directory_to_fp16 = MODULE.convert_directory_to_fp16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Write variants elsewhere; defaults to the model directory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_names",
        choices=DEFAULT_FP16_GRAPH_MODELS,
        help="Convert only this supported shell; may be repeated.",
    )
    args = parser.parse_args()
    model_names = args.model_names or DEFAULT_FP16_GRAPH_MODELS
    for output in convert_directory_to_fp16(
        args.model_directory,
        output_directory=args.output_directory,
        model_names=model_names,
    ):
        print(f"OK {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
