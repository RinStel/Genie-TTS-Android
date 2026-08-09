"""Create optional FP16-compute graphs from the distributed FP32 shells.

The character models distribute FP16 weights, but their ONNX shells expose
FLOAT tensors.  This module keeps the public graph IO as FLOAT and converts
the graph body and external initializers to FLOAT16.  Keeping the boundary
stable is intentional: Android's native decoder loop can then reuse the
existing FP32 KV-cache bridge while QNN/CPU execute the large graph body in
half precision.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxconverter_common import float16


DEFAULT_FP16_GRAPH_MODELS = (
    "t2s_first_stage_decoder_fp32.onnx",
    "t2s_stage_decoder_fp32.onnx",
    "vits_fp32.onnx",
)


def convert_graph_to_fp16(
    source_model: str | Path,
    output_model: str | Path,
) -> Path:
    """Convert one FP32 shell to an FP16-body, FP32-IO graph.

    The source directory may contain either the original FP32 external data or
    the distributed FP16 file.  The latter is preferred so conversion does
    not expand a large model merely to shrink it again.
    """

    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = onnx.load_model(source_path, load_external_data=False)
    converted = float16.convert_float_to_float16(
        onnx.load_model(source_path, load_external_data=False),
        keep_io_types=False,
    )
    _fix_float_attributes(converted, original)
    _wrap_float_io(converted)
    _relink_external_initializers(converted, original, source_path.parent, output_path.parent)

    onnx.save_model(converted, output_path)
    onnx.checker.check_model(output_path, full_check=False)
    return output_path


def convert_directory_to_fp16(
    model_directory: str | Path,
    output_directory: str | Path | None = None,
    model_names: Iterable[str] = DEFAULT_FP16_GRAPH_MODELS,
) -> list[Path]:
    """Convert the supported synthesis graphs in one character directory."""

    source_directory = Path(model_directory)
    target_directory = Path(output_directory) if output_directory is not None else source_directory
    target_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for model_name in model_names:
        source_path = source_directory / model_name
        output_name = model_name.replace("_fp32.onnx", "_fp16.onnx")
        outputs.append(convert_graph_to_fp16(source_path, target_directory / output_name))
    return outputs


def _fix_float_attributes(model: onnx.ModelProto, original: onnx.ModelProto) -> None:
    """Repair dtype attributes that the generic converter cannot infer.

    Cast and RandomNormalLike store their target dtype in attributes rather
    than tensor metadata.  The converter changes the surrounding graph but
    does not rewrite these attributes for original nodes.  Casts inserted for
    blocked operators are deliberately left untouched because they are the
    FP32 islands used by ONNX Runtime for those operators.
    """

    original_node_names = {node.name for node in original.graph.node}
    for node in model.graph.node:
        if node.name not in original_node_names:
            continue
        if node.op_type == "Cast":
            _set_attribute_dtype(node, "to", TensorProto.FLOAT, TensorProto.FLOAT16)
        elif node.op_type in {"RandomNormal", "RandomNormalLike"}:
            _set_attribute_dtype(node, "dtype", TensorProto.FLOAT, TensorProto.FLOAT16)


def _set_attribute_dtype(
    node: onnx.NodeProto,
    attribute_name: str,
    source_dtype: int,
    target_dtype: int,
) -> None:
    for attribute in node.attribute:
        if attribute.name == attribute_name and attribute.i == source_dtype:
            attribute.i = target_dtype


def _wrap_float_io(model: onnx.ModelProto) -> None:
    """Expose FP32 input/output tensors around the converted graph body."""

    graph = model.graph
    float16_inputs = {
        value_info.name
        for value_info in graph.input
        if value_info.type.HasField("tensor_type")
        and value_info.type.tensor_type.elem_type == TensorProto.FLOAT16
    }
    float16_outputs = {
        value_info.name
        for value_info in graph.output
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16
    }
    input_aliases = {name: f"{name}__fp16" for name in float16_inputs}
    output_aliases = {name: f"{name}__fp16" for name in float16_outputs}

    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = input_aliases.get(name, output_aliases.get(name, name))
        for index, name in enumerate(node.output):
            if name in output_aliases:
                node.output[index] = output_aliases[name]

    for value_info in graph.value_info:
        if value_info.name in input_aliases:
            value_info.name = input_aliases[value_info.name]
        if value_info.name in output_aliases:
            value_info.name = output_aliases[value_info.name]

    input_casts: list[onnx.NodeProto] = []
    for value_info in graph.input:
        if value_info.name not in input_aliases:
            continue
        alias = input_aliases[value_info.name]
        value_info.type.tensor_type.elem_type = TensorProto.FLOAT
        input_casts.append(
            helper.make_node(
                "Cast",
                [value_info.name],
                [alias],
                name=f"fp16_input_cast_{value_info.name}",
                to=TensorProto.FLOAT16,
            ),
        )

    output_casts: list[onnx.NodeProto] = []
    for value_info in graph.output:
        if value_info.name not in output_aliases:
            continue
        alias = output_aliases[value_info.name]
        value_info.type.tensor_type.elem_type = TensorProto.FLOAT
        output_casts.append(
            helper.make_node(
                "Cast",
                [alias],
                [value_info.name],
                name=f"fp16_output_cast_{value_info.name}",
                to=TensorProto.FLOAT,
            ),
        )

    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(input_casts + nodes + output_casts)


def _relink_external_initializers(
    model: onnx.ModelProto,
    original: onnx.ModelProto,
    source_directory: Path,
    output_directory: Path,
) -> None:
    original_external = {
        tensor.name: {entry.key: entry.value for entry in tensor.external_data}
        for tensor in original.graph.initializer
        if tensor.data_location == TensorProto.EXTERNAL
    }
    tensors_by_location: dict[str, list[onnx.TensorProto]] = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        if tensor.name not in original_external:
            raise ValueError(f"Converted initializer has no source metadata: {tensor.name}")
        source_location = original_external[tensor.name]["location"]
        tensors_by_location.setdefault(source_location, []).append(tensor)

    for source_location, tensors in tensors_by_location.items():
        output_location = _fp16_location(source_location)
        distributed_path = source_directory / output_location
        output_path = output_directory / output_location
        if distributed_path.is_file():
            _relink_from_existing_fp16(
                tensors,
                original_external,
                distributed_path,
                output_path,
            )
            continue
        _relink_from_fp32(
            tensors,
            original_external,
            source_directory / source_location,
            output_path,
        )


def _relink_from_existing_fp16(
    tensors: list[onnx.TensorProto],
    original_external: dict[str, dict[str, str]],
    source_path: Path,
    output_path: Path,
) -> None:
    source_size = source_path.stat().st_size
    max_end = 0
    for tensor in tensors:
        metadata = original_external[tensor.name]
        offset = int(metadata.get("offset", "0"))
        length = int(metadata["length"])
        _validate_fp32_range(tensor, offset, length)
        fp16_offset = offset // 2
        fp16_length = length // 2
        max_end = max(max_end, fp16_offset + fp16_length)
        _set_external_data(tensor, output_path.name, fp16_offset, fp16_length)
    if source_size < max_end:
        raise ValueError(f"FP16 external data is truncated: {source_path}")
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)


def _relink_from_fp32(
    tensors: list[onnx.TensorProto],
    original_external: dict[str, dict[str, str]],
    source_path: Path,
    output_path: Path,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Neither FP16 nor FP32 external data is available for {source_path.name}"
        )
    source_bytes = source_path.read_bytes()
    output_bytes = bytearray()
    for tensor in tensors:
        metadata = original_external[tensor.name]
        offset = int(metadata.get("offset", "0"))
        length = int(metadata["length"])
        _validate_fp32_range(tensor, offset, length)
        raw = source_bytes[offset:offset + length]
        values = np.frombuffer(raw, dtype=np.float32)
        expected_count = _tensor_element_count(tensor)
        if values.size != expected_count:
            raise ValueError(f"FP32 external data shape mismatch: {tensor.name}")
        payload = values.astype(np.float16).tobytes()
        new_offset = len(output_bytes)
        output_bytes.extend(payload)
        _set_external_data(tensor, output_path.name, new_offset, len(payload))
    output_path.write_bytes(output_bytes)


def _set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: int,
    length: int,
) -> None:
    tensor.data_type = TensorProto.FLOAT16
    del tensor.external_data[:]
    tensor.external_data.add(key="location", value=location)
    tensor.external_data.add(key="offset", value=str(offset))
    tensor.external_data.add(key="length", value=str(length))


def _validate_fp32_range(tensor: onnx.TensorProto, offset: int, length: int) -> None:
    if offset < 0 or length <= 0 or offset % 4 != 0 or length % 4 != 0:
        raise ValueError(f"Invalid FP32 external range: {tensor.name}")
    if length != _tensor_element_count(tensor) * 4:
        raise ValueError(f"FP32 external length does not match shape: {tensor.name}")


def _tensor_element_count(tensor: onnx.TensorProto) -> int:
    count = 1
    for dimension in tensor.dims:
        count *= int(dimension)
    return count


def _fp16_location(location: str) -> str:
    path = Path(location)
    if path.name.endswith("_fp32.bin"):
        return str(path.with_name(path.name.replace("_fp32.bin", "_fp16.bin")))
    return str(path.with_name(f"{path.stem}_fp16{path.suffix}"))
