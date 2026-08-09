"""Export and inspect the V2ProPlus multi-reference prompt contract.

The upstream decoder encodes every reference independently and reduces the
resulting conditioning tensors before the SoVITS text path.  This exporter
puts that reduction in the exported prompt graph, rather than making the
Android runtime guess how to combine reference waveforms.
"""

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import onnx
from onnx import helper, numpy_helper


UPSTREAM_REPOSITORY = "https://github.com/RVC-Boss/GPT-SoVITS"
UPSTREAM_SOURCE_REVISION = "d523079fc05d9a8028d6085bffe4a2757c32abb6"
INTERFACE_NAME = "genie.interface"
INTERFACE_VERSION = "multi_reference_v1"
DEFAULT_MAX_REFERENCE_COUNT = 8
DEFAULT_PLACEHOLDER_AUDIO_SAMPLES = 2048
REFERENCE_INPUT_NAMES = ("ref_audio", "sv_emb")
REFERENCE_OUTPUT_NAMES = ("ge", "ge_advanced")
REFERENCE_COUNT_INPUT = "reference_count"
PLACEHOLDER_AUDIO_SAMPLES_KEY = "genie.multi_reference.placeholder_audio_samples"


@dataclass(frozen=True)
class MultiReferenceSpec:
    interface: str
    max_reference_count: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    reduction: str
    upstream_repository: str
    upstream_revision: str
    placeholder_audio_samples: int = DEFAULT_PLACEHOLDER_AUDIO_SAMPLES

    @property
    def max_auxiliary_reference_count(self) -> int:
        """Return capacity left after the primary reference occupies slot 0."""
        return max(0, self.max_reference_count - 1)

    @property
    def is_supported(self) -> bool:
        expected_inputs = tuple(
            [REFERENCE_COUNT_INPUT]
            + [
                f"{input_name}_{index}"
                for index in range(self.max_reference_count)
                for input_name in REFERENCE_INPUT_NAMES
            ]
        )
        return (
            self.interface == INTERFACE_VERSION
            and self.max_reference_count > 0
            and self.input_names == expected_inputs
            and self.reduction == "independent_prompt_conditioning_mean"
            and set(self.output_names) == set(REFERENCE_OUTPUT_NAMES)
            and self.placeholder_audio_samples > 0
        )


def export_prompt_encoder(
    source_path: str | Path,
    output_path: str | Path,
    *,
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
    fp16_weight_path: str | Path | None = None,
) -> MultiReferenceSpec:
    """Create a multi-reference prompt encoder without embedding weights.

    ``max_reference_count`` counts total graph slots.  Slot zero is reserved
    for the primary reference; remaining slots accept ordered auxiliary audio.
    The source model is loaded with ``load_external_data=False`` so its
    external weight locations and offsets are copied unchanged.  The caller
    must copy the referenced weight file beside the output ONNX file, unless
    ``fp16_weight_path`` is supplied.  The latter is the upstream distribution
    format: the graph keeps FP32 external offsets, while the runtime expands
    the matching FP16 weight file in memory before creating the session.
    """
    if max_reference_count < 1:
        raise ValueError("max_reference_count must be positive")

    source = Path(source_path)
    output = Path(output_path)
    model = onnx.load_model(source, load_external_data=False)
    _validate_prompt_encoder(model)
    fp16_weight = Path(fp16_weight_path) if fp16_weight_path is not None else None
    if fp16_weight is not None:
        _validate_fp16_weight_layout(model, fp16_weight)

    missing_external_data = sorted(
        location
        for location in _external_data_locations(model)
        if not (source.parent / location).is_file()
    )
    if missing_external_data and fp16_weight is None:
        raise FileNotFoundError(
            "Missing external prompt encoder weights: "
            + ", ".join(missing_external_data)
        )

    _add_sequence_safe_reference_graph(model, max_reference_count)
    placeholder_audio_samples = _minimum_placeholder_audio_samples(model)
    _set_metadata(
        model,
        {
            INTERFACE_NAME: INTERFACE_VERSION,
            "genie.multi_reference.max_count": str(max_reference_count),
            "genie.multi_reference.inputs": ",".join(
                [REFERENCE_COUNT_INPUT]
                + [
                    f"{input_name}_{index}"
                    for index in range(max_reference_count)
                    for input_name in REFERENCE_INPUT_NAMES
                ]
            ),
            "genie.multi_reference.outputs": ",".join(REFERENCE_OUTPUT_NAMES),
            "genie.multi_reference.reduction": "independent_prompt_conditioning_mean",
            PLACEHOLDER_AUDIO_SAMPLES_KEY: str(placeholder_audio_samples),
            "genie.upstream.repository": UPSTREAM_REPOSITORY,
            "genie.upstream.revision": UPSTREAM_SOURCE_REVISION,
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    missing_output_data = sorted(
        location
        for location in _external_data_locations(model)
        if not (output.parent / location).is_file()
    )
    if missing_output_data and fp16_weight is None:
        raise FileNotFoundError(
            "Missing external prompt encoder weights beside the output ONNX: "
            + ", ".join(missing_output_data)
        )
    # Path-based validation tries to open the logical FP32 external file.  In
    # the FP16 distribution that file is intentionally not materialized.
    if not missing_output_data:
        onnx.checker.check_model(output, full_check=False)
    return read_multi_reference_spec(output)


def read_multi_reference_spec(model_path: str | Path) -> Optional[MultiReferenceSpec]:
    """Read the versioned capability contract from an ONNX model."""
    model = onnx.load_model(model_path, load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get(INTERFACE_NAME) != INTERFACE_VERSION:
        return None

    max_count = int(metadata.get("genie.multi_reference.max_count", "0"))
    placeholder_audio_samples = _positive_int_or_default(
        metadata.get(PLACEHOLDER_AUDIO_SAMPLES_KEY),
        DEFAULT_PLACEHOLDER_AUDIO_SAMPLES,
    )
    inputs = tuple(
        value for value in metadata.get("genie.multi_reference.inputs", "").split(",") if value
    )
    spec = MultiReferenceSpec(
        interface=metadata[INTERFACE_NAME],
        max_reference_count=max_count,
        input_names=inputs,
        output_names=tuple(
            value for value in metadata.get("genie.multi_reference.outputs", "").split(",") if value
        ),
        reduction=metadata.get("genie.multi_reference.reduction", ""),
        upstream_repository=metadata.get("genie.upstream.repository", ""),
        upstream_revision=metadata.get("genie.upstream.revision", ""),
        placeholder_audio_samples=placeholder_audio_samples,
    )
    return spec if spec.is_supported else None


def write_fp16_external_data_manifest(
    prompt_encoder_path: str | Path,
    fp16_weight_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Write the small runtime map needed to load FP16 external weights.

    The ONNX shell stores FP32 external offsets because its operators require
    FP32 tensors.  The manifest lets mobile runtimes reconstruct those tensors
    directly from the distributed FP16 file without materializing a second
    weight file on disk.
    """
    model = onnx.load_model(prompt_encoder_path, load_external_data=False)
    fp16_weight = Path(fp16_weight_path)
    _validate_fp16_weight_layout(model, fp16_weight)
    initializers = []
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {entry.key: entry.value for entry in tensor.external_data}
        initializers.append(
            {
                "name": tensor.name,
                "offset": int(fields.get("offset", "0")),
                "length": int(fields["length"]),
                "shape": list(tensor.dims),
                "data_type": int(tensor.data_type),
            }
        )
    if not initializers:
        raise ValueError("Prompt encoder does not contain external initializers")

    manifest = {
        "version": 1,
        "weight_file": fp16_weight.name,
        "logical_data_type": "float32",
        "initializers": initializers,
    }
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_auxiliary_reference_paths(
    paths: Iterable[str | Path],
    *,
    max_reference_count: int,
    primary_reference_path: str | Path | None = None,
) -> list[str]:
    """Validate ordered auxiliary paths against total graph reference slots.

    The primary reference occupies slot zero, so the usable auxiliary capacity
    is ``max_reference_count - 1``.
    """
    if max_reference_count < 1:
        raise ValueError("max_reference_count must be positive")

    normalized: list[str] = []
    seen: set[str] = set()
    primary_canonical = (
        str(Path(primary_reference_path).expanduser().resolve(strict=False))
        if primary_reference_path is not None
        else None
    )
    for raw_path in paths:
        raw_value = str(raw_path)
        if not raw_value.strip():
            raise ValueError("Auxiliary reference paths cannot be blank")
        path = Path(raw_value).expanduser()
        canonical = str(path.resolve(strict=False))
        if primary_canonical is not None and canonical == primary_canonical:
            raise ValueError("Auxiliary reference path duplicates the primary reference")
        if canonical in seen:
            raise ValueError(f"Duplicate auxiliary reference path: {raw_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Auxiliary reference audio not found: {path}")
        seen.add(canonical)
        normalized.append(canonical)

    max_auxiliary_count = max(0, max_reference_count - 1)
    if len(normalized) > max_auxiliary_count:
        raise ValueError(
            f"At most {max_auxiliary_count} auxiliary references are supported; got {len(normalized)}"
        )
    return normalized


def _validate_prompt_encoder(model: onnx.ModelProto) -> None:
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    missing_inputs = set(REFERENCE_INPUT_NAMES) - input_names
    missing_outputs = set(REFERENCE_OUTPUT_NAMES) - output_names
    if missing_inputs or missing_outputs:
        raise ValueError(
            "Prompt encoder contract mismatch: "
            f"missing inputs={sorted(missing_inputs)}, outputs={sorted(missing_outputs)}"
        )

    for value in model.graph.input:
        if value.name not in REFERENCE_INPUT_NAMES:
            continue
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT or len(tensor_type.shape.dim) != 2:
            raise ValueError(f"Unsupported prompt encoder input shape: {value.name}")


def _add_sequence_safe_reference_graph(
    model: onnx.ModelProto,
    max_reference_count: int,
) -> None:
    """Unroll independent prompt encoders without padding variable-length audio."""
    source_nodes = [copy.deepcopy(node) for node in model.graph.node]
    source_inputs = {
        value.name: copy.deepcopy(value)
        for value in model.graph.input
        if value.name in REFERENCE_INPUT_NAMES
    }
    source_outputs = {
        value.name: copy.deepcopy(value)
        for value in model.graph.output
        if value.name in REFERENCE_OUTPUT_NAMES
    }
    if set(source_inputs) != set(REFERENCE_INPUT_NAMES):
        raise ValueError("Prompt encoder graph must expose both reference inputs")
    if set(source_outputs) != set(REFERENCE_OUTPUT_NAMES):
        raise ValueError("Prompt encoder graph must expose both reference outputs")

    del model.graph.input[:]
    del model.graph.output[:]
    del model.graph.node[:]
    # The old value_info entries refer to the single-reference names. They are
    # optional, and retaining them would leave stale names after unrolling.
    del model.graph.value_info[:]
    model.graph.input.append(
        helper.make_tensor_value_info(REFERENCE_COUNT_INPUT, onnx.TensorProto.INT64, [1])
    )

    masked_outputs: dict[str, list[str]] = {name: [] for name in REFERENCE_OUTPUT_NAMES}
    for index in range(max_reference_count):
        value_map = {
            input_name: f"{input_name}_{index}"
            for input_name in REFERENCE_INPUT_NAMES
        }
        for input_name, source_input in source_inputs.items():
            input_value = copy.deepcopy(source_input)
            input_value.name = value_map[input_name]
            model.graph.input.append(input_value)

        for node_index, source_node in enumerate(source_nodes):
            node = copy.deepcopy(source_node)
            node.name = f"{source_node.name or source_node.op_type}_{node_index}_reference_{index}"
            node.input[:] = [value_map.get(name, name) for name in source_node.input]
            output_map = {
                name: f"{name}_reference_{index}"
                for name in source_node.output
            }
            # ONNX control-flow subgraphs use lexical scope: a branch can read
            # values produced by the enclosing graph without listing them as
            # branch inputs. Rewrite those references along with top-level
            # inputs/outputs, otherwise an unrolled If points at stale names.
            scoped_value_map = dict(value_map)
            scoped_value_map.update(output_map)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    _rewrite_graph_scope(
                        attribute.g,
                        scoped_value_map,
                        f"reference_{index}_node_{node_index}_{attribute.name}",
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, graph in enumerate(attribute.graphs):
                        _rewrite_graph_scope(
                            graph,
                            scoped_value_map,
                            f"reference_{index}_node_{node_index}_{attribute.name}_{graph_index}",
                        )
            mapped_outputs = []
            for name in source_node.output:
                mapped_outputs.append(output_map[name])
            node.output[:] = mapped_outputs
            model.graph.node.append(node)
            value_map.update(output_map)

        slot_constant = f"reference_slot_{index}"
        valid_flag = f"reference_valid_{index}"
        valid_float = f"reference_valid_float_{index}"
        model.graph.node.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[slot_constant],
                name=f"reference_slot_constant_{index}",
                value=helper.make_tensor(
                    name=f"reference_slot_value_{index}",
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[index],
                ),
            )
        )
        model.graph.node.append(
            helper.make_node(
                "Less",
                inputs=[slot_constant, REFERENCE_COUNT_INPUT],
                outputs=[valid_flag],
                name=f"reference_valid_{index}",
            )
        )
        model.graph.node.append(
            helper.make_node(
                "Cast",
                inputs=[valid_flag],
                outputs=[valid_float],
                name=f"reference_valid_cast_{index}",
                to=onnx.TensorProto.FLOAT,
            )
        )
        for output_name in REFERENCE_OUTPUT_NAMES:
            masked_name = f"{output_name}_masked_{index}"
            model.graph.node.append(
                helper.make_node(
                    "Mul",
                    inputs=[value_map[output_name], valid_float],
                    outputs=[masked_name],
                    name=f"reference_mask_{output_name}_{index}",
                )
            )
            masked_outputs[output_name].append(masked_name)

    count_float = "reference_count_float"
    model.graph.node.append(
        helper.make_node(
            "Cast",
            inputs=[REFERENCE_COUNT_INPUT],
            outputs=[count_float],
            name="reference_count_cast",
            to=onnx.TensorProto.FLOAT,
        )
    )
    for output_name, values in masked_outputs.items():
        total = values[0]
        for index, value in enumerate(values[1:], start=1):
            summed = f"{output_name}_sum_{index}"
            model.graph.node.append(
                helper.make_node(
                    "Add",
                    inputs=[total, value],
                    outputs=[summed],
                    name=f"reference_sum_{output_name}_{index}",
                )
            )
            total = summed
        model.graph.node.append(
            helper.make_node(
                "Div",
                inputs=[total, count_float],
                outputs=[output_name],
                name=f"reference_mean_{output_name}",
            )
        )
        output_value = source_outputs[output_name]
        output_value.name = output_name
        output_value.type.tensor_type.shape.dim[0].ClearField("dim_param")
        output_value.type.tensor_type.shape.dim[0].dim_value = 1
        model.graph.output.append(output_value)


def _rewrite_graph_scope(
    graph: onnx.GraphProto,
    outer_value_map: Mapping[str, str],
    scope_name: str,
) -> None:
    """Rewrite a control-flow graph while preserving lexical outer references."""
    value_map = dict(outer_value_map)

    # Graph inputs are local and shadow any outer value with the same name.
    for input_value in graph.input:
        original_name = input_value.name
        mapped_name = f"{original_name}_{scope_name}"
        value_map[original_name] = mapped_name
        input_value.name = mapped_name

    # Initializers are also local to the graph. Keep their names stable because
    # they are not duplicated across reference slots by this exporter.
    for initializer in graph.initializer:
        value_map[initializer.name] = initializer.name

    for node_index, node in enumerate(graph.node):
        node.input[:] = [value_map.get(name, name) for name in node.input]
        output_map = {
            name: f"{name}_{scope_name}_node_{node_index}"
            for name in node.output
        }
        scoped_value_map = dict(value_map)
        scoped_value_map.update(output_map)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _rewrite_graph_scope(
                    attribute.g,
                    scoped_value_map,
                    f"{scope_name}_node_{node_index}_{attribute.name}",
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph_index, child_graph in enumerate(attribute.graphs):
                    _rewrite_graph_scope(
                        child_graph,
                        scoped_value_map,
                        f"{scope_name}_node_{node_index}_{attribute.name}_{graph_index}",
                    )
        node.name = f"{node.name or node.op_type}_{scope_name}_{node_index}"
        node.output[:] = [output_map[name] for name in node.output]
        value_map.update(output_map)

    for value in graph.output:
        value.name = value_map.get(value.name, value.name)
    for value in graph.value_info:
        value.name = value_map.get(value.name, value.name)


def _set_metadata(model: onnx.ModelProto, updates: Mapping[str, str]) -> None:
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(updates)
    del model.metadata_props[:]
    for key, value in sorted(existing.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def _minimum_placeholder_audio_samples(model: onnx.ModelProto) -> int:
    """Find a safe zero-audio length for all STFT nodes in the source graph."""
    frame_lengths: list[int] = []
    for graph in _iter_graphs(model.graph):
        producers = {
            output_name: node
            for node in graph.node
            for output_name in node.output
        }
        initializers = {initializer.name: initializer for initializer in graph.initializer}
        for node in graph.node:
            if node.op_type != "STFT" or len(node.input) <= 2:
                continue
            frame_length = _resolve_scalar_value(
                node.input[2],
                producers=producers,
                initializers=initializers,
            )
            if frame_length is not None and frame_length > 0:
                frame_lengths.append(frame_length)
    return max(frame_lengths, default=DEFAULT_PLACEHOLDER_AUDIO_SAMPLES)


def _iter_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    yield from _iter_graphs(child_graph)


def _resolve_scalar_value(
    name: str,
    *,
    producers: Mapping[str, onnx.NodeProto],
    initializers: Mapping[str, onnx.TensorProto],
    visited: set[str] | None = None,
) -> int | None:
    visited = visited or set()
    if name in visited:
        return None
    visited.add(name)

    initializer = initializers.get(name)
    if initializer is not None:
        values = numpy_helper.to_array(initializer).reshape(-1)
        return int(values[0]) if values.size == 1 else None

    producer = producers.get(name)
    if producer is None:
        return None
    if producer.op_type == "Constant":
        value_attribute = next(
            (attribute for attribute in producer.attribute if attribute.name == "value"),
            None,
        )
        if value_attribute is not None and value_attribute.type == onnx.AttributeProto.TENSOR:
            values = numpy_helper.to_array(value_attribute.t).reshape(-1)
            return int(values[0]) if values.size == 1 else None
    if producer.op_type in {"Cast", "Identity"} and producer.input:
        return _resolve_scalar_value(
            producer.input[0],
            producers=producers,
            initializers=initializers,
            visited=visited,
        )
    return None


def _positive_int_or_default(value: str | None, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _external_data_locations(model: onnx.ModelProto) -> set[str]:
    return {
        entry.value
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }


def _validate_fp16_weight_layout(model: onnx.ModelProto, fp16_path: Path) -> None:
    """Ensure an FP16 file can satisfy the graph's logical FP32 offsets."""
    if not fp16_path.is_file():
        raise FileNotFoundError(f"Missing FP16 prompt encoder weights: {fp16_path}")
    fp16_size = fp16_path.stat().st_size
    if fp16_size % 2:
        raise ValueError(f"Invalid FP16 prompt encoder weight length: {fp16_path}")

    required_fp32_size = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {entry.key: entry.value for entry in tensor.external_data}
        offset = int(fields.get("offset", "0"))
        length = int(fields.get("length", "0"))
        required_fp32_size = max(required_fp32_size, offset + length)

    if required_fp32_size > fp16_size * 2:
        raise ValueError(
            "FP16 prompt encoder weights are shorter than the graph's external layout: "
            f"required {required_fp32_size // 2} FP16 bytes, found {fp16_size}"
        )
