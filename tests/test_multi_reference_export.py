from pathlib import Path
import importlib.util
import sys

import onnx
import numpy as np
import pytest
from onnx import TensorProto, helper

_MODULE_PATH = Path(__file__).parents[1] / "src/genie_tts/Converter/v2ProPlus/MultiReferenceExporter.py"
_SPEC = importlib.util.spec_from_file_location("multi_reference_exporter", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

INTERFACE_VERSION = _MODULE.INTERFACE_VERSION
REFERENCE_INPUT_NAMES = _MODULE.REFERENCE_INPUT_NAMES
export_prompt_encoder = _MODULE.export_prompt_encoder
read_multi_reference_spec = _MODULE.read_multi_reference_spec
validate_auxiliary_reference_paths = _MODULE.validate_auxiliary_reference_paths
write_fp16_external_data_manifest = _MODULE.write_fp16_external_data_manifest


def _write_prompt_encoder(path: Path) -> None:
    ref_audio = helper.make_tensor_value_info("ref_audio", TensorProto.FLOAT, [1, "audio_length"])
    sv_emb = helper.make_tensor_value_info("sv_emb", TensorProto.FLOAT, [1, 3])
    ge = helper.make_tensor_value_info("ge", TensorProto.FLOAT, [1, 3])
    ge_advanced = helper.make_tensor_value_info("ge_advanced", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["ref_audio"],
                ["ref_audio_mean"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node("Add", ["sv_emb", "ref_audio_mean"], ["ge"]),
            helper.make_node("Mul", ["sv_emb", "ref_audio_mean"], ["ge_advanced"]),
        ],
        "prompt_encoder",
        [ref_audio, sv_emb],
        [ge, ge_advanced],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def _write_prompt_encoder_with_if(path: Path) -> None:
    ref_audio = helper.make_tensor_value_info("ref_audio", TensorProto.FLOAT, [1, "audio_length"])
    sv_emb = helper.make_tensor_value_info("sv_emb", TensorProto.FLOAT, [1, 3])
    branch_ref = helper.make_tensor_value_info("branch_ref", TensorProto.FLOAT, [1, "audio_length"])
    then_branch = helper.make_graph(
        [helper.make_node("Identity", ["ref_audio"], ["branch_ref"])],
        "then_branch",
        [],
        [branch_ref],
    )
    else_branch = helper.make_graph(
        [helper.make_node("Identity", ["ref_audio"], ["branch_ref"])],
        "else_branch",
        [],
        [branch_ref],
    )
    ge = helper.make_tensor_value_info("ge", TensorProto.FLOAT, [1, 3])
    ge_advanced = helper.make_tensor_value_info("ge_advanced", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["condition"],
                value=helper.make_tensor("condition_value", TensorProto.BOOL, [], [True]),
            ),
            helper.make_node("If", ["condition"], ["selected_audio"], then_branch=then_branch, else_branch=else_branch),
            helper.make_node(
                "ReduceMean",
                ["selected_audio"],
                ["audio_mean"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node("Add", ["sv_emb", "audio_mean"], ["ge"]),
            helper.make_node("Mul", ["sv_emb", "audio_mean"], ["ge_advanced"]),
        ],
        "prompt_encoder_with_if",
        [ref_audio, sv_emb],
        [ge, ge_advanced],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def test_export_writes_versioned_dynamic_contract(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    _write_prompt_encoder(source)

    spec = export_prompt_encoder(source, output, max_reference_count=3)
    model = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(model)

    assert spec.interface == INTERFACE_VERSION
    assert spec.max_reference_count == 3
    assert spec.placeholder_audio_samples == 2048
    assert spec.input_names[0] == "reference_count"
    assert spec.input_names[1:] == (
        "ref_audio_0",
        "sv_emb_0",
        "ref_audio_1",
        "sv_emb_1",
        "ref_audio_2",
        "sv_emb_2",
    )
    assert read_multi_reference_spec(output) == spec
    assert model.graph.input[0].name == "reference_count"
    assert model.graph.input[1].name == "ref_audio_0"
    assert model.graph.input[1].type.tensor_type.shape.dim[1].dim_param == "audio_length"
    assert {node.op_type for node in model.graph.node} >= {"Less", "Cast", "Div"}
    assert [output.name for output in model.graph.output] == ["ge", "ge_advanced"]


def test_exported_graph_reduces_one_two_and_three_references(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    _write_prompt_encoder(source)
    export_prompt_encoder(source, output, max_reference_count=3)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for count in (1, 2, 3):
        feed = {"reference_count": [count]}
        expected = np.zeros(3, dtype=np.float32)
        expected_advanced = np.zeros(3, dtype=np.float32)
        for index in range(3):
            audio_value = float(index + 1)
            feed[f"ref_audio_{index}"] = [[audio_value] * (index + 1)]
            values = np.full((1, 3), index + 1, dtype=np.float32)
            feed[f"sv_emb_{index}"] = values
            if index < count:
                expected += values[0] + audio_value
                expected_advanced += values[0] * audio_value
        expected /= count
        expected_advanced /= count
        outputs = session.run(None, feed)
        np.testing.assert_allclose(outputs[0][0], expected)
        np.testing.assert_allclose(outputs[1][0], expected_advanced)


def test_export_rewrites_outer_scope_references_inside_if_subgraphs(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder_if.onnx"
    output = tmp_path / "multi" / "prompt_encoder_if.onnx"
    _write_prompt_encoder_with_if(source)

    export_prompt_encoder(source, output, max_reference_count=2)
    model = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(model)

    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "reference_count": [1],
            "ref_audio_0": [[2.0, 2.0]],
            "sv_emb_0": [[1.0, 1.0, 1.0]],
            "ref_audio_1": [[7.0]],
            "sv_emb_1": [[9.0, 9.0, 9.0]],
        },
    )
    np.testing.assert_allclose(outputs[0], [[3.0, 3.0, 3.0]])
    np.testing.assert_allclose(outputs[1], [[2.0, 2.0, 2.0]])


def test_export_rejects_missing_external_prompt_weights(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    _write_prompt_encoder(source)
    model = onnx.load(source, load_external_data=False)
    initializer = helper.make_tensor("external_weight", TensorProto.FLOAT, [1], [1.0])
    initializer.data_location = TensorProto.EXTERNAL
    initializer.ClearField("raw_data")
    initializer.external_data.add(key="location", value="missing_weights.bin")
    model.graph.initializer.append(initializer)
    onnx.save(model, source)

    with pytest.raises(FileNotFoundError, match="external prompt encoder weights"):
        export_prompt_encoder(source, output, max_reference_count=2)


def test_export_rejects_output_without_external_prompt_weights(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    _write_prompt_encoder(source)
    weights = tmp_path / "prompt_encoder_fp32.bin"
    weights.write_bytes(b"weights")
    model = onnx.load(source, load_external_data=False)
    initializer = helper.make_tensor("external_weight", TensorProto.FLOAT, [1], [1.0])
    initializer.data_location = TensorProto.EXTERNAL
    initializer.ClearField("raw_data")
    initializer.external_data.add(key="location", value=weights.name)
    model.graph.initializer.append(initializer)
    onnx.save(model, source)

    with pytest.raises(FileNotFoundError, match="beside the output ONNX"):
        export_prompt_encoder(source, output, max_reference_count=2)


def test_export_accepts_fp16_weight_layout_without_materializing_fp32_file(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    fp16_weights = tmp_path / "prompt_encoder_fp16.bin"
    _write_prompt_encoder(source)
    model = onnx.load(source, load_external_data=False)
    initializer = helper.make_tensor("external_weight", TensorProto.FLOAT, [2], [1.0, 2.0])
    initializer.data_location = TensorProto.EXTERNAL
    initializer.ClearField("raw_data")
    initializer.ClearField("float_data")
    initializer.external_data.add(key="location", value="prompt_encoder_fp32.bin")
    initializer.external_data.add(key="offset", value="0")
    initializer.external_data.add(key="length", value="8")
    model.graph.initializer.append(initializer)
    onnx.save(model, source)
    fp16_weights.write_bytes(b"\x00" * 4)

    spec = export_prompt_encoder(
        source,
        output,
        max_reference_count=2,
        fp16_weight_path=fp16_weights,
    )

    assert spec.is_supported
    assert output.is_file()
    assert not (output.parent / "prompt_encoder_fp32.bin").exists()


def test_writes_fp16_external_initializer_manifest(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    fp16_weights = tmp_path / "prompt_encoder_fp16.bin"
    manifest_path = tmp_path / "prompt_encoder_fp16_manifest.json"
    _write_prompt_encoder(source)
    model = onnx.load(source, load_external_data=False)
    initializer = helper.make_tensor("external_weight", TensorProto.FLOAT, [2], [1.0, 2.0])
    initializer.data_location = TensorProto.EXTERNAL
    initializer.ClearField("raw_data")
    initializer.ClearField("float_data")
    initializer.external_data.add(key="location", value="prompt_encoder_fp32.bin")
    initializer.external_data.add(key="offset", value="0")
    initializer.external_data.add(key="length", value="8")
    model.graph.initializer.append(initializer)
    onnx.save(model, source)
    fp16_weights.write_bytes(b"\x00" * 4)

    manifest = write_fp16_external_data_manifest(source, fp16_weights, manifest_path)

    assert manifest["version"] == 1
    assert manifest["weight_file"] == fp16_weights.name
    assert manifest["initializers"] == [
        {
            "name": "external_weight",
            "offset": 0,
            "length": 8,
            "shape": [2],
            "data_type": TensorProto.FLOAT,
        }
    ]
    assert manifest_path.is_file()


def test_legacy_prompt_encoder_has_no_multi_reference_capability(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    _write_prompt_encoder(source)

    assert read_multi_reference_spec(source) is None


def test_incomplete_multi_reference_metadata_is_not_capable(tmp_path: Path) -> None:
    source = tmp_path / "prompt_encoder.onnx"
    output = tmp_path / "multi" / "prompt_encoder.onnx"
    _write_prompt_encoder(source)
    export_prompt_encoder(source, output, max_reference_count=2)

    model = onnx.load(output, load_external_data=False)
    for metadata in model.metadata_props:
        if metadata.key == "genie.multi_reference.inputs":
            metadata.value = "reference_count,ref_audio_0,sv_emb_0"
    onnx.save(model, output)

    assert read_multi_reference_spec(output) is None


def test_auxiliary_paths_preserve_order_and_validate_limits(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.touch()
    second.touch()

    paths = validate_auxiliary_reference_paths([second, first], max_reference_count=3)

    assert paths == [str(second.resolve()), str(first.resolve())]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_auxiliary_reference_paths([first, first], max_reference_count=3)
    with pytest.raises(ValueError, match="blank"):
        validate_auxiliary_reference_paths([""], max_reference_count=3)
    with pytest.raises(ValueError, match="At most"):
        validate_auxiliary_reference_paths([first, second], max_reference_count=2)
    with pytest.raises(ValueError, match="primary"):
        validate_auxiliary_reference_paths(
            [first],
            max_reference_count=2,
            primary_reference_path=first,
        )
