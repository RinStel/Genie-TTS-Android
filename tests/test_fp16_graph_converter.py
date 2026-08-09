from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


_MODULE_PATH = Path(__file__).parents[1] / "src/genie_tts/Converter/v2ProPlus/Fp16GraphConverter.py"
_SPEC = importlib.util.spec_from_file_location("fp16_graph_converter", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

convert_graph_to_fp16 = _MODULE.convert_graph_to_fp16


def _write_external_matmul_model(model_path: Path, fp16_only: bool) -> None:
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
    weight.ClearField("float_data")
    weight.ClearField("int32_data")
    weight.ClearField("raw_data")
    weight.data_location = TensorProto.EXTERNAL
    weight.external_data.add(key="location", value="weight_fp32.bin")
    weight.external_data.add(key="offset", value="0")
    weight.external_data.add(key="length", value="16")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "matmul",
        [input_info],
        [output_info],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)
    if fp16_only:
        model_path.with_name("weight_fp16.bin").write_bytes(
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes(),
        )
    else:
        model_path.with_name("weight_fp32.bin").write_bytes(
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes(),
        )


def test_converts_external_graph_to_fp16_with_fp32_io(tmp_path: Path) -> None:
    source = tmp_path / "matmul_fp32.onnx"
    output = tmp_path / "out" / "matmul_fp16.onnx"
    source.parent.mkdir(exist_ok=True)
    _write_external_matmul_model(source, fp16_only=False)

    convert_graph_to_fp16(source, output)

    model = onnx.load(output, load_external_data=False)
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(t.data_type == TensorProto.FLOAT16 for t in model.graph.initializer)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    result = session.run(None, {"x": np.asarray([[1.0, 1.0]], dtype=np.float32)})[0]
    np.testing.assert_allclose(result, np.asarray([[4.0, 6.0]], dtype=np.float32), rtol=1e-2, atol=1e-2)


def test_uses_distributed_fp16_external_data_without_materializing_fp32(tmp_path: Path) -> None:
    source = tmp_path / "matmul_fp32.onnx"
    output = tmp_path / "out" / "matmul_fp16.onnx"
    source.parent.mkdir(exist_ok=True)
    _write_external_matmul_model(source, fp16_only=True)

    convert_graph_to_fp16(source, output)

    assert (output.parent / "weight_fp16.bin").is_file()
    assert not (output.parent / "weight_fp32.bin").exists()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    result = session.run(None, {"x": np.asarray([[1.0, 1.0]], dtype=np.float32)})[0]
    np.testing.assert_allclose(result, np.asarray([[4.0, 6.0]], dtype=np.float32), rtol=1e-2, atol=1e-2)
