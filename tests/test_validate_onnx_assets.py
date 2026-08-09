from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tools.validate_onnx_assets import main


def _write_model(path: Path, *, external_location: str | None = None) -> None:
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["value"], ["output"])],
        "fixture",
        [value],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    if external_location is not None:
        initializer = helper.make_tensor("external_weight", TensorProto.FLOAT, [1], [1.0])
        initializer.data_location = TensorProto.EXTERNAL
        initializer.ClearField("raw_data")
        initializer.external_data.add(key="location", value=external_location)
        model.graph.initializer.append(initializer)
    onnx.save(model, path)


def test_validator_accepts_a_complete_model(tmp_path: Path, capsys) -> None:
    model = tmp_path / "model.onnx"
    _write_model(model)

    assert main([str(model)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validator_reports_missing_external_weights(tmp_path: Path, capsys) -> None:
    model = tmp_path / "model.onnx"
    _write_model(model, external_location="weights.bin")

    assert main([str(model)]) == 1
    output = capsys.readouterr().out
    assert "MISSING_EXTERNAL_DATA" in output
    assert "weights.bin" in output
