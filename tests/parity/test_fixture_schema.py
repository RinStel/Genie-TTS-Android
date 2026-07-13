import hashlib
import json
import subprocess
import struct
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import generate_python_fixture
from fixture_schema import (
    FIXTURE_SCHEMA_VERSION,
    FixtureSchemaError,
    decode_fixture,
    serialize_float_tensor,
    serialize_integer_tensor,
)


def test_generator_loads_checkout_frontend_without_package_initializers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository_src = Path(__file__).resolve().parents[2] / "src"
    installed_package = tmp_path / "site-packages" / "genie_tts"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text(
        "raise AssertionError('installed genie_tts package executed')\n",
        encoding="utf-8",
    )

    for module_name in tuple(sys.modules):
        if (
            module_name == "genie_tts"
            or module_name.startswith("genie_tts.")
            or module_name == generate_python_fixture.ISOLATED_PACKAGE_NAME
            or module_name.startswith(f"{generate_python_fixture.ISOLATED_PACKAGE_NAME}.")
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [str(installed_package.parent)] + list(sys.path),
    )

    chinese_to_phones = generate_python_fixture.load_chinese_to_phones()

    assert callable(chinese_to_phones)
    package_name = generate_python_fixture.ISOLATED_PACKAGE_NAME
    assert "genie_tts" not in sys.modules
    assert Path(sys.modules[package_name].__path__[0]).resolve() == repository_src / "genie_tts"
    assert "genie_tts.Internal" not in sys.modules
    resources = sys.modules[f"{package_name}.Core.Resources"]
    assert not hasattr(resources, "download_genie_data")
    assert getattr(resources, "__file__", None) is None


def test_generator_refuses_to_overwrite_source_fixture() -> None:
    source = Path(__file__).parent / "fixtures" / "chinese_frontend_cases.json"

    with pytest.raises(ValueError, match="separate output"):
        generate_python_fixture.require_separate_output_path(source, source)


def test_tensor_serialization_is_deterministic_and_preserves_values() -> None:
    integer = serialize_integer_tensor(shape=(1, 3), values=(7, -2, 99))
    floating = serialize_float_tensor(shape=(1, 3), values=(1.5, -0.0, float("inf")))

    assert integer == {
        "dtype": "int64",
        "shape": [1, 3],
        "values": [7, -2, 99],
    }
    assert floating["shape"] == [1, 3]
    assert floating["dtype"] == "float32"
    assert floating["sha256"] == hashlib.sha256(
        struct.pack("<fff", 1.5, -0.0, float("inf"))
    ).hexdigest()
    assert floating["statistics"] == {
        "finite_count": 2,
        "non_finite_count": 1,
        "min": -0.0,
        "max": 1.5,
        "mean": 0.75,
    }
    assert "values" not in floating
    assert json.dumps(floating, ensure_ascii=False, sort_keys=True) == json.dumps(
        serialize_float_tensor(shape=(1, 3), values=(1.5, -0.0, float("inf"))),
        ensure_ascii=False,
        sort_keys=True,
    )


@pytest.mark.parametrize("version", (None, 0, FIXTURE_SCHEMA_VERSION + 1))
def test_fixture_schema_rejects_missing_or_unsupported_versions(version: int | None) -> None:
    fixture = {"cases": []}
    if version is not None:
        fixture["schema_version"] = version

    with pytest.raises(FixtureSchemaError, match="schema_version"):
        decode_fixture(fixture)


@pytest.mark.parametrize("version", (True, 1.0, "1"))
def test_fixture_schema_rejects_non_integer_version_types(version) -> None:
    fixture = {
        "schema_version": version,
        "fixture_type": "chinese_frontend",
        "cases": [{"id": "case", "category": "smoke", "input_text": "你好。"}],
    }

    with pytest.raises(FixtureSchemaError, match="schema_version"):
        decode_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fixture_type", "unknown"),
        ("fixture_type", 1),
        ("cases", {}),
        ("cases", []),
    ),
)
def test_fixture_schema_rejects_invalid_type_and_cases(field: str, value) -> None:
    fixture = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_type": "chinese_frontend",
        "cases": [{"id": "case", "category": "smoke", "input_text": "你好。"}],
    }
    fixture[field] = value

    with pytest.raises(FixtureSchemaError):
        decode_fixture(fixture)


def test_chinese_frontend_fixture_has_required_coverage() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "chinese_frontend_cases.json"

    fixture = decode_fixture(json.loads(fixture_path.read_text(encoding="utf-8")))

    assert fixture["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert {case["category"] for case in fixture["cases"]} == {
        "contextual_polyphones",
        "neutral_tone",
        "third_tone_chain",
        "numbers",
        "punctuation",
        "erhua",
    }


def test_shape_rejects_boolean_and_float_dimensions() -> None:
    with pytest.raises(FixtureSchemaError, match="dimensions"):
        serialize_integer_tensor(shape=(True, 2), values=(1, 2))
    with pytest.raises(FixtureSchemaError, match="dimensions"):
        serialize_float_tensor(shape=(1.0, 2), values=(1.0, 2.0))


def test_full_contract_fixture_validates_all_parity_boundaries() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "full_contract_case.json"

    fixture = decode_fixture(json.loads(fixture_path.read_text(encoding="utf-8")))
    expected = fixture["cases"][0]["expected"]

    assert expected["phones"] == ["n", "i3", "h", "ao3", "."]
    assert expected["phone_ids"]["values"][1:4] == [-2147483648, 42, 2147483647]
    assert expected["semantic_tokens"]["values"] == [-2147483648, 2147483647]
    assert expected["tensors"]["roberta_features"]["sha256"] == (
        "937d530fa41fcbc6164988a1952b69213dc6b6f9cdb8a939027135cf9979a710"
    )
    assert expected["tensors"]["bert_phone_ids"]["values"] == [7, 11]
    assert expected["timing"]["stages"][0]["elapsed_ms"] == 12
    assert expected["trace"][0]["provider"] == "cpu"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda document: document["cases"][0]["expected"].pop("phones"),
        lambda document: document["cases"][0]["expected"].update(
            {"trace": [{"model_role": "frontend", "provider": "unsafe"}]}
        ),
        lambda document: document["cases"][0]["expected"].update(
            {"timing": {"stages": [{"name": "frontend", "elapsed_ms": "12"}]}}
        ),
        lambda document: document["cases"][0]["expected"]["tensors"][
            "roberta_features"
        ].pop("sha256"),
        lambda document: document["cases"][0]["expected"]["semantic_tokens"].update(
            {"values": [True, 2]}
        ),
        lambda document: document["cases"][0]["expected"]["trace"][0].update(
            {"path": "C:/private/reference.wav"}
        ),
    ),
)
def test_full_contract_fixture_rejects_invalid_optional_stage_metadata(mutate) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "full_contract_case.json"
    document = deepcopy(json.loads(fixture_path.read_text(encoding="utf-8")))
    mutate(document)

    with pytest.raises(FixtureSchemaError):
        decode_fixture(document)


def test_generator_runs_without_top_level_package_or_core_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in tuple(sys.modules):
        if (
            module_name == "genie_tts"
            or module_name.startswith("genie_tts.")
            or module_name == generate_python_fixture.ISOLATED_PACKAGE_NAME
            or module_name.startswith(f"{generate_python_fixture.ISOLATED_PACKAGE_NAME}.")
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    def unexpected_input(_: str) -> str:
        raise AssertionError("fixture generation must not request GenieData download")

    monkeypatch.setattr("builtins.input", unexpected_input)
    generated = generate_python_fixture.generate_fixture(
        {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_type": "chinese_frontend",
            "cases": [{"id": "smoke", "category": "smoke", "input_text": "你好。"}],
        }
    )

    assert generated["cases"][0]["expected"]["phones"]
    assert "genie_tts.Internal" not in sys.modules
    resources = sys.modules[f"{generate_python_fixture.ISOLATED_PACKAGE_NAME}.Core.Resources"]
    assert not hasattr(resources, "download_genie_data")


def test_generator_cli_writes_independent_output(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "chinese_frontend_cases.json"
    output = tmp_path / "generated-chinese-frontend.json"

    result = subprocess.run(
        [sys.executable, str(generate_python_fixture.__file__), str(source), str(output)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = decode_fixture(json.loads(output.read_text(encoding="utf-8")))
    assert generated["cases"][0]["expected"]["phones"]
    assert json.loads(source.read_text(encoding="utf-8"))["cases"][0].get("expected") is None
