import hashlib
import importlib.util
import json
import struct
import sys
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


def test_generator_prepends_repository_src_before_resolving_genie_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_src = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setattr(
        sys,
        "path",
        [path for path in sys.path if Path(path).resolve() != repository_src],
    )

    generate_python_fixture.prepend_repository_src()

    assert Path(sys.path[0]).resolve() == repository_src
    spec = importlib.util.find_spec("genie_tts")
    assert spec is not None
    assert Path(spec.origin).resolve() == repository_src / "genie_tts" / "__init__.py"


def test_generator_prioritizes_checkout_src_over_site_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository_src = Path(__file__).resolve().parents[2] / "src"
    installed_package = tmp_path / "site-packages" / "genie_tts"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("SOURCE = 'installed'\n", encoding="utf-8")

    monkeypatch.delitem(sys.modules, "genie_tts", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [str(installed_package.parent)]
        + [path for path in sys.path if Path(path).resolve() != repository_src],
    )

    generate_python_fixture.prepend_repository_src()

    spec = importlib.util.find_spec("genie_tts")
    assert spec is not None
    assert Path(sys.path[0]).resolve() == repository_src
    assert Path(spec.origin).resolve() == repository_src / "genie_tts" / "__init__.py"


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
