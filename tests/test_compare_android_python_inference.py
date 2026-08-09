import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "tools/compare_android_python_inference.py"
_SPEC = importlib.util.spec_from_file_location("compare_android_python_inference", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_trace_parser_keeps_hashes_and_model_roles(tmp_path: Path) -> None:
    log = tmp_path / "android.log"
    log.write_text(
        "\n".join(
            [
                "role=prompt_encoder provider=cpu cache=n/a",
                "semantic_hash: " + "a" * 64,
                "audio_hash: " + "b" * 64,
                "semantic_shape: [1, 1, 4]",
            ]
        ),
        encoding="utf-8",
    )

    trace = _MODULE.parse_trace(log)

    assert trace.hashes == {"semantic": "a" * 64, "audio": "b" * 64}
    assert trace.model_roles == {"prompt_encoder": "cpu"}


def test_float_stats_allow_small_hash_difference_within_tolerance(tmp_path: Path) -> None:
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(
        "audio_hash: " + "a" * 64 + "\n"
        "semantic_tokens: 1\n"
        "audio_samples: 2\n"
        "audio_stats: finite_count=2 non_finite_count=0 min=-1 max=1 mean=0\n",
        encoding="utf-8",
    )
    right.write_text(
        "audio_hash: " + "b" * 64 + "\n"
        "semantic_tokens: 1\n"
        "audio_samples: 2\n"
        "audio_stats: finite_count=2 non_finite_count=0 min=-0.99999 max=1.00001 mean=0.000001\n",
        encoding="utf-8",
    )

    messages = _MODULE.compare_trace(
        _MODULE.parse_trace(left),
        _MODULE.parse_trace(right),
        strict=True,
        float_atol=2e-5,
        float_rtol=0.0,
    )

    assert messages == []


def test_float_stats_report_out_of_tolerance_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(
        "audio_stats: finite_count=2 non_finite_count=0 min=-1 max=1 mean=0\n",
        encoding="utf-8",
    )
    right.write_text(
        "audio_stats: finite_count=2 non_finite_count=0 min=-0.9 max=1 mean=0\n",
        encoding="utf-8",
    )

    messages = _MODULE.compare_trace(
        _MODULE.parse_trace(left),
        _MODULE.parse_trace(right),
        strict=True,
        float_atol=1e-4,
        float_rtol=0.0,
    )

    assert any(message.startswith("FAIL stats audio") for message in messages)
