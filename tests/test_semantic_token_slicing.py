import importlib
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _install_namespace(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module


def _load_slice_semantic_tokens():
    package_name = "_genie_tts_semantic_slicing"
    package_root = Path(__file__).resolve().parents[1] / "src" / "genie_tts"
    _install_namespace(package_name, package_root)
    _install_namespace(f"{package_name}.Core", package_root / "Core")
    _install_namespace(f"{package_name}.Audio", package_root / "Audio")

    reference_audio = ModuleType(f"{package_name}.Audio.ReferenceAudio")
    reference_audio.ReferenceAudio = object
    sys.modules[reference_audio.__name__] = reference_audio

    phones_and_bert = ModuleType(f"{package_name}.GetPhonesAndBert")
    phones_and_bert.get_phones_and_bert = lambda *args, **kwargs: None
    sys.modules[phones_and_bert.__name__] = phones_and_bert

    return importlib.import_module(f"{package_name}.Core.Inference").slice_semantic_tokens


slice_semantic_tokens = _load_slice_semantic_tokens()


@pytest.mark.parametrize(
    ("decoder_output", "completed_count", "expected"),
    [
        (
            np.array([[99, 1024]], dtype=np.int64),
            1,
            np.array([[]], dtype=np.int64),
        ),
        (
            np.array([[99, 11, 12, 1024]], dtype=np.int64),
            3,
            np.array([[11, 12]], dtype=np.int64),
        ),
        (
            np.array([[99, 21, 22, 23]], dtype=np.int64),
            3,
            np.array([[21, 22]], dtype=np.int64),
        ),
        (
            np.array([[99, 31, 32, 0]], dtype=np.int64),
            3,
            np.array([[31, 32]], dtype=np.int64),
        ),
    ],
    ids=["one_step_eos", "normal_eos", "max_steps", "terminal_replacement"],
)
def test_slice_semantic_tokens_excludes_terminal_decoder_output(
    decoder_output: np.ndarray,
    completed_count: int,
    expected: np.ndarray,
) -> None:
    np.testing.assert_array_equal(
        slice_semantic_tokens(decoder_output, completed_count),
        expected,
    )


def test_slice_semantic_tokens_returns_empty_for_zero_completed_outputs() -> None:
    decoder_output = np.array([[99, 31, 32]], dtype=np.int64)

    actual = slice_semantic_tokens(decoder_output, completed_count=0)

    np.testing.assert_array_equal(actual, np.array([[]], dtype=np.int64))
