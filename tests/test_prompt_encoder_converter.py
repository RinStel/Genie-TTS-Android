import importlib.util
import sys
from types import ModuleType
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).parents[1] / "src/genie_tts/Converter/v2ProPlus/PromptEncoderConverter.py"
_PACKAGE = "genie_tts_prompt_converter_test"
for _name in (_PACKAGE, f"{_PACKAGE}.Converter", f"{_PACKAGE}.Converter.v2ProPlus"):
    _module = ModuleType(_name)
    _module.__path__ = []
    sys.modules[_name] = _module
_load_state_dict = ModuleType(f"{_PACKAGE}.Converter.load_state_dict")
_load_state_dict.load_sovits_model = lambda *_args, **_kwargs: {"weight": {}}
sys.modules[_load_state_dict.__name__] = _load_state_dict

_SPEC = importlib.util.spec_from_file_location(
    f"{_PACKAGE}.Converter.v2ProPlus.PromptEncoderConverter",
    _MODULE_PATH,
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_reconstructs_fp32_external_weights_for_multi_reference_export(tmp_path: Path) -> None:
    converter = _MODULE.PromptEncoderConverter.__new__(_MODULE.PromptEncoderConverter)
    converter.fp16_bin_path = str(tmp_path / "prompt_encoder_fp16.bin")
    converter.reconstructed_fp32_bin_path = str(tmp_path / "prompt_encoder_fp32.bin")
    expected = np.asarray([0.0, 1.5, -2.25, 65504.0], dtype=np.float16)
    expected.tofile(converter.fp16_bin_path)

    converter.step3_reconstruct_fp32_bin()

    actual = np.fromfile(converter.reconstructed_fp32_bin_path, dtype=np.float32)
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=0, atol=0)
