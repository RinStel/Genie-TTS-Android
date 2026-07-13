"""Generate deterministic Chinese frontend parity expectations from Python."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from fixture_schema import decode_fixture, serialize_integer_tensor

ISOLATED_PACKAGE_NAME = "_genie_tts_parity"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "src" / "genie_tts"
_TEMP_RESOURCE_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None


def require_separate_output_path(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Generator source and output must use separate output paths.")


def load_chinese_to_phones() -> Callable[[str], tuple[str, list[str], list[int], list[int]]]:
    """Load only the checkout Chinese frontend under an isolated namespace."""
    module_name = f"{ISOLATED_PACKAGE_NAME}.G2P.Chinese.ChineseG2P"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.chinese_to_phones

    _install_namespace(ISOLATED_PACKAGE_NAME, _PACKAGE_ROOT)
    _install_namespace(f"{ISOLATED_PACKAGE_NAME}.Core", _PACKAGE_ROOT / "Core")
    _install_namespace(f"{ISOLATED_PACKAGE_NAME}.G2P", _PACKAGE_ROOT / "G2P")
    _install_namespace(
        f"{ISOLATED_PACKAGE_NAME}.G2P.Chinese",
        _PACKAGE_ROOT / "G2P" / "Chinese",
    )

    resources = ModuleType(f"{ISOLATED_PACKAGE_NAME}.Core.Resources")
    resources.__package__ = f"{ISOLATED_PACKAGE_NAME}.Core"
    resources.Chinese_G2P_DIR = str(_resolve_chinese_g2p_directory())
    sys.modules[resources.__name__] = resources

    module = importlib.import_module(module_name)
    return module.chinese_to_phones


def generate_fixture(source: dict[str, Any]) -> dict[str, Any]:
    """Generate frontend data and normalize an existing full-contract baseline.

    This generator never creates model outputs. Full-contract sources must provide
    their own complete stage baseline, which is preserved and revalidated.
    """
    decode_fixture(source)
    chinese_to_phones = load_chinese_to_phones()

    generated_cases = []
    for case in source["cases"]:
        normalized_text, phones, phone_ids, word2ph = chinese_to_phones(case["input_text"])
        frontend_expected = {
            "normalized_text": normalized_text,
            "phones": phones,
            "phone_ids": serialize_integer_tensor(
                shape=(1, len(phone_ids)),
                values=phone_ids,
            ),
            "word2ph": serialize_integer_tensor(
                shape=(len(word2ph),),
                values=word2ph,
            ),
        }
        expected = frontend_expected
        if source["fixture_type"] == "genie_tts_parity_contract":
            expected = {**case["expected"], **frontend_expected}
        generated_cases.append(
            {
                **case,
                "expected": expected,
            }
        )
    generated = {**source, "cases": generated_cases}
    decode_fixture(generated)
    return generated


def _install_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module


def _resolve_chinese_g2p_directory() -> Path:
    configured = os.getenv("Chinese_G2P_DIR")
    candidates = [
        Path(configured) if configured else None,
        _REPOSITORY_ROOT / "GenieData" / "G2P" / "ChineseG2P",
        _REPOSITORY_ROOT / "Android" / "app" / "src" / "main" / "assets" / "chinese_g2p",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "opencpop-strict.txt").is_file() and (
            candidate / "polyphonic.pickle"
        ).is_file():
            return candidate
        if (candidate / "opencpop-strict.txt").is_file() and (
            candidate / "polyphonic.json"
        ).is_file():
            return _materialize_pickle_resources(candidate)
    raise FileNotFoundError(
        "Chinese G2P resources were not found. Set Chinese_G2P_DIR or provide repo Android assets."
    )


def _materialize_pickle_resources(source: Path) -> Path:
    global _TEMP_RESOURCE_DIRECTORY
    if _TEMP_RESOURCE_DIRECTORY is None:
        _TEMP_RESOURCE_DIRECTORY = tempfile.TemporaryDirectory(prefix="genie-parity-g2p-")
        target = Path(_TEMP_RESOURCE_DIRECTORY.name)
        shutil.copyfile(source / "opencpop-strict.txt", target / "opencpop-strict.txt")
        polyphonic = json.loads((source / "polyphonic.json").read_text(encoding="utf-8"))
        with (target / "polyphonic.pickle").open("wb") as output:
            pickle.dump(polyphonic, output, protocol=pickle.HIGHEST_PROTOCOL)
    return Path(_TEMP_RESOURCE_DIRECTORY.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input Chinese frontend case JSON.")
    parser.add_argument("output", type=Path, help="Generated parity fixture JSON.")
    args = parser.parse_args()

    require_separate_output_path(args.source, args.output)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    generated = generate_fixture(source)
    args.output.write_text(
        json.dumps(generated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
