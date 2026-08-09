import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tools.package_android_character_model import package_character_model


def create_character(root: Path, *, variant: str = "v2ProPlus", character_id: str = "Alice") -> Path:
    character = root / "CharacterModels" / variant / character_id
    (character / "tts_models").mkdir(parents=True)
    (character / "tts_models" / "t2s_first_stage_decoder_fp32.onnx").write_bytes(b"model")
    (character / "prompt_wav").mkdir()
    (character / "prompt_wav" / "normal.wav").write_bytes(b"wav")
    (character / "prompt_wav.json").write_text(
        json.dumps(
            {"Normal": {"wav": "normal.wav", "text": "reference text"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return character


def test_packages_character_directory_with_android_root(tmp_path: Path) -> None:
    character = create_character(tmp_path)
    output = tmp_path / "alice-character.zip"

    package_info = package_character_model(character, output)

    assert package_info.variant == "v2ProPlus"
    assert package_info.character_id == "Alice"
    assert output.is_file()
    with ZipFile(output) as archive:
        assert archive.getinfo(
            "CharacterModels/v2ProPlus/Alice/tts_models/t2s_first_stage_decoder_fp32.onnx",
        ).compress_type == ZIP_DEFLATED
        assert "CharacterModels/v2ProPlus/Alice/prompt_wav/normal.wav" in archive.namelist()
        manifest = json.loads(
            archive.read("CharacterModels/v2ProPlus/Alice/character_manifest.json"),
        )
        assert manifest["package_type"] == "character_model"
        assert manifest["character_id"] == "Alice"


def test_uses_explicit_variant_and_character_id_for_standalone_directory(tmp_path: Path) -> None:
    character = create_character(tmp_path, variant="ignored", character_id="SourceName")
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    for child in character.iterdir():
        child.rename(standalone / child.name)
    output = tmp_path / "standalone.zip"

    package_info = package_character_model(
        standalone,
        output,
        variant="v2",
        character_id="Alice_CN",
    )

    assert package_info.variant == "v2"
    assert package_info.character_id == "Alice_CN"
    with ZipFile(output) as archive:
        assert "CharacterModels/v2/Alice_CN/tts_models/t2s_first_stage_decoder_fp32.onnx" in archive.namelist()


def test_rejects_character_without_onnx_or_valid_normal_reference(tmp_path: Path) -> None:
    character = tmp_path / "CharacterModels" / "v2ProPlus" / "Broken"
    (character / "tts_models").mkdir(parents=True)
    (character / "prompt_wav").mkdir()
    (character / "prompt_wav" / "normal.wav").write_bytes(b"wav")
    (character / "prompt_wav.json").write_text(
        '{"Normal": {"wav": "normal.wav", "text": " "}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ONNX"):
        package_character_model(character, tmp_path / "broken.zip")


def test_rejects_reference_audio_path_outside_character_directory(tmp_path: Path) -> None:
    character = create_character(tmp_path)
    (character / "prompt_wav.json").write_text(
        '{"Normal": {"wav": "../outside.wav", "text": "reference"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference audio"):
        package_character_model(character, tmp_path / "broken-reference.zip")


def test_generates_normal_prompt_for_legacy_single_reference_filename(tmp_path: Path) -> None:
    character = create_character(tmp_path)
    (character / "prompt_wav.json").unlink()
    (character / "prompt_wav" / "normal.wav").rename(
        character / "prompt_wav" / "穗-对啊，我又不是要做戏子的.wav",
    )
    output = tmp_path / "legacy.zip"

    package_character_model(character, output)

    with ZipFile(output) as archive:
        prompt = json.loads(
            archive.read("CharacterModels/v2ProPlus/Alice/prompt_wav.json"),
        )
        assert prompt["Normal"]["wav"] == "穗-对啊，我又不是要做戏子的.wav"
        assert prompt["Normal"]["text"] == "对啊，我又不是要做戏子的"
