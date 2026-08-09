"""Package one Genie-TTS character model for the Android resource importer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


@dataclass(frozen=True)
class CharacterPackageInfo:
    output: Path
    variant: str
    character_id: str
    file_count: int
    total_bytes: int


def package_character_model(
    character_dir: Path | str,
    output_zip: Path | str,
    *,
    variant: str | None = None,
    character_id: str | None = None,
    reference_text: str | None = None,
) -> CharacterPackageInfo:
    """Validate and package a character directory using the Android layout."""

    source_root = Path(character_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"character directory does not exist: {source_root}")

    variant, character_id = _resolve_package_identity(source_root, variant, character_id)
    _validate_component(variant, "variant")
    _validate_component(character_id, "character_id")

    tts_models = source_root / "tts_models"
    if not tts_models.is_dir():
        raise ValueError(f"character directory must contain tts_models: {source_root}")
    if not any(
        path.is_file() and path.suffix.lower() == ".onnx"
        for path in tts_models.rglob("*")
    ):
        raise ValueError(f"character directory has no ONNX model under tts_models: {source_root}")

    generated_prompt_json = _normal_reference_payload(source_root, reference_text)
    files = _collect_files(source_root)
    output = Path(output_zip).expanduser().resolve()
    if _is_relative_to(output, source_root):
        raise ValueError("output ZIP must be outside the character directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    package_root = Path("CharacterModels") / variant / character_id
    archive_files = [
        (path, (package_root / path.relative_to(source_root)).as_posix())
        for path in files
        if path != source_root / "character_manifest.json"
    ]
    generated_prompt_name = (package_root / "prompt_wav.json").as_posix()
    archive_names = [archive_name for _, archive_name in archive_files]
    if generated_prompt_json is not None:
        archive_names.append(generated_prompt_name)
    manifest = {
        "format_version": 1,
        "package_type": "character_model",
        "variant": variant,
        "character_id": character_id,
        "files": archive_names,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_name = (package_root / "character_manifest.json").as_posix()

    total_bytes = sum(path.stat().st_size for path, _ in archive_files) + len(manifest_bytes)
    if generated_prompt_json is not None:
        total_bytes += len(generated_prompt_json)
    file_count = len(archive_files) + (1 if generated_prompt_json is not None else 0) + 1
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with ZipFile(temporary_path, mode="w", compression=ZIP_STORED, allowZip64=True) as archive:
            for path, archive_name in archive_files:
                archive.write(path, archive_name, compress_type=ZIP_STORED)
            if generated_prompt_json is not None:
                archive.writestr(generated_prompt_name, generated_prompt_json, compress_type=ZIP_STORED)
            archive.writestr(manifest_name, manifest_bytes, compress_type=ZIP_STORED)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return CharacterPackageInfo(
        output=output,
        variant=variant,
        character_id=character_id,
        file_count=file_count,
        total_bytes=total_bytes,
    )


def _resolve_package_identity(
    source_root: Path,
    variant: str | None,
    character_id: str | None,
) -> tuple[str, str]:
    if (variant is None) != (character_id is None):
        raise ValueError("variant and character_id must be provided together")
    if variant is not None and character_id is not None:
        return variant, character_id

    parts = source_root.parts
    marker_index = next(
        (index for index, part in enumerate(parts) if part.lower() == "charactermodels"),
        None,
    )
    if marker_index is None or len(parts) - marker_index != 3:
        raise ValueError(
            "cannot infer variant and character_id; pass --variant and --character-id "
            "or use .../CharacterModels/<variant>/<character_id>",
        )
    return parts[marker_index + 1], parts[marker_index + 2]


def _validate_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or ":" in value:
        raise ValueError(f"invalid {label}: {value!r}")


def _normal_reference_payload(source_root: Path, reference_text: str | None) -> bytes | None:
    prompt_json = source_root / "prompt_wav.json"
    if prompt_json.is_file():
        try:
            data = json.loads(prompt_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prompt_wav.json: {prompt_json}") from error

        normal = data.get("Normal") if isinstance(data, dict) else None
        wav_name = normal.get("wav") if isinstance(normal, dict) else None
        normal_text = normal.get("text") if isinstance(normal, dict) else None
        if not isinstance(wav_name, str) or not wav_name.strip():
            raise ValueError("Normal reference must contain a wav file name")
        if not isinstance(normal_text, str) or not normal_text.strip():
            raise ValueError("Normal reference must contain non-blank text")
        generated_prompt_json = None
    else:
        prompt_root = source_root / "prompt_wav"
        audio_files = sorted(
            path for path in prompt_root.glob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        )
        if len(audio_files) != 1:
            raise ValueError(
                "missing prompt_wav.json; provide it or keep exactly one reference audio "
                "and pass --reference-text",
            )
        wav_name = audio_files[0].name
        stem = audio_files[0].stem
        inferred_text = stem.split("-", 1)[1].strip() if "-" in stem else ""
        normal_text = (reference_text or inferred_text).strip()
        if not normal_text:
            raise ValueError(
                "missing Normal reference text; pass --reference-text when the audio filename "
                "does not contain '<prefix>-<text>'",
            )
        generated_prompt_json = (
            json.dumps(
                {"Normal": {"wav": wav_name, "text": normal_text}},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")

    prompt_root = (source_root / "prompt_wav").resolve()
    audio_path = (prompt_root / wav_name).resolve()
    if not _is_relative_to(audio_path, prompt_root) or not audio_path.is_file():
        raise ValueError(f"reference audio is outside prompt_wav or missing: {wav_name}")
    return generated_prompt_json


def _collect_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic links are not allowed in character packages: {path}")
        if path.is_file():
            files.append(path)
    return files


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("character_dir", type=Path, help="character directory containing tts_models")
    parser.add_argument("--output", required=True, type=Path, help="output ZIP path")
    parser.add_argument("--variant", help="model variant, required for standalone directories")
    parser.add_argument("--character-id", help="character ID, required for standalone directories")
    parser.add_argument(
        "--reference-text",
        help="reference text for legacy directories without prompt_wav.json",
    )
    args = parser.parse_args()
    try:
        info = package_character_model(
            args.character_dir,
            args.output,
            variant=args.variant,
            character_id=args.character_id,
            reference_text=args.reference_text,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Created {info.output} ({info.file_count} files, {info.total_bytes} bytes) "
        f"for {info.variant}/{info.character_id}.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
