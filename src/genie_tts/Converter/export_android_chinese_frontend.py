"""Export the Python Chinese frontend's contextual lexicon for Android."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import jieba_fast
from g2pM import G2pM


SCHEMA_VERSION = 1
_CJK_WORD = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_jieba_dictionary() -> Path:
    dictionary = Path(jieba_fast.__file__).resolve().parent / "dict.txt"
    if not dictionary.is_file():
        raise FileNotFoundError(f"jieba dictionary not found: {dictionary}")
    return dictionary


def _read_lexicon(
    dictionary: Path,
    polyphonic: dict[str, list[str]],
) -> list[tuple[str, int, str, list[str]]]:
    g2pm = G2pM()
    entries: list[tuple[str, int, str, list[str]]] = []
    for line in dictionary.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) < 2 or not _CJK_WORD.fullmatch(fields[0]):
            continue
        word = fields[0]
        pinyins = [pinyin.replace("u:", "v") for pinyin in g2pm(word, char_split=True)]
        override = polyphonic.get(word)
        if override is not None and len(override) == len(word):
            pinyins = override
        else:
            for index, char in enumerate(word):
                char_override = polyphonic.get(char)
                if char_override and index < len(pinyins):
                    pinyins[index] = char_override[0]
        entries.append((word, int(fields[1]), fields[2] if len(fields) == 3 else "", pinyins))
    return sorted(entries, key=lambda entry: entry[0])


def _read_traditional_mapping(source_path: Path) -> list[tuple[str, str]]:
    """Read the same static mapping used by Python TextNormalizer."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "simplified_charcters",
            "traditional_characters",
        }:
            continue
        values[target.id] = ast.literal_eval(node.value)

    simplified = values.get("simplified_charcters")
    traditional = values.get("traditional_characters")
    if simplified is None or traditional is None or len(simplified) != len(traditional):
        raise ValueError(f"Invalid traditional/simplified mapping: {source_path}")
    return sorted({(traditional_char, simplified_char) for simplified_char, traditional_char in zip(simplified, traditional)})


def export_frontend(source_dir: Path, output_dir: Path) -> dict[str, object]:
    required = {
        "polyphonic.json": source_dir / "polyphonic.json",
        "single_char_pinyin.json": source_dir / "single_char_pinyin.json",
        "opencpop-strict.txt": source_dir / "opencpop-strict.txt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Chinese frontend resources: " + ", ".join(missing))

    dictionary = _find_jieba_dictionary()
    repository_root = Path(__file__).resolve().parents[3]
    char_convert_path = repository_root / "src/genie_tts/G2P/Chinese/Normalization/char_convert.py"
    polyphonic = json.loads(required["polyphonic.json"].read_text(encoding="utf-8"))
    entries = _read_lexicon(dictionary, polyphonic)
    traditional_mapping = _read_traditional_mapping(char_convert_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    lexicon_path = output_dir / "contextual_lexicon.tsv"
    lexicon_path.write_text(
        "\n".join(
            f"{word}\t{frequency}\t{tag}\t{','.join(pinyins)}"
            for word, frequency, tag, pinyins in entries
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "traditional_to_simplified.tsv").write_text(
        "".join(f"{traditional}\t{simplified}\n" for traditional, simplified in traditional_mapping),
        encoding="utf-8",
    )

    source_hashes = {name: _sha256(path) for name, path in required.items()}
    source_hashes["jieba_fast/dict.txt"] = _sha256(dictionary)
    for relative_path in (
        "src/genie_tts/G2P/Chinese/ChineseG2P.py",
        "src/genie_tts/G2P/Chinese/ToneSandhi.py",
        "src/genie_tts/G2P/Chinese/Erhua.py",
        "src/genie_tts/G2P/Chinese/CorrectPronunciation.py",
        "src/genie_tts/G2P/Chinese/Normalization/char_convert.py",
        "src/genie_tts/G2P/Chinese/Normalization/text_normlization.py",
    ):
        source_path = repository_root / relative_path
        if source_path.is_file():
            source_hashes[relative_path] = _sha256(source_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "frontend": "genie_tts_chinese",
        "source_hashes": source_hashes,
        "resources": {
            "polyphonic_pinyin": "polyphonic.json",
            "single_char_pinyin": "single_char_pinyin.json",
            "opencpop": "opencpop-strict.txt",
            "contextual_lexicon": lexicon_path.name,
            "traditional_to_simplified": "traditional_to_simplified.tsv",
        },
        "traditional_mapping": {
            "entry_count": len(traditional_mapping),
        },
        "contextual_lexicon": {
            "entry_count": len(entries),
            "max_word_length": max((len(word) for word, _, _, _ in entries), default=1),
            "total_frequency": sum(frequency for _, frequency, _, _ in entries),
            "has_word_pinyin": True,
        },
    }
    (output_dir / "frontend_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_traditional_mapping(output_dir: Path) -> int:
    """Export the small normalization resource without rebuilding the lexicon."""
    repository_root = Path(__file__).resolve().parents[3]
    source_path = repository_root / "src/genie_tts/G2P/Chinese/Normalization/char_convert.py"
    mapping = _read_traditional_mapping(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traditional_to_simplified.tsv").write_text(
        "".join(f"{traditional}\t{simplified}\n" for traditional, simplified in mapping),
        encoding="utf-8",
    )
    return len(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only-traditional-mapping",
        action="store_true",
        help="Export only the normalization mapping without rebuilding the lexicon.",
    )
    args = parser.parse_args()
    if args.only_traditional_mapping:
        print(json.dumps({"traditional_mapping_entries": export_traditional_mapping(args.output_dir)}))
        return
    manifest = export_frontend(args.source_dir, args.output_dir)
    print(json.dumps(manifest["contextual_lexicon"], sort_keys=True))


if __name__ == "__main__":
    main()
