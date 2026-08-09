import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "Android" / "app" / "src" / "main" / "assets" / "chinese_g2p"


def test_exported_frontend_manifest_matches_assets() -> None:
    manifest = json.loads((ASSET_DIR / "frontend_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["frontend"] == "genie_tts_chinese"
    assert manifest["contextual_lexicon"]["entry_count"] > 100_000
    mapping_name = manifest["resources"]["traditional_to_simplified"]
    mapping_lines = (ASSET_DIR / mapping_name).read_text(encoding="utf-8").splitlines()
    assert len(mapping_lines) == manifest["traditional_mapping"]["entry_count"]
    assert "臺\t台" in mapping_lines
    assert all(line.count("\t") == 1 for line in mapping_lines)

    for filename, expected_hash in manifest["source_hashes"].items():
        if filename.startswith("jieba_fast/"):
            continue
        source_path = ROOT / filename if filename.startswith("src/") else ASSET_DIR / filename
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_contextual_lexicon_is_sorted_and_typed() -> None:
    entries = (ASSET_DIR / "contextual_lexicon.tsv").read_text(encoding="utf-8").splitlines()
    words = []
    for line in entries:
        word, frequency, tag, pinyin = line.split("\t")
        words.append(word)
        assert word
        assert int(frequency) > 0
        assert "\t" not in tag
        assert len(pinyin.split(",")) == len(word)
    assert words == sorted(words)
