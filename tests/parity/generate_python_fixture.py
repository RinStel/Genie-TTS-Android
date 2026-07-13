"""Generate deterministic Chinese frontend parity expectations from Python."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fixture_schema import decode_fixture, serialize_integer_tensor


def prepend_repository_src() -> Path:
    repository_src = Path(__file__).resolve().parents[2] / "src"
    repository_src_string = str(repository_src)
    sys.path[:] = [path for path in sys.path if Path(path).resolve() != repository_src]
    sys.path.insert(0, repository_src_string)
    return repository_src


def require_separate_output_path(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Generator source and output must use separate output paths.")


def generate_fixture(source: dict[str, Any]) -> dict[str, Any]:
    """Run the Python frontend for each source case without loading model weights."""
    decode_fixture(source)
    prepend_repository_src()
    from genie_tts.G2P.Chinese.ChineseG2P import chinese_to_phones

    generated_cases = []
    for case in source["cases"]:
        normalized_text, phones, phone_ids, word2ph = chinese_to_phones(case["input_text"])
        generated_cases.append(
            {
                **case,
                "expected": {
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
                },
            }
        )
    return {**source, "cases": generated_cases}


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
