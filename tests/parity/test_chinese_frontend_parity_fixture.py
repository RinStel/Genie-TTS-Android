import json
from pathlib import Path

import generate_python_fixture


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_python_frontend_expected_fixture_is_reproducible() -> None:
    source = json.loads((FIXTURE_DIR / "chinese_frontend_cases.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURE_DIR / "chinese_frontend_expected.json").read_text(encoding="utf-8"))

    assert generate_python_fixture.generate_fixture(source) == expected
