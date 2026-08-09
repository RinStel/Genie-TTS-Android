import json
from pathlib import Path

import numpy as np
import soxr


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reference_resampler_soxr_hq.json"


def test_resampler_fixture_is_soxr_hq_and_shape_stable() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["schema_version"] == 1
    assert fixture["quality"] == "soxr_hq"
    assert set(fixture["cases"]) == {
        "impulse",
        "sweep",
        "speech_like",
        "stereo_mix",
        "edge_length",
    }

    for case in fixture["cases"].values():
        source = np.asarray(case["source_samples"], dtype=np.float32)
        expected = np.asarray(case["expected_samples"], dtype=np.float32)
        generated = soxr.resample(
            source,
            case["source_rate"],
            case["target_rate"],
            quality="HQ",
        ).astype(np.float32)
        assert expected.shape == generated.shape
        np.testing.assert_array_equal(expected, generated)
