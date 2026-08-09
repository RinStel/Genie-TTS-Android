from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "capture_android_inference.py"
SPEC = importlib.util.spec_from_file_location("capture_android_inference", MODULE_PATH)
assert SPEC and SPEC.loader
capture_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture_tool
SPEC.loader.exec_module(capture_tool)


def test_parse_adb_devices_ignores_header_and_keeps_states() -> None:
    devices = capture_tool.parse_adb_devices(
        """List of devices attached
ABC123\tdevice product:foo model:bar
offline-1\toffline
"""
    )

    assert [(device.serial, device.state) for device in devices] == [
        ("ABC123", "device"),
        ("offline-1", "offline"),
    ]


def test_build_adb_command_places_serial_before_subcommand() -> None:
    assert capture_tool.build_adb_command("adb", "ABC123", ("forward", "tcp:16580", "tcp:16580")) == [
        "adb",
        "-s",
        "ABC123",
        "forward",
        "tcp:16580",
        "tcp:16580",
    ]


def test_load_request_requires_text(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"language":"zh"}', encoding="utf-8")

    with pytest.raises(ValueError, match="text"):
        capture_tool.load_request(request)


def test_output_path_is_indexed_for_warm_runs(tmp_path: Path) -> None:
    output = tmp_path / "android.wav"

    assert capture_tool._output_path(output, 0, 2).name == "android-1.wav"
    assert capture_tool._output_path(output, 1, 2).name == "android-2.wav"
    assert capture_tool._output_path(output, 0, 1) == output


def test_capture_does_not_hide_inference_failure_during_forward_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"text":"hello"}', encoding="utf-8")
    config = capture_tool.CaptureConfig(
        adb="adb",
        serial=None,
        package="dev.rinstel.genie_tts",
        request=request,
        android_log=tmp_path / "android.log",
        android_wav=tmp_path / "android.wav",
        device_port=16580,
        host_port=16580,
        timeout_seconds=1.0,
        health_timeout_seconds=1.0,
        apk=None,
        start_app=False,
        repeat=1,
        log_tags=("GenieTtsTiming:V",),
    )

    class FakeLogProcess:
        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> None:
            pass

    monkeypatch.setattr(capture_tool, "select_device", lambda adb, serial: "serial")
    monkeypatch.setattr(capture_tool, "_wait_for_health", lambda base_url, timeout: None)
    monkeypatch.setattr(capture_tool.subprocess, "Popen", lambda *args, **kwargs: FakeLogProcess())

    def fake_run_adb(adb, serial, args, *, timeout=None):
        if args[:2] == ("forward", "--remove"):
            raise RuntimeError("device disconnected during cleanup")
        return None

    monkeypatch.setattr(capture_tool, "_run_adb", fake_run_adb)
    monkeypatch.setattr(
        capture_tool,
        "_post_inference",
        lambda base_url, body, timeout: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        capture_tool.capture(config)
