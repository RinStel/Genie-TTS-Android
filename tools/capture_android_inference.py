"""Capture an Android inference trace through adb forwarding.

The app intentionally binds its HTTP server to 127.0.0.1.  ``adb forward``
exposes that loopback port to the host without widening the app's network
surface.  The resulting log and WAV files can be passed to
``compare_android_python_inference.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_PACKAGE = "dev.rinstel.genie_tts"
DEFAULT_DEVICE_PORT = 16580
DEFAULT_LOG_TAGS = ("GenieTtsTiming:V", "NativeDecoderLoop:V")


@dataclass(frozen=True)
class AdbDevice:
    serial: str
    state: str


@dataclass(frozen=True)
class CaptureConfig:
    adb: str
    serial: str | None
    package: str
    request: Path
    android_log: Path
    android_wav: Path
    device_port: int
    host_port: int
    timeout_seconds: float
    health_timeout_seconds: float
    apk: Path | None
    start_app: bool
    repeat: int
    log_tags: tuple[str, ...]


def build_adb_command(adb: str, serial: str | None, args: Sequence[str]) -> list[str]:
    command = [adb]
    if serial:
        command.extend(("-s", serial))
    command.extend(args)
    return command


def parse_adb_devices(output: str) -> list[AdbDevice]:
    devices: list[AdbDevice] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            devices.append(AdbDevice(serial=fields[0], state=fields[1]))
    return devices


def select_device(adb: str, serial: str | None) -> str:
    result = subprocess.run(
        [adb, "devices"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"adb devices failed: {result.stderr.strip()}")

    devices = parse_adb_devices(result.stdout)
    if serial:
        selected = next((device for device in devices if device.serial == serial), None)
        if selected is None:
            raise RuntimeError(f"adb device {serial!r} was not found.")
        if selected.state != "device":
            raise RuntimeError(f"adb device {serial!r} is not ready: {selected.state}")
        return serial

    ready = [device.serial for device in devices if device.state == "device"]
    if len(ready) != 1:
        if not ready:
            raise RuntimeError("No ready adb device is available.")
        raise RuntimeError("Multiple adb devices are ready; pass --serial explicitly.")
    return ready[0]


def load_request(path: Path) -> bytes:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON request {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("The inference request must be a JSON object.")
    if not str(payload.get("text", "")).strip():
        raise ValueError("The inference request must contain a non-empty 'text' field.")
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _run_adb(adb: str, serial: str, args: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    command = build_adb_command(adb, serial, args)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result


def _wait_for_health(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "connection refused"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"Android HTTP service did not become healthy: {last_error}")


def _post_inference(base_url: str, body: bytes, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(
        f"{base_url}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = response.read()
            if response.status != 200:
                raise RuntimeError(f"Android inference returned HTTP {response.status}.")
            if not result:
                raise RuntimeError("Android inference returned an empty response.")
            return result
    except urllib.error.HTTPError as error:
        error.read()
        raise RuntimeError(f"Android inference returned HTTP {error.code}.") from error


def _output_path(path: Path, repeat_index: int, repeat_count: int) -> Path:
    if repeat_count == 1:
        return path
    return path.with_name(f"{path.stem}-{repeat_index + 1}{path.suffix}")


def capture(config: CaptureConfig) -> list[Path]:
    serial = select_device(config.adb, config.serial)
    body = load_request(config.request)
    config.android_log.parent.mkdir(parents=True, exist_ok=True)
    config.android_wav.parent.mkdir(parents=True, exist_ok=True)

    if config.apk is not None:
        _run_adb(config.adb, serial, ("install", "-r", str(config.apk)), timeout=config.timeout_seconds)

    _run_adb(
        config.adb,
        serial,
        ("forward", f"tcp:{config.host_port}", f"tcp:{config.device_port}"),
        timeout=config.timeout_seconds,
    )
    log_process: subprocess.Popen[str] | None = None
    output_paths: list[Path] = []
    base_url = f"http://127.0.0.1:{config.host_port}"
    try:
        _run_adb(config.adb, serial, ("logcat", "-c"), timeout=config.timeout_seconds)
        with config.android_log.open("w", encoding="utf-8") as log_file:
            log_process = subprocess.Popen(
                build_adb_command(
                    config.adb,
                    serial,
                    ("logcat", "-v", "threadtime", "-s", *config.log_tags),
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if config.start_app:
                _run_adb(
                    config.adb,
                    serial,
                    ("shell", "monkey", "-p", config.package, "1"),
                    timeout=config.timeout_seconds,
                )
            _wait_for_health(base_url, config.health_timeout_seconds)
            for repeat_index in range(config.repeat):
                output_path = _output_path(config.android_wav, repeat_index, config.repeat)
                output_path.write_bytes(_post_inference(base_url, body, config.timeout_seconds))
                output_paths.append(output_path)
            time.sleep(0.25)
    finally:
        if log_process is not None:
            log_process.terminate()
            try:
                log_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                log_process.kill()
                log_process.wait()
        # Cleanup must not hide the inference or logcat failure that caused
        # the capture to fail, especially when the device disconnects.
        try:
            _run_adb(
                config.adb,
                serial,
                ("forward", "--remove", f"tcp:{config.host_port}"),
                timeout=config.timeout_seconds,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass
    return output_paths


def parse_args(argv: Iterable[str]) -> CaptureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON body for POST /infer")
    parser.add_argument("--android-log", type=Path, required=True)
    parser.add_argument("--android-wav", type=Path, required=True)
    parser.add_argument("--serial")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--apk", type=Path)
    parser.add_argument("--device-port", type=int, default=DEFAULT_DEVICE_PORT)
    parser.add_argument("--host-port", type=int, default=DEFAULT_DEVICE_PORT)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument(
        "--log-tag",
        dest="log_tags",
        action="append",
        default=None,
        help="logcat filter spec; may be repeated",
    )
    args = parser.parse_args(list(argv))
    if not 1024 <= args.device_port <= 65535:
        parser.error("--device-port must be between 1024 and 65535")
    if not 1024 <= args.host_port <= 65535:
        parser.error("--host-port must be between 1024 and 65535")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return CaptureConfig(
        adb=args.adb,
        serial=args.serial,
        package=args.package,
        request=args.request,
        android_log=args.android_log,
        android_wav=args.android_wav,
        device_port=args.device_port,
        host_port=args.host_port,
        timeout_seconds=args.timeout,
        health_timeout_seconds=args.health_timeout,
        apk=args.apk,
        start_app=not args.no_start,
        repeat=args.repeat,
        log_tags=tuple(args.log_tags or DEFAULT_LOG_TAGS),
    )


def main(argv: Iterable[str] | None = None) -> int:
    try:
        config = parse_args(argv or sys.argv[1:])
        outputs = capture(config)
    except (OSError, RuntimeError, TimeoutError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    print(config.android_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
