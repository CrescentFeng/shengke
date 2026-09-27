from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .contracts import AudioProbe, MediaFormat
from .errors import AudioProbeError, AudioToolUnavailable
from .validation import require_regular_file


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _find_tool(executable: str) -> str:
    found = shutil.which(executable)
    if not found:
        raise AudioToolUnavailable(f"executable not found: {executable}")
    return found


def probe_file(path: str | Path, *, ffprobe: str = "ffprobe", timeout: float = 30.0) -> AudioProbe:
    source = require_regular_file(path)
    executable = _find_tool(ffprobe)
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_long_name,sample_rate,channels,channel_layout,duration,bit_rate,sample_fmt:format=format_name,format_long_name,duration,bit_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProbeError(f"ffprobe timed out after {timeout}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise AudioProbeError(f"ffprobe failed: {detail or 'unknown error'}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        container = payload.get("format", {})
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise AudioProbeError("ffprobe returned no readable audio stream") from exc

    duration = _optional_float(stream.get("duration"))
    if duration is None:
        duration = _optional_float(container.get("duration"))
    bit_rate = _optional_int(stream.get("bit_rate"))
    if bit_rate is None:
        bit_rate = _optional_int(container.get("bit_rate"))
    return AudioProbe(
        codec_name=stream["codec_name"],
        codec_long_name=stream.get("codec_long_name"),
        sample_rate_hz=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        channel_layout=stream.get("channel_layout"),
        duration_seconds=duration,
        bit_rate=bit_rate,
        sample_format=stream.get("sample_fmt"),
        media_format=MediaFormat(
            format_name=container.get("format_name", "unknown"),
            format_long_name=container.get("format_long_name"),
            duration_seconds=_optional_float(container.get("duration")),
            bit_rate=_optional_int(container.get("bit_rate")),
        ),
    )
