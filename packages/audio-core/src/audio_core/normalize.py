from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .errors import AudioNormalizeError, AudioToolUnavailable
from .validation import require_distinct_output, require_regular_file


class NormalizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate_hz: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    sample_format: str = Field(default="s16", pattern=r"^(s16|s24|s32|flt)$")

    @property
    def pcm_codec(self) -> str:
        return {
            "s16": "pcm_s16le",
            "s24": "pcm_s24le",
            "s32": "pcm_s32le",
            "flt": "pcm_f32le",
        }[self.sample_format]


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    spec: NormalizeSpec | None = None,
    ffmpeg: str = "ffmpeg",
    timeout: float = 300.0,
) -> Path:
    source = require_regular_file(input_path)
    destination = require_distinct_output(source, output_path)
    executable = shutil.which(ffmpeg)
    if not executable:
        raise AudioToolUnavailable(f"executable not found: {ffmpeg}")
    spec = spec or NormalizeSpec()

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".wav", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-map_metadata",
        "-1",
        "-ar",
        str(spec.sample_rate_hz),
        "-ac",
        str(spec.channels),
        "-c:a",
        spec.pcm_codec,
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise AudioNormalizeError(f"ffmpeg failed: {detail or 'unknown error'}")
        if temporary.stat().st_size == 0:
            raise AudioNormalizeError("ffmpeg produced an empty output")
        os.replace(temporary, destination)
        return destination
    except subprocess.TimeoutExpired as exc:
        raise AudioNormalizeError(f"ffmpeg timed out after {timeout}s") from exc
    finally:
        temporary.unlink(missing_ok=True)
