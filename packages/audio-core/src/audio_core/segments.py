from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import AudioSegmentExportError, AudioToolUnavailable
from .validation import require_distinct_output, require_regular_file


class SegmentExportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    sample_rate_hz: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)

    @model_validator(mode="after")
    def interval_is_valid(self) -> SegmentExportSpec:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


def extract_audio_segment(
    input_path: str | Path,
    output_path: str | Path,
    *,
    spec: SegmentExportSpec,
    ffmpeg: str = "ffmpeg",
    timeout: float = 300.0,
) -> Path:
    source = require_regular_file(input_path)
    destination = require_distinct_output(source, output_path)
    executable = shutil.which(ffmpeg)
    if not executable:
        raise AudioToolUnavailable(f"executable not found: {ffmpeg}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".wav", dir=destination.parent
    )
    os.close(file_descriptor)
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
        "-ss",
        f"{spec.start_ms / 1000:.3f}",
        "-t",
        f"{(spec.end_ms - spec.start_ms) / 1000:.3f}",
        "-vn",
        "-map_metadata",
        "-1",
        "-ar",
        str(spec.sample_rate_hz),
        "-ac",
        str(spec.channels),
        "-c:a",
        "pcm_s16le",
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
            raise AudioSegmentExportError(
                f"ffmpeg segment extraction failed: {detail or 'unknown error'}"
            )
        if temporary.stat().st_size <= 44:
            raise AudioSegmentExportError("ffmpeg produced an empty audio segment")
        os.replace(temporary, destination)
        return destination
    except subprocess.TimeoutExpired as exc:
        raise AudioSegmentExportError(
            f"ffmpeg segment extraction timed out after {timeout}s"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
