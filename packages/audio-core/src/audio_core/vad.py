from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import AudioToolUnavailable, AudioVadDependencyError, AudioVadError
from .validation import require_regular_file


class VadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["adaptive_energy_v1", "webrtc_v1"] = "adaptive_energy_v1"
    sample_rate_hz: int = Field(default=16000)
    frame_ms: int = Field(default=30)
    minimum_speech_ms: int = Field(default=150, ge=30, le=10000)
    maximum_silence_ms: int = Field(default=300, ge=0, le=10000)
    padding_ms: int = Field(default=100, ge=0, le=5000)
    minimum_speech_dbfs: float = Field(default=-42.0, ge=-100.0, le=0.0)
    noise_margin_db: float = Field(default=10.0, ge=0.0, le=40.0)
    webrtc_aggressiveness: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def frame_configuration_is_valid(self) -> VadSpec:
        if self.sample_rate_hz not in {8000, 16000, 32000, 48000}:
            raise ValueError("sample_rate_hz must be 8000, 16000, 32000, or 48000")
        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("frame_ms must be 10, 20, or 30")
        return self


class DetectedSpeechSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)
    mean_dbfs: float

    @model_validator(mode="after")
    def interval_is_valid(self) -> DetectedSpeechSegment:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class VadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = "adaptive_energy_v1"
    sample_rate_hz: int = Field(gt=0)
    frame_ms: int = Field(gt=0)
    threshold_dbfs: float | None = None
    aggressiveness: int | None = Field(default=None, ge=0, le=3)
    duration_ms: int = Field(ge=0)
    speech_duration_ms: int = Field(ge=0)
    speech_ratio: float = Field(ge=0.0, le=1.0)
    segments: list[DetectedSpeechSegment] = Field(default_factory=list)


def _decode_pcm(source: Path, executable: str, sample_rate_hz: int, timeout: float) -> array:
    file_descriptor, temporary_name = tempfile.mkstemp(prefix="audio-vad-", suffix=".pcm")
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
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-f",
        "s16le",
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
            raise AudioVadError(f"ffmpeg PCM decode failed: {detail or 'unknown error'}")
        samples = array("h")
        with temporary.open("rb") as stream:
            samples.fromfile(stream, temporary.stat().st_size // samples.itemsize)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            raise AudioVadError("ffmpeg decoded no audio samples")
        return samples
    except subprocess.TimeoutExpired as exc:
        raise AudioVadError(f"ffmpeg PCM decode timed out after {timeout}s") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _frame_dbfs(samples: array, frame_samples: int) -> list[float]:
    levels: list[float] = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if not frame:
            continue
        mean_square = sum(sample * sample for sample in frame) / len(frame)
        if mean_square <= 0:
            levels.append(-120.0)
        else:
            levels.append(max(-120.0, 20.0 * math.log10(math.sqrt(mean_square) / 32768.0)))
    return levels


def _merge_speech_frames(flags: list[bool], maximum_gap_frames: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_speech in enumerate(flags):
        if is_speech and start is None:
            start = index
        if not is_speech and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    if not runs:
        return []

    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= maximum_gap_frames:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _webrtc_flags(samples: array, spec: VadSpec) -> list[bool]:
    try:
        import webrtcvad
    except ImportError as exc:
        raise AudioVadDependencyError(
            "webrtc_v1 requires the 'vad-webrtc' optional dependencies"
        ) from exc

    detector = webrtcvad.Vad(spec.webrtc_aggressiveness)
    frame_samples = spec.sample_rate_hz * spec.frame_ms // 1000
    flags: list[bool] = []
    for offset in range(0, len(samples), frame_samples):
        frame = array("h", samples[offset : offset + frame_samples])
        if len(frame) < frame_samples:
            frame.extend([0] * (frame_samples - len(frame)))
        if sys.byteorder != "little":
            frame.byteswap()
        try:
            flags.append(detector.is_speech(frame.tobytes(), spec.sample_rate_hz))
        except Exception as exc:
            raise AudioVadError(f"WebRTC VAD inference failed: {exc}") from exc
    return flags


def _build_result(
    *,
    spec: VadSpec,
    samples: array,
    levels: list[float],
    flags: list[bool],
    threshold_dbfs: float | None,
) -> VadResult:
    maximum_gap_frames = math.ceil(spec.maximum_silence_ms / spec.frame_ms)
    minimum_speech_frames = math.ceil(spec.minimum_speech_ms / spec.frame_ms)
    padding_frames = math.ceil(spec.padding_ms / spec.frame_ms)
    intervals = [
        interval
        for interval in _merge_speech_frames(flags, maximum_gap_frames)
        if interval[1] - interval[0] >= minimum_speech_frames
    ]
    duration_ms = round(len(samples) * 1000 / spec.sample_rate_hz)
    speech_duration_ms = min(
        duration_ms,
        sum(end - start for start, end in intervals) * spec.frame_ms,
    )

    padded: list[tuple[int, int]] = []
    padded_sources: list[list[tuple[int, int]]] = []
    for start, end in intervals:
        candidate = (max(0, start - padding_frames), min(len(levels), end + padding_frames))
        if padded and candidate[0] <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], candidate[1]))
            padded_sources[-1].append((start, end))
        else:
            padded.append(candidate)
            padded_sources.append([(start, end)])

    segments: list[DetectedSpeechSegment] = []
    for (start, end), source_intervals in zip(padded, padded_sources, strict=True):
        start_ms = start * spec.frame_ms
        end_ms = min(duration_ms, end * spec.frame_ms)
        source_indexes = [
            index
            for source_start, source_end in source_intervals
            for index in range(source_start, source_end)
        ]
        speech_levels = [levels[index] for index in source_indexes if flags[index]]
        mean_dbfs = sum(speech_levels) / len(speech_levels) if speech_levels else -120.0
        if threshold_dbfs is None:
            confidence = sum(flags[index] for index in source_indexes) / len(source_indexes)
        else:
            confidence = min(0.99, max(0.5, 0.5 + (mean_dbfs - threshold_dbfs) / 40.0))
        segments.append(
            DetectedSpeechSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
                mean_dbfs=mean_dbfs,
            )
        )
    return VadResult(
        backend=spec.backend,
        sample_rate_hz=spec.sample_rate_hz,
        frame_ms=spec.frame_ms,
        threshold_dbfs=threshold_dbfs,
        aggressiveness=(spec.webrtc_aggressiveness if spec.backend == "webrtc_v1" else None),
        duration_ms=duration_ms,
        speech_duration_ms=speech_duration_ms,
        speech_ratio=speech_duration_ms / duration_ms if duration_ms else 0.0,
        segments=segments,
    )


def detect_voice_activity(
    path: str | Path,
    *,
    spec: VadSpec | None = None,
    ffmpeg: str = "ffmpeg",
    timeout: float = 300.0,
) -> VadResult:
    source = require_regular_file(path)
    executable = shutil.which(ffmpeg)
    if not executable:
        raise AudioToolUnavailable(f"executable not found: {ffmpeg}")
    spec = spec or VadSpec()
    samples = _decode_pcm(source, executable, spec.sample_rate_hz, timeout)
    frame_samples = spec.sample_rate_hz * spec.frame_ms // 1000
    levels = _frame_dbfs(samples, frame_samples)
    if spec.backend == "webrtc_v1":
        return _build_result(
            spec=spec,
            samples=samples,
            levels=levels,
            flags=_webrtc_flags(samples, spec),
            threshold_dbfs=None,
        )

    sorted_levels = sorted(levels)
    noise_floor = sorted_levels[int((len(sorted_levels) - 1) * 0.2)]
    dynamic_range = sorted_levels[-1] - noise_floor
    threshold = spec.minimum_speech_dbfs
    if dynamic_range >= 6.0:
        threshold = max(threshold, noise_floor + spec.noise_margin_db)

    return _build_result(
        spec=spec,
        samples=samples,
        levels=levels,
        flags=[level >= threshold for level in levels],
        threshold_dbfs=threshold,
    )
