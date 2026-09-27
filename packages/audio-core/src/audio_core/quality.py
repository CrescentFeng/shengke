from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import AudioQualityMetrics, Finding, FindingSeverity
from .errors import AudioAnalysisError, AudioToolUnavailable
from .validation import require_regular_file

_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+|inf|nan)"


class QualityAnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    silence_noise_db: float = Field(default=-50.0, ge=-100.0, le=0.0)
    minimum_silence_seconds: float = Field(default=0.5, gt=0.0, le=60.0)
    clipping_peak_dbfs: float = Field(default=-0.1, ge=-3.0, le=0.0)


class AudioQualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_integrated_lufs: float = Field(default=-24.0, ge=-70.0, le=0.0)
    maximum_integrated_lufs: float = Field(default=-14.0, ge=-70.0, le=0.0)
    maximum_true_peak_dbtp: float = Field(default=-1.0, ge=-20.0, le=0.0)
    maximum_clipping_ratio: float = Field(default=0.0001, ge=0.0, le=1.0)
    maximum_silence_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def loudness_range_is_valid(self) -> AudioQualityPolicy:
        if self.minimum_integrated_lufs >= self.maximum_integrated_lufs:
            raise ValueError("minimum_integrated_lufs must be below maximum_integrated_lufs")
        return self


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _last_metric(output: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}:\s*({_NUMBER})", output, flags=re.IGNORECASE)
    return _finite_float(matches[-1]) if matches else None


def _run_filter(source: Path, audio_filter: str, executable: str, timeout: float) -> str:
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-af",
        audio_filter,
        "-f",
        "null",
        "-",
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
        raise AudioAnalysisError(f"ffmpeg analysis timed out after {timeout}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise AudioAnalysisError(f"ffmpeg analysis failed: {detail or 'unknown error'}")
    return completed.stderr


def _parse_loudnorm(output: str) -> tuple[float | None, float | None, float | None]:
    matches = re.findall(r'\{\s*"input_i".*?\}', output, flags=re.DOTALL)
    if not matches:
        raise AudioAnalysisError("ffmpeg loudnorm returned no JSON measurements")
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise AudioAnalysisError("ffmpeg loudnorm returned invalid JSON") from exc
    return (
        _finite_float(payload.get("input_i")),
        _finite_float(payload.get("input_lra")),
        _finite_float(payload.get("input_tp")),
    )


def analyze_quality(
    path: str | Path,
    *,
    duration_seconds: float,
    spec: QualityAnalysisSpec | None = None,
    ffmpeg: str = "ffmpeg",
    timeout: float = 300.0,
) -> AudioQualityMetrics:
    source = require_regular_file(path)
    executable = shutil.which(ffmpeg)
    if not executable:
        raise AudioToolUnavailable(f"executable not found: {ffmpeg}")
    if duration_seconds <= 0:
        raise AudioAnalysisError("duration_seconds must be positive")
    spec = spec or QualityAnalysisSpec()

    loudness_output = _run_filter(
        source,
        "loudnorm=I=-23:LRA=7:TP=-2:print_format=json",
        executable,
        timeout,
    )
    integrated_lufs, loudness_range_lu, true_peak_dbtp = _parse_loudnorm(loudness_output)

    stats_output = _run_filter(source, "astats=metadata=1:reset=0", executable, timeout)
    peak_dbfs = _last_metric(stats_output, "Peak level dB")
    rms_dbfs = _last_metric(stats_output, "RMS level dB")
    peak_count = _last_metric(stats_output, "Abs Peak count") or 0.0
    sample_count_value = _last_metric(stats_output, "Number of samples")
    sample_count = int(sample_count_value) if sample_count_value is not None else None
    clipping_ratio = 0.0
    if peak_dbfs is not None and peak_dbfs >= spec.clipping_peak_dbfs and sample_count:
        clipping_ratio = min(1.0, peak_count / sample_count)

    silence_filter = (
        f"silencedetect=noise={spec.silence_noise_db:g}dB:d={spec.minimum_silence_seconds:g}"
    )
    silence_output = _run_filter(source, silence_filter, executable, timeout)
    durations = re.findall(rf"silence_duration:\s*({_NUMBER})", silence_output)
    silence_duration = min(
        duration_seconds,
        sum(value for item in durations if (value := _finite_float(item)) is not None),
    )

    return AudioQualityMetrics(
        integrated_lufs=integrated_lufs,
        loudness_range_lu=loudness_range_lu,
        true_peak_dbtp=true_peak_dbtp,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_duration_seconds=silence_duration,
        silence_ratio=silence_duration / duration_seconds,
        sample_count=sample_count,
    )


def evaluate_quality(
    asset_id: UUID,
    metrics: AudioQualityMetrics,
    policy: AudioQualityPolicy | None = None,
) -> list[Finding]:
    policy = policy or AudioQualityPolicy()
    findings: list[Finding] = []
    loudness = metrics.integrated_lufs
    if loudness is not None and loudness < policy.minimum_integrated_lufs:
        findings.append(
            Finding(
                category="integrated_loudness_low",
                severity=FindingSeverity.WARNING,
                message="Integrated loudness is below the configured minimum.",
                evidence={
                    "asset_id": str(asset_id),
                    "measured_lufs": loudness,
                    "minimum_lufs": policy.minimum_integrated_lufs,
                },
            )
        )
    if loudness is not None and loudness > policy.maximum_integrated_lufs:
        findings.append(
            Finding(
                category="integrated_loudness_high",
                severity=FindingSeverity.WARNING,
                message="Integrated loudness is above the configured maximum.",
                evidence={
                    "asset_id": str(asset_id),
                    "measured_lufs": loudness,
                    "maximum_lufs": policy.maximum_integrated_lufs,
                },
            )
        )
    if (
        metrics.true_peak_dbtp is not None
        and metrics.true_peak_dbtp > policy.maximum_true_peak_dbtp
    ):
        findings.append(
            Finding(
                category="true_peak_high",
                severity=FindingSeverity.HIGH,
                message="True peak exceeds the configured ceiling.",
                evidence={
                    "asset_id": str(asset_id),
                    "measured_dbtp": metrics.true_peak_dbtp,
                    "maximum_dbtp": policy.maximum_true_peak_dbtp,
                },
            )
        )
    if metrics.clipping_ratio > policy.maximum_clipping_ratio:
        findings.append(
            Finding(
                category="clipping_detected",
                severity=FindingSeverity.HIGH,
                message="Clipped samples exceed the configured ratio.",
                evidence={
                    "asset_id": str(asset_id),
                    "measured_ratio": metrics.clipping_ratio,
                    "maximum_ratio": policy.maximum_clipping_ratio,
                },
            )
        )
    if metrics.silence_ratio > policy.maximum_silence_ratio:
        findings.append(
            Finding(
                category="excessive_silence",
                severity=FindingSeverity.WARNING,
                message="Silence exceeds the configured ratio.",
                evidence={
                    "asset_id": str(asset_id),
                    "measured_ratio": metrics.silence_ratio,
                    "maximum_ratio": policy.maximum_silence_ratio,
                },
            )
        )
    return findings
