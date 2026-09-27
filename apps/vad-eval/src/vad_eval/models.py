from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from audio_core.vad import VadSpec
from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceInterval(StrictModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def interval_is_valid(self) -> ReferenceInterval:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class VadBenchmarkSample(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    audio: Path
    reference_segments: list[ReferenceInterval]
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def segments_are_ordered_and_non_overlapping(self) -> VadBenchmarkSample:
        for previous, current in zip(
            self.reference_segments, self.reference_segments[1:], strict=False
        ):
            if current.start_ms < previous.end_ms:
                raise ValueError("reference_segments must be ordered and non-overlapping")
        return self


class VadCandidate(StrictModel):
    name: str = Field(min_length=1, max_length=256)
    spec: VadSpec


class VadSampleEvaluation(StrictModel):
    candidate: str
    id: str
    audio: str
    success: bool
    error: str | None = None
    duration_ms: int = Field(default=0, ge=0)
    inference_seconds: float = Field(default=0.0, ge=0.0)
    rtf: float | None = Field(default=None, ge=0.0)
    reference_segments: int = Field(default=0, ge=0)
    predicted_segments: int = Field(default=0, ge=0)
    reference_speech_ms: int = Field(default=0, ge=0)
    predicted_speech_ms: int = Field(default=0, ge=0)
    true_positive_ms: int = Field(default=0, ge=0)
    false_positive_ms: int = Field(default=0, ge=0)
    false_negative_ms: int = Field(default=0, ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    iou: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    boundary_pairs: int = Field(default=0, ge=0)
    tags: dict[str, str] = Field(default_factory=dict)


class VadCandidateSummary(StrictModel):
    candidate: str
    backend: str
    parameters: dict[str, int | float | str | bool]
    sample_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    iou: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    mean_rtf: float | None = Field(default=None, ge=0.0)


class RuntimeInfo(StrictModel):
    platform: str
    python_version: str
    machine: str


class VadBenchmarkReport(StrictModel):
    schema_version: str = "1.0"
    manifest: str
    created_at: datetime = Field(default_factory=utc_now)
    runtime: RuntimeInfo
    best_candidate: str | None = None
    candidates: list[VadCandidateSummary]
    samples: list[VadSampleEvaluation]
