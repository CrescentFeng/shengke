from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from audio_core.contracts import SpeakerTurn
from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiarizationBenchmarkSample(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    audio: Path
    reference_turns: list[SpeakerTurn] = Field(default_factory=list)
    rttm: Path | None = None
    recording_id: str | None = Field(default=None, min_length=1, max_length=256)
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rttm_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    audio_kind: Literal["real", "synthetic", "unknown"] = "unknown"
    annotation_kind: Literal["human", "derived", "synthetic", "unknown"] = "unknown"
    num_speakers: int | None = Field(default=None, ge=1, le=100)
    min_speakers: int | None = Field(default=None, ge=1, le=100)
    max_speakers: int | None = Field(default=None, ge=1, le=100)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def constraints_are_valid(self) -> DiarizationBenchmarkSample:
        if not self.reference_turns and self.rttm is None:
            raise ValueError("reference_turns or rttm is required")
        if self.annotation_kind == "human" and self.audio_kind != "real":
            raise ValueError("human annotations require audio_kind=real")
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError("num_speakers cannot be combined with speaker bounds")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers")
        if self.reference_turns != sorted(
            self.reference_turns,
            key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker),
        ):
            raise ValueError("reference_turns must be sorted")
        return self


class DiarizationSampleEvaluation(StrictModel):
    id: str
    audio: str
    timeline: Literal["regular", "exclusive"]
    success: bool
    error: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    inference_seconds: float = Field(default=0.0, ge=0.0)
    rtf: float | None = Field(default=None, ge=0.0)
    reference_speakers: int = Field(default=0, ge=0)
    hypothesis_speakers: int = Field(default=0, ge=0)
    reference_turns: int = Field(default=0, ge=0)
    hypothesis_turns: int = Field(default=0, ge=0)
    effective_num_speakers: int | None = Field(default=None, ge=1, le=100)
    effective_min_speakers: int | None = Field(default=None, ge=1, le=100)
    effective_max_speakers: int | None = Field(default=None, ge=1, le=100)
    speaker_count_correct: bool | None = None
    speaker_count_error: int | None = None
    scored_seconds: float = Field(default=0.0, ge=0.0)
    reference_speaker_seconds: float = Field(default=0.0, ge=0.0)
    false_alarm_seconds: float = Field(default=0.0, ge=0.0)
    missed_detection_seconds: float = Field(default=0.0, ge=0.0)
    confusion_seconds: float = Field(default=0.0, ge=0.0)
    der: float | None = Field(default=None, ge=0.0)
    jer: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_reference_count: int = Field(default=0, ge=0)
    boundary_matched_count: int = Field(default=0, ge=0)
    boundary_within_tolerance_count: int = Field(default=0, ge=0)
    boundary_errors_ms: list[float] = Field(default_factory=list)
    boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    boundary_p95_ms: float | None = Field(default=None, ge=0.0)
    boundary_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: dict[str, str] = Field(default_factory=dict)


class DiarizationBenchmarkSummary(StrictModel):
    sample_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    aggregate_der: float | None = Field(default=None, ge=0.0)
    mean_jer: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_rtf: float | None = Field(default=None, ge=0.0)
    speaker_count_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_absolute_speaker_count_error: float | None = Field(default=None, ge=0.0)
    boundary_reference_count: int = Field(default=0, ge=0)
    boundary_matched_count: int = Field(default=0, ge=0)
    boundary_within_tolerance_count: int = Field(default=0, ge=0)
    boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    boundary_p95_ms: float | None = Field(default=None, ge=0.0)
    boundary_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    false_alarm_seconds: float = Field(default=0.0, ge=0.0)
    missed_detection_seconds: float = Field(default=0.0, ge=0.0)
    confusion_seconds: float = Field(default=0.0, ge=0.0)
    reference_speaker_seconds: float = Field(default=0.0, ge=0.0)


class RuntimeInfo(StrictModel):
    platform: str
    python_version: str
    machine: str


class DiarizationBenchmarkReport(StrictModel):
    schema_version: str = "2.0"
    backend: str
    model: str
    backend_version: str
    manifest: str
    frame_ms: int = Field(gt=0)
    collar_ms: int = Field(ge=0)
    skip_overlap: bool
    use_exclusive: bool
    speaker_constraint: Literal["manifest", "oracle_num", "bounded", "unconstrained"] = (
        "manifest"
    )
    speaker_bound_padding: int = Field(default=1, ge=0, le=99)
    boundary_tolerance_ms: int = Field(default=250, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    runtime: RuntimeInfo
    summary: DiarizationBenchmarkSummary
    samples: list[DiarizationSampleEvaluation]


class DatasetAudit(StrictModel):
    sample_count: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0.0)
    real_audio_count: int = Field(ge=0)
    human_annotation_count: int = Field(ge=0)
    verified_real_human: bool


class ProductionThresholds(StrictModel):
    minimum_samples: int = Field(default=20, ge=1)
    minimum_duration_seconds: float = Field(default=1800.0, gt=0.0)
    maximum_der: float = Field(default=0.25, ge=0.0)
    maximum_jer: float = Field(default=0.35, ge=0.0, le=1.0)
    minimum_speaker_count_accuracy: float = Field(default=0.9, ge=0.0, le=1.0)
    maximum_boundary_p95_ms: float = Field(default=500.0, ge=0.0)
    maximum_mean_rtf: float = Field(default=0.75, ge=0.0)


class ConfigurationAssessment(StrictModel):
    name: str
    use_exclusive: bool
    speaker_constraint: str
    aggregate_der: float | None = None
    mean_jer: float | None = None
    speaker_count_accuracy: float | None = None
    boundary_p95_ms: float | None = None
    mean_rtf: float | None = None
    failed_count: int = Field(ge=0)
    meets_metric_thresholds: bool
    failures: list[str] = Field(default_factory=list)


class ProductionRecommendation(StrictModel):
    status: Literal["accepted", "rejected", "insufficient_data"]
    candidate_configuration: str | None = None
    reasons: list[str] = Field(default_factory=list)
    thresholds: ProductionThresholds


class DiarizationBenchmarkMatrixReport(StrictModel):
    schema_version: str = "1.0"
    backend: str
    model: str
    manifest: str
    created_at: datetime = Field(default_factory=utc_now)
    dataset: DatasetAudit
    reports: dict[str, DiarizationBenchmarkReport]
    assessments: list[ConfigurationAssessment]
    recommendation: ProductionRecommendation
