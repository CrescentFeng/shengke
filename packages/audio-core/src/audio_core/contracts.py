from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class JobState(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class Operation(StrEnum):
    PROBE = "probe"
    NORMALIZE = "normalize"
    VAD = "vad"
    LOUDNESS = "loudness"
    DENOISE = "denoise"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    CLASSIFY = "classify"
    SYNTHESIZE = "synthesize"


class ToolRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class MediaFormat(StrictModel):
    format_name: str = Field(min_length=1)
    format_long_name: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    bit_rate: int | None = Field(default=None, ge=0)


class AudioProbe(StrictModel):
    codec_name: str
    codec_long_name: str | None = None
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    channel_layout: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    bit_rate: int | None = Field(default=None, ge=0)
    sample_format: str | None = None
    media_format: MediaFormat


class AudioQualityMetrics(StrictModel):
    integrated_lufs: float | None = None
    loudness_range_lu: float | None = Field(default=None, ge=0)
    true_peak_dbtp: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    clipping_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    silence_duration_seconds: float = Field(default=0.0, ge=0.0)
    silence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_count: int | None = Field(default=None, ge=0)


class AudioAsset(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    uri: str = Field(min_length=1)
    filename: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    probe: AudioProbe | None = None
    source_asset_id: UUID | None = None
    created_at: datetime = Field(default_factory=utc_now)


class StoredArtifact(StrictModel):
    uri: str = Field(min_length=1)
    key: str = Field(min_length=1)
    content_type: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AudioSegment(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID
    slice_asset_id: UUID | None = None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    transcript: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    labels: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def interval_is_valid(self) -> AudioSegment:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        for label, confidence in self.labels.items():
            if not label:
                raise ValueError("segment labels must not be empty")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("segment label confidence must be between 0 and 1")
        return self


class TranscriptSpan(StrictModel):
    segment_id: UUID
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def interval_is_valid(self) -> TranscriptSpan:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class TranscriptDocument(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    backend: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    language: str | None = Field(default=None, max_length=32)
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    text: str
    spans: list[TranscriptSpan] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class TranscriptEnvelope(StrictModel):
    transcript_uri: str = Field(min_length=1)
    transcript: TranscriptDocument


class SpeakerTurn(StrictModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker: str = Field(min_length=1, max_length=128)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def interval_is_valid(self) -> SpeakerTurn:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class DiarizationDocument(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    asset_id: UUID
    backend: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)
    exclusive: bool = True
    speaker_count: int = Field(ge=0)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    regular_turns: list[SpeakerTurn] = Field(default_factory=list)
    exclusive_turns: list[SpeakerTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def speaker_count_matches_turns(self) -> DiarizationDocument:
        actual = len({turn.speaker for turn in self.turns})
        if self.speaker_count != actual:
            raise ValueError("speaker_count must equal the distinct speakers in turns")
        if self.turns != sorted(self.turns, key=lambda turn: (turn.start_ms, turn.end_ms)):
            raise ValueError("turns must be sorted by start_ms and end_ms")
        if self.regular_turns != sorted(
            self.regular_turns, key=lambda turn: (turn.start_ms, turn.end_ms)
        ):
            raise ValueError("regular_turns must be sorted by start_ms and end_ms")
        if self.exclusive_turns != sorted(
            self.exclusive_turns, key=lambda turn: (turn.start_ms, turn.end_ms)
        ):
            raise ValueError("exclusive_turns must be sorted by start_ms and end_ms")
        return self


class DiarizationEnvelope(StrictModel):
    diarization_uri: str = Field(min_length=1)
    diarization: DiarizationDocument


class ToolRun(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    tool_name: str = Field(min_length=1, max_length=128)
    tool_version: str = Field(min_length=1, max_length=128)
    operation: Operation
    status: ToolRunStatus = ToolRunStatus.PENDING
    input_asset_ids: list[UUID] = Field(default_factory=list)
    output_asset_ids: list[UUID] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> ToolRun:
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        return self


class Finding(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    category: str = Field(min_length=1, max_length=128)
    severity: FindingSeverity
    message: str = Field(min_length=1)
    segment_id: UUID | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class AudioReport(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    input_asset_id: UUID
    processing_chain_id: str = Field(min_length=1, max_length=256)
    assets: list[AudioAsset] = Field(default_factory=list)
    segments: list[AudioSegment] = Field(default_factory=list)
    tool_runs: list[ToolRun] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AudioReportEnvelope(StrictModel):
    report_uri: str = Field(min_length=1)
    report: AudioReport


class CreateAudioJob(StrictModel):
    input_uri: str = Field(min_length=1, max_length=4096)
    operations: list[Operation] = Field(min_length=1)
    policy_id: str = Field(default="default", min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def operations_are_unique(self) -> CreateAudioJob:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("operations must not contain duplicates")
        return self


class JobEvent(StrictModel):
    state: JobState
    at: datetime = Field(default_factory=utc_now)
    message: str | None = None


class AudioJob(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    input_uri: str
    operations: list[Operation]
    policy_id: str
    idempotency_key: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    state: JobState = JobState.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    events: list[JobEvent] = Field(default_factory=list)

    @classmethod
    def from_request(cls, request: CreateAudioJob) -> AudioJob:
        job = cls(**request.model_dump())
        job.events.append(JobEvent(state=JobState.PENDING, message="job created"))
        return job
