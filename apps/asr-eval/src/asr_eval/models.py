from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceSpan(StrictModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def interval_is_valid(self) -> ReferenceSpan:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class BenchmarkSample(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    audio: Path
    reference_text: str
    language: str | None = Field(default=None, max_length=32)
    reference_spans: list[ReferenceSpan] = Field(default_factory=list)
    hotwords: list[str] = Field(default_factory=list, max_length=100)
    tags: dict[str, str] = Field(default_factory=dict)


class SampleEvaluation(StrictModel):
    id: str
    audio: str
    success: bool
    error: str | None = None
    reference_text: str
    hypothesis_text: str = ""
    reference_language: str | None = None
    detected_language: str | None = None
    language_correct: bool | None = None
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    inference_seconds: float = Field(ge=0.0)
    rtf: float | None = Field(default=None, ge=0.0)
    word_errors: int = Field(default=0, ge=0)
    reference_words: int = Field(default=0, ge=0)
    wer: float | None = Field(default=None, ge=0.0)
    character_errors: int = Field(default=0, ge=0)
    reference_characters: int = Field(default=0, ge=0)
    cer: float | None = Field(default=None, ge=0.0)
    timestamp_boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    timestamp_pairs: int = Field(default=0, ge=0)
    peak_process_memory_mb: float | None = Field(default=None, ge=0.0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)
    tags: dict[str, str] = Field(default_factory=dict)


class BenchmarkSummary(StrictModel):
    sample_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    total_audio_seconds: float = Field(ge=0.0)
    total_inference_seconds: float = Field(ge=0.0)
    aggregate_wer: float | None = Field(default=None, ge=0.0)
    aggregate_cer: float | None = Field(default=None, ge=0.0)
    mean_rtf: float | None = Field(default=None, ge=0.0)
    p95_rtf: float | None = Field(default=None, ge=0.0)
    language_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp_boundary_mae_ms: float | None = Field(default=None, ge=0.0)
    peak_process_memory_mb: float | None = Field(default=None, ge=0.0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)


class RuntimeInfo(StrictModel):
    platform: str
    python_version: str
    machine: str
    processor: str


class AsrBenchmarkReport(StrictModel):
    schema_version: str = "1.0"
    backend: str
    model: str
    backend_version: str
    manifest: str
    created_at: datetime = Field(default_factory=utc_now)
    runtime: RuntimeInfo
    summary: BenchmarkSummary
    samples: list[SampleEvaluation]
