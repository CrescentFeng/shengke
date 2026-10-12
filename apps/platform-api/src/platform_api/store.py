from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol, runtime_checkable
from uuid import UUID

from audio_core.contracts import (
    TERMINAL_STATES,
    AudioAsset,
    AudioJob,
    AudioReportEnvelope,
    AudioSegment,
    CreateAudioJob,
    DiarizationEnvelope,
    JobEvent,
    JobState,
    TranscriptEnvelope,
)
from audio_core.errors import InvalidStateTransition

ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.PENDING: {JobState.VALIDATING, JobState.CANCELLED},
    JobState.VALIDATING: {JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
    JobState.QUEUED: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


class JobNotFound(KeyError):
    pass


class JobStore(Protocol):
    def ping(self) -> bool: ...

    def create(self, request: CreateAudioJob) -> tuple[AudioJob, bool]: ...

    def get(self, job_id: UUID) -> AudioJob: ...

    def list_tts_jobs(self, limit: int, offset: int) -> dict: ...

    def transition(
        self,
        job_id: UUID,
        target: JobState,
        *,
        message: str | None = None,
        progress: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> AudioJob: ...

    def cancel(self, job_id: UUID, *, message: str = "cancel requested") -> AudioJob: ...


@runtime_checkable
class ResultStore(Protocol):
    def list_assets(self, job_id: UUID) -> list[AudioAsset]: ...

    def list_segments(self, job_id: UUID) -> list[AudioSegment]: ...

    def get_transcript(self, job_id: UUID) -> TranscriptEnvelope: ...

    def get_diarization(self, job_id: UUID) -> DiarizationEnvelope: ...

    def get_report(self, job_id: UUID) -> AudioReportEnvelope: ...


class InMemoryJobStore:
    """Development-only store with idempotency and guarded transitions."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, AudioJob] = {}
        self._idempotency: dict[str, UUID] = {}
        self._lock = RLock()

    def ping(self) -> bool:
        return True

    def list_tts_jobs(self, limit: int, offset: int) -> dict:
        from audio_core.shengke_release import job_summary

        with self._lock:
            jobs = sorted(
                (
                    j
                    for j in self._jobs.values()
                    if "tts" in j.parameters and "synthesize" in j.operations
                ),
                key=lambda j: (j.created_at, str(j.id)),
                reverse=True,
            )
            return {
                "items": [job_summary(j) for j in jobs[offset : offset + limit]],
                "total": len(jobs),
                "limit": limit,
                "offset": offset,
            }

    def create(self, request: CreateAudioJob) -> tuple[AudioJob, bool]:
        with self._lock:
            existing_id = self._idempotency.get(request.idempotency_key)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if (
                    existing.input_uri != request.input_uri
                    or existing.operations != request.operations
                    or existing.policy_id != request.policy_id
                    or existing.parameters != request.parameters
                ):
                    raise ValueError("idempotency key is already used for a different request")
                return deepcopy(existing), False
            job = AudioJob.from_request(request)
            self._jobs[job.id] = job
            self._idempotency[job.idempotency_key] = job.id
            return deepcopy(job), True

    def get(self, job_id: UUID) -> AudioJob:
        with self._lock:
            try:
                return deepcopy(self._jobs[job_id])
            except KeyError as exc:
                raise JobNotFound(str(job_id)) from exc

    def transition(
        self,
        job_id: UUID,
        target: JobState,
        *,
        message: str | None = None,
        progress: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> AudioJob:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFound(str(job_id))
            job = self._jobs[job_id]
            if target not in ALLOWED_TRANSITIONS[job.state]:
                raise InvalidStateTransition(f"cannot transition from {job.state} to {target}")
            now = datetime.now(UTC)
            job.state = target
            job.updated_at = now
            if progress is not None:
                job.progress = progress
            elif target == JobState.SUCCEEDED:
                job.progress = 1.0
            job.error_code = error_code
            job.error_message = error_message
            job.events.append(JobEvent(state=target, at=now, message=message))
            return deepcopy(job)

    def cancel(self, job_id: UUID, *, message: str = "cancel requested") -> AudioJob:
        job = self.get(job_id)
        if job.state in TERMINAL_STATES:
            return job
        return self.transition(job_id, JobState.CANCELLED, message=message)
