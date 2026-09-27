from __future__ import annotations

from audio_core.contracts import AudioJob, CreateAudioJob, JobState

from .store import JobStore


def submit_job(store: JobStore, request: CreateAudioJob) -> tuple[AudioJob, bool]:
    job, created = store.create(request)
    if not created:
        return job, False
    job = store.transition(job.id, JobState.VALIDATING, message="request validated")
    job = store.transition(job.id, JobState.QUEUED, message="job queued")
    return job, True
