from __future__ import annotations

import os
from typing import Annotated
from uuid import UUID

from audio_core.contracts import (
    AudioAsset,
    AudioJob,
    AudioReportEnvelope,
    AudioSegment,
    CreateAudioJob,
    DiarizationEnvelope,
    TranscriptEnvelope,
)
from audio_core.errors import InvalidStateTransition
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from .dependencies import get_store
from .service import submit_job
from .store import JobNotFound, JobStore, ResultStore

app = FastAPI(
    title="Audio AI Platform",
    version="0.9.0",
    description="Shared job and contract layer for audio understanding, QC, and TTS.",
)


@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["system"])
def readyz() -> dict[str, str]:
    store = get_store()
    try:
        ready = store.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="metadata store is not ready") from exc
    if not ready:
        raise HTTPException(status_code=503, detail="metadata store is not ready")
    return {"status": "ready", "store": type(store).__name__}


@app.post("/v1/audio/jobs", response_model=AudioJob, tags=["jobs"])
def create_job(
    request: CreateAudioJob,
    response: Response,
    store: Annotated[JobStore, Depends(get_store)],
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AudioJob:
    if idempotency_key is not None and idempotency_key != request.idempotency_key:
        raise HTTPException(status_code=400, detail="header/body idempotency key mismatch")
    try:
        job, created = submit_job(store, request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return job


@app.get("/v1/audio/jobs/{job_id}", response_model=AudioJob, tags=["jobs"])
def get_job(job_id: UUID, store: Annotated[JobStore, Depends(get_store)]) -> AudioJob:
    try:
        return store.get(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.post("/v1/audio/jobs/{job_id}/cancel", response_model=AudioJob, tags=["jobs"])
def cancel_job(job_id: UUID, store: Annotated[JobStore, Depends(get_store)]) -> AudioJob:
    try:
        return store.cancel(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except InvalidStateTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/audio/jobs/{job_id}/assets", response_model=list[AudioAsset], tags=["jobs"])
def list_assets(job_id: UUID, store: Annotated[JobStore, Depends(get_store)]) -> list[AudioAsset]:
    if not isinstance(store, ResultStore):
        raise HTTPException(status_code=501, detail="persistent result store is not configured")
    try:
        return store.list_assets(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/v1/audio/jobs/{job_id}/segments", response_model=list[AudioSegment], tags=["jobs"])
def list_segments(
    job_id: UUID, store: Annotated[JobStore, Depends(get_store)]
) -> list[AudioSegment]:
    if not isinstance(store, ResultStore):
        raise HTTPException(status_code=501, detail="persistent result store is not configured")
    try:
        return store.list_segments(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get(
    "/v1/audio/jobs/{job_id}/transcript",
    response_model=TranscriptEnvelope,
    tags=["jobs"],
)
def get_transcript(
    job_id: UUID, store: Annotated[JobStore, Depends(get_store)]
) -> TranscriptEnvelope:
    if not isinstance(store, ResultStore):
        raise HTTPException(status_code=501, detail="persistent result store is not configured")
    try:
        return store.get_transcript(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="transcript not found") from exc


@app.get(
    "/v1/audio/jobs/{job_id}/diarization",
    response_model=DiarizationEnvelope,
    tags=["jobs"],
)
def get_diarization(
    job_id: UUID, store: Annotated[JobStore, Depends(get_store)]
) -> DiarizationEnvelope:
    if not isinstance(store, ResultStore):
        raise HTTPException(status_code=501, detail="persistent result store is not configured")
    try:
        return store.get_diarization(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="diarization not found") from exc


@app.get(
    "/v1/audio/jobs/{job_id}/report",
    response_model=AudioReportEnvelope,
    tags=["jobs"],
)
def get_report(job_id: UUID, store: Annotated[JobStore, Depends(get_store)]) -> AudioReportEnvelope:
    if not isinstance(store, ResultStore):
        raise HTTPException(status_code=501, detail="persistent result store is not configured")
    try:
        return store.get_report(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="report not found") from exc


def run() -> None:
    import uvicorn

    uvicorn.run(
        "platform_api.main:app",
        host=os.getenv("AUDIO_PLATFORM_HOST", "0.0.0.0"),
        port=int(os.getenv("AUDIO_PLATFORM_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
