from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from audio_core.contracts import (
    AudioAsset,
    AudioJob,
    AudioReport,
    AudioReportEnvelope,
    AudioSegment,
    CreateAudioJob,
    DiarizationDocument,
    DiarizationEnvelope,
    JobEvent,
    JobState,
    ToolRun,
    TranscriptDocument,
    TranscriptEnvelope,
)
from audio_core.errors import InvalidStateTransition

from .store import ALLOWED_TRANSITIONS, JobNotFound


class PostgresDependencyError(RuntimeError):
    pass


class PostgresJobStore:
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise PostgresDependencyError(
                "psycopg is required for PostgreSQL persistence; install project dependencies"
            ) from exc
        self._psycopg = psycopg
        self._dict_row = dict_row
        self._dsn = dsn

    def ping(self) -> bool:
        with self._connection() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._psycopg.connect(self._dsn, row_factory=self._dict_row) as connection:
            yield connection

    @staticmethod
    def _row_to_job(row: dict[str, Any], events: list[dict[str, Any]]) -> AudioJob:
        return AudioJob(
            id=row["id"],
            input_uri=row["input_uri"],
            operations=row["operations"],
            policy_id=row["policy_id"],
            idempotency_key=row["idempotency_key"],
            parameters=row["parameters"],
            state=row["state"],
            progress=float(row["progress"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            events=[
                JobEvent(state=event["state"], at=event["created_at"], message=event["message"])
                for event in events
            ],
        )

    def _load_events(self, connection: Any, job_id: UUID) -> list[dict[str, Any]]:
        return connection.execute(
            """
            SELECT state, message, created_at
            FROM job_events
            WHERE job_id = %s
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()

    def create(self, request: CreateAudioJob) -> tuple[AudioJob, bool]:
        payload = request.model_dump(mode="json")
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO audio_jobs (
                    input_uri, operations, policy_id, idempotency_key, parameters, state
                ) VALUES (%s, %s::jsonb, %s, %s, %s::jsonb, 'pending')
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING *
                """,
                (
                    request.input_uri,
                    json.dumps(payload["operations"]),
                    request.policy_id,
                    request.idempotency_key,
                    json.dumps(payload["parameters"]),
                ),
            ).fetchone()
            created = row is not None
            if created:
                connection.execute(
                    "INSERT INTO job_events (job_id, state, message) VALUES (%s, 'pending', %s)",
                    (row["id"], "job created"),
                )
            else:
                row = connection.execute(
                    "SELECT * FROM audio_jobs WHERE idempotency_key = %s",
                    (request.idempotency_key,),
                ).fetchone()
                assert row is not None
                if (
                    row["input_uri"] != request.input_uri
                    or row["operations"] != payload["operations"]
                    or row["policy_id"] != request.policy_id
                    or row["parameters"] != payload["parameters"]
                ):
                    raise ValueError("idempotency key is already used for a different request")
            events = self._load_events(connection, row["id"])
            return self._row_to_job(row, events), created

    def get(self, job_id: UUID) -> AudioJob:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM audio_jobs WHERE id = %s", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            return self._row_to_job(row, self._load_events(connection, job_id))

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
        with self._connection() as connection:
            current = connection.execute(
                "SELECT state FROM audio_jobs WHERE id = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(str(job_id))
            current_state = JobState(current["state"])
            if target not in ALLOWED_TRANSITIONS[current_state]:
                raise InvalidStateTransition(f"cannot transition from {current_state} to {target}")
            next_progress = 1.0 if target == JobState.SUCCEEDED else progress
            row = connection.execute(
                """
                UPDATE audio_jobs
                SET state = %s,
                    progress = COALESCE(%s, progress),
                    error_code = %s,
                    error_message = %s,
                    updated_at = now(),
                    locked_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE locked_at
                    END,
                    locked_by = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE locked_by
                    END
                WHERE id = %s
                RETURNING *
                """,
                (
                    target.value,
                    next_progress,
                    error_code,
                    error_message,
                    target.value,
                    target.value,
                    job_id,
                ),
            ).fetchone()
            connection.execute(
                "INSERT INTO job_events (job_id, state, message) VALUES (%s, %s, %s)",
                (job_id, target.value, message),
            )
            assert row is not None
            return self._row_to_job(row, self._load_events(connection, job_id))

    def cancel(self, job_id: UUID, *, message: str = "cancel requested") -> AudioJob:
        job = self.get(job_id)
        if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            return job
        return self.transition(job_id, JobState.CANCELLED, message=message)

    def claim_next(self, worker_id: str) -> AudioJob | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                WITH candidate AS (
                    SELECT id
                    FROM audio_jobs
                    WHERE state = 'queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE audio_jobs AS jobs
                SET state = 'running',
                    locked_at = now(),
                    locked_by = %s,
                    updated_at = now()
                FROM candidate
                WHERE jobs.id = candidate.id
                RETURNING jobs.*
                """,
                (worker_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "INSERT INTO job_events (job_id, state, message) VALUES (%s, 'running', %s)",
                (row["id"], f"claimed by {worker_id}"),
            )
            return self._row_to_job(row, self._load_events(connection, row["id"]))

    def requeue_stale(self, stale_after_seconds: int) -> int:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        with self._connection() as connection:
            rows = connection.execute(
                """
                UPDATE audio_jobs
                SET state = 'queued',
                    locked_at = NULL,
                    locked_by = NULL,
                    updated_at = now()
                WHERE state = 'running'
                  AND locked_at < now() - make_interval(secs => %s)
                RETURNING id
                """,
                (stale_after_seconds,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO job_events (job_id, state, message)
                    VALUES (%s, 'queued', 'stale worker lease recovered')
                    """,
                    (row["id"],),
                )
            return len(rows)

    def add_asset(self, job_id: UUID, asset: AudioAsset, role: str) -> None:
        payload = asset.model_dump(mode="json")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audio_assets (
                    id, job_id, role, uri, filename, sha256, size_bytes, probe,
                    source_asset_id, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    asset.id,
                    job_id,
                    role,
                    asset.uri,
                    asset.filename,
                    asset.sha256,
                    asset.size_bytes,
                    json.dumps(payload["probe"]),
                    asset.source_asset_id,
                    asset.created_at,
                ),
            )

    def add_tool_run(self, job_id: UUID, run: ToolRun) -> None:
        payload = run.model_dump(mode="json")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tool_runs (
                    id, job_id, tool_name, tool_version, operation, status,
                    input_asset_ids, output_asset_ids, parameters, metrics,
                    started_at, finished_at, error_code, error_message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    output_asset_ids = EXCLUDED.output_asset_ids,
                    metrics = EXCLUDED.metrics,
                    finished_at = EXCLUDED.finished_at,
                    error_code = EXCLUDED.error_code,
                    error_message = EXCLUDED.error_message
                """,
                (
                    run.id,
                    job_id,
                    run.tool_name,
                    run.tool_version,
                    run.operation.value,
                    run.status.value,
                    json.dumps(payload["input_asset_ids"]),
                    json.dumps(payload["output_asset_ids"]),
                    json.dumps(payload["parameters"]),
                    json.dumps(payload["metrics"]),
                    run.started_at,
                    run.finished_at,
                    run.error_code,
                    run.error_message,
                ),
            )

    def add_segment(self, job_id: UUID, segment: AudioSegment) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audio_segments (
                    id, job_id, asset_id, slice_asset_id, start_ms, end_ms,
                    speaker, language, transcript, confidence, labels
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    slice_asset_id = EXCLUDED.slice_asset_id,
                    speaker = EXCLUDED.speaker,
                    language = EXCLUDED.language,
                    transcript = EXCLUDED.transcript,
                    confidence = EXCLUDED.confidence,
                    labels = EXCLUDED.labels
                """,
                (
                    segment.id,
                    job_id,
                    segment.asset_id,
                    segment.slice_asset_id,
                    segment.start_ms,
                    segment.end_ms,
                    segment.speaker,
                    segment.language,
                    segment.transcript,
                    segment.confidence,
                    json.dumps(segment.labels),
                ),
            )

    def save_report(self, report: AudioReport, report_uri: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audio_reports (
                    id, job_id, input_asset_id, processing_chain_id, report_uri, payload
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (job_id) DO UPDATE SET
                    processing_chain_id = EXCLUDED.processing_chain_id,
                    report_uri = EXCLUDED.report_uri,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    report.id,
                    report.job_id,
                    report.input_asset_id,
                    report.processing_chain_id,
                    report_uri,
                    report.model_dump_json(),
                ),
            )

    def save_transcript(self, transcript: TranscriptDocument, transcript_uri: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audio_transcripts (
                    id, job_id, backend, model, language, transcript_uri, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    backend = EXCLUDED.backend,
                    model = EXCLUDED.model,
                    language = EXCLUDED.language,
                    transcript_uri = EXCLUDED.transcript_uri,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    transcript.id,
                    transcript.job_id,
                    transcript.backend,
                    transcript.model,
                    transcript.language,
                    transcript_uri,
                    transcript.model_dump_json(),
                    transcript.created_at,
                ),
            )

    def save_diarization(
        self, diarization: DiarizationDocument, diarization_uri: str
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audio_diarizations (
                    id, job_id, input_asset_id, backend, model, exclusive,
                    speaker_count, diarization_uri, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    backend = EXCLUDED.backend,
                    model = EXCLUDED.model,
                    exclusive = EXCLUDED.exclusive,
                    speaker_count = EXCLUDED.speaker_count,
                    diarization_uri = EXCLUDED.diarization_uri,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    diarization.id,
                    diarization.job_id,
                    diarization.asset_id,
                    diarization.backend,
                    diarization.model,
                    diarization.exclusive,
                    diarization.speaker_count,
                    diarization_uri,
                    diarization.model_dump_json(),
                    diarization.created_at,
                ),
            )

    def list_assets(self, job_id: UUID) -> list[AudioAsset]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM audio_jobs WHERE id = %s", (job_id,)
            ).fetchone()
            if exists is None:
                raise JobNotFound(str(job_id))
            rows = connection.execute(
                """
                SELECT id, uri, filename, sha256, size_bytes, probe, source_asset_id, created_at
                FROM audio_assets
                WHERE job_id = %s
                ORDER BY created_at, id
                """,
                (job_id,),
            ).fetchall()
            return [AudioAsset(**row) for row in rows]

    def list_segments(self, job_id: UUID) -> list[AudioSegment]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM audio_jobs WHERE id = %s", (job_id,)
            ).fetchone()
            if exists is None:
                raise JobNotFound(str(job_id))
            rows = connection.execute(
                """
                SELECT id, asset_id, slice_asset_id, start_ms, end_ms, speaker,
                       language, transcript, confidence, labels
                FROM audio_segments
                WHERE job_id = %s
                ORDER BY start_ms, end_ms, id
                """,
                (job_id,),
            ).fetchall()
            return [AudioSegment(**row) for row in rows]

    def get_report(self, job_id: UUID) -> AudioReportEnvelope:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT report_uri, payload FROM audio_reports WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            return AudioReportEnvelope(
                report_uri=row["report_uri"],
                report=AudioReport.model_validate(row["payload"]),
            )

    def get_transcript(self, job_id: UUID) -> TranscriptEnvelope:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT transcript_uri, payload FROM audio_transcripts WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            return TranscriptEnvelope(
                transcript_uri=row["transcript_uri"],
                transcript=TranscriptDocument.model_validate(row["payload"]),
            )

    def get_diarization(self, job_id: UUID) -> DiarizationEnvelope:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT diarization_uri, payload FROM audio_diarizations WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            return DiarizationEnvelope(
                diarization_uri=row["diarization_uri"],
                diarization=DiarizationDocument.model_validate(row["payload"]),
            )
