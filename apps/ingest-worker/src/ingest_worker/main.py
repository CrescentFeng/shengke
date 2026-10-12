from __future__ import annotations

import argparse
import signal
import threading
from contextlib import contextmanager
from uuid import uuid4

from audio_core.contracts import Operation
from platform_api.postgres_store import PostgresJobStore

from .asr import build_asr_backend
from .diarization import build_diarization_backend
from .processor import IngestProcessor
from .s3_store import S3ArtifactStore
from .settings import WorkerSettings
from .tts import TtsProcessor


@contextmanager
def maintain_lease(metadata, job_id, worker_id, interval):
    done = threading.Event()

    def refresh():
        while not done.wait(interval):
            try:
                if not metadata.heartbeat(job_id, worker_id):
                    break
            except Exception:
                # Completion still checks ownership under a database row lock.
                break

    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-ingest-worker")
    parser.add_argument("--once", action="store_true", help="process at most one queued job")
    return parser


def run() -> None:
    args = build_parser().parse_args()
    settings = WorkerSettings.from_env()
    metadata = PostgresJobStore(settings.database_url)
    metadata.requeue_stale(settings.stale_after_seconds)
    artifacts = S3ArtifactStore(settings)
    artifacts.ensure_buckets()
    asr = build_asr_backend(
        settings.asr_backend,
        model_name=settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        beam_size=settings.asr_beam_size,
    )
    diarization = build_diarization_backend(
        settings.diarization_backend,
        model_name=settings.diarization_model,
        token=settings.huggingface_token,
        device=settings.diarization_device,
    )
    processor = IngestProcessor(
        metadata,
        artifacts,
        ffmpeg=settings.ffmpeg,
        ffprobe=settings.ffprobe,
        asr=asr,
        asr_language=settings.asr_language,
        diarization=diarization,
    )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while not stop.is_set():
        claim_id = f"{settings.worker_id}:{uuid4()}"
        job = metadata.claim_next(claim_id)
        if job is None:
            if args.once:
                return
            stop.wait(settings.poll_seconds)
            continue
        try:
            with maintain_lease(
                metadata, job.id, claim_id, min(30, settings.stale_after_seconds / 3)
            ):
                if Operation.SYNTHESIZE in job.operations:
                    TtsProcessor(metadata, artifacts, worker_id=claim_id).process(job)
                else:
                    processor.process(job)
        except Exception as exc:
            metadata.fail_claimed(job.id, claim_id, type(exc).__name__, str(exc)[:2000])
        if args.once:
            return


if __name__ == "__main__":
    run()
