from __future__ import annotations

import argparse
import signal
import threading

from audio_core.contracts import TERMINAL_STATES, JobState
from platform_api.postgres_store import PostgresJobStore

from .asr import build_asr_backend
from .diarization import build_diarization_backend
from .processor import IngestProcessor
from .s3_store import S3ArtifactStore
from .settings import WorkerSettings


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
        job = metadata.claim_next(settings.worker_id)
        if job is None:
            if args.once:
                return
            stop.wait(settings.poll_seconds)
            continue
        try:
            processor.process(job)
        except Exception as exc:
            current = metadata.get(job.id)
            if current.state not in TERMINAL_STATES:
                metadata.transition(
                    job.id,
                    JobState.FAILED,
                    message="ingest pipeline failed",
                    error_code=type(exc).__name__,
                    error_message=str(exc)[:2000],
                )
        if args.once:
            return


if __name__ == "__main__":
    run()
