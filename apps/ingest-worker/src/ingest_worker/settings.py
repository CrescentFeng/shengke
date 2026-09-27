from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WorkerSettings:
    database_url: str
    s3_endpoint: str
    s3_region: str
    s3_access_key: str
    s3_secret_key: str
    artifact_bucket: str
    input_bucket: str
    s3_use_ssl: bool
    allowed_input_root: Path
    worker_id: str
    poll_seconds: float
    stale_after_seconds: int
    ffmpeg: str
    ffprobe: str
    asr_backend: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_beam_size: int
    asr_language: str | None
    diarization_backend: str
    diarization_model: str
    diarization_device: str
    huggingface_token: str | None

    @classmethod
    def from_env(cls) -> WorkerSettings:
        return cls(
            database_url=_required("AUDIO_PLATFORM_DATABASE_URL"),
            s3_endpoint=_required("AUDIO_PLATFORM_S3_ENDPOINT"),
            s3_region=os.getenv("AUDIO_PLATFORM_S3_REGION", "us-east-1"),
            s3_access_key=_required("AUDIO_PLATFORM_S3_ACCESS_KEY"),
            s3_secret_key=_required("AUDIO_PLATFORM_S3_SECRET_KEY"),
            artifact_bucket=os.getenv("AUDIO_PLATFORM_S3_ARTIFACT_BUCKET", "audio-artifacts"),
            input_bucket=os.getenv("AUDIO_PLATFORM_S3_INPUT_BUCKET", "audio-input"),
            s3_use_ssl=_bool("AUDIO_PLATFORM_S3_USE_SSL"),
            allowed_input_root=Path(os.getenv("AUDIO_PLATFORM_ALLOWED_INPUT_ROOT", "/data/input")),
            worker_id=os.getenv(
                "AUDIO_PLATFORM_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}"
            ),
            poll_seconds=float(os.getenv("AUDIO_PLATFORM_WORKER_POLL_SECONDS", "2")),
            stale_after_seconds=int(os.getenv("AUDIO_PLATFORM_WORKER_STALE_AFTER_SECONDS", "900")),
            ffmpeg=os.getenv("AUDIO_PLATFORM_FFMPEG", "ffmpeg"),
            ffprobe=os.getenv("AUDIO_PLATFORM_FFPROBE", "ffprobe"),
            asr_backend=os.getenv("AUDIO_PLATFORM_ASR_BACKEND", "disabled"),
            asr_model=os.getenv("AUDIO_PLATFORM_ASR_MODEL", "small"),
            asr_device=os.getenv("AUDIO_PLATFORM_ASR_DEVICE", "cpu"),
            asr_compute_type=os.getenv("AUDIO_PLATFORM_ASR_COMPUTE_TYPE", "int8"),
            asr_beam_size=int(os.getenv("AUDIO_PLATFORM_ASR_BEAM_SIZE", "5")),
            asr_language=os.getenv("AUDIO_PLATFORM_ASR_LANGUAGE") or None,
            diarization_backend=os.getenv(
                "AUDIO_PLATFORM_DIARIZATION_BACKEND", "disabled"
            ),
            diarization_model=os.getenv(
                "AUDIO_PLATFORM_DIARIZATION_MODEL",
                "pyannote/speaker-diarization-community-1",
            ),
            diarization_device=os.getenv("AUDIO_PLATFORM_DIARIZATION_DEVICE", "cpu"),
            huggingface_token=os.getenv("AUDIO_PLATFORM_HF_TOKEN") or None,
        )
