"""Independent Shengke release identity; does not change the shared platform version."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RELEASE = json.loads((ROOT / "configs/shengke/release.json").read_text())


def build_id():
    digest = hashlib.sha256()
    paths = set()
    for directory in (
        "apps/platform-api/src",
        "apps/ingest-worker/src",
        "apps/shengke-ui/src",
        "packages/audio-core/src",
        "services/tts-runtime",
    ):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "scripts").glob("shengke_*.py"))
    paths.add(ROOT / "configs/shengke/release.json")
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


BUILD_ID = build_id()


def job_summary(job):
    spec = job.parameters["tts"]
    return {
        "id": str(job.id),
        "created_at": job.created_at.isoformat(),
        "state": job.state.value,
        "text": spec.get("text", ""),
        "model": spec.get("model", "cosyvoice2"),
        "emotion": spec.get("emotion", "neutral"),
        "inference_profile": spec.get("inference_profile", "baseline"),
        "voice_id": spec.get("voice_id"),
        "error_message": job.error_message,
    }
