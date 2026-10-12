"""Run the local Shengke API, worker and optional UI with isolated model services."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment():
    env = os.environ.copy()
    defaults = {
        "AUDIO_PLATFORM_DATABASE_URL": "postgresql://audio:audio@127.0.0.1:55433/audio_platform",
        "AUDIO_PLATFORM_S3_ENDPOINT": "http://127.0.0.1:9000",
        "AUDIO_PLATFORM_S3_ACCESS_KEY": "audioadmin",
        "AUDIO_PLATFORM_S3_SECRET_KEY": "change-me-now",
        "AUDIO_PLATFORM_S3_REGION": "us-east-1",
        "AUDIO_PLATFORM_HOST": "127.0.0.1",
        "AUDIO_PLATFORM_PORT": "8000",
        "AUDIO_PLATFORM_ALLOWED_INPUT_ROOT": str(ROOT / "data/input"),
        "AUDIO_PLATFORM_ASR_BACKEND": "disabled",
        "AUDIO_PLATFORM_DIARIZATION_BACKEND": "disabled",
        "SHENGKE_MODEL_ENDPOINTS": '{"cosyvoice2":"http://127.0.0.1:9881"}',
        "PYTHONUNBUFFERED": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    bypass = env.get("no_proxy", env.get("NO_PROXY", ""))
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(filter(None, [bypass, "127.0.0.1", "localhost", "::1"]))
    sources = [
        "packages/audio-core/src",
        "apps/platform-api/src",
        "apps/ingest-worker/src",
        "apps/shengke-ui/src",
        "apps/tts-eval/src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / path) for path in sources)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ui-python", help="Python with Gradio 5 installed; omit to run API/worker only"
    )
    args = parser.parse_args()
    env = environment()
    # Fail before starting background processes when infrastructure is unavailable.
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    from ingest_worker.s3_store import S3ArtifactStore
    from ingest_worker.settings import WorkerSettings
    from platform_api.postgres_store import PostgresJobStore

    original = os.environ.copy()
    os.environ.update(env)
    try:
        db = PostgresJobStore(env["AUDIO_PLATFORM_DATABASE_URL"])
        db.ping()
        with db._connection() as conn:
            conn.execute("SELECT id FROM tts_voices LIMIT 1")
        S3ArtifactStore(WorkerSettings.from_env()).ensure_buckets()
    finally:
        os.environ.clear()
        os.environ.update(original)
    logs = ROOT / "artifacts/shengke/logs"
    logs.mkdir(parents=True, exist_ok=True)
    commands = [
        ("api", [sys.executable, "-m", "platform_api.main"]),
        ("worker", [sys.executable, "-m", "ingest_worker.main"]),
    ]
    if args.ui_python:
        commands.append(("ui", [args.ui_python, "-m", "shengke_ui.main"]))
    processes, handles = [], []

    def stop(_signal=None, _frame=None):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command in commands:
            handle = (logs / f"{name}.log").open("a")
            handles.append(handle)
            processes.append(
                (
                    name,
                    subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
                    ),
                )
            )
        print("Shengke API: http://127.0.0.1:8000/docs", flush=True)
        if args.ui_python:
            print("Shengke UI: http://127.0.0.1:7860", flush=True)
        print(f"Logs: {logs}", flush=True)
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited ({process.returncode}); inspect {logs / name}.log"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
