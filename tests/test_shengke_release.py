"""Release boundaries: history isolation, pagination and local service ownership."""

import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from audio_core.contracts import CreateAudioJob, JobState
from fastapi.testclient import TestClient
from platform_api.dependencies import get_store
from platform_api.main import app
from platform_api.store import InMemoryJobStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import shengke_service as service  # noqa: E402


def create(store, tts=True):
    spec = CreateAudioJob(
        input_uri="s3://test/reference.wav",
        idempotency_key=str(uuid4()),
        operations=["synthesize"] if tts else ["probe"],
        parameters={"tts": {"text": "你好", "reference_text": "你好"}} if tts else {},
    )
    return store.create(spec)[0]


def test_history_api_isolated_paginated_and_legacy_compatible():
    store = InMemoryJobStore()
    create(store, False)
    a, b, c = [create(store) for _ in range(3)]
    store.cancel(b.id)
    store.transition(c.id, JobState.VALIDATING)
    store.transition(c.id, JobState.FAILED, error_message="missing model")
    # Legacy rows may lack a profile/voice id; they must remain visible.
    store._jobs[a.id].parameters["tts"].pop("inference_profile", None)
    app.dependency_overrides[get_store] = lambda: store
    try:
        with TestClient(app) as client:
            page = client.get("/v1/tts/jobs?limit=2").json()
            assert page["total"] == 3
            assert [j["state"] for j in page["items"]] == ["failed", "cancelled"]
            tail = client.get("/v1/tts/jobs?limit=2&offset=2").json()
            assert tail["items"][0]["id"] == str(a.id)
            assert tail["items"][0]["inference_profile"] == "baseline"
            assert client.get("/v1/tts/jobs?offset=99").json()["items"] == []
            for query in ["limit=0", "limit=101", "offset=-1"]:
                assert client.get("/v1/tts/jobs?" + query).status_code == 422
            info = client.get("/v1/tts/info").json()
            assert info["version"] == "1.0.0"
            assert len(info["build_id"]) == 64
            assert info["instance_id"] == client.get("/v1/tts/info").json()["instance_id"]
            assert client.get("/openapi.json").json()["info"]["version"] == "0.9.0"
    finally:
        app.dependency_overrides.clear()


def test_stop_without_managed_instance_never_signals_process(tmp_path):
    with (
        patch.object(service, "RUNTIME", tmp_path),
        patch.object(service, "current", return_value={"stage": "stopped"}),
        patch.object(service, "control") as control,
        patch.object(service.sys, "argv", ["service", "stop"]),
    ):
        service.main()
        control.assert_not_called()


def test_busy_port_refuses_without_starting_docker_or_process(tmp_path):
    with (
        patch.object(service, "RUNTIME", tmp_path),
        patch.object(service, "current", return_value={"stage": "stopped"}),
        patch.object(service, "commands"),
        patch.object(service, "busy_ports", return_value=[7860]),
        patch.object(service, "ensure_docker") as docker,
        patch.object(service.subprocess, "Popen") as process,
        patch.object(service.sys, "argv", ["service", "start"]),
    ):
        with pytest.raises(RuntimeError, match="占用"):
            service.main()
        docker.assert_not_called()
        process.assert_not_called()


def test_missing_model_stops_before_launch():
    import json

    import shengke_stack

    config = json.loads((ROOT / "configs/shengke/local-macos.json").read_text())
    config["rhythm_weights"] = "/does-not-exist/shengke.pt"
    with pytest.raises(ValueError, match="不存在"):
        shengke_stack.commands(config)


def test_docker_unavailable_gives_recovery_message():
    with patch.object(service.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match="Docker"):
            service.ensure_docker({"PATH": ""})


def test_ready_instance_reused_without_new_process(tmp_path):
    with (
        patch.object(service, "RUNTIME", tmp_path),
        patch.object(service, "current", return_value={"stage": "ready", "instance_id": "owned"}),
        patch.object(service, "healthy", return_value=True),
        patch.object(service.subprocess, "Popen") as process,
        patch.object(service.sys, "argv", ["service", "start"]),
    ):
        service.main()
        process.assert_not_called()


def test_control_close_during_shutdown_is_not_json_error(tmp_path):
    with (
        patch.object(service, "SOCKET", tmp_path / "missing.sock"),
        patch.object(service, "control", side_effect=ConnectionResetError),
    ):
        assert service.current()["stage"] == "stopped"


def test_docker_timeout_does_not_start_stack():
    from types import SimpleNamespace

    with (
        patch.object(service.shutil, "which", return_value="/docker"),
        patch.object(service.sys, "platform", "darwin"),
        patch.object(service.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run,
        patch.object(service.time, "monotonic", side_effect=[0, 100]),
    ):
        with pytest.raises(RuntimeError, match="未就绪"):
            service.ensure_docker({"PATH": ""}, timeout=1)
        assert run.call_count == 2
