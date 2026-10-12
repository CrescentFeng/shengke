from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid5

from audio_core.contracts import (
    AudioAsset,
    AudioReport,
    JobState,
    Operation,
    ToolRun,
    ToolRunStatus,
)
from audio_core.hashing import sha256_file
from audio_core.probe import probe_file
from audio_core.quality import analyze_quality
from audio_core.tts import MODEL_IDS, TtsSynthesisSpec
from audio_core.tts_quality import inspect_reference


class TtsBackendError(RuntimeError):
    pass


def model_endpoints() -> dict[str, str]:
    value = json.loads(os.getenv("SHENGKE_MODEL_ENDPOINTS", "{}"))
    if not isinstance(value, dict) or not set(value) <= set(MODEL_IDS):
        raise ValueError("SHENGKE_MODEL_ENDPOINTS must map supported model IDs to HTTP URLs")
    for url in value.values():
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("model endpoint must be an HTTP URL")
    return value


class HttpTtsBackend:
    def __init__(self, endpoints: dict[str, str] | None = None, timeout: float = 600):
        self.endpoints = endpoints if endpoints is not None else model_endpoints()
        self.timeout = timeout

    def synthesize(self, spec: TtsSynthesisSpec, reference: Path, output: Path) -> dict:
        endpoint = self.endpoints.get(spec.model)
        if not endpoint:
            raise TtsBackendError(f"模型 {spec.model} 尚未配置推理服务")
        payload = {
            "spec": spec.model_dump(mode="json"),
            "reference_base64": base64.b64encode(reference.read_bytes()).decode(),
        }
        headers = {"Content-Type": "application/json"}
        token = os.getenv("SHENGKE_MODEL_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/synthesize",
            json.dumps(payload).encode(),
            headers,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise TtsBackendError("模型响应过大")
            result = json.loads(raw)
            audio = base64.b64decode(result["audio_base64"], validate=True)
            metrics = result["metrics"]
            if not audio or metrics.get("model") != spec.model:
                raise TtsBackendError("模型返回的音频或模型标识无效")
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode(errors="replace")
            raise TtsBackendError(f"推理服务错误 HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError, KeyError) as exc:
            raise TtsBackendError(f"推理服务不可用或返回格式无效: {exc}") from exc
        output.write_bytes(audio)
        return {**metrics, "request_elapsed_s": time.perf_counter() - started}


class TtsProcessor:
    def __init__(self, metadata, artifacts, backend=None, *, worker_id: str):
        self.metadata = metadata
        self.artifacts = artifacts
        self.backend = backend or HttpTtsBackend()
        self.worker_id = worker_id

    def process(self, job):
        spec = TtsSynthesisSpec.model_validate(job.parameters["tts"])
        started = datetime.now(UTC)
        with tempfile.TemporaryDirectory(prefix="shengke-") as directory:
            root = Path(directory)
            reference = self.artifacts.fetch_input(job.input_uri, root / "reference.wav")
            ref_probe, ref_quality, warnings = inspect_reference(reference)
            ref_hash = sha256_file(reference)
            original = AudioAsset(
                id=uuid5(job.id, "tts:reference"),
                uri=job.input_uri,
                filename="reference.wav",
                sha256=ref_hash,
                size_bytes=reference.stat().st_size,
                probe=ref_probe,
            )
            output = root / "speech.wav"
            metrics = self.backend.synthesize(spec, reference, output)
            probe = probe_file(output)
            if not probe.duration_seconds or probe.duration_seconds > 300:
                raise TtsBackendError("合成结果时长异常")
            quality = analyze_quality(output, duration_seconds=probe.duration_seconds)
            if quality.rms_dbfs is None or quality.silence_ratio >= 0.98:
                raise TtsBackendError("合成结果没有可听音频")
            if self.metadata.get(job.id).state != JobState.RUNNING:
                return None
            output_hash = sha256_file(output)
            output_uri = self.artifacts.upload_file(
                output,
                f"jobs/{job.id}/tts/{output_hash}.wav",
                "audio/wav",
            )
            asset = AudioAsset(
                id=uuid5(job.id, "tts:output"),
                uri=output_uri,
                filename="speech.wav",
                sha256=output_hash,
                size_bytes=output.stat().st_size,
                probe=probe,
                source_asset_id=original.id,
            )
            run = ToolRun(
                id=uuid5(job.id, "tool:synthesize"),
                tool_name=spec.model,
                tool_version=str(metrics.get("revision", "unverified")),
                operation=Operation.SYNTHESIZE,
                status=ToolRunStatus.SUCCEEDED,
                input_asset_ids=[original.id],
                output_asset_ids=[asset.id],
                parameters={
                    **spec.model_dump(mode="json"),
                    "weights_sha256": metrics.get("weights_sha256", {}),
                },
                metrics={
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, str, bool)) or value is None
                },
                started_at=started,
                finished_at=datetime.now(UTC),
            )
            report = AudioReport(
                id=uuid5(job.id, "tts:report"),
                job_id=job.id,
                input_asset_id=original.id,
                processing_chain_id="shengke-tts-v1",
                assets=[original, asset],
                tool_runs=[run],
                summary={
                    "tts": metrics,
                    "output_asset_id": str(asset.id),
                    "reference_quality": ref_quality.model_dump(mode="json"),
                    "quality": quality.model_dump(mode="json"),
                    "warnings": warnings,
                },
            )
            report_path = root / "report.json"
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            report_uri = self.artifacts.upload_file(
                report_path,
                f"jobs/{job.id}/tts/{sha256_file(report_path)}.json",
                "application/json",
            )
            # One transaction checks lease ownership and publishes all metadata.
            self.metadata.complete_tts(job.id, self.worker_id, report, report_uri)
            return report
