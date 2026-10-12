from __future__ import annotations

import base64
import copy
import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from uuid import uuid4

from audio_core.contracts import (
    AudioAsset,
    AudioReport,
    AudioReportEnvelope,
    CreateAudioJob,
    JobState,
)
from audio_core.tts import ListeningScore, TtsSynthesisSpec
from audio_core.tts_quality import inspect_reference
from fastapi.testclient import TestClient
from ingest_worker.tts import HttpTtsBackend, TtsBackendError, TtsProcessor
from platform_api.dependencies import get_store
from platform_api.main import app
from platform_api.store import InMemoryJobStore
from platform_api.tts_routes import summarize as listening_summary
from platform_api.tts_store import get_tts_artifacts, get_tts_store
from pydantic import ValidationError
from tts_eval.cli import summarize as benchmark_summary


def wav(path, duration=3.5, silent=False):
    rate = 16000
    frames = bytearray()
    for i in range(int(rate * duration)):
        value = 0 if silent else int(6000 * math.sin(2 * math.pi * 220 * i / rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


class MemoryTtsStore:
    def __init__(self):
        self.voices, self.evaluations, self.scores = {}, {}, {}

    def save_voice(self, profile):
        self.voices[str(profile.id)] = profile.model_dump(mode="json")

    def list_voices(self):
        return list(self.voices.values())

    def get_voice(self, voice_id):
        return self.voices[str(voice_id)]

    def save_evaluation(self, value):
        self.evaluations[value["id"]] = copy.deepcopy(value)

    def get_evaluation(self, evaluation_id):
        return copy.deepcopy(self.evaluations[str(evaluation_id)])

    def save_score(self, evaluation_id, score):
        self.scores[(str(evaluation_id), score.listener_id, str(score.sample_id))] = (
            score.model_dump(mode="json")
        )

    def list_scores(self, evaluation_id):
        return [score for key, score in self.scores.items() if key[0] == str(evaluation_id)]


class Files:
    def __init__(self, root):
        self.root = root

    def upload_file(self, path, key, content_type=None):
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return destination.as_uri()

    def fetch_input(self, uri, destination):
        from urllib.parse import unquote, urlparse

        shutil.copyfile(unquote(urlparse(uri).path), destination)
        return destination


class Jobs(InMemoryJobStore):
    def __init__(self):
        super().__init__()
        self.reports = {}

    def get_report(self, job_id):
        return self.reports[job_id]

    def complete_tts(self, job_id, worker_id, report, uri):
        self.transition(job_id, JobState.SUCCEEDED)
        self.reports[job_id] = AudioReportEnvelope(report=report, report_uri=uri)


class TtsContractsTests(unittest.TestCase):
    def test_pause_trial_is_opt_in_and_gpt_only(self):
        spec = {"text": "你好。", "reference_text": "参考原文"}
        self.assertEqual(TtsSynthesisSpec(**spec).inference_profile, "baseline")
        self.assertEqual(
            TtsSynthesisSpec(
                **spec, model="gpt-sovits-v2", inference_profile="gpt_pause_v3"
            ).inference_profile,
            "gpt_pause_v3",
        )
        with self.assertRaises(ValidationError):
            TtsSynthesisSpec(**spec, model="cosyvoice2", inference_profile="gpt_pause_v3")

    def test_delivery_trials_are_not_pooled_with_baseline_ratings(self):
        evaluation = {
            "samples": [
                {"id": "old", "model": "cosyvoice2"},
                {"id": "new", "model": "cosyvoice2", "inference_profile": "pilot_v2"},
            ]
        }
        report = listening_summary(
            evaluation,
            [
                {"sample_id": "old", "listener_id": "tester", "naturalness": 2},
                {"sample_id": "new", "listener_id": "tester", "naturalness": 4},
            ],
        )
        means = {
            r["inference_profile"]: r["scores"]["naturalness"]["mean"] for r in report["groups"]
        }
        self.assertEqual(means, {"baseline": 2, "pilot_v2": 4})

    def test_capabilities_are_enforced(self):
        for kwargs in [
            {"text": "你好", "reference_text": " "},
            {"text": " ", "reference_text": "你好"},
            {"text": "你好", "reference_text": "你好", "model": "gpt-sovits-v2", "emotion": "sad"},
            {"text": "你好", "model": "vits-csmsc", "emotion": "happy"},
            {"text": "好" * 201, "reference_text": "你好"},
            {"text": "你好", "model": "vits-csmsc", "inference_profile": "pilot_v2"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                TtsSynthesisSpec(**kwargs)

    def test_synthesis_must_not_silently_ignore_parameters(self):
        for operations, tts in [
            (["synthesize"], {}),
            (["synthesize", "normalize"], {"text": "你好", "reference_text": "好"}),
        ]:
            with self.assertRaises(ValidationError):
                CreateAudioJob(
                    input_uri="s3://bucket/ref.wav",
                    operations=operations,
                    idempotency_key="testing-tts",
                    parameters={"tts": tts},
                )

    def test_scores_are_real_integer_ratings(self):
        for value in [0, 6, True, 3.5, "5"]:
            with self.assertRaises(ValidationError):
                ListeningScore(listener_id="tester", sample_id=uuid4(), naturalness=value)

    def test_unavailable_model_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.wav"
            with self.assertRaises(TtsBackendError):
                HttpTtsBackend({}).synthesize(
                    TtsSynthesisSpec(text="你好", reference_text="好"), path, path
                )
            self.assertFalse(path.exists())

    def test_benchmark_keeps_failure_denominator_and_null_gpu(self):
        rows = [
            {"model": "cosyvoice2", "status": "failed"},
            {"model": "cosyvoice2", "status": "succeeded", "metrics": {"rtf": 2.0}},
        ]
        summary = benchmark_summary(rows)[0]
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertIsNone(summary["metrics"]["peak_cuda_allocated_bytes"])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class TtsApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.wav = self.root / "reference.wav"
        wav(self.wav)
        self.store, self.jobs = MemoryTtsStore(), Jobs()
        self.files = Files(self.root / "objects")
        app.dependency_overrides[get_tts_store] = lambda: self.store
        app.dependency_overrides[get_tts_artifacts] = lambda: self.files
        app.dependency_overrides[get_store] = lambda: self.jobs
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        self.directory.cleanup()

    def upload(self):
        return self.client.post(
            "/v1/tts/voices",
            json={
                "name": "fixture",
                "filename": "reference.wav",
                "audio_base64": base64.b64encode(self.wav.read_bytes()).decode(),
                "reference_text": "你好",
                "source": "synthetic test fixture",
                "license": "test only",
            },
        )

    def test_upload_preserves_original_and_rejects_invalid_reference(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["asset"]["probe"]["sample_rate_hz"], 16000)
        self.assertEqual(len(self.client.get("/v1/tts/voices").json()), 1)
        voice_id = response.json()["id"]
        audio = self.client.get(f"/v1/tts/voices/{voice_id}/audio")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.content, self.wav.read_bytes())
        self.assertEqual(self.client.get(f"/v1/tts/voices/{uuid4()}/audio").status_code, 404)
        for duration, silent in [(2.9, False), (10.1, False), (3.5, True)]:
            wav(self.wav, duration, silent)
            self.assertEqual(self.upload().status_code, 422)

    def test_qc_accepts_exact_duration_boundaries(self):
        for duration in [3, 10]:
            wav(self.wav, duration)
            probe, _, _ = inspect_reference(self.wav)
            self.assertEqual(probe.duration_seconds, duration)

    def make_completed_job(self, model="cosyvoice2"):
        request = CreateAudioJob(
            input_uri=self.wav.as_uri(),
            operations=["synthesize"],
            idempotency_key=str(uuid4()),
            parameters={
                "tts": {
                    "model": model,
                    "text": "你好",
                    "reference_text": "你好",
                }
            },
        )
        job, _ = self.jobs.create(request)
        for state in [JobState.VALIDATING, JobState.QUEUED, JobState.RUNNING, JobState.SUCCEEDED]:
            self.jobs.transition(job.id, state)
        asset = AudioAsset(uri=self.wav.as_uri())
        self.jobs.reports[job.id] = AudioReportEnvelope(
            report_uri="file:///unused.json",
            report=AudioReport(
                job_id=job.id,
                input_asset_id=uuid4(),
                processing_chain_id="test",
                assets=[asset],
                summary={"output_asset_id": str(asset.id)},
            ),
        )
        return job.id

    def test_blind_order_stable_and_ratings_upsert_without_fake_mos(self):
        job_ids = [str(self.make_completed_job()) for _ in range(4)]
        response = self.client.post(
            "/v1/tts/evaluations", json={"name": "test", "job_ids": job_ids}
        )
        self.assertEqual(response.status_code, 201, response.text)
        evaluation_id = response.json()["id"]
        endpoint = f"/v1/tts/evaluations/{evaluation_id}"
        first = self.client.get(endpoint, params={"listener_id": "listener1"}).json()
        second = self.client.get(endpoint, params={"listener_id": "listener1"}).json()
        self.assertEqual(first, second)
        for sample in first["samples"]:
            self.assertNotIn("model", sample)
            self.assertNotIn("job_id", sample)
            self.assertNotIn("uri", sample)
            self.assertNotIn("inference_profile", sample)
        sample_id = first["samples"][0]["id"]
        empty = self.client.get(endpoint + "/report").json()
        self.assertEqual(empty["status"], "not_performed")
        self.assertEqual(empty["listeners"], 0)
        self.assertEqual(empty["groups"][0]["ratings"], 0)
        self.assertIsNone(empty["groups"][0]["scores"]["naturalness"]["mean"])
        score = {"listener_id": "listener1", "sample_id": sample_id, "naturalness": 3}
        self.assertEqual(self.client.post(endpoint + "/scores", json=score).status_code, 200)
        progress = self.client.get(
            endpoint + "/progress", params={"listener_id": "listener1"}
        ).json()
        self.assertEqual((progress["completed"], progress["total"]), (1, 4))
        self.assertEqual(progress["ratings"][sample_id]["naturalness"], 3)
        self.assertNotIn("model", progress["ratings"][sample_id])
        self.assertNotIn("inference_profile", progress["ratings"][sample_id])
        other = self.client.get(endpoint + "/progress", params={"listener_id": "other"}).json()
        self.assertEqual(other["ratings"], {})
        score["naturalness"] = 5
        self.client.post(endpoint + "/scores", json=score)
        result = self.client.get(endpoint + "/report").json()
        self.assertEqual(result["listeners"], 1)
        self.assertEqual(result["status"], "pilot_not_formal_mos")
        self.assertEqual(result["groups"][0]["ratings"], 1)
        self.assertEqual(result["groups"][0]["scores"]["naturalness"]["mean"], 5)
        score["sample_id"] = str(uuid4())
        self.assertEqual(self.client.post(endpoint + "/scores", json=score).status_code, 422)
        audio = self.client.get(endpoint + f"/samples/{sample_id}/audio")
        self.assertEqual(audio.content[:4], b"RIFF")

    def test_baseline_rejects_clone_similarity_score(self):
        job_id = self.make_completed_job("vits-csmsc")
        evaluation_id = self.client.post(
            "/v1/tts/evaluations",
            json={
                "name": "baseline",
                "job_ids": [str(job_id)],
            },
        ).json()["id"]
        endpoint = f"/v1/tts/evaluations/{evaluation_id}"
        sample_id = self.client.get(endpoint, params={"listener_id": "l1"}).json()["samples"][0][
            "id"
        ]
        response = self.client.post(
            endpoint + "/scores",
            json={
                "listener_id": "l1",
                "sample_id": sample_id,
                "naturalness": 3,
                "similarity": 4,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_cancellation_while_inference_runs_does_not_publish_result(self):
        request = CreateAudioJob(
            input_uri=self.wav.as_uri(),
            operations=["synthesize"],
            idempotency_key=str(uuid4()),
            parameters={
                "tts": {
                    "text": "你好",
                    "reference_text": "你好",
                }
            },
        )
        job, _ = self.jobs.create(request)
        for state in [JobState.VALIDATING, JobState.QUEUED, JobState.RUNNING]:
            job = self.jobs.transition(job.id, state)
        owner = self

        class CancellingBackend:
            def synthesize(self, spec, reference, output):
                shutil.copyfile(owner.wav, output)
                owner.jobs.cancel(job.id)
                return {"model": "cosyvoice2"}

        result = TtsProcessor(self.jobs, self.files, CancellingBackend(), worker_id="test").process(
            job
        )
        self.assertIsNone(result)
        self.assertNotIn(job.id, self.jobs.reports)
        self.assertEqual(self.jobs.get(job.id).state, JobState.CANCELLED)

    def test_success_retains_nested_weight_provenance_without_breaking_tool_run(self):
        request = CreateAudioJob(
            input_uri=self.wav.as_uri(),
            operations=["synthesize"],
            idempotency_key=str(uuid4()),
            parameters={"tts": {"text": "你好", "reference_text": "你好"}},
        )
        job, _ = self.jobs.create(request)
        for state in [JobState.VALIDATING, JobState.QUEUED, JobState.RUNNING]:
            job = self.jobs.transition(job.id, state)
        owner = self

        class Backend:
            def synthesize(self, spec, reference, output):
                shutil.copyfile(owner.wav, output)
                return {
                    "model": "cosyvoice2",
                    "weights_sha256": {"model.pt": "a" * 64},
                    "rtf": 2.5,
                    "peak_cuda_allocated_bytes": None,
                }

        report = TtsProcessor(self.jobs, self.files, Backend(), worker_id="test").process(job)
        self.assertEqual(self.jobs.get(job.id).state, JobState.SUCCEEDED)
        self.assertEqual(report.summary["tts"]["weights_sha256"], {"model.pt": "a" * 64})
        self.assertNotIn("weights_sha256", report.tool_runs[0].metrics)
        self.assertEqual(report.tool_runs[0].metrics["rtf"], 2.5)


if __name__ == "__main__":
    unittest.main()


def test_short_happy_profile_capability_boundary():
    import pytest

    source = {
        "text": "你终于回来了。",
        "reference_text": "这是一段参考录音。",
        "inference_profile": "cosy_happy_v4",
    }
    assert TtsSynthesisSpec(**source, model="cosyvoice2", emotion="happy")
    for model, emotion in [("cosyvoice2", "neutral"), ("gpt-sovits-v2", "neutral")]:
        with pytest.raises(ValueError):
            TtsSynthesisSpec(**source, model=model, emotion=emotion)


def test_gentle_rhythm_is_explicit_and_happy_cosy_only():
    import pytest

    source = {"text": "你好。", "reference_text": "参考录音。"}
    assert TtsSynthesisSpec(**source).inference_profile == "baseline"
    assert (
        TtsSynthesisSpec(**source, emotion="happy", inference_profile="cosy_rhythm_v5").model
        == "cosyvoice2"
    )
    for model, emotion in [
        ("cosyvoice2", "neutral"),
        ("cosyvoice2", "sad"),
        ("gpt-sovits-v2", "neutral"),
        ("vits-csmsc", "neutral"),
    ]:
        with pytest.raises(ValueError):
            TtsSynthesisSpec(
                **source, model=model, emotion=emotion, inference_profile="cosy_rhythm_v5"
            )


def test_gpt_rhythm_rejects_other_engines_and_ambiguous_text():
    import pytest

    base = {
        "text": "你好，欢迎回来。",
        "reference_text": "参考文本。",
        "model": "gpt-sovits-v2",
        "inference_profile": "gpt_rhythm_v6",
    }
    assert TtsSynthesisSpec(**base)
    for change in [
        {"model": "cosyvoice2"},
        {"text": "版本是3.14，欢迎使用。"},
        {"text": "Hello，欢迎回来。"},
    ]:
        with pytest.raises(ValueError):
            TtsSynthesisSpec(**{**base, **change})
