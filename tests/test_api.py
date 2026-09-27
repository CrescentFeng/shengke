from __future__ import annotations

import importlib.util
import unittest
from uuid import uuid4


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from platform_api.main import app

        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

        ready = self.client.get("/readyz")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["store"], "InMemoryJobStore")

    def test_create_get_and_cancel(self):
        key = f"request-{uuid4()}"
        payload = {
            "input_uri": "s3://private-audio/example.wav",
            "operations": ["probe", "normalize"],
            "idempotency_key": key,
        }
        created = self.client.post(
            "/v1/audio/jobs",
            json=payload,
            headers={"Idempotency-Key": key},
        )
        self.assertEqual(created.status_code, 201)
        job_id = created.json()["id"]
        fetched = self.client.get(f"/v1/audio/jobs/{job_id}")
        self.assertEqual(fetched.status_code, 200)
        cancelled = self.client.post(f"/v1/audio/jobs/{job_id}/cancel")
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["state"], "cancelled")

        assets = self.client.get(f"/v1/audio/jobs/{job_id}/assets")
        self.assertEqual(assets.status_code, 501)
        segments = self.client.get(f"/v1/audio/jobs/{job_id}/segments")
        self.assertEqual(segments.status_code, 501)
        transcript = self.client.get(f"/v1/audio/jobs/{job_id}/transcript")
        self.assertEqual(transcript.status_code, 501)
        diarization = self.client.get(f"/v1/audio/jobs/{job_id}/diarization")
        self.assertEqual(diarization.status_code, 501)

    def test_idempotent_retry_returns_same_job(self):
        key = f"request-{uuid4()}"
        payload = {
            "input_uri": "s3://private-audio/example.wav",
            "operations": ["probe"],
            "idempotency_key": key,
        }
        first = self.client.post("/v1/audio/jobs", json=payload)
        second = self.client.post("/v1/audio/jobs", json=payload)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["id"], second.json()["id"])


if __name__ == "__main__":
    unittest.main()
