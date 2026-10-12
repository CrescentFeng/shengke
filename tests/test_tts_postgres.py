"""Opt-in PostgreSQL lease tests, isolated in a fresh disposable schema."""

from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import uuid4

from audio_core.contracts import AudioAsset, AudioReport, CreateAudioJob, JobState
from audio_core.errors import InvalidStateTransition
from platform_api.postgres_store import PostgresJobStore
from platform_api.store import JobNotFound


@unittest.skipUnless(os.getenv("SHENGKE_TEST_DATABASE_URL"), "opt-in PostgreSQL test DSN required")
class TtsPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        self.dsn = os.environ["SHENGKE_TEST_DATABASE_URL"]
        self.schema = "shengke_test_" + uuid4().hex
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
            conn.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
            )
            root = Path(__file__).resolve().parents[1]
            # gen_random_uuid is built into supported PostgreSQL 16; avoid changing extensions.
            ddl = (root / "infra/postgres/init.sql").read_text()
            conn.execute(ddl.replace("CREATE EXTENSION IF NOT EXISTS pgcrypto;", ""))
        self.store = PostgresJobStore(
            make_conninfo(self.dsn, options=f"-c search_path={self.schema},public")
        )
        self.addCleanup(self.drop_schema)

    def drop_schema(self):
        import psycopg
        from psycopg import sql

        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    def test_history_excludes_non_tts_and_keeps_cancelled(self):
        tts = self.claimed()
        self.store.cancel(tts.id)
        self.store.create(
            CreateAudioJob(
                input_uri="s3://test/other.wav", operations=["probe"], idempotency_key=str(uuid4())
            )
        )
        page = self.store.list_tts_jobs(20, 0)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], str(tts.id))
        self.assertEqual(page["items"][0]["state"], "cancelled")
        self.assertEqual(self.store.list_tts_jobs(20, 10)["items"], [])

    def claimed(self, worker="owner"):
        job, _ = self.store.create(
            CreateAudioJob(
                input_uri="s3://test/reference.wav",
                operations=["synthesize"],
                idempotency_key=str(uuid4()),
                parameters={"tts": {"text": "你好", "reference_text": "你好"}},
            )
        )
        self.store.transition(job.id, JobState.VALIDATING)
        self.store.transition(job.id, JobState.QUEUED)
        claimed = self.store.claim_next(worker)
        self.assertEqual(claimed.id, job.id)
        return claimed

    def report(self, job):
        reference = AudioAsset(uri=job.input_uri)
        output = AudioAsset(uri="s3://test/output.wav", source_asset_id=reference.id)
        return AudioReport(
            job_id=job.id,
            input_asset_id=reference.id,
            processing_chain_id="lease-test",
            assets=[reference, output],
            summary={"output_asset_id": str(output.id)},
        )

    def test_reclaimed_lease_rejects_old_completion_and_failure(self):
        job = self.claimed("old")
        self.assertFalse(self.store.heartbeat(job.id, "other"))
        with self.store._connection() as conn:
            conn.execute(
                "UPDATE audio_jobs SET locked_at=now()-interval '1 hour' WHERE id=%s", (job.id,)
            )
        self.assertEqual(self.store.requeue_stale(60), 1)
        self.store.claim_next("new")
        self.assertFalse(self.store.heartbeat(job.id, "old"))
        self.assertTrue(self.store.heartbeat(job.id, "new"))
        self.assertEqual(self.store.requeue_stale(60), 0)
        report = self.report(job)
        with self.assertRaises(InvalidStateTransition):
            self.store.complete_tts(job.id, "old", report, "s3://test/old.json")
        self.store.fail_claimed(job.id, "old", "TEST", "late error")
        self.assertEqual(self.store.get(job.id).state, JobState.RUNNING)
        self.assertEqual(self.store.list_assets(job.id), [])
        self.store.complete_tts(job.id, "new", report, "s3://test/new.json")
        self.assertEqual(self.store.get(job.id).state, JobState.SUCCEEDED)
        self.assertEqual(len(self.store.list_assets(job.id)), 2)
        self.assertEqual(self.store.get_report(job.id).report_uri, "s3://test/new.json")

    def test_invalid_report_rolls_back_all_metadata(self):
        import psycopg

        job = self.claimed()
        report = self.report(job)
        # Fail the second INSERT after the reference INSERT has succeeded.
        report.assets[1].source_asset_id = uuid4()
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.store.complete_tts(job.id, "owner", report, "s3://test/report.json")
        self.assertEqual(self.store.get(job.id).state, JobState.RUNNING)
        self.assertEqual(self.store.list_assets(job.id), [])
        with self.assertRaises(JobNotFound):
            self.store.get_report(job.id)

    def test_cancel_and_complete_race_has_only_one_terminal_result(self):
        for _ in range(5):
            job = self.claimed()
            report, barrier = self.report(job), Barrier(2)

            def cancel(job=job, barrier=barrier):
                barrier.wait()
                try:
                    self.store.cancel(job.id)
                except InvalidStateTransition:
                    pass  # Completion acquired the row lock first.

            def complete(job=job, report=report, barrier=barrier):
                barrier.wait()
                try:
                    self.store.complete_tts(job.id, "owner", report, "s3://test/result.json")
                except InvalidStateTransition:
                    pass  # Cancellation acquired the row lock first.

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(cancel), pool.submit(complete)]
                for future in futures:
                    future.result()
            final = self.store.get(job.id)
            terminals = [
                e for e in final.events if e.state in {JobState.CANCELLED, JobState.SUCCEEDED}
            ]
            self.assertEqual(len(terminals), 1)
            self.assertEqual(
                len(self.store.list_assets(job.id)), 2 if final.state == JobState.SUCCEEDED else 0
            )


if __name__ == "__main__":
    unittest.main()
