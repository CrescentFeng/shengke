import unittest

from audio_core.contracts import CreateAudioJob, JobState, Operation
from platform_api.service import submit_job
from platform_api.store import InMemoryJobStore


class ServiceTests(unittest.TestCase):
    def test_submit_moves_new_job_to_queue(self):
        store = InMemoryJobStore()
        job, created = submit_job(
            store,
            CreateAudioJob(
                input_uri="s3://audio-input/example.wav",
                operations=[Operation.PROBE, Operation.NORMALIZE],
                idempotency_key="submit-0001",
            ),
        )
        self.assertTrue(created)
        self.assertEqual(job.state, JobState.QUEUED)
        self.assertEqual(
            [event.state for event in job.events],
            [JobState.PENDING, JobState.VALIDATING, JobState.QUEUED],
        )

    def test_idempotent_retry_does_not_requeue(self):
        store = InMemoryJobStore()
        request = CreateAudioJob(
            input_uri="s3://audio-input/example.wav",
            operations=[Operation.PROBE],
            idempotency_key="submit-0002",
        )
        first, _ = submit_job(store, request)
        second, created = submit_job(store, request)
        self.assertFalse(created)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.state, JobState.QUEUED)


if __name__ == "__main__":
    unittest.main()
