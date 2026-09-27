import unittest

from audio_core.contracts import CreateAudioJob, JobState, Operation
from audio_core.errors import InvalidStateTransition
from platform_api.store import InMemoryJobStore


def request(key: str = "request-0001") -> CreateAudioJob:
    return CreateAudioJob(
        input_uri="s3://private-audio/example.wav",
        operations=[Operation.PROBE, Operation.NORMALIZE],
        idempotency_key=key,
    )


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryJobStore()

    def test_create_is_idempotent(self):
        first, first_created = self.store.create(request())
        second, second_created = self.store.create(request())
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first.id, second.id)

    def test_idempotency_collision_is_rejected(self):
        self.store.create(request())
        changed = request()
        changed.input_uri = "s3://private-audio/other.wav"
        with self.assertRaises(ValueError):
            self.store.create(changed)

    def test_valid_state_path(self):
        job, _ = self.store.create(request())
        self.store.transition(job.id, JobState.VALIDATING)
        self.store.transition(job.id, JobState.QUEUED)
        self.store.transition(job.id, JobState.RUNNING, progress=0.5)
        done = self.store.transition(job.id, JobState.SUCCEEDED)
        self.assertEqual(done.progress, 1.0)
        self.assertEqual(done.state, JobState.SUCCEEDED)

    def test_invalid_state_jump_is_rejected(self):
        job, _ = self.store.create(request())
        with self.assertRaises(InvalidStateTransition):
            self.store.transition(job.id, JobState.SUCCEEDED)

    def test_cancel_is_idempotent_for_terminal_job(self):
        job, _ = self.store.create(request())
        cancelled = self.store.cancel(job.id)
        again = self.store.cancel(job.id)
        self.assertEqual(cancelled.id, again.id)
        self.assertEqual(again.state, JobState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
