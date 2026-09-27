import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from audio_core.contracts import (
    AudioQualityMetrics,
    AudioSegment,
    CreateAudioJob,
    JobState,
    Operation,
    ToolRun,
    ToolRunStatus,
    TranscriptSpan,
)
from audio_core.quality import AudioQualityPolicy, evaluate_quality
from pydantic import ValidationError


class ContractTests(unittest.TestCase):
    def test_create_request_rejects_duplicate_operations(self):
        with self.assertRaises(ValidationError):
            CreateAudioJob(
                input_uri="s3://bucket/input.wav",
                operations=[Operation.PROBE, Operation.PROBE],
                idempotency_key="request-0001",
            )

    def test_job_state_values_are_stable(self):
        self.assertEqual(JobState.PENDING.value, "pending")
        self.assertEqual(Operation.NORMALIZE.value, "normalize")

    def test_segment_rejects_reversed_interval(self):
        with self.assertRaises(ValidationError):
            AudioSegment(asset_id=uuid4(), start_ms=1000, end_ms=999)

    def test_segment_rejects_invalid_label_confidence(self):
        with self.assertRaises(ValidationError):
            AudioSegment(
                asset_id=uuid4(),
                start_ms=0,
                end_ms=1000,
                labels={"speech": 1.1},
            )

    def test_tool_run_rejects_reversed_timestamps(self):
        started = datetime.now(UTC)
        with self.assertRaises(ValidationError):
            ToolRun(
                tool_name="ffmpeg",
                tool_version="6.1.1",
                operation=Operation.NORMALIZE,
                status=ToolRunStatus.FAILED,
                started_at=started,
                finished_at=started - timedelta(seconds=1),
            )

    def test_quality_metrics_reject_invalid_ratio(self):
        with self.assertRaises(ValidationError):
            AudioQualityMetrics(silence_ratio=1.1)

    def test_quality_policy_emits_traceable_findings(self):
        metrics = AudioQualityMetrics(
            integrated_lufs=-30.0,
            true_peak_dbtp=-0.2,
            clipping_ratio=0.01,
            silence_ratio=0.75,
        )
        findings = evaluate_quality(uuid4(), metrics, AudioQualityPolicy())
        self.assertEqual(
            {finding.category for finding in findings},
            {
                "integrated_loudness_low",
                "true_peak_high",
                "clipping_detected",
                "excessive_silence",
            },
        )

    def test_transcript_span_rejects_reversed_interval(self):
        with self.assertRaises(ValidationError):
            TranscriptSpan(segment_id=uuid4(), start_ms=500, end_ms=400, text="invalid")


if __name__ == "__main__":
    unittest.main()
