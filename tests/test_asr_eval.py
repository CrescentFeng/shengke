from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_eval.models import BenchmarkSample, ReferenceSpan
from asr_eval.runner import AsrBenchmarkRunner, load_manifest, write_report
from audio_core.contracts import AudioProbe, MediaFormat
from ingest_worker.asr import AsrChunk, AsrResult


class FakeBackend:
    name = "fake-asr"
    model_name = "fake-small"
    version = "test"

    def transcribe(
        self,
        path: Path,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> AsrResult:
        del path
        self.hotwords = hotwords
        return AsrResult(
            text="hello world",
            language=language or "en",
            language_confidence=0.95,
            chunks=[AsrChunk(start_ms=120, end_ms=880, text="hello world")],
        )


class FailingBackend(FakeBackend):
    def transcribe(
        self,
        path: Path,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> AsrResult:
        del path, language, hotwords
        raise RuntimeError("synthetic inference failure")


def _probe() -> AudioProbe:
    return AudioProbe(
        codec_name="pcm_s16le",
        sample_rate_hz=16000,
        channels=1,
        duration_seconds=1.0,
        media_format=MediaFormat(format_name="wav", duration_seconds=1.0),
    )


class AsrBenchmarkTests(unittest.TestCase):
    def test_manifest_resolves_relative_audio_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"placeholder")
            entry = {
                "id": "sample-1",
                "audio": "sample.wav",
                "reference_text": "hello world",
                "language": "en",
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

            samples = load_manifest(manifest)
            self.assertEqual(samples[0].audio, (root / "sample.wav").resolve())

            manifest.write_text(
                json.dumps(entry) + "\n" + json.dumps(entry) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample id"):
                load_manifest(manifest)

    @patch("asr_eval.runner.probe_file", return_value=_probe())
    def test_runner_aggregates_metrics_and_writes_both_reports(self, _probe_file):
        sample = BenchmarkSample(
            id="sample-1",
            audio=Path("/tmp/sample.wav"),
            reference_text="hello world",
            language="en-US",
            reference_spans=[ReferenceSpan(start_ms=100, end_ms=900, text="hello world")],
            tags={"domain": "meeting"},
            hotwords=["hello world"],
        )
        backend = FakeBackend()
        report = AsrBenchmarkRunner(backend).run([sample], manifest="manifest.jsonl")

        self.assertEqual(report.summary.succeeded_count, 1)
        self.assertEqual(report.summary.failed_count, 0)
        self.assertEqual(report.summary.aggregate_wer, 0.0)
        self.assertEqual(report.summary.aggregate_cer, 0.0)
        self.assertEqual(report.summary.language_accuracy, 1.0)
        self.assertEqual(report.summary.timestamp_boundary_mae_ms, 20.0)
        self.assertIsNotNone(report.summary.mean_rtf)
        self.assertEqual(report.samples[0].hypothesis_text, "hello world")
        self.assertEqual(backend.hotwords, ["hello world"])

        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_report(report, directory)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["backend"], "fake-asr")
            self.assertIn("ASR 基准评测报告", markdown_path.read_text(encoding="utf-8"))

    @patch("asr_eval.runner.probe_file", return_value=_probe())
    def test_runner_records_failure_without_aborting_report(self, _probe_file):
        sample = BenchmarkSample(
            id="broken",
            audio=Path("/tmp/broken.wav"),
            reference_text="hello",
        )
        report = AsrBenchmarkRunner(FailingBackend()).run(
            [sample], manifest="manifest.jsonl"
        )
        self.assertEqual(report.summary.failed_count, 1)
        self.assertIsNone(report.summary.aggregate_wer)
        self.assertIsNone(report.summary.aggregate_cer)
        self.assertEqual(report.samples[0].success, False)
        self.assertIn("synthetic inference failure", report.samples[0].error or "")


if __name__ == "__main__":
    unittest.main()
