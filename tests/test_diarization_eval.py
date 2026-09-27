from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_core.contracts import AudioProbe, MediaFormat, SpeakerTurn
from diarization_eval.matrix import require_real_human_dataset, run_matrix
from diarization_eval.metrics import calculate_metrics
from diarization_eval.models import DiarizationBenchmarkSample, ProductionThresholds
from diarization_eval.runner import (
    DiarizationBenchmarkRunner,
    load_manifest,
    write_report,
)
from ingest_worker.diarization import DiarizationResult


class FakeBackend:
    name = "fake-diarization"
    model_name = "fixture-speakers"
    version = "test"

    def __init__(self):
        self.specs = []

    def diarize(self, path, *, spec):
        del path
        self.spec = spec
        self.specs.append(spec)
        return DiarizationResult(
            turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="PREDICTED_B"),
                SpeakerTurn(start_ms=1000, end_ms=2000, speaker="PREDICTED_A"),
            ],
            exclusive=True,
        )


class OracleOnlyBackend(FakeBackend):
    def diarize(self, path, *, spec):
        if (
            spec.num_speakers is None
            and spec.min_speakers is None
            and spec.max_speakers is None
        ):
            self.specs.append(spec)
            return DiarizationResult(
                turns=[
                    SpeakerTurn(start_ms=0, end_ms=2000, speaker="ONE_SPEAKER")
                ],
                exclusive=spec.use_exclusive,
            )
        return super().diarize(path, spec=spec)


def _probe() -> AudioProbe:
    return AudioProbe(
        codec_name="pcm_s16le",
        sample_rate_hz=16000,
        channels=1,
        duration_seconds=2.0,
        media_format=MediaFormat(format_name="wav", duration_seconds=2.0),
    )


class DiarizationMetricTests(unittest.TestCase):
    def test_permuted_speaker_labels_have_zero_der_and_jer(self):
        reference = [
            SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE"),
            SpeakerTurn(start_ms=1000, end_ms=2000, speaker="BOB"),
        ]
        hypothesis = [
            SpeakerTurn(start_ms=0, end_ms=1000, speaker="SPEAKER_01"),
            SpeakerTurn(start_ms=1000, end_ms=2000, speaker="SPEAKER_00"),
        ]
        metrics = calculate_metrics(reference, hypothesis, duration_ms=2000)
        self.assertEqual(metrics.der, 0.0)
        self.assertEqual(metrics.jer, 0.0)

    def test_der_breakdown_includes_miss_and_false_alarm(self):
        reference = [SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE")]
        hypothesis = [
            SpeakerTurn(start_ms=0, end_ms=500, speaker="SPEAKER_00"),
            SpeakerTurn(start_ms=1000, end_ms=1500, speaker="SPEAKER_00"),
        ]
        metrics = calculate_metrics(reference, hypothesis, duration_ms=1500)
        self.assertAlmostEqual(metrics.missed_detection_seconds, 0.5)
        self.assertAlmostEqual(metrics.false_alarm_seconds, 0.5)
        self.assertEqual(metrics.confusion_seconds, 0.0)
        self.assertAlmostEqual(metrics.der, 1.0)
        self.assertAlmostEqual(metrics.jer, 2 / 3)

    def test_boundary_metrics_use_mapped_speaker_labels(self):
        reference = [SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE")]
        hypothesis = [
            SpeakerTurn(start_ms=100, end_ms=1100, speaker="SPEAKER_09")
        ]
        metrics = calculate_metrics(
            reference,
            hypothesis,
            duration_ms=1200,
            boundary_tolerance_ms=150,
        )
        self.assertEqual(metrics.boundary_reference_count, 2)
        self.assertEqual(metrics.boundary_matched_count, 2)
        self.assertEqual(metrics.boundary_errors_ms, (100.0, 100.0))
        self.assertEqual(metrics.boundary_mae_ms, 100.0)
        self.assertEqual(metrics.boundary_p95_ms, 100.0)
        self.assertEqual(metrics.boundary_recall, 1.0)


class DiarizationRunnerTests(unittest.TestCase):
    def test_manifest_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "meeting-1",
                        "audio": "sample.wav",
                        "reference_turns": [
                            {"start_ms": 0, "end_ms": 1000, "speaker": "A"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_manifest(manifest)[0].audio,
                (root / "sample.wav").resolve(),
            )

    def test_manifest_loads_recording_from_rttm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meeting.wav").write_bytes(b"placeholder")
            (root / "reference.rttm").write_text(
                "SPEAKER meeting-1 1 0.250 1.500 <NA> <NA> ALICE <NA> <NA>\n"
                "SPEAKER other 1 0.000 9.000 <NA> <NA> OTHER <NA> <NA>\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "audio": "meeting.wav",
                        "rttm": "reference.rttm",
                        "recording_id": "meeting-1",
                        "audio_kind": "real",
                        "annotation_kind": "human",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = load_manifest(manifest)[0]
            self.assertEqual(sample.reference_turns[0].start_ms, 250)
            self.assertEqual(sample.reference_turns[0].end_ms, 1750)
            self.assertEqual(sample.reference_turns[0].speaker, "ALICE")

    def test_manifest_rejects_changed_audio_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "meeting.wav"
            audio.write_bytes(b"original")
            expected = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "meeting-1",
                        "audio": "meeting.wav",
                        "audio_sha256": expected,
                        "reference_turns": [
                            {"start_ms": 0, "end_ms": 1000, "speaker": "A"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "audio checksum mismatch"):
                load_manifest(manifest)

    @patch("diarization_eval.runner.probe_file", return_value=_probe())
    def test_runner_generates_zero_error_report(self, _probe_file):
        sample = DiarizationBenchmarkSample(
            id="meeting-1",
            audio=Path("/tmp/sample.wav"),
            reference_turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE"),
                SpeakerTurn(start_ms=1000, end_ms=2000, speaker="BOB"),
            ],
            num_speakers=2,
        )
        backend = FakeBackend()
        report = DiarizationBenchmarkRunner(backend).run(
            [sample], manifest="manifest.jsonl"
        )
        self.assertEqual(report.summary.aggregate_der, 0.0)
        self.assertEqual(report.summary.mean_jer, 0.0)
        self.assertEqual(backend.spec.num_speakers, 2)
        self.assertEqual(report.samples[0].timeline, "exclusive")

        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_report(report, directory)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["backend"],
                "fake-diarization",
            )
            self.assertIn(
                "说话人分离基准报告",
                markdown_path.read_text(encoding="utf-8"),
            )

    @patch("diarization_eval.runner.probe_file", return_value=_probe())
    def test_accuracy_matrix_runs_six_configurations(self, _probe_file):
        sample = DiarizationBenchmarkSample(
            id="meeting-1",
            audio=Path("/tmp/sample.wav"),
            rttm=Path("/tmp/sample.rttm"),
            audio_sha256="a" * 64,
            rttm_sha256="b" * 64,
            reference_turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE"),
                SpeakerTurn(start_ms=1000, end_ms=2000, speaker="BOB"),
            ],
            audio_kind="real",
            annotation_kind="human",
            tags={
                "dataset": "fixture",
                "dataset_version": "1",
                "split": "test",
                "license": "test-only",
            },
        )
        backend = FakeBackend()
        report = run_matrix(
            backend,
            [sample],
            manifest="manifest.jsonl",
            thresholds=ProductionThresholds(
                minimum_samples=1,
                minimum_duration_seconds=1,
            ),
        )
        self.assertEqual(len(report.reports), 6)
        self.assertEqual(len(backend.specs), 3)
        self.assertEqual(report.dataset.verified_real_human, True)
        self.assertEqual(report.recommendation.status, "accepted")
        self.assertTrue(
            all(
                benchmark.summary.speaker_count_accuracy == 1.0
                for benchmark in report.reports.values()
            )
        )

    def test_accuracy_matrix_rejects_synthetic_or_unverified_data(self):
        sample = DiarizationBenchmarkSample(
            id="synthetic",
            audio=Path("/tmp/sample.wav"),
            reference_turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="SYNTHETIC")
            ],
            audio_kind="synthetic",
            annotation_kind="synthetic",
        )
        with self.assertRaisesRegex(ValueError, "verified real/human RTTM"):
            require_real_human_dataset([sample])

    @patch("diarization_eval.runner.probe_file", return_value=_probe())
    def test_oracle_success_cannot_produce_acceptance(self, _probe_file):
        sample = DiarizationBenchmarkSample(
            id="meeting-1",
            audio=Path("/tmp/sample.wav"),
            rttm=Path("/tmp/sample.rttm"),
            audio_sha256="a" * 64,
            rttm_sha256="b" * 64,
            reference_turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="ALICE"),
                SpeakerTurn(start_ms=1000, end_ms=2000, speaker="BOB"),
            ],
            audio_kind="real",
            annotation_kind="human",
            tags={
                "dataset": "fixture",
                "dataset_version": "1",
                "split": "test",
                "license": "test-only",
            },
        )
        report = run_matrix(
            OracleOnlyBackend(),
            [sample],
            manifest="manifest.jsonl",
            thresholds=ProductionThresholds(
                minimum_samples=1,
                minimum_duration_seconds=1,
                maximum_der=0.1,
                maximum_jer=0.1,
                minimum_speaker_count_accuracy=1.0,
            ),
        )
        self.assertEqual(report.recommendation.status, "rejected")
        self.assertIn("unconstrained", " ".join(report.recommendation.reasons))

    def test_accuracy_matrix_rejects_missing_provenance(self):
        sample = DiarizationBenchmarkSample(
            id="unverified-real",
            audio=Path("/tmp/sample.wav"),
            reference_turns=[
                SpeakerTurn(start_ms=0, end_ms=1000, speaker="PERSON")
            ],
            audio_kind="real",
            annotation_kind="human",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 required"):
            require_real_human_dataset([sample])


if __name__ == "__main__":
    unittest.main()
