from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_core.vad import DetectedSpeechSegment, VadResult, VadSpec
from vad_eval.metrics import boundary_mae, duration_metrics, merge_intervals
from vad_eval.models import ReferenceInterval, VadBenchmarkSample, VadCandidate
from vad_eval.runner import VadBenchmarkRunner, load_manifest, write_report


def fake_detector(path, *, spec, ffmpeg):
    del path, ffmpeg
    shift = 20 if spec.minimum_speech_dbfs == -42 else 100
    return VadResult(
        backend=spec.backend,
        sample_rate_hz=16000,
        frame_ms=30,
        threshold_dbfs=spec.minimum_speech_dbfs,
        duration_ms=1000,
        speech_duration_ms=600,
        speech_ratio=0.6,
        segments=[
            DetectedSpeechSegment(
                start_ms=100 + shift,
                end_ms=900 - shift,
                confidence=0.9,
                mean_dbfs=-20,
            )
        ],
    )


class VadMetricTests(unittest.TestCase):
    def test_merge_and_duration_metrics_use_exact_interval_overlap(self):
        self.assertEqual(
            merge_intervals([(100, 300), (250, 500)], duration_ms=1000),
            [(100, 500)],
        )
        metrics = duration_metrics(
            [(100, 500)],
            [(200, 600)],
            duration_ms=1000,
        )
        self.assertEqual(metrics.true_positive_ms, 300)
        self.assertEqual(metrics.false_positive_ms, 100)
        self.assertEqual(metrics.false_negative_ms, 100)
        self.assertEqual(metrics.precision, 0.75)
        self.assertEqual(metrics.recall, 0.75)
        self.assertEqual(metrics.iou, 0.6)

    def test_boundary_mae_pairs_intervals_in_order(self):
        value, count = boundary_mae([(100, 900)], [(120, 880)])
        self.assertEqual(count, 1)
        self.assertEqual(value, 20.0)


class VadRunnerTests(unittest.TestCase):
    def test_manifest_model_rejects_overlapping_reference_segments(self):
        with self.assertRaisesRegex(ValueError, "ordered and non-overlapping"):
            VadBenchmarkSample(
                id="overlap",
                audio=Path("/tmp/sample.wav"),
                reference_segments=[
                    ReferenceInterval(start_ms=100, end_ms=500),
                    ReferenceInterval(start_ms=400, end_ms=900),
                ],
            )

    def test_manifest_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "audio": "sample.wav",
                        "reference_segments": [{"start_ms": 100, "end_ms": 900}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = load_manifest(manifest)[0]
            self.assertEqual(sample.audio, (root / "sample.wav").resolve())

    def test_runner_selects_highest_f1_candidate_and_writes_reports(self):
        sample = VadBenchmarkSample(
            id="sample-1",
            audio=Path("/tmp/sample.wav"),
            reference_segments=[ReferenceInterval(start_ms=100, end_ms=900)],
        )
        candidates = [
            VadCandidate(
                name="near",
                spec=VadSpec(minimum_speech_dbfs=-42),
            ),
            VadCandidate(
                name="far",
                spec=VadSpec(minimum_speech_dbfs=-36),
            ),
        ]
        report = VadBenchmarkRunner(detector=fake_detector).run(
            [sample], candidates, manifest="manifest.jsonl"
        )
        self.assertEqual(report.best_candidate, "near")
        self.assertGreater(report.candidates[0].f1 or 0, report.candidates[1].f1 or 0)

        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_report(report, directory)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["best_candidate"],
                "near",
            )
            self.assertIn("VAD 校准报告", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
