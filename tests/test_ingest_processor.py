from __future__ import annotations

import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from audio_core.contracts import AudioJob, CreateAudioJob, JobState, Operation, SpeakerTurn
from ingest_worker.asr import AsrChunk, AsrResult
from ingest_worker.diarization import DiarizationResult
from ingest_worker.processor import IngestProcessor


def write_sine(path: Path, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray(struct.pack("<h", 0) * int(sample_rate * 0.3))
        for index in range(int(sample_rate * 0.4)):
            value = int(6000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        frames.extend(struct.pack("<h", 0) * int(sample_rate * 0.3))
        output.writeframes(frames)


class FakeArtifacts:
    def __init__(self, source: Path, root: Path) -> None:
        self.source = source
        self.root = root
        self.uploaded: list[str] = []

    def fetch_input(self, _uri: str, destination: Path) -> Path:
        shutil.copyfile(self.source, destination)
        return destination

    def upload_file(self, source: Path, key: str, content_type: str | None = None) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        self.uploaded.append(key)
        return f"s3://test-artifacts/{key}"


class FakeMetadata:
    def __init__(self) -> None:
        self.assets = []
        self.runs = []
        self.reports = []
        self.transcripts = []
        self.diarizations = []
        self.segments = {}
        self.transitions = []

    def add_asset(self, job_id, asset, role):
        self.assets.append((job_id, asset, role))

    def add_tool_run(self, job_id, run):
        self.runs.append((job_id, run))

    def add_segment(self, job_id, segment):
        self.segments[segment.id] = (job_id, segment)

    def save_report(self, report, report_uri):
        self.reports.append((report, report_uri))

    def save_transcript(self, transcript, transcript_uri):
        self.transcripts.append((transcript, transcript_uri))

    def save_diarization(self, diarization, diarization_uri):
        self.diarizations.append((diarization, diarization_uri))

    def transition(self, job_id, target, **kwargs):
        self.transitions.append((job_id, target, kwargs))
        return None


class FakeAsr:
    name = "fake-asr"
    model_name = "fixture-v1"
    version = "1"

    def transcribe(self, path, *, language=None, hotwords=None):
        self.last_path = path
        self.last_hotwords = hotwords
        return AsrResult(
            text="hello audio",
            language=language or "en",
            language_confidence=0.95,
            chunks=[
                AsrChunk(start_ms=20, end_ms=100, text="hello", confidence=0.9),
                AsrChunk(start_ms=100, end_ms=180, text="audio", confidence=0.85),
            ],
        )


class FakeDiarization:
    name = "fake-diarization"
    model_name = "fixture-speakers-v1"
    version = "1"

    def diarize(self, path, *, spec):
        self.last_path = path
        self.last_spec = spec
        return DiarizationResult(
            exclusive=True,
            turns=[
                SpeakerTurn(start_ms=0, end_ms=450, speaker="SPEAKER_00"),
                SpeakerTurn(start_ms=450, end_ms=1000, speaker="SPEAKER_01"),
            ],
        )


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
class IngestProcessorTests(unittest.TestCase):
    def test_pipeline_persists_original_normalized_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            write_sine(source)
            metadata = FakeMetadata()
            artifacts = FakeArtifacts(source, root / "objects")
            request = CreateAudioJob(
                input_uri="s3://audio-input/source.wav",
                operations=[
                    Operation.PROBE,
                    Operation.VAD,
                    Operation.DIARIZE,
                    Operation.TRANSCRIBE,
                    Operation.LOUDNESS,
                    Operation.NORMALIZE,
                ],
                idempotency_key="worker-0001",
                parameters={
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "quality_analysis": {"minimum_silence_seconds": 0.1},
                    "asr": {"hotwords": ["Audio AI", "客服质检"]},
                },
            )
            job = AudioJob.from_request(request)
            job.state = JobState.RUNNING
            report = IngestProcessor(
                metadata,
                artifacts,
                asr=FakeAsr(),
                asr_language="en",
                diarization=FakeDiarization(),
            ).process(job)

            self.assertEqual(len(metadata.assets), 3)
            self.assertEqual(
                [entry[2] for entry in metadata.assets],
                ["original", "speech_segment", "normalized"],
            )
            self.assertEqual(len(metadata.segments), 1)
            self.assertEqual(len(metadata.runs), 6)
            self.assertEqual(len(metadata.reports), 1)
            self.assertEqual(len(metadata.transcripts), 1)
            self.assertEqual(len(metadata.diarizations), 1)
            self.assertEqual(metadata.transitions[-1][1], JobState.SUCCEEDED)
            self.assertEqual(report.assets[-1].probe.sample_rate_hz, 16000)
            self.assertEqual(len(report.segments), 1)
            self.assertEqual(report.segments[0].slice_asset_id, report.assets[1].id)
            self.assertEqual(report.segments[0].transcript, "hello audio")
            self.assertEqual(report.segments[0].speaker, "SPEAKER_01")
            transcript = metadata.transcripts[0][0]
            self.assertEqual(transcript.text, "hello audio")
            self.assertEqual(transcript.language, "en")
            self.assertEqual(len(transcript.spans), 2)
            self.assertEqual(transcript.spans[0].speaker, "SPEAKER_01")
            self.assertGreater(report.segments[0].start_ms, 0)
            self.assertEqual(
                transcript.spans[0].start_ms,
                report.segments[0].start_ms + 20,
            )
            self.assertIsNotNone(report.summary["quality"])
            self.assertIsNotNone(report.summary["vad"])
            self.assertEqual(report.summary["diarization"]["speaker_count"], 2)
            self.assertEqual(report.summary["transcript"]["language"], "en")
            self.assertEqual(report.tool_runs[3].metrics["hotword_count"], 2)
            self.assertEqual(report.tool_runs[1].operation, Operation.VAD)
            self.assertEqual(report.tool_runs[2].operation, Operation.DIARIZE)
            self.assertEqual(report.tool_runs[3].operation, Operation.TRANSCRIBE)
            self.assertTrue(any(key.endswith(".json") for key in artifacts.uploaded))


if __name__ == "__main__":
    unittest.main()
