from __future__ import annotations

import hashlib
import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audio_core.errors import InvalidAudioInput
from audio_core.hashing import sha256_file
from audio_core.normalize import NormalizeSpec, normalize_audio
from audio_core.probe import probe_file
from audio_core.quality import QualityAnalysisSpec, analyze_quality
from audio_core.segments import SegmentExportSpec, extract_audio_segment
from audio_core.vad import VadSpec, detect_voice_activity


def write_sine(path: Path, *, sample_rate: int = 8000, duration: float = 0.25) -> None:
    frame_count = int(sample_rate * duration)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        output.writeframes(frames)


def write_silence_then_clipped(path: Path, *, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack("<h", 0) * (sample_rate // 2))
        output.writeframes(struct.pack("<h", 32767) * (sample_rate // 2))


def write_speech_pattern(path: Path, *, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray(struct.pack("<h", 0) * int(sample_rate * 0.3))
        for index in range(int(sample_rate * 0.6)):
            value = int(9000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        frames.extend(struct.pack("<h", 0) * int(sample_rate * 0.5))
        output.writeframes(frames)


class AudioCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input.wav"
        write_sine(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_sha256_matches_standard_library(self):
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(self.source), expected)

    def test_missing_input_is_rejected(self):
        with self.assertRaises(InvalidAudioInput):
            sha256_file(self.root / "missing.wav")

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is not installed")
    def test_probe_wav(self):
        result = probe_file(self.source)
        self.assertEqual(result.codec_name, "pcm_s16le")
        self.assertEqual(result.sample_rate_hz, 8000)
        self.assertEqual(result.channels, 1)
        self.assertAlmostEqual(result.duration_seconds or 0, 0.25, places=2)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg is not installed",
    )
    def test_normalize_creates_derivative_without_overwriting_source(self):
        original_hash = sha256_file(self.source)
        destination = self.root / "derived" / "normalized.wav"
        result = normalize_audio(
            self.source,
            destination,
            spec=NormalizeSpec(sample_rate_hz=16000, channels=1),
        )
        self.assertEqual(result, destination.resolve())
        self.assertEqual(sha256_file(self.source), original_hash)
        probe = probe_file(destination)
        self.assertEqual(probe.sample_rate_hz, 16000)
        self.assertEqual(probe.channels, 1)

    def test_normalize_refuses_to_overwrite_original(self):
        with self.assertRaises(InvalidAudioInput):
            normalize_audio(self.source, self.source)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_quality_analysis_detects_silence_and_clipping(self):
        source = self.root / "quality.wav"
        write_silence_then_clipped(source)
        result = analyze_quality(
            source,
            duration_seconds=1.0,
            spec=QualityAnalysisSpec(minimum_silence_seconds=0.1),
        )
        self.assertIsNotNone(result.integrated_lufs)
        self.assertGreaterEqual(result.silence_ratio, 0.45)
        self.assertLessEqual(result.silence_ratio, 0.55)
        self.assertGreater(result.clipping_ratio, 0.45)
        self.assertIsNotNone(result.peak_dbfs)
        self.assertAlmostEqual(result.peak_dbfs, 0.0, places=2)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg is not installed",
    )
    def test_vad_detects_and_exports_millisecond_segment(self):
        source = self.root / "speech-pattern.wav"
        write_speech_pattern(source)
        result = detect_voice_activity(
            source,
            spec=VadSpec(
                minimum_speech_ms=150,
                maximum_silence_ms=120,
                padding_ms=90,
            ),
        )
        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertGreaterEqual(segment.start_ms, 150)
        self.assertLessEqual(segment.start_ms, 300)
        self.assertGreaterEqual(segment.end_ms, 900)
        self.assertLessEqual(segment.end_ms, 1050)

        destination = self.root / "segments" / "speech.wav"
        extract_audio_segment(
            source,
            destination,
            spec=SegmentExportSpec(start_ms=segment.start_ms, end_ms=segment.end_ms),
        )
        exported = probe_file(destination)
        expected_duration = (segment.end_ms - segment.start_ms) / 1000
        self.assertAlmostEqual(exported.duration_seconds or 0, expected_duration, places=2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_vad_returns_no_segments_for_silence(self):
        source = self.root / "silence.wav"
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<h", 0) * 16000)
        result = detect_voice_activity(source)
        self.assertEqual(result.segments, [])
        self.assertEqual(result.speech_duration_ms, 0)
        self.assertEqual(result.speech_ratio, 0.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_webrtc_backend_uses_pluggable_frame_classifier(self):
        source = self.root / "speech-pattern-webrtc.wav"
        write_speech_pattern(source)

        class FakeVad:
            def __init__(self, mode):
                self.mode = mode

            def is_speech(self, frame, sample_rate):
                self.sample_rate = sample_rate
                return any(frame)

        with patch.dict(
            "sys.modules",
            {"webrtcvad": SimpleNamespace(Vad=FakeVad)},
        ):
            result = detect_voice_activity(
                source,
                spec=VadSpec(
                    backend="webrtc_v1",
                    webrtc_aggressiveness=3,
                    minimum_speech_ms=150,
                    maximum_silence_ms=120,
                    padding_ms=90,
                ),
            )

        self.assertEqual(result.backend, "webrtc_v1")
        self.assertEqual(result.aggressiveness, 3)
        self.assertIsNone(result.threshold_dbfs)
        self.assertEqual(len(result.segments), 1)
        self.assertGreaterEqual(result.segments[0].start_ms, 150)
        self.assertLessEqual(result.segments[0].end_ms, 1050)


if __name__ == "__main__":
    unittest.main()
