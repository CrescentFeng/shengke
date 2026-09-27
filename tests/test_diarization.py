from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from audio_core.contracts import (
    AudioSegment,
    DiarizationDocument,
    SpeakerTurn,
)
from ingest_worker.diarization import (
    DiarizationSpec,
    PyannoteDiarizationBackend,
    _turns_from_annotation,
    assign_speakers,
    build_diarization_backend,
)


class FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, *, yield_label):
        self.yield_label = yield_label
        for index, (turn, speaker) in enumerate(self.turns):
            yield turn, index, speaker


class DiarizationTests(unittest.TestCase):
    def test_spec_rejects_conflicting_speaker_constraints(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            DiarizationSpec(num_speakers=2, min_speakers=1)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            DiarizationSpec(min_speakers=3, max_speakers=2)

    def test_disabled_backend_requires_no_optional_dependency(self):
        backend = build_diarization_backend(
            "disabled",
            model_name="unused",
            token=None,
            device="cpu",
        )
        self.assertIsNone(backend)

    def test_pyannote_annotation_is_converted_to_sorted_milliseconds(self):
        annotation = FakeAnnotation(
            [
                (FakeTurn(1.2, 2.0), "SPEAKER_01"),
                (FakeTurn(0.1, 1.0), "SPEAKER_00"),
            ]
        )
        turns = _turns_from_annotation(annotation)
        self.assertEqual(
            [(turn.start_ms, turn.end_ms, turn.speaker) for turn in turns],
            [(100, 1000, "SPEAKER_00"), (1200, 2000, "SPEAKER_01")],
        )

    def test_pyannote_backend_prefers_exclusive_timeline(self):
        regular = FakeAnnotation([(FakeTurn(0.0, 1.0), "REGULAR")])
        exclusive = FakeAnnotation([(FakeTurn(0.0, 1.0), "EXCLUSIVE")])
        backend = object.__new__(PyannoteDiarizationBackend)
        backend.model_name = "fixture"
        backend.version = "test"
        backend._pipeline = lambda _path, **_kwargs: SimpleNamespace(
            speaker_diarization=regular,
            exclusive_speaker_diarization=exclusive,
        )

        result = backend.diarize(Path("fixture.wav"), spec=DiarizationSpec())
        self.assertTrue(result.exclusive)
        self.assertEqual(result.turns[0].speaker, "EXCLUSIVE")
        self.assertEqual(result.regular_turns[0].speaker, "REGULAR")
        self.assertEqual(result.exclusive_turns[0].speaker, "EXCLUSIVE")

    def test_segment_gets_speaker_with_most_overlap(self):
        segment = AudioSegment(
            asset_id=uuid4(),
            start_ms=100,
            end_ms=900,
            labels={"speech": 0.9},
        )
        updated = assign_speakers(
            [segment],
            [
                SpeakerTurn(start_ms=0, end_ms=450, speaker="SPEAKER_00"),
                SpeakerTurn(start_ms=450, end_ms=1000, speaker="SPEAKER_01"),
            ],
        )[0]
        self.assertEqual(updated.speaker, "SPEAKER_01")
        self.assertEqual(updated.labels["speaker_coverage"], 450 / 800)

    def test_document_rejects_incorrect_speaker_count(self):
        with self.assertRaisesRegex(ValueError, "distinct speakers"):
            DiarizationDocument(
                job_id=uuid4(),
                asset_id=uuid4(),
                backend="fixture",
                model="fixture",
                speaker_count=2,
                turns=[SpeakerTurn(start_ms=0, end_ms=100, speaker="SPEAKER_00")],
            )


if __name__ == "__main__":
    unittest.main()
