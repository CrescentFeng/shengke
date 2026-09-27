from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from ingest_worker.asr import (
    AsrChunk,
    AsrDependencyError,
    AsrRequestOptions,
    FasterWhisperBackend,
    FunAsrBackend,
    build_asr_backend,
)


class AsrAdapterTests(unittest.TestCase):
    def test_hotwords_are_trimmed_and_deduplicated(self):
        options = AsrRequestOptions(hotwords=[" Audio AI ", "audio ai", "客服质检"])
        self.assertEqual(options.hotwords, ["Audio AI", "客服质检"])
        with self.assertRaisesRegex(ValueError, "blank terms"):
            AsrRequestOptions(hotwords=[" "])

    def test_chunk_rejects_reversed_interval(self):
        with self.assertRaises(ValueError):
            AsrChunk(start_ms=100, end_ms=100, text="invalid")

    def test_disabled_backend_requires_no_optional_dependency(self):
        backend = build_asr_backend(
            "disabled",
            model_name="small",
            device="cpu",
            compute_type="int8",
            beam_size=5,
        )
        self.assertIsNone(backend)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            build_asr_backend(
                "unknown",
                model_name="small",
                device="cpu",
                compute_type="int8",
                beam_size=5,
            )

    def test_funasr_rich_timestamps_are_converted_from_seconds(self):
        chunks = FunAsrBackend._chunks(
            {
                "text": "你好",
                "timestamps": [
                    {"token": "你", "start_time": 0.1, "end_time": 0.25},
                    {"token": "好", "start_time": 0.25, "end_time": 0.5},
                ],
            }
        )
        self.assertEqual(
            [(chunk.start_ms, chunk.end_ms) for chunk in chunks], [(100, 250), (250, 500)]
        )

    def test_faster_whisper_receives_space_separated_hotwords(self):
        class FakeModel:
            def transcribe(self, _path, **kwargs):
                self.kwargs = kwargs
                return iter(()), SimpleNamespace(language="zh", language_probability=0.9)

        backend = object.__new__(FasterWhisperBackend)
        backend.beam_size = 5
        backend._model = FakeModel()
        backend.transcribe(Path("fixture.wav"), hotwords=["音频智能", "客服质检"])
        self.assertEqual(backend._model.kwargs["hotwords"], "音频智能 客服质检")

    def test_funasr_receives_singular_hotword_parameter(self):
        class FakeModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [{"text": "音频智能"}]

        backend = object.__new__(FunAsrBackend)
        backend._model = FakeModel()
        backend.transcribe(Path("fixture.wav"), hotwords=["音频智能", "客服质检"])
        self.assertEqual(backend._model.kwargs["hotword"], "音频智能 客服质检")

    def test_missing_faster_whisper_has_clear_error(self):
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            with self.assertRaises(AsrDependencyError):
                build_asr_backend(
                    "faster-whisper",
                    model_name="small",
                    device="cpu",
                    compute_type="int8",
                    beam_size=5,
                )


if __name__ == "__main__":
    unittest.main()
