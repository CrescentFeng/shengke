from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "shengke_runtime", ROOT / "services/tts-runtime/server.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeProvenanceTests(unittest.TestCase):
    def test_zero_semantic_span_retries_full_text_without_decoding_reference(self):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.model_id = "gpt-sovits-v2"
        calls, decoded = [], []
        decoder = SimpleNamespace(
            infer_panel_naive_batched=Mock(return_value=([np.arange(265)], [0]))
        )
        model = SimpleNamespace(t2s_model=SimpleNamespace(model=decoder))

        def run(parameters):
            calls.append(dict(parameters))
            try:
                model.t2s_model.model.infer_panel_naive_batched()
            except runtime.GptEarlyTerminationError:
                # Mirror upstream yielding silence and rebuilding on error.
                yield 32000, np.zeros(32000, dtype=np.int16)
                model.t2s_model.model = SimpleNamespace(
                    infer_panel_naive_batched=Mock(return_value=([np.arange(364)], [100]))
                )
                raise
            decoded.append(True)
            yield 32000, np.ones(320, dtype=np.float32)

        model.run = run
        backend.model = model
        spec = {
            "model": "gpt-sovits-v2",
            "text": "清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。",
            "reference_text": "参考文本保持不变。",
            "inference_profile": "gpt_pause_v3",
        }
        sr, audio = backend.generate(spec, Path("reference.wav"))
        self.assertEqual([c["text_split_method"] for c in calls], ["cut5", "cut0"])
        self.assertEqual({**calls[0], "text_split_method": "cut0"}, calls[1])
        self.assertEqual(decoded, [True])
        self.assertEqual((sr, len(audio)), (32000, 320))  # No failed-run silence included.
        self.assertEqual(backend.generation_delivery["text_split_method"], "cut0")
        self.assertEqual(
            backend.generation_delivery["fallback_reason"], "early_semantic_termination"
        )

    def test_invalid_semantic_span_fails_without_publishing_or_looping(self):
        for length in [0, -1, 11]:
            with self.subTest(length=length):
                backend = runtime.Backend.__new__(runtime.Backend)
                backend.model_id = "gpt-sovits-v2"
                decoder = SimpleNamespace(
                    infer_panel_naive_batched=Mock(return_value=([np.arange(10)], [length]))
                )

                def run(parameters, decoder=decoder):
                    decoder.infer_panel_naive_batched()
                    self.fail("invalid semantic span reached waveform decoding")
                    yield

                backend.model = SimpleNamespace(t2s_model=SimpleNamespace(model=decoder), run=run)
                with self.assertRaises(runtime.GptEarlyTerminationError):
                    backend.generate(
                        {"model": "gpt-sovits-v2", "text": "你好。", "reference_text": "参考"},
                        Path("ref.wav"),
                    )

    def test_failed_whole_text_retry_does_not_return_upstream_silence(self):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.model_id = "gpt-sovits-v2"
        decoder = SimpleNamespace(
            infer_panel_naive_batched=Mock(return_value=([np.arange(265)], [0]))
        )
        attempts = []

        def run(parameters):
            attempts.append(parameters["text_split_method"])
            try:
                decoder.infer_panel_naive_batched()
            except runtime.GptEarlyTerminationError:
                yield 32000, np.zeros(32000, dtype=np.int16)
                raise
            self.fail("invalid span reached waveform decoding")

        backend.model = SimpleNamespace(t2s_model=SimpleNamespace(model=decoder), run=run)
        with self.assertRaises(runtime.GptEarlyTerminationError):
            backend.generate(
                {
                    "model": "gpt-sovits-v2",
                    "text": "清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。",
                    "reference_text": "参考文本保持不变。",
                    "inference_profile": "gpt_pause_v3",
                },
                Path("reference.wav"),
            )
        self.assertEqual(attempts, ["cut5", "cut0"])

    def test_delivery_reaches_cosyvoice_without_changing_text_or_reference(self):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.model_id = "cosyvoice2"
        tensor = Mock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = np.ones(240)
        backend.model = Mock(sample_rate=24000)
        backend.model.inference_instruct2.return_value = [{"tts_speech": tensor}]
        backend.model.inference_zero_shot.return_value = [{"tts_speech": tensor}]
        spec = {"model": "cosyvoice2", "text": "你终于回来了。", "reference_text": "你好。"}
        backend.generate({**spec, "emotion": "happy"}, Path("ref.wav"))
        original = backend.model.inference_instruct2.call_args
        self.assertEqual(original.args[1], runtime.PROMPTS["happy"] + "<|endofprompt|>")
        self.assertEqual(original.kwargs["speed"], 1.0)
        backend.generate(
            {**spec, "emotion": "happy", "inference_profile": "pilot_v2"}, Path("ref.wav")
        )
        candidate = backend.model.inference_instruct2.call_args
        self.assertNotEqual(candidate.args[1], original.args[1])
        self.assertEqual(candidate.args[1].count("<|endofprompt|>"), 1)
        self.assertEqual((candidate.args[0], candidate.args[2]), (spec["text"], "ref.wav"))
        backend.generate({**spec, "inference_profile": "pilot_v2"}, Path("ref.wav"))
        neutral = backend.model.inference_zero_shot.call_args
        self.assertEqual(neutral.args, (spec["text"], spec["reference_text"], "ref.wav"))
        self.assertEqual(neutral.kwargs["speed"], 0.92)

    def test_gpt_candidate_preserves_whole_sentence_and_controls_native_speed(self):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.model_id = "gpt-sovits-v2"
        backend.model = Mock()
        backend.model.run.return_value = [(32000, np.ones(320, dtype=np.float32))]
        spec = {
            "model": "gpt-sovits-v2",
            "text": "清晨的阳光，落在书桌旁。",
            "reference_text": "你好。",
            "inference_profile": "pilot_v2",
        }
        backend.generate(spec, Path("ref.wav"))
        args = backend.model.run.call_args.args[0]
        self.assertEqual(args["speed_factor"], 0.9)
        self.assertEqual(args["text_split_method"], "cut0")
        self.assertEqual(args["text"], spec["text"])
        self.assertEqual(args["seed"], 42)

    def test_unknown_or_unsupported_profiles_fail_instead_of_silent_fallback(self):
        for spec in [
            {"model": "cosyvoice2", "inference_profile": "typo"},
            {"model": "vits-csmsc", "inference_profile": "pilot_v2"},
            {"model": "cosyvoice2", "inference_profile": "gpt_pause_v3"},
        ]:
            with self.assertRaises(ValueError):
                runtime.delivery_settings(spec)

    def test_pause_trial_keeps_speed_and_reference_and_uses_native_splitting(self):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.model_id = "gpt-sovits-v2"
        backend.model = Mock()
        backend.model.run.return_value = [(32000, np.ones(320, dtype=np.float32))]
        text = "清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。"
        spec = {"model": "gpt-sovits-v2", "text": text, "reference_text": "参考原文"}
        backend.generate(spec, Path("ref.wav"))
        original = backend.model.run.call_args.args[0]
        backend.generate({**spec, "inference_profile": "gpt_pause_v3"}, Path("ref.wav"))
        candidate = backend.model.run.call_args.args[0]
        self.assertEqual(original["text_split_method"], "cut0")
        self.assertEqual(candidate["text_split_method"], "cut5")
        self.assertEqual(candidate["speed_factor"], 1.0)
        self.assertEqual(candidate["fragment_interval"], 0.3)
        # The sole changed inference argument is the native segmentation method.
        self.assertEqual({**original, "text_split_method": "cut5"}, candidate)

    def test_pause_trial_does_not_split_short_or_ambiguous_clauses(self):
        for text in [
            "你好，欢迎来到声刻。",
            "清晨的阳光穿过窗帘，谢谢。",
            "你终于回来了，我一直在这里等你。",
            "清晨的阳光穿过窗帘。",
            "版本是3.14，请保持数字的小数点。",
            "清晨的阳光穿过窗帘\n落在书桌旁的一杯热茶上。",
        ]:
            with self.subTest(text=text):
                settings = runtime.delivery_settings(
                    {"model": "gpt-sovits-v2", "inference_profile": "gpt_pause_v3", "text": text}
                )
                self.assertEqual(settings["text_split_method"], "cut0")
                self.assertEqual(settings["segmentation_reason"], "retain_whole_utterance")

    def test_vocoder_and_bert_weights_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [
                "bert.bin",
                "vocoder.pkl",
                "incomplete.pth.partial",
                "config.yaml",
                "._bert.bin",
            ]:
                (root / name).write_bytes(b"test fixture")
            self.assertEqual(set(runtime.fingerprints(root)), {"bert.bin", "vocoder.pkl"})

    def test_baseline_model_cannot_be_mislabeled(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            runtime.validate_espnet_configuration(
                "transformer-csmsc", {"tts": "vits", "g2p": "pypinyin_g2p_phone"}
            )

    def test_vocoder_must_match_acoustic_features(self):
        acoustic = {
            "tts": "transformer",
            "g2p": "pypinyin_g2p_phone",
            "feats_extract_conf": {
                "fs": 24000,
                "hop_length": 300,
                "n_mels": 80,
                "n_fft": 2048,
                "win_length": 1200,
                "fmin": 80,
                "fmax": 7600,
            },
        }
        vocoder = {
            "sampling_rate": 24000,
            "hop_size": 300,
            "num_mels": 80,
            "fft_size": 2048,
            "win_length": 1200,
            "fmin": 80,
            "fmax": 7600,
        }
        runtime.validate_espnet_configuration("transformer-csmsc", acoustic, vocoder)
        vocoder["sampling_rate"] = 22050
        with self.assertRaisesRegex(ValueError, "feature mismatch"):
            runtime.validate_espnet_configuration("transformer-csmsc", acoustic, vocoder)


if __name__ == "__main__":
    unittest.main()


def test_short_happy_instruction_changes_only_instruction():
    source = {"model": "cosyvoice2", "emotion": "happy", "text": "你终于回来了。"}
    baseline = runtime.delivery_settings(source)
    candidate = runtime.delivery_settings({**source, "inference_profile": "cosy_happy_v4"})
    assert candidate["instruction"] == "用开心的语气说这句话。<|endofprompt|>"
    assert candidate["speed"] == baseline["speed"] == 1.0
    assert set(candidate) == set(baseline)
    for model, emotion in [("cosyvoice2", "neutral"), ("gpt-sovits-v2", "happy")]:
        with unittest.TestCase().assertRaises(ValueError):
            runtime.delivery_settings(
                {"model": model, "emotion": emotion, "inference_profile": "cosy_happy_v4"}
            )


def test_rhythm_trial_preserves_native_generation_and_processes_only_opt_in():
    backend = runtime.Backend.__new__(runtime.Backend)
    backend.model_id = "cosyvoice2"
    tensor = Mock()
    original = np.ones(240, dtype=np.float32)
    tensor.detach.return_value.cpu.return_value.numpy.return_value = original
    backend.model = Mock(sample_rate=24000)
    backend.model.inference_instruct2.return_value = [{"tts_speech": tensor}]
    source = {"model": "cosyvoice2", "emotion": "happy", "text": "你终于回来了。"}
    processed = np.ones(280, dtype=np.float32)
    with patch.object(runtime, "stretch_rhythm", return_value=(processed, {"tempo": 0.85})) as fx:
        _, baseline = backend.generate(source, Path("ref.wav"))
        fx.assert_not_called()
        assert np.array_equal(baseline, original)
        _, result = backend.generate(
            {**source, "inference_profile": "cosy_rhythm_v5"}, Path("ref.wav")
        )
        assert result is processed
        assert backend.model.inference_instruct2.call_args.kwargs["speed"] == 1.0
        assert backend.model.inference_instruct2.call_args.args == (
            source["text"],
            "用开心的语气说这句话。<|endofprompt|>",
            "ref.wav",
        )
        assert fx.call_args.args[1:] == (24000, 0.85)
        assert backend.generation_delivery["postprocessing"]["tempo"] == 0.85
    with patch.object(runtime, "stretch_rhythm", side_effect=RuntimeError("failed")):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "failed"):
            backend.generate({**source, "inference_profile": "cosy_rhythm_v5"}, Path("ref.wav"))


def test_rhythm_processing_extends_audio_without_shifting_test_tone():
    if importlib.util.find_spec("soundfile") is None:
        raise unittest.SkipTest("Run this audio integration check in the model environment")
    rate = 24000
    waveform = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
    result, info = runtime.stretch_rhythm(waveform, rate, 0.85)
    assert abs(len(result) / rate - 2 / 0.85) < 0.08
    assert np.isfinite(result).all()
    peak_hz = np.fft.rfftfreq(len(result), 1 / rate)[np.argmax(abs(np.fft.rfft(result)))]
    assert abs(peak_hz - 440) < 3
    assert info["input_audio_duration_s"] == 2
    assert len(info["input_wav_sha256"]) == 64
    with patch.object(runtime.subprocess, "run", side_effect=FileNotFoundError):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "FFmpeg"):
            runtime.stretch_rhythm(waveform, rate, 0.85)


def test_gpt_rhythm_processes_only_successful_retry_and_keeps_baseline_separate():
    backend = runtime.Backend.__new__(runtime.Backend)
    backend.model_id = "gpt-sovits-v2"
    waveform = np.ones(320, dtype=np.float32) * 0.1
    spec = {
        "model": "gpt-sovits-v2",
        "text": "清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。",
        "reference_text": "参考文本",
        "inference_profile": "gpt_rhythm_v6",
    }
    backend.gpt_chunks = Mock(side_effect=[runtime.GptEarlyTerminationError(), [(32000, waveform)]])
    transformed = np.ones(400, dtype=np.float32) * 0.1
    with patch.object(
        runtime, "apply_gpt_rhythm", return_value=(transformed, {"tempo": 0.9})
    ) as fx:
        rate, result = backend.generate(spec, Path("ref.wav"))
        assert result is transformed and rate == 32000
        fx.assert_called_once()
        assert np.array_equal(fx.call_args.args[0], waveform)
        assert backend.generation_delivery["fallback_reason"] == "early_semantic_termination"
        assert backend.generation_delivery["postprocessing"] == {"tempo": 0.9}
        backend.gpt_chunks = Mock(return_value=[(32000, waveform)])
        fx.reset_mock()
        backend.generate({**spec, "inference_profile": "baseline"}, Path("ref.wav"))
        fx.assert_not_called()


def test_gpt_rhythm_missing_configuration_does_not_return_plain_audio():
    if importlib.util.find_spec("soundfile") is None:
        raise unittest.SkipTest("Run in model environment with soundfile")
    with patch.dict(runtime.os.environ, {}, clear=True):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "尚未配置"):
            runtime.apply_gpt_rhythm(np.ones(320), 32000, "你好。")
