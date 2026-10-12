"""Run with the isolated Gradio environment; no synthesis or human scores are submitted."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import gradio  # noqa: F401
except ImportError:
    raise unittest.SkipTest("isolated Gradio environment required") from None


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "shengke_ui_test", ROOT / "apps/shengke-ui/src/shengke_ui/main.py"
)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class ShengkeUiTests(unittest.TestCase):
    def test_restore_is_read_only_and_preserves_trial_parameters(self):
        spec = {
            "model": "gpt-sovits-v2",
            "emotion": "neutral",
            "text": "你好，声刻。",
            "voice_id": "v",
            "inference_profile": "gpt_rhythm_v6",
        }
        with patch.object(ui, "request", return_value={"parameters": {"tts": spec}}) as request:
            result = ui.restore_job("job", [{"id": "v"}])
            request.assert_called_once_with("/v1/audio/jobs/job")
            assert result[6]["value"] is True
            assert result[4]["interactive"] is False
            assert result[9:12] == (None, None, "")

    def test_failure_clears_previous_audio_and_reenables_generate(self):
        with patch.object(ui, "create_job", side_effect=RuntimeError("offline")):
            outputs = list(ui.generate_guarded([], None, "你好", "cosyvoice2", "neutral"))
        assert outputs[0][2:4] == (None, None)
        assert outputs[0][4]["interactive"] is False
        assert "offline" in outputs[1][1]
        assert outputs[1][2:4] == (None, None)
        assert outputs[-1][4]["interactive"] is True

    def test_example_missing_does_not_submit_or_download(self):
        with patch.object(ui, "request") as request:
            with self.assertRaisesRegex(Exception, "示例尚未导入"):
                ui.use_example([])
            request.assert_not_called()

    def test_history_failed_task_does_not_reuse_audio(self):
        with (
            patch.object(ui, "poll", return_value={"state": "failed", "error_message": "早停保护"}),
            patch.object(ui, "audio_file") as audio,
        ):
            outputs = list(ui.history_result("job"))
            assert outputs[-1][:2] == (None, None)
            assert outputs[-1][3] == "早停保护"
            audio.assert_not_called()

    def test_ui_build_and_double_submit_policy(self):
        app = ui.build_app()
        assert any(c["props"].get("label") == "历史与下载" for c in app.config["components"])
        generate = [
            d for d in app.config["dependencies"] if d.get("api_name") == "generate_guarded"
        ][0]
        assert generate["trigger_mode"] == "once"


if __name__ == "__main__":
    unittest.main()
