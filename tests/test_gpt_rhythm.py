"""Test uncertainty handling; silence is never inferred just from low speech energy."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "gpt_rhythm", Path(__file__).resolve().parents[1] / "services/tts-runtime/gpt_rhythm.py"
)
rhythm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rhythm)
TEXT = "春天的微风吹过树梢，孩子们开心地跑向操场。"


def words(text):
    return [
        {"word": c, "start": i * 0.2, "end": (i + 1) * 0.2, "probability": 0.99}
        for i, c in enumerate(rhythm.chinese(text))
    ]


def test_insert_keeps_frames_and_skips_existing_silence():
    source = np.full(5000, 0.02, dtype=np.float32)
    decisions = rhythm.plan(TEXT, words(TEXT), 5)["boundaries"]
    out = rhythm.insert_pauses(source, 1000, decisions)
    assert decisions[0]["action"] == "insert"
    assert len(out) == len(source) + 198
    assert np.array_equal(source, np.full(5000, 0.02, dtype=np.float32))
    source[1690:1900] = 0
    decisions = rhythm.plan(TEXT, words(TEXT), 5)["boundaries"]
    out = rhythm.insert_pauses(source, 1000, decisions)
    assert decisions[0]["action"] == "keep"
    assert np.array_equal(out, source)


@pytest.mark.parametrize("case", ["mismatch", "deletion", "low_confidence", "same_token"])
def test_uncertain_boundary_never_edits_audio(case):
    observed = words(TEXT)
    if case == "mismatch":
        observed = words("这是完全不同的一段文字，不能推断停顿位置。")
    elif case == "deletion":
        del observed[8]
    elif case == "low_confidence":
        observed[8]["probability"] = 0.2
    else:
        observed[8]["word"] += observed[9]["word"]
        observed[8]["end"] = observed[9]["end"]
        del observed[9]
    decisions = rhythm.plan(TEXT, observed, 6)["boundaries"]
    assert decisions[0]["action"] == "skip"
    source = np.ones(6000, dtype=np.float32) * 0.01
    assert np.array_equal(rhythm.insert_pauses(source, 1000, decisions), source)


def test_invalid_timing_and_decimal_input_rejected():
    observed = words(TEXT)
    observed[8]["end"] = float("nan")
    with pytest.raises(ValueError, match="timing"):
        rhythm.plan(TEXT, observed, 6)
    with pytest.raises(ValueError):
        rhythm.boundaries("版本是3.14，继续测试。")
    assert rhythm.boundaries("这是没有句内标点的完整句子。") == []


def test_close_multiple_punctuation_is_not_duplicated():
    assert rhythm.boundaries("你好？！欢迎回来。") == [2]
    text = "你好，欢迎回来，今天我们一起散步。"
    decisions = rhythm.plan(text, words(text), 5)["boundaries"]
    source = np.ones(5000, dtype=np.float32) * 0.01
    out = rhythm.insert_pauses(source, 1000, decisions)
    assert len(out) == len(source) + 198 * 2
    assert all(p["action"] == "insert" for p in decisions)
