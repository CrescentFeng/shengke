"""Content diagnostics warn about gross mismatches without certifying quality."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "shengke_check_content", Path(__file__).parents[1] / "scripts/shengke_check_content.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_same_content_with_spacing_and_punctuation_is_not_a_quality_pass():
    result = module.compare("你好，世界！", "你 好。世 界", 0.6)
    assert result["raw_character_error_ratio"] == 0
    assert result["status"] == "no_gross_mismatch_detected"
    assert result["human_acceptance"] == "not_assessed"


def test_reference_leak_and_missing_target_require_review():
    result = module.compare("清晨的阳光穿过窗帘", "发射约二十分钟后飞行样机打开降落伞", 0.6)
    assert result["status"] == "review_required"
    assert module.compare("你好", "", 0.6)["status"] == "review_required"


def test_single_pronunciation_difference_is_preserved_without_auto_rejection():
    result = module.compare("清晨的阳光穿过窗帘", "清水的阳光穿过窗帘", 0.6)
    assert result["edit_distance"] == 1
    assert result["human_acceptance"] == "not_assessed"


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        module.compare("你好", "你好", threshold)


def test_empty_expected_text():
    with pytest.raises(ValueError):
        module.compare(" ！", "你好", 0.6)
