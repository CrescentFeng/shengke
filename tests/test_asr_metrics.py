from __future__ import annotations

import unittest

from asr_eval.metrics import (
    character_error_rate,
    normalize_language,
    normalize_text,
    timestamp_boundary_mae,
    word_error_rate,
)
from asr_eval.models import ReferenceSpan
from ingest_worker.asr import AsrChunk


class AsrMetricTests(unittest.TestCase):
    def test_normalization_is_unicode_and_punctuation_aware(self):
        self.assertEqual(normalize_text("Ｈｅｌｌｏ， WORLD!"), "hello world")

    def test_word_error_rate_counts_deletion(self):
        result = word_error_rate("hello brave world", "hello world")
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.reference_units, 3)
        self.assertAlmostEqual(result.rate, 1 / 3)

    def test_character_error_rate_supports_chinese(self):
        result = character_error_rate("你好世界", "你好世")
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.reference_units, 4)
        self.assertEqual(result.rate, 0.25)

    def test_empty_reference_has_defined_rate(self):
        self.assertEqual(word_error_rate("", "").rate, 0.0)
        self.assertEqual(word_error_rate("", "unexpected").rate, 1.0)

    def test_timestamp_boundary_mae_pairs_segments_in_order(self):
        result, pair_count = timestamp_boundary_mae(
            [ReferenceSpan(start_ms=100, end_ms=900, text="hello")],
            [AsrChunk(start_ms=120, end_ms=880, text="hello")],
        )
        self.assertEqual(pair_count, 1)
        self.assertEqual(result, 20.0)

    def test_language_normalization_handles_common_aliases(self):
        self.assertEqual(normalize_language("zh-CN"), "zh")
        self.assertEqual(normalize_language("Mandarin"), "zh")
        self.assertEqual(normalize_language("EN_us"), "en")


if __name__ == "__main__":
    unittest.main()
