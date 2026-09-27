from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from ingest_worker.asr import AsrChunk

from .models import ReferenceSpan


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    reference_units: int
    rate: float


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_unit != hypothesis_unit
            )
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRate:
    errors = edit_distance(reference, hypothesis)
    denominator = len(reference)
    if denominator == 0:
        return ErrorRate(errors=errors, reference_units=0, rate=0.0 if errors == 0 else 1.0)
    return ErrorRate(errors=errors, reference_units=denominator, rate=errors / denominator)


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return error_rate(normalize_text(reference).split(), normalize_text(hypothesis).split())


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    reference_units = list(normalize_text(reference).replace(" ", ""))
    hypothesis_units = list(normalize_text(hypothesis).replace(" ", ""))
    return error_rate(reference_units, hypothesis_units)


def timestamp_boundary_mae(
    reference: list[ReferenceSpan], hypothesis: list[AsrChunk]
) -> tuple[float | None, int]:
    pair_count = min(len(reference), len(hypothesis))
    if pair_count == 0:
        return None, 0
    absolute_errors: list[int] = []
    for expected, actual in zip(reference[:pair_count], hypothesis[:pair_count], strict=True):
        absolute_errors.append(abs(expected.start_ms - actual.start_ms))
        absolute_errors.append(abs(expected.end_ms - actual.end_ms))
    return sum(absolute_errors) / len(absolute_errors), pair_count


def normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().casefold().replace("_", "-").split("-", 1)[0]
    aliases = {
        "chinese": "zh",
        "mandarin": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
    }
    return aliases.get(normalized, normalized)
