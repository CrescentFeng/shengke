from __future__ import annotations

import math
from dataclasses import dataclass

from audio_core.contracts import SpeakerTurn


@dataclass(frozen=True)
class DiarizationMetrics:
    scored_seconds: float
    reference_speaker_seconds: float
    false_alarm_seconds: float
    missed_detection_seconds: float
    confusion_seconds: float
    der: float
    jer: float
    boundary_reference_count: int
    boundary_matched_count: int
    boundary_within_tolerance_count: int
    boundary_errors_ms: tuple[float, ...]
    boundary_mae_ms: float | None
    boundary_p95_ms: float | None
    boundary_recall: float | None


def _active_speakers(turns: list[SpeakerTurn], position_ms: float) -> set[str]:
    return {
        turn.speaker
        for turn in turns
        if turn.start_ms <= position_ms < turn.end_ms
    }


def _maximum_assignment(weights: list[list[float]]) -> dict[int, int]:
    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if row_count == 0 or column_count == 0:
        return {}
    size = max(row_count, column_count)
    maximum = max((max(row, default=0.0) for row in weights), default=0.0)
    costs = [
        [
            maximum
            - (weights[row][column] if row < row_count and column < column_count else 0.0)
            for column in range(size)
        ]
        for row in range(size)
    ]

    potentials_rows = [0.0] * (size + 1)
    potentials_columns = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    potentials_rows[matched_row[candidate]] += delta
                    potentials_columns[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            next_column = previous_column[column]
            matched_row[column] = matched_row[next_column]
            column = next_column
            if column == 0:
                break

    column_to_row: dict[int, int] = {}
    for column in range(1, size + 1):
        row = matched_row[column]
        if 1 <= row <= row_count and column <= column_count:
            column_to_row[column - 1] = row - 1
    return column_to_row


def _match_boundary_errors(
    reference_boundaries: list[int], hypothesis_boundaries: list[int]
) -> list[float]:
    remaining = list(hypothesis_boundaries)
    errors: list[float] = []
    for boundary in sorted(reference_boundaries):
        if not remaining:
            break
        best_index = min(
            range(len(remaining)),
            key=lambda index: (abs(remaining[index] - boundary), remaining[index]),
        )
        errors.append(float(abs(remaining.pop(best_index) - boundary)))
    return errors


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def calculate_metrics(
    reference: list[SpeakerTurn],
    hypothesis: list[SpeakerTurn],
    *,
    duration_ms: int,
    frame_ms: int = 10,
    collar_ms: int = 0,
    skip_overlap: bool = False,
    boundary_tolerance_ms: int = 250,
) -> DiarizationMetrics:
    if duration_ms < 0:
        raise ValueError("duration_ms must not be negative")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if collar_ms < 0:
        raise ValueError("collar_ms must not be negative")
    if boundary_tolerance_ms < 0:
        raise ValueError("boundary_tolerance_ms must not be negative")

    reference_speakers = sorted({turn.speaker for turn in reference})
    hypothesis_speakers = sorted({turn.speaker for turn in hypothesis})
    reference_index = {speaker: index for index, speaker in enumerate(reference_speakers)}
    hypothesis_index = {speaker: index for index, speaker in enumerate(hypothesis_speakers)}
    boundaries = [boundary for turn in reference for boundary in (turn.start_ms, turn.end_ms)]
    scored_frames: list[tuple[set[int], set[int]]] = []
    for frame_start in range(0, duration_ms, frame_ms):
        midpoint = frame_start + min(frame_ms, duration_ms - frame_start) / 2
        if collar_ms and any(abs(midpoint - boundary) < collar_ms for boundary in boundaries):
            continue
        expected = {
            reference_index[speaker] for speaker in _active_speakers(reference, midpoint)
        }
        actual = {
            hypothesis_index[speaker] for speaker in _active_speakers(hypothesis, midpoint)
        }
        if skip_overlap and len(expected) > 1:
            continue
        scored_frames.append((expected, actual))

    cooccurrence = [
        [0.0 for _hypothesis in hypothesis_speakers]
        for _reference in reference_speakers
    ]
    for expected, actual in scored_frames:
        for reference_speaker in expected:
            for hypothesis_speaker in actual:
                cooccurrence[reference_speaker][hypothesis_speaker] += 1
    hypothesis_to_reference = _maximum_assignment(cooccurrence)

    false_alarm_frames = 0
    missed_frames = 0
    confusion_frames = 0
    reference_speaker_frames = 0
    for expected, actual in scored_frames:
        mapped = {
            hypothesis_to_reference[hypothesis_speaker]
            for hypothesis_speaker in actual
            if hypothesis_speaker in hypothesis_to_reference
        }
        correct = len(expected & mapped)
        reference_speaker_frames += len(expected)
        missed_frames += max(0, len(expected) - len(actual))
        false_alarm_frames += max(0, len(actual) - len(expected))
        confusion_frames += max(0, min(len(expected), len(actual)) - correct)

    reference_frame_sets = [set() for _speaker in reference_speakers]
    hypothesis_frame_sets = [set() for _speaker in hypothesis_speakers]
    for frame_index, (expected, actual) in enumerate(scored_frames):
        for speaker in expected:
            reference_frame_sets[speaker].add(frame_index)
        for speaker in actual:
            hypothesis_frame_sets[speaker].add(frame_index)
    jaccard_weights: list[list[float]] = []
    for expected_frames in reference_frame_sets:
        row: list[float] = []
        for actual_frames in hypothesis_frame_sets:
            union = expected_frames | actual_frames
            row.append(len(expected_frames & actual_frames) / len(union) if union else 0.0)
        jaccard_weights.append(row)
    jaccard_mapping = _maximum_assignment(jaccard_weights)
    reference_to_hypothesis = {
        reference_speaker: hypothesis_speaker
        for hypothesis_speaker, reference_speaker in jaccard_mapping.items()
    }
    if reference_speakers:
        jaccard_errors = []
        for reference_speaker, expected_frames in enumerate(reference_frame_sets):
            hypothesis_speaker = reference_to_hypothesis.get(reference_speaker)
            if hypothesis_speaker is None:
                jaccard_errors.append(1.0)
                continue
            actual_frames = hypothesis_frame_sets[hypothesis_speaker]
            union = expected_frames | actual_frames
            similarity = len(expected_frames & actual_frames) / len(union) if union else 1.0
            jaccard_errors.append(1.0 - similarity)
        jer = sum(jaccard_errors) / len(jaccard_errors)
    else:
        jer = 0.0 if not hypothesis_speakers else 1.0

    reference_to_hypothesis = {
        reference_speaker: hypothesis_speaker
        for hypothesis_speaker, reference_speaker in hypothesis_to_reference.items()
    }
    boundary_errors: list[float] = []
    boundary_reference_count = len(reference) * 2
    for reference_speaker, reference_name in enumerate(reference_speakers):
        hypothesis_speaker = reference_to_hypothesis.get(reference_speaker)
        if hypothesis_speaker is None:
            continue
        hypothesis_name = hypothesis_speakers[hypothesis_speaker]
        reference_speaker_turns = [
            turn for turn in reference if turn.speaker == reference_name
        ]
        hypothesis_speaker_turns = [
            turn for turn in hypothesis if turn.speaker == hypothesis_name
        ]
        boundary_errors.extend(
            _match_boundary_errors(
                [turn.start_ms for turn in reference_speaker_turns],
                [turn.start_ms for turn in hypothesis_speaker_turns],
            )
        )
        boundary_errors.extend(
            _match_boundary_errors(
                [turn.end_ms for turn in reference_speaker_turns],
                [turn.end_ms for turn in hypothesis_speaker_turns],
            )
        )
    boundary_within_tolerance_count = sum(
        error <= boundary_tolerance_ms for error in boundary_errors
    )
    boundary_mae_ms = (
        sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
    )
    boundary_recall = (
        boundary_within_tolerance_count / boundary_reference_count
        if boundary_reference_count
        else None
    )

    frame_seconds = frame_ms / 1000
    error_frames = false_alarm_frames + missed_frames + confusion_frames
    der = (
        error_frames / reference_speaker_frames
        if reference_speaker_frames
        else (0.0 if error_frames == 0 else 1.0)
    )
    return DiarizationMetrics(
        scored_seconds=len(scored_frames) * frame_seconds,
        reference_speaker_seconds=reference_speaker_frames * frame_seconds,
        false_alarm_seconds=false_alarm_frames * frame_seconds,
        missed_detection_seconds=missed_frames * frame_seconds,
        confusion_seconds=confusion_frames * frame_seconds,
        der=der,
        jer=jer,
        boundary_reference_count=boundary_reference_count,
        boundary_matched_count=len(boundary_errors),
        boundary_within_tolerance_count=boundary_within_tolerance_count,
        boundary_errors_ms=tuple(boundary_errors),
        boundary_mae_ms=boundary_mae_ms,
        boundary_p95_ms=_percentile(boundary_errors, 0.95),
        boundary_recall=boundary_recall,
    )
