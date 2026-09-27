from __future__ import annotations

from dataclasses import dataclass

Interval = tuple[int, int]


@dataclass(frozen=True)
class DurationMetrics:
    reference_ms: int
    predicted_ms: int
    true_positive_ms: int
    false_positive_ms: int
    false_negative_ms: int
    precision: float
    recall: float
    f1: float
    iou: float


def merge_intervals(intervals: list[Interval], *, duration_ms: int) -> list[Interval]:
    clipped = sorted(
        (max(0, start), min(duration_ms, end))
        for start, end in intervals
        if min(duration_ms, end) > max(0, start)
    )
    merged: list[Interval] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_duration(left: list[Interval], right: list[Interval]) -> int:
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def duration_metrics(
    reference: list[Interval], predicted: list[Interval], *, duration_ms: int
) -> DurationMetrics:
    expected = merge_intervals(reference, duration_ms=duration_ms)
    actual = merge_intervals(predicted, duration_ms=duration_ms)
    reference_ms = interval_duration(expected)
    predicted_ms = interval_duration(actual)
    true_positive_ms = intersection_duration(expected, actual)
    false_positive_ms = predicted_ms - true_positive_ms
    false_negative_ms = reference_ms - true_positive_ms
    if reference_ms == 0 and predicted_ms == 0:
        precision = recall = 1.0
    else:
        precision = true_positive_ms / predicted_ms if predicted_ms else 0.0
        recall = true_positive_ms / reference_ms if reference_ms else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union_ms = true_positive_ms + false_positive_ms + false_negative_ms
    return DurationMetrics(
        reference_ms=reference_ms,
        predicted_ms=predicted_ms,
        true_positive_ms=true_positive_ms,
        false_positive_ms=false_positive_ms,
        false_negative_ms=false_negative_ms,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=true_positive_ms / union_ms if union_ms else 1.0,
    )


def boundary_mae(reference: list[Interval], predicted: list[Interval]) -> tuple[float | None, int]:
    pair_count = min(len(reference), len(predicted))
    if pair_count == 0:
        return None, 0
    total_error = 0
    for expected, actual in zip(reference[:pair_count], predicted[:pair_count], strict=True):
        total_error += abs(expected[0] - actual[0]) + abs(expected[1] - actual[1])
    return total_error / (pair_count * 2), pair_count
