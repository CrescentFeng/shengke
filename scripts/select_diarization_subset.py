from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from audio_core.probe import probe_file
from diarization_eval.models import DiarizationBenchmarkSample
from diarization_eval.runner import load_manifest


@dataclass(frozen=True)
class Candidate:
    sample: DiarizationBenchmarkSample
    raw: dict[str, object]
    duration_seconds: float
    speaker_count: int
    duration_bucket: int
    speaker_bucket: str

    @property
    def bucket(self) -> tuple[str, int]:
        return self.speaker_bucket, self.duration_bucket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic speaker-count/duration-stratified accuracy subset."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--minimum-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--minimum-speakers", type=int, default=2)
    parser.add_argument("--seed", default="voxconverse-0.3-acceptance-v1")
    return parser


def _speaker_bucket(count: int) -> str:
    return str(count) if count <= 4 else "5+"


def _duration_bucket(duration: float, ordered: list[float]) -> int:
    if not ordered:
        return 0
    thresholds = [
        ordered[min(len(ordered) - 1, math.floor(len(ordered) * quantile))]
        for quantile in (0.25, 0.5, 0.75)
    ]
    return sum(duration > threshold for threshold in thresholds)


def _stable_rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def select_candidates(
    candidates: list[Candidate],
    *,
    minimum_samples: int,
    minimum_duration_seconds: float,
    seed: str,
) -> list[Candidate]:
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if minimum_duration_seconds <= 0:
        raise ValueError("minimum_duration_seconds must be positive")
    groups: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.bucket].append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: (_stable_rank(seed, item.sample.id), item.sample.id))

    selected: list[Candidate] = []
    bucket_order = sorted(groups)
    indices = {bucket: 0 for bucket in bucket_order}
    total_duration = 0.0
    while len(selected) < minimum_samples or total_duration < minimum_duration_seconds:
        progressed = False
        for bucket in bucket_order:
            index = indices[bucket]
            group = groups[bucket]
            if index >= len(group):
                continue
            candidate = group[index]
            indices[bucket] += 1
            selected.append(candidate)
            total_duration += candidate.duration_seconds
            progressed = True
            if (
                len(selected) >= minimum_samples
                and total_duration >= minimum_duration_seconds
            ):
                break
        if not progressed:
            raise ValueError(
                "dataset cannot satisfy requested minimum sample count and duration"
            )
    return sorted(selected, key=lambda item: item.sample.id)


def main() -> None:
    arguments = build_parser().parse_args()
    manifest = arguments.manifest.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if arguments.minimum_speakers < 2:
        raise SystemExit("--minimum-speakers must be at least 2 for accuracy acceptance")
    raw_entries = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_by_id = {entry["id"]: entry for entry in raw_entries}
    samples = load_manifest(manifest)
    durations: dict[str, float] = {}
    for sample in samples:
        duration = probe_file(sample.audio).duration_seconds
        if duration is None or duration <= 0:
            raise SystemExit(f"audio duration unavailable: {sample.id}")
        durations[sample.id] = duration
    ordered_durations = sorted(durations.values())
    candidates: list[Candidate] = []
    for sample in samples:
        speaker_count = len({turn.speaker for turn in sample.reference_turns})
        if speaker_count < arguments.minimum_speakers:
            continue
        duration = durations[sample.id]
        candidates.append(
            Candidate(
                sample=sample,
                raw=raw_by_id[sample.id],
                duration_seconds=duration,
                speaker_count=speaker_count,
                duration_bucket=_duration_bucket(duration, ordered_durations),
                speaker_bucket=_speaker_bucket(speaker_count),
            )
        )
    selected = select_candidates(
        candidates,
        minimum_samples=arguments.minimum_samples,
        minimum_duration_seconds=arguments.minimum_duration_seconds,
        seed=arguments.seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    selected_entries: list[dict[str, object]] = []
    for candidate in selected:
        entry = dict(candidate.raw)
        tags = dict(entry.get("tags", {}))
        tags.update(
            {
                "selection": "stratified-speaker-duration-v1",
                "selection_seed": arguments.seed,
                "duration_seconds": f"{candidate.duration_seconds:.6f}",
                "reference_speakers": str(candidate.speaker_count),
                "duration_quartile": str(candidate.duration_bucket + 1),
            }
        )
        entry["tags"] = tags
        selected_entries.append(entry)
    output.write_text(
        "".join(
            json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
            for entry in selected_entries
        ),
        encoding="utf-8",
    )

    report_path = output.with_suffix(".selection.json")
    selection_report = {
        "algorithm": "stratified-speaker-duration-v1",
        "seed": arguments.seed,
        "source_manifest": os.path.relpath(manifest, start=report_path.parent),
        "candidate_count": len(candidates),
        "source_sample_count": len(samples),
        "minimum_speakers": arguments.minimum_speakers,
        "selected_count": len(selected),
        "selected_duration_seconds": sum(item.duration_seconds for item in selected),
        "speaker_count_distribution": dict(
            sorted(Counter(item.speaker_bucket for item in selected).items())
        ),
        "duration_quartile_distribution": dict(
            sorted(Counter(str(item.duration_bucket + 1) for item in selected).items())
        ),
        "selected_ids": [item.sample.id for item in selected],
    }
    report_path.write_text(
        json.dumps(selection_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report_path)
    print(
        f"selected={len(selected)} "
        f"duration_seconds={selection_report['selected_duration_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
