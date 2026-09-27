from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audio_core.hashing import sha256_file
from diarization_eval.rttm import load_rttm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a checksum-pinned real/human diarization JSONL manifest."
    )
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--rttm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--license", dest="license_name", required=True)
    parser.add_argument("--limit", type=int)
    return parser


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base)


def main() -> None:
    arguments = build_parser().parse_args()
    audio_root = arguments.audio_root.expanduser().resolve()
    rttm_root = arguments.rttm_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not audio_root.is_dir():
        raise SystemExit(f"audio root is not a directory: {audio_root}")
    if not rttm_root.is_dir():
        raise SystemExit(f"RTTM root is not a directory: {rttm_root}")
    if arguments.limit is not None and arguments.limit <= 0:
        raise SystemExit("--limit must be positive")

    audio_by_id: dict[str, Path] = {}
    for extension in ("*.wav", "*.flac"):
        for audio_path in audio_root.rglob(extension):
            if audio_path.stem in audio_by_id:
                raise SystemExit(f"duplicate audio recording id: {audio_path.stem}")
            audio_by_id[audio_path.stem] = audio_path.resolve()
    rttm_by_id: dict[str, Path] = {}
    for rttm_path in rttm_root.rglob("*.rttm"):
        if rttm_path.stem in rttm_by_id:
            raise SystemExit(f"duplicate RTTM recording id: {rttm_path.stem}")
        rttm_by_id[rttm_path.stem] = rttm_path.resolve()

    recording_ids = sorted(set(audio_by_id) & set(rttm_by_id))
    if arguments.limit is not None:
        recording_ids = recording_ids[: arguments.limit]
    if not recording_ids:
        raise SystemExit("no matching audio/RTTM recording ids found")

    output.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    total_speakers: set[tuple[str, str]] = set()
    for recording_id in recording_ids:
        audio_path = audio_by_id[recording_id]
        rttm_path = rttm_by_id[recording_id]
        turns = load_rttm(rttm_path, recording_id=recording_id)
        total_speakers.update((recording_id, turn.speaker) for turn in turns)
        entries.append(
            {
                "id": recording_id,
                "audio": _relative(audio_path, output.parent),
                "rttm": _relative(rttm_path, output.parent),
                "recording_id": recording_id,
                "audio_sha256": sha256_file(audio_path),
                "rttm_sha256": sha256_file(rttm_path),
                "audio_kind": "real",
                "annotation_kind": "human",
                "tags": {
                    "dataset": arguments.dataset,
                    "dataset_version": arguments.dataset_version,
                    "split": arguments.split,
                    "license": arguments.license_name,
                },
            }
        )
    output.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    print(output)
    print(f"recordings={len(entries)} speakers={len(total_speakers)}")


if __name__ == "__main__":
    main()
