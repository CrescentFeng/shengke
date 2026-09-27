from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from audio_core.vad import VadSpec

from .models import VadCandidate
from .runner import VadBenchmarkRunner, load_manifest, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate VAD backends on a JSONL manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--backend",
        choices=("adaptive_energy_v1", "webrtc_v1", "all"),
        default="adaptive_energy_v1",
    )
    parser.add_argument("--energy-threshold", type=float, action="append")
    parser.add_argument("--webrtc-modes", default="0,1,2,3")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--minimum-speech-ms", type=int, default=150)
    parser.add_argument("--maximum-silence-ms", type=int, default=300)
    parser.add_argument("--padding-ms", type=int, default=100)
    parser.add_argument("--noise-margin-db", type=float, default=10.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path)
    return parser


def _common(arguments: argparse.Namespace) -> dict[str, int | float]:
    return {
        "sample_rate_hz": arguments.sample_rate,
        "frame_ms": arguments.frame_ms,
        "minimum_speech_ms": arguments.minimum_speech_ms,
        "maximum_silence_ms": arguments.maximum_silence_ms,
        "padding_ms": arguments.padding_ms,
        "noise_margin_db": arguments.noise_margin_db,
    }


def build_candidates(arguments: argparse.Namespace) -> list[VadCandidate]:
    common = _common(arguments)
    candidates: list[VadCandidate] = []
    if arguments.backend in {"adaptive_energy_v1", "all"}:
        for threshold in arguments.energy_threshold or [-48.0, -42.0, -36.0]:
            candidates.append(
                VadCandidate(
                    name=f"adaptive-energy-{threshold:g}dbfs",
                    spec=VadSpec(
                        backend="adaptive_energy_v1",
                        minimum_speech_dbfs=threshold,
                        **common,
                    ),
                )
            )
    if arguments.backend in {"webrtc_v1", "all"}:
        try:
            modes = [int(value.strip()) for value in arguments.webrtc_modes.split(",")]
        except ValueError as exc:
            raise SystemExit("--webrtc-modes must be comma-separated integers from 0 to 3") from exc
        for mode in modes:
            candidates.append(
                VadCandidate(
                    name=f"webrtc-mode-{mode}",
                    spec=VadSpec(
                        backend="webrtc_v1",
                        webrtc_aggressiveness=mode,
                        **common,
                    ),
                )
            )
    return candidates


def main() -> None:
    arguments = build_parser().parse_args()
    samples = load_manifest(arguments.manifest)
    candidates = build_candidates(arguments)
    report = VadBenchmarkRunner(ffmpeg=arguments.ffmpeg).run(
        samples, candidates, manifest=arguments.manifest
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path("artifacts/vad-eval") / timestamp
    json_path, markdown_path = write_report(report, output)
    print(json_path)
    print(markdown_path)
    if any(candidate.failed_count for candidate in report.candidates):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
