from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .hashing import sha256_file
from .normalize import NormalizeSpec, normalize_audio
from .probe import probe_file
from .quality import AudioQualityPolicy, QualityAnalysisSpec, analyze_quality, evaluate_quality
from .vad import VadSpec, detect_voice_activity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="inspect one audio file")
    probe.add_argument("input", type=Path)
    probe.add_argument("--ffprobe", default="ffprobe")

    normalize = subparsers.add_parser("normalize", help="create a normalized WAV derivative")
    normalize.add_argument("input", type=Path)
    normalize.add_argument("output", type=Path)
    normalize.add_argument("--sample-rate", type=int, default=16000)
    normalize.add_argument("--channels", type=int, default=1)
    normalize.add_argument("--sample-format", choices=["s16", "s24", "s32", "flt"], default="s16")
    normalize.add_argument("--ffmpeg", default="ffmpeg")

    quality = subparsers.add_parser("quality", help="measure loudness, clipping, and silence")
    quality.add_argument("input", type=Path)
    quality.add_argument("--silence-noise-db", type=float, default=-50.0)
    quality.add_argument("--minimum-silence", type=float, default=0.5)
    quality.add_argument("--ffmpeg", default="ffmpeg")
    quality.add_argument("--ffprobe", default="ffprobe")

    vad = subparsers.add_parser("vad", help="detect speech and return millisecond segments")
    vad.add_argument("input", type=Path)
    vad.add_argument(
        "--backend",
        choices=("adaptive_energy_v1", "webrtc_v1"),
        default="adaptive_energy_v1",
    )
    vad.add_argument("--minimum-speech-ms", type=int, default=150)
    vad.add_argument("--maximum-silence-ms", type=int, default=300)
    vad.add_argument("--padding-ms", type=int, default=100)
    vad.add_argument("--minimum-speech-dbfs", type=float, default=-42.0)
    vad.add_argument("--webrtc-aggressiveness", type=int, choices=range(4), default=2)
    vad.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        result = {
            "sha256": sha256_file(args.input),
            "size_bytes": args.input.stat().st_size,
            "probe": probe_file(args.input, ffprobe=args.ffprobe).model_dump(mode="json"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "vad":
        result = detect_voice_activity(
            args.input,
            spec=VadSpec(
                backend=args.backend,
                minimum_speech_ms=args.minimum_speech_ms,
                maximum_silence_ms=args.maximum_silence_ms,
                padding_ms=args.padding_ms,
                minimum_speech_dbfs=args.minimum_speech_dbfs,
                webrtc_aggressiveness=args.webrtc_aggressiveness,
            ),
            ffmpeg=args.ffmpeg,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    if args.command == "quality":
        probe = probe_file(args.input, ffprobe=args.ffprobe)
        if not probe.duration_seconds:
            raise SystemExit("input has no measurable duration")
        metrics = analyze_quality(
            args.input,
            duration_seconds=probe.duration_seconds,
            spec=QualityAnalysisSpec(
                silence_noise_db=args.silence_noise_db,
                minimum_silence_seconds=args.minimum_silence,
            ),
            ffmpeg=args.ffmpeg,
        )
        findings = evaluate_quality(uuid4(), metrics, AudioQualityPolicy())
        result = {
            "metrics": metrics.model_dump(mode="json"),
            "findings": [finding.model_dump(mode="json") for finding in findings],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    spec = NormalizeSpec(
        sample_rate_hz=args.sample_rate,
        channels=args.channels,
        sample_format=args.sample_format,
    )
    output = normalize_audio(args.input, args.output, spec=spec, ffmpeg=args.ffmpeg)
    print(output)


if __name__ == "__main__":
    main()
