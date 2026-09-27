from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from ingest_worker.diarization import build_diarization_backend

from .matrix import require_real_human_dataset, run_matrix, write_matrix_report
from .models import ProductionThresholds
from .runner import DiarizationBenchmarkRunner, load_manifest, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker diarization with frame-based DER and JER."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=("pyannote",), default="pyannote")
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token", default=os.getenv("AUDIO_PLATFORM_HF_TOKEN"))
    parser.add_argument("--frame-ms", type=int, default=10)
    parser.add_argument("--collar-ms", type=int, default=0)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--regular", action="store_true", help="disable exclusive timeline")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run regular/exclusive across oracle, bounded, and unconstrained speaker settings",
    )
    parser.add_argument("--speaker-bound-padding", type=int, default=1)
    parser.add_argument("--boundary-tolerance-ms", type=int, default=250)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--minimum-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--maximum-der", type=float, default=0.25)
    parser.add_argument("--maximum-jer", type=float, default=0.35)
    parser.add_argument("--minimum-speaker-count-accuracy", type=float, default=0.9)
    parser.add_argument("--maximum-boundary-p95-ms", type=float, default=500.0)
    parser.add_argument("--maximum-mean-rtf", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    samples = load_manifest(arguments.manifest)
    if arguments.matrix:
        try:
            require_real_human_dataset(samples)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    backend = build_diarization_backend(
        arguments.backend,
        model_name=arguments.model,
        token=arguments.token,
        device=arguments.device,
    )
    if backend is None:
        raise SystemExit("diarization backend is disabled")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path("artifacts/diarization-eval") / timestamp
    if arguments.matrix:
        thresholds = ProductionThresholds(
            minimum_samples=arguments.minimum_samples,
            minimum_duration_seconds=arguments.minimum_duration_seconds,
            maximum_der=arguments.maximum_der,
            maximum_jer=arguments.maximum_jer,
            minimum_speaker_count_accuracy=arguments.minimum_speaker_count_accuracy,
            maximum_boundary_p95_ms=arguments.maximum_boundary_p95_ms,
            maximum_mean_rtf=arguments.maximum_mean_rtf,
        )
        report = run_matrix(
            backend,
            samples,
            manifest=arguments.manifest,
            frame_ms=arguments.frame_ms,
            collar_ms=arguments.collar_ms,
            skip_overlap=arguments.skip_overlap,
            speaker_bound_padding=arguments.speaker_bound_padding,
            boundary_tolerance_ms=arguments.boundary_tolerance_ms,
            thresholds=thresholds,
        )
        json_path, markdown_path = write_matrix_report(report, output)
        failed_count = sum(
            benchmark.summary.failed_count for benchmark in report.reports.values()
        )
    else:
        report = DiarizationBenchmarkRunner(
            backend,
            frame_ms=arguments.frame_ms,
            collar_ms=arguments.collar_ms,
            skip_overlap=arguments.skip_overlap,
            use_exclusive=not arguments.regular,
            boundary_tolerance_ms=arguments.boundary_tolerance_ms,
        ).run(samples, manifest=arguments.manifest)
        json_path, markdown_path = write_report(report, output)
        failed_count = report.summary.failed_count
    print(json_path)
    print(markdown_path)
    if failed_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
