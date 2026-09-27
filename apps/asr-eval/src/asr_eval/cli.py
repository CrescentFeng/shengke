from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

from ingest_worker.asr import build_asr_backend

from .runner import AsrBenchmarkRunner, load_manifest, write_report


def _default_output(backend: str, model: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "model"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/asr-eval") / f"{backend}-{safe_model}-{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a repeatable ASR benchmark from a JSONL manifest."
    )
    parser.add_argument("manifest", type=Path, help="JSONL benchmark manifest")
    parser.add_argument(
        "--backend",
        choices=("faster-whisper", "funasr"),
        default="faster-whisper",
    )
    parser.add_argument("--model", default="small", help="model name or local model path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--output", type=Path, help="report output directory")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.beam_size < 1:
        raise SystemExit("--beam-size must be greater than zero")
    samples = load_manifest(arguments.manifest)
    backend = build_asr_backend(
        arguments.backend,
        model_name=arguments.model,
        device=arguments.device,
        compute_type=arguments.compute_type,
        beam_size=arguments.beam_size,
    )
    if backend is None:
        raise SystemExit("ASR backend is disabled")
    runner = AsrBenchmarkRunner(backend)
    report = runner.run(samples, manifest=arguments.manifest)
    output_directory = arguments.output or _default_output(
        arguments.backend, arguments.model
    )
    json_path, markdown_path = write_report(report, output_directory)
    print(json_path)
    print(markdown_path)
    if report.summary.failed_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
