from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Iterable
from pathlib import Path

from audio_core.probe import probe_file
from ingest_worker.asr import AsrBackend, AsrRequestOptions
from pydantic import ValidationError

from .metrics import (
    character_error_rate,
    normalize_language,
    timestamp_boundary_mae,
    word_error_rate,
)
from .models import (
    AsrBenchmarkReport,
    BenchmarkSample,
    BenchmarkSummary,
    RuntimeInfo,
    SampleEvaluation,
)
from .monitoring import GpuMemorySampler, peak_process_memory_mb


def load_manifest(path: str | Path) -> list[BenchmarkSample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a regular file: {manifest_path}")

    samples: list[BenchmarkSample] = []
    sample_ids: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            sample = BenchmarkSample.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"invalid manifest entry at {manifest_path}:{line_number}: {exc}"
            ) from exc
        if sample.id in sample_ids:
            raise ValueError(
                f"duplicate sample id at {manifest_path}:{line_number}: {sample.id}"
            )
        audio_path = sample.audio.expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise ValueError(
                f"audio is not a regular file at {manifest_path}:{line_number}: {audio_path}"
            )
        samples.append(sample.model_copy(update={"audio": audio_path}))
        sample_ids.add(sample.id)

    if not samples:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    return samples


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _aggregate_rate(errors: int, reference_units: int) -> float | None:
    if reference_units == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_units


def _maximum(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


class AsrBenchmarkRunner:
    def __init__(self, backend: AsrBackend) -> None:
        self.backend = backend

    def evaluate(self, sample: BenchmarkSample) -> SampleEvaluation:
        duration_seconds = 0.0
        inference_seconds = 0.0
        try:
            probe = probe_file(sample.audio)
            duration_seconds = probe.duration_seconds or 0.0
            options = AsrRequestOptions(
                language=sample.language,
                hotwords=sample.hotwords,
            )
            with GpuMemorySampler() as gpu_memory:
                started_at = time.perf_counter()
                result = self.backend.transcribe(
                    sample.audio,
                    language=options.language,
                    hotwords=options.hotwords,
                )
                inference_seconds = time.perf_counter() - started_at

            word_result = word_error_rate(sample.reference_text, result.text)
            character_result = character_error_rate(sample.reference_text, result.text)
            timestamp_mae, timestamp_pairs = timestamp_boundary_mae(
                sample.reference_spans, result.chunks
            )
            expected_language = normalize_language(sample.language)
            detected_language = normalize_language(result.language)
            language_correct = (
                expected_language == detected_language
                if expected_language is not None and detected_language is not None
                else None
            )
            return SampleEvaluation(
                id=sample.id,
                audio=str(sample.audio),
                success=True,
                reference_text=sample.reference_text,
                hypothesis_text=result.text,
                reference_language=sample.language,
                detected_language=result.language,
                language_correct=language_correct,
                language_confidence=result.language_confidence,
                duration_seconds=duration_seconds,
                inference_seconds=inference_seconds,
                rtf=(inference_seconds / duration_seconds if duration_seconds > 0 else None),
                word_errors=word_result.errors,
                reference_words=word_result.reference_units,
                wer=word_result.rate,
                character_errors=character_result.errors,
                reference_characters=character_result.reference_units,
                cer=character_result.rate,
                timestamp_boundary_mae_ms=timestamp_mae,
                timestamp_pairs=timestamp_pairs,
                peak_process_memory_mb=peak_process_memory_mb(),
                peak_gpu_memory_mb=gpu_memory.peak_mb,
                tags=sample.tags,
            )
        except Exception as exc:
            return SampleEvaluation(
                id=sample.id,
                audio=str(sample.audio),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                reference_text=sample.reference_text,
                reference_language=sample.language,
                duration_seconds=duration_seconds,
                inference_seconds=inference_seconds,
                peak_process_memory_mb=peak_process_memory_mb(),
                tags=sample.tags,
            )

    def run(
        self, samples: Iterable[BenchmarkSample], *, manifest: str | Path
    ) -> AsrBenchmarkReport:
        evaluated = [self.evaluate(sample) for sample in samples]
        succeeded = [sample for sample in evaluated if sample.success]
        rtfs = [sample.rtf for sample in succeeded if sample.rtf is not None]
        language_results = [
            sample.language_correct
            for sample in succeeded
            if sample.language_correct is not None
        ]
        timestamp_weight = sum(sample.timestamp_pairs * 2 for sample in succeeded)
        timestamp_error_sum = sum(
            (sample.timestamp_boundary_mae_ms or 0.0) * sample.timestamp_pairs * 2
            for sample in succeeded
        )
        summary = BenchmarkSummary(
            sample_count=len(evaluated),
            succeeded_count=len(succeeded),
            failed_count=len(evaluated) - len(succeeded),
            total_audio_seconds=sum(sample.duration_seconds for sample in succeeded),
            total_inference_seconds=sum(sample.inference_seconds for sample in succeeded),
            aggregate_wer=(
                _aggregate_rate(
                    sum(sample.word_errors for sample in succeeded),
                    sum(sample.reference_words for sample in succeeded),
                )
                if succeeded
                else None
            ),
            aggregate_cer=(
                _aggregate_rate(
                    sum(sample.character_errors for sample in succeeded),
                    sum(sample.reference_characters for sample in succeeded),
                )
                if succeeded
                else None
            ),
            mean_rtf=sum(rtfs) / len(rtfs) if rtfs else None,
            p95_rtf=_percentile_95(rtfs),
            language_accuracy=(
                sum(language_results) / len(language_results) if language_results else None
            ),
            timestamp_boundary_mae_ms=(
                timestamp_error_sum / timestamp_weight if timestamp_weight else None
            ),
            peak_process_memory_mb=_maximum(
                sample.peak_process_memory_mb for sample in evaluated
            ),
            peak_gpu_memory_mb=_maximum(sample.peak_gpu_memory_mb for sample in evaluated),
        )
        return AsrBenchmarkReport(
            backend=self.backend.name,
            model=self.backend.model_name,
            backend_version=self.backend.version,
            manifest=str(Path(manifest).expanduser().resolve()),
            runtime=RuntimeInfo(
                platform=platform.platform(),
                python_version=platform.python_version(),
                machine=platform.machine(),
                processor=platform.processor(),
            ),
            summary=summary,
            samples=evaluated,
        )


def _format_optional(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: AsrBenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "# ASR 基准评测报告",
        "",
        f"- 后端：`{report.backend}`",
        f"- 模型：`{report.model}`",
        f"- 后端版本：`{report.backend_version}`",
        f"- 清单：`{report.manifest}`",
        f"- 生成时间：`{report.created_at.isoformat()}`",
        f"- 运行环境：`{report.runtime.platform}` / Python `{report.runtime.python_version}`",
        "",
        "## 汇总",
        "",
        "| 样本 | 成功 | 失败 | WER | CER | 平均 RTF | P95 RTF | 语种准确率 | "
        "时间戳边界 MAE (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                str(summary.sample_count),
                str(summary.succeeded_count),
                str(summary.failed_count),
                _format_optional(summary.aggregate_wer),
                _format_optional(summary.aggregate_cer),
                _format_optional(summary.mean_rtf),
                _format_optional(summary.p95_rtf),
                _format_optional(summary.language_accuracy),
                _format_optional(summary.timestamp_boundary_mae_ms, 2),
            ]
        )
        + " |",
        "",
        f"总音频时长：`{summary.total_audio_seconds:.3f}s`；总推理耗时："
        f"`{summary.total_inference_seconds:.3f}s`；进程峰值内存："
        f"`{_format_optional(summary.peak_process_memory_mb, 2)} MB`；GPU 峰值显存："
        f"`{_format_optional(summary.peak_gpu_memory_mb, 2)} MB`。",
        "",
        "## 样本明细",
        "",
        "| ID | 状态 | 时长 (s) | 推理 (s) | RTF | WER | CER | 语种 | 时间戳 MAE (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for sample in report.samples:
        language = (
            f"{sample.reference_language or '—'} → {sample.detected_language or '—'}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(sample.id),
                    "成功" if sample.success else "失败",
                    f"{sample.duration_seconds:.3f}",
                    f"{sample.inference_seconds:.3f}",
                    _format_optional(sample.rtf),
                    _format_optional(sample.wer),
                    _format_optional(sample.cer),
                    _escape_table_cell(language),
                    _format_optional(sample.timestamp_boundary_mae_ms, 2),
                ]
            )
            + " |"
        )

    failed = [sample for sample in report.samples if not sample.success]
    if failed:
        lines.extend(["", "## 失败样本", ""])
        for sample in failed:
            lines.append(f"- `{sample.id}`：{sample.error}")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: AsrBenchmarkReport, output_directory: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
