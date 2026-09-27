from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from audio_core.vad import VadResult, detect_voice_activity
from pydantic import ValidationError

from .metrics import boundary_mae, duration_metrics
from .models import (
    RuntimeInfo,
    VadBenchmarkReport,
    VadBenchmarkSample,
    VadCandidate,
    VadCandidateSummary,
    VadSampleEvaluation,
)

Detector = Callable[..., VadResult]


def load_manifest(path: str | Path) -> list[VadBenchmarkSample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a regular file: {manifest_path}")
    samples: list[VadBenchmarkSample] = []
    sample_ids: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            sample = VadBenchmarkSample.model_validate(json.loads(line))
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


def _ratios(true_positive_ms: int, false_positive_ms: int, false_negative_ms: int):
    precision_denominator = true_positive_ms + false_positive_ms
    recall_denominator = true_positive_ms + false_negative_ms
    if precision_denominator == 0 and recall_denominator == 0:
        precision = recall = 1.0
    else:
        precision = true_positive_ms / precision_denominator if precision_denominator else 0.0
        recall = true_positive_ms / recall_denominator if recall_denominator else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = true_positive_ms + false_positive_ms + false_negative_ms
    iou = true_positive_ms / union if union else 1.0
    return precision, recall, f1, iou


class VadBenchmarkRunner:
    def __init__(self, *, detector: Detector = detect_voice_activity, ffmpeg: str = "ffmpeg"):
        self.detector = detector
        self.ffmpeg = ffmpeg

    def evaluate(
        self, sample: VadBenchmarkSample, candidate: VadCandidate
    ) -> VadSampleEvaluation:
        started_at = time.perf_counter()
        try:
            result = self.detector(sample.audio, spec=candidate.spec, ffmpeg=self.ffmpeg)
            inference_seconds = time.perf_counter() - started_at
            reference = [
                (segment.start_ms, segment.end_ms) for segment in sample.reference_segments
            ]
            predicted = [(segment.start_ms, segment.end_ms) for segment in result.segments]
            metrics = duration_metrics(reference, predicted, duration_ms=result.duration_ms)
            boundary_error, boundary_pairs = boundary_mae(reference, predicted)
            duration_seconds = result.duration_ms / 1000
            return VadSampleEvaluation(
                candidate=candidate.name,
                id=sample.id,
                audio=str(sample.audio),
                success=True,
                duration_ms=result.duration_ms,
                inference_seconds=inference_seconds,
                rtf=(inference_seconds / duration_seconds if duration_seconds > 0 else None),
                reference_segments=len(reference),
                predicted_segments=len(predicted),
                reference_speech_ms=metrics.reference_ms,
                predicted_speech_ms=metrics.predicted_ms,
                true_positive_ms=metrics.true_positive_ms,
                false_positive_ms=metrics.false_positive_ms,
                false_negative_ms=metrics.false_negative_ms,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                iou=metrics.iou,
                boundary_mae_ms=boundary_error,
                boundary_pairs=boundary_pairs,
                tags=sample.tags,
            )
        except Exception as exc:
            return VadSampleEvaluation(
                candidate=candidate.name,
                id=sample.id,
                audio=str(sample.audio),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                inference_seconds=time.perf_counter() - started_at,
                reference_segments=len(sample.reference_segments),
                tags=sample.tags,
            )

    def run(
        self,
        samples: Iterable[VadBenchmarkSample],
        candidates: Iterable[VadCandidate],
        *,
        manifest: str | Path,
    ) -> VadBenchmarkReport:
        sample_list = list(samples)
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("at least one VAD candidate is required")
        candidate_names = [candidate.name for candidate in candidate_list]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("VAD candidate names must be unique")
        evaluations = [
            self.evaluate(sample, candidate)
            for candidate in candidate_list
            for sample in sample_list
        ]
        summaries: list[VadCandidateSummary] = []
        for candidate in candidate_list:
            candidate_results = [
                result for result in evaluations if result.candidate == candidate.name
            ]
            succeeded = [result for result in candidate_results if result.success]
            true_positive_ms = sum(result.true_positive_ms for result in succeeded)
            false_positive_ms = sum(result.false_positive_ms for result in succeeded)
            false_negative_ms = sum(result.false_negative_ms for result in succeeded)
            precision, recall, f1, iou = _ratios(
                true_positive_ms, false_positive_ms, false_negative_ms
            )
            boundary_count = sum(result.boundary_pairs * 2 for result in succeeded)
            boundary_error = sum(
                (result.boundary_mae_ms or 0.0) * result.boundary_pairs * 2
                for result in succeeded
            )
            rtfs = [result.rtf for result in succeeded if result.rtf is not None]
            summaries.append(
                VadCandidateSummary(
                    candidate=candidate.name,
                    backend=candidate.spec.backend,
                    parameters=candidate.spec.model_dump(),
                    sample_count=len(candidate_results),
                    succeeded_count=len(succeeded),
                    failed_count=len(candidate_results) - len(succeeded),
                    precision=precision if succeeded else None,
                    recall=recall if succeeded else None,
                    f1=f1 if succeeded else None,
                    iou=iou if succeeded else None,
                    boundary_mae_ms=(
                        boundary_error / boundary_count if boundary_count else None
                    ),
                    mean_rtf=sum(rtfs) / len(rtfs) if rtfs else None,
                )
            )
        complete = [summary for summary in summaries if summary.failed_count == 0]
        best = max(
            complete,
            key=lambda summary: (
                summary.f1 if summary.f1 is not None else -1,
                summary.iou if summary.iou is not None else -1,
                -(summary.mean_rtf if summary.mean_rtf is not None else float("inf")),
            ),
            default=None,
        )
        return VadBenchmarkReport(
            manifest=str(Path(manifest).expanduser().resolve()),
            runtime=RuntimeInfo(
                platform=platform.platform(),
                python_version=platform.python_version(),
                machine=platform.machine(),
            ),
            best_candidate=best.candidate if best else None,
            candidates=summaries,
            samples=evaluations,
        )


def _value(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(report: VadBenchmarkReport) -> str:
    lines = [
        "# VAD 校准报告",
        "",
        f"- 清单：`{report.manifest}`",
        f"- 生成时间：`{report.created_at.isoformat()}`",
        f"- 最佳候选：`{report.best_candidate or '无完整候选'}`",
        f"- 环境：`{report.runtime.platform}` / Python `{report.runtime.python_version}`",
        "",
        "## 候选汇总",
        "",
        "| 候选 | 后端 | 成功/总数 | Precision | Recall | F1 | IoU | 边界 MAE (ms) | 平均 RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report.candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.candidate.replace("|", "\\|"),
                    summary.backend,
                    f"{summary.succeeded_count}/{summary.sample_count}",
                    _value(summary.precision),
                    _value(summary.recall),
                    _value(summary.f1),
                    _value(summary.iou),
                    _value(summary.boundary_mae_ms, 2),
                    _value(summary.mean_rtf),
                ]
            )
            + " |"
        )
    failed = [sample for sample in report.samples if not sample.success]
    if failed:
        lines.extend(["", "## 失败样本", ""])
        for sample in failed:
            lines.append(f"- `{sample.candidate}/{sample.id}`：{sample.error}")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: VadBenchmarkReport, output_directory: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
