from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from audio_core.hashing import sha256_file
from audio_core.probe import probe_file
from ingest_worker.diarization import DiarizationBackend, DiarizationSpec
from pydantic import ValidationError

from .metrics import calculate_metrics
from .models import (
    DiarizationBenchmarkReport,
    DiarizationBenchmarkSample,
    DiarizationBenchmarkSummary,
    DiarizationSampleEvaluation,
    RuntimeInfo,
)
from .rttm import load_rttm


def load_manifest(path: str | Path) -> list[DiarizationBenchmarkSample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a regular file: {manifest_path}")
    samples: list[DiarizationBenchmarkSample] = []
    sample_ids: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            sample = DiarizationBenchmarkSample.model_validate(json.loads(line))
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
        if sample.audio_sha256 and sha256_file(audio_path) != sample.audio_sha256:
            raise ValueError(
                f"audio checksum mismatch at {manifest_path}:{line_number}: {audio_path}"
            )
        rttm_path = sample.rttm
        reference_turns = sample.reference_turns
        if rttm_path is not None:
            rttm_path = rttm_path.expanduser()
            if not rttm_path.is_absolute():
                rttm_path = manifest_path.parent / rttm_path
            rttm_path = rttm_path.resolve()
            if sample.rttm_sha256 and sha256_file(rttm_path) != sample.rttm_sha256:
                raise ValueError(
                    f"RTTM checksum mismatch at {manifest_path}:{line_number}: {rttm_path}"
                )
            reference_turns = load_rttm(
                rttm_path,
                recording_id=sample.recording_id or sample.id,
            )
        samples.append(
            sample.model_copy(
                update={
                    "audio": audio_path,
                    "rttm": rttm_path,
                    "reference_turns": reference_turns,
                }
            )
        )
        sample_ids.add(sample.id)
    if not samples:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    return samples


class DiarizationBenchmarkRunner:
    def __init__(
        self,
        backend: DiarizationBackend,
        *,
        frame_ms: int = 10,
        collar_ms: int = 0,
        skip_overlap: bool = False,
        use_exclusive: bool = True,
        speaker_constraint: Literal[
            "manifest", "oracle_num", "bounded", "unconstrained"
        ] = "manifest",
        speaker_bound_padding: int = 1,
        boundary_tolerance_ms: int = 250,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if collar_ms < 0:
            raise ValueError("collar_ms must not be negative")
        if speaker_bound_padding < 0:
            raise ValueError("speaker_bound_padding must not be negative")
        if boundary_tolerance_ms < 0:
            raise ValueError("boundary_tolerance_ms must not be negative")
        self.backend = backend
        self.frame_ms = frame_ms
        self.collar_ms = collar_ms
        self.skip_overlap = skip_overlap
        self.use_exclusive = use_exclusive
        self.speaker_constraint = speaker_constraint
        self.speaker_bound_padding = speaker_bound_padding
        self.boundary_tolerance_ms = boundary_tolerance_ms

    def _speaker_spec(self, sample: DiarizationBenchmarkSample) -> DiarizationSpec:
        reference_speakers = len({turn.speaker for turn in sample.reference_turns})
        if self.speaker_constraint == "oracle_num":
            return DiarizationSpec(
                num_speakers=reference_speakers,
                use_exclusive=self.use_exclusive,
            )
        if self.speaker_constraint == "bounded":
            return DiarizationSpec(
                min_speakers=max(1, reference_speakers - self.speaker_bound_padding),
                max_speakers=reference_speakers + self.speaker_bound_padding,
                use_exclusive=self.use_exclusive,
            )
        if self.speaker_constraint == "unconstrained":
            return DiarizationSpec(use_exclusive=self.use_exclusive)
        return DiarizationSpec(
            num_speakers=sample.num_speakers,
            min_speakers=sample.min_speakers,
            max_speakers=sample.max_speakers,
            use_exclusive=self.use_exclusive,
        )

    def evaluate(self, sample: DiarizationBenchmarkSample) -> DiarizationSampleEvaluation:
        duration_seconds = 0.0
        inference_seconds = 0.0
        try:
            probe = probe_file(sample.audio)
            duration_seconds = probe.duration_seconds or 0.0
            started_at = time.perf_counter()
            spec = self._speaker_spec(sample)
            result = self.backend.diarize(sample.audio, spec=spec)
            inference_seconds = time.perf_counter() - started_at
            duration_ms = round(duration_seconds * 1000)
            if duration_ms == 0:
                duration_ms = max(
                    [turn.end_ms for turn in sample.reference_turns + result.turns],
                    default=0,
                )
                duration_seconds = duration_ms / 1000
            metrics = calculate_metrics(
                sample.reference_turns,
                result.turns,
                duration_ms=duration_ms,
                frame_ms=self.frame_ms,
                collar_ms=self.collar_ms,
                skip_overlap=self.skip_overlap,
                boundary_tolerance_ms=self.boundary_tolerance_ms,
            )
            reference_speaker_count = len(
                {turn.speaker for turn in sample.reference_turns}
            )
            hypothesis_speaker_count = len({turn.speaker for turn in result.turns})
            return DiarizationSampleEvaluation(
                id=sample.id,
                audio=str(sample.audio),
                timeline="exclusive" if self.use_exclusive else "regular",
                success=True,
                duration_seconds=duration_seconds,
                inference_seconds=inference_seconds,
                rtf=(inference_seconds / duration_seconds if duration_seconds else None),
                reference_speakers=reference_speaker_count,
                hypothesis_speakers=hypothesis_speaker_count,
                reference_turns=len(sample.reference_turns),
                hypothesis_turns=len(result.turns),
                effective_num_speakers=spec.num_speakers,
                effective_min_speakers=spec.min_speakers,
                effective_max_speakers=spec.max_speakers,
                speaker_count_correct=(reference_speaker_count == hypothesis_speaker_count),
                speaker_count_error=hypothesis_speaker_count - reference_speaker_count,
                scored_seconds=metrics.scored_seconds,
                reference_speaker_seconds=metrics.reference_speaker_seconds,
                false_alarm_seconds=metrics.false_alarm_seconds,
                missed_detection_seconds=metrics.missed_detection_seconds,
                confusion_seconds=metrics.confusion_seconds,
                der=metrics.der,
                jer=metrics.jer,
                boundary_reference_count=metrics.boundary_reference_count,
                boundary_matched_count=metrics.boundary_matched_count,
                boundary_within_tolerance_count=metrics.boundary_within_tolerance_count,
                boundary_errors_ms=list(metrics.boundary_errors_ms),
                boundary_mae_ms=metrics.boundary_mae_ms,
                boundary_p95_ms=metrics.boundary_p95_ms,
                boundary_recall=metrics.boundary_recall,
                tags=sample.tags,
            )
        except Exception as exc:
            return DiarizationSampleEvaluation(
                id=sample.id,
                audio=str(sample.audio),
                timeline="exclusive" if self.use_exclusive else "regular",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=duration_seconds,
                inference_seconds=inference_seconds,
                reference_speakers=len(
                    {turn.speaker for turn in sample.reference_turns}
                ),
                reference_turns=len(sample.reference_turns),
                tags=sample.tags,
            )

    def run(
        self,
        samples: Iterable[DiarizationBenchmarkSample],
        *,
        manifest: str | Path,
    ) -> DiarizationBenchmarkReport:
        evaluated = [self.evaluate(sample) for sample in samples]
        succeeded = [sample for sample in evaluated if sample.success]
        false_alarm = sum(sample.false_alarm_seconds for sample in succeeded)
        missed = sum(sample.missed_detection_seconds for sample in succeeded)
        confusion = sum(sample.confusion_seconds for sample in succeeded)
        reference = sum(sample.reference_speaker_seconds for sample in succeeded)
        rtfs = [sample.rtf for sample in succeeded if sample.rtf is not None]
        jers = [sample.jer for sample in succeeded if sample.jer is not None]
        speaker_count_results = [
            sample.speaker_count_correct
            for sample in succeeded
            if sample.speaker_count_correct is not None
        ]
        speaker_count_errors = [
            abs(sample.speaker_count_error)
            for sample in succeeded
            if sample.speaker_count_error is not None
        ]
        boundary_errors = [
            error for sample in succeeded for error in sample.boundary_errors_ms
        ]
        boundary_reference_count = sum(
            sample.boundary_reference_count for sample in succeeded
        )
        boundary_matched_count = sum(
            sample.boundary_matched_count for sample in succeeded
        )
        boundary_within_tolerance_count = sum(
            sample.boundary_within_tolerance_count for sample in succeeded
        )
        summary = DiarizationBenchmarkSummary(
            sample_count=len(evaluated),
            succeeded_count=len(succeeded),
            failed_count=len(evaluated) - len(succeeded),
            aggregate_der=(
                (false_alarm + missed + confusion) / reference
                if reference
                else (0.0 if succeeded and false_alarm + missed + confusion == 0 else None)
            ),
            mean_jer=sum(jers) / len(jers) if jers else None,
            mean_rtf=sum(rtfs) / len(rtfs) if rtfs else None,
            speaker_count_accuracy=(
                sum(bool(result) for result in speaker_count_results)
                / len(speaker_count_results)
                if speaker_count_results
                else None
            ),
            mean_absolute_speaker_count_error=(
                sum(speaker_count_errors) / len(speaker_count_errors)
                if speaker_count_errors
                else None
            ),
            boundary_reference_count=boundary_reference_count,
            boundary_matched_count=boundary_matched_count,
            boundary_within_tolerance_count=boundary_within_tolerance_count,
            boundary_mae_ms=(
                sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
            ),
            boundary_p95_ms=(
                sorted(boundary_errors)[max(0, math.ceil(0.95 * len(boundary_errors)) - 1)]
                if boundary_errors
                else None
            ),
            boundary_recall=(
                boundary_within_tolerance_count / boundary_reference_count
                if boundary_reference_count
                else None
            ),
            false_alarm_seconds=false_alarm,
            missed_detection_seconds=missed,
            confusion_seconds=confusion,
            reference_speaker_seconds=reference,
        )
        return DiarizationBenchmarkReport(
            backend=self.backend.name,
            model=self.backend.model_name,
            backend_version=self.backend.version,
            manifest=str(Path(manifest).expanduser().resolve()),
            frame_ms=self.frame_ms,
            collar_ms=self.collar_ms,
            skip_overlap=self.skip_overlap,
            use_exclusive=self.use_exclusive,
            speaker_constraint=self.speaker_constraint,
            speaker_bound_padding=self.speaker_bound_padding,
            boundary_tolerance_ms=self.boundary_tolerance_ms,
            runtime=RuntimeInfo(
                platform=platform.platform(),
                python_version=platform.python_version(),
                machine=platform.machine(),
            ),
            summary=summary,
            samples=evaluated,
        )


def _value(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(report: DiarizationBenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "# 说话人分离基准报告",
        "",
        f"- 后端：`{report.backend}`",
        f"- 模型：`{report.model}`",
        f"- 清单：`{report.manifest}`",
        f"- 帧长：`{report.frame_ms}ms`；边界 collar：`{report.collar_ms}ms`；"
        f"跳过重叠：`{report.skip_overlap}`；exclusive：`{report.use_exclusive}`",
        "",
        "## 汇总",
        "",
        f"- Speaker 约束：`{report.speaker_constraint}`；边界容差："
        f"`{report.boundary_tolerance_ms}ms`",
        "",
        "| 样本 | 成功 | 失败 | DER | JER | Speaker 数准确率 | "
        "边界 MAE/P95 (ms) | 边界召回 | 平均 RTF |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                str(summary.sample_count),
                str(summary.succeeded_count),
                str(summary.failed_count),
                _value(summary.aggregate_der),
                _value(summary.mean_jer),
                _value(summary.speaker_count_accuracy),
                f"{_value(summary.boundary_mae_ms, 1)}/{_value(summary.boundary_p95_ms, 1)}",
                _value(summary.boundary_recall),
                _value(summary.mean_rtf),
            ]
        )
        + " |",
        "",
        "## 样本明细",
        "",
        "| ID | 时间轴 | 状态 | 参考/预测 speaker | 参考/预测 turn | "
        "DER | JER | 边界 P95 (ms) | RTF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in report.samples:
        lines.append(
            "| "
            + " | ".join(
                [
                    sample.id.replace("|", "\\|"),
                    sample.timeline,
                    "成功" if sample.success else "失败",
                    f"{sample.reference_speakers}/{sample.hypothesis_speakers}",
                    f"{sample.reference_turns}/{sample.hypothesis_turns}",
                    _value(sample.der),
                    _value(sample.jer),
                    _value(sample.boundary_p95_ms, 1),
                    _value(sample.rtf),
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
    report: DiarizationBenchmarkReport, output_directory: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
