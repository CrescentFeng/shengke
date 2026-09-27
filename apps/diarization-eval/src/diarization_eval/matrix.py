from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ingest_worker.diarization import (
    DiarizationBackend,
    DiarizationResult,
    DiarizationSpec,
)

from .models import (
    ConfigurationAssessment,
    DatasetAudit,
    DiarizationBenchmarkMatrixReport,
    DiarizationBenchmarkReport,
    DiarizationBenchmarkSample,
    ProductionRecommendation,
    ProductionThresholds,
)
from .runner import DiarizationBenchmarkRunner, _value, write_report


class _TimelineCacheBackend:
    """Reuse one pipeline call because Community-1 returns both timelines."""

    def __init__(self, backend: DiarizationBackend) -> None:
        self._backend = backend
        self.name = backend.name
        self.model_name = backend.model_name
        self.version = backend.version
        self._results: dict[tuple[str, int | None, int | None, int | None], DiarizationResult] = {}

    def diarize(self, path: Path, *, spec: DiarizationSpec) -> DiarizationResult:
        key = (
            str(path.resolve()),
            spec.num_speakers,
            spec.min_speakers,
            spec.max_speakers,
        )
        result = self._results.get(key)
        if result is None:
            result = self._backend.diarize(path, spec=spec)
            self._results[key] = result
        regular_turns = result.regular_turns or result.turns
        exclusive_turns = result.exclusive_turns or result.turns
        return result.model_copy(
            update={
                "turns": exclusive_turns if spec.use_exclusive else regular_turns,
                "exclusive": spec.use_exclusive and bool(result.exclusive_turns),
            }
        )


def _copy_shared_runtime(
    source: DiarizationBenchmarkReport,
    target: DiarizationBenchmarkReport,
) -> DiarizationBenchmarkReport:
    runtime_by_id = {
        sample.id: (sample.inference_seconds, sample.rtf)
        for sample in source.samples
        if sample.success
    }
    samples = [
        sample.model_copy(
            update={
                "inference_seconds": runtime_by_id[sample.id][0],
                "rtf": runtime_by_id[sample.id][1],
            }
        )
        if sample.success and sample.id in runtime_by_id
        else sample
        for sample in target.samples
    ]
    return target.model_copy(
        update={
            "samples": samples,
            "summary": target.summary.model_copy(
                update={"mean_rtf": source.summary.mean_rtf}
            ),
        }
    )


def require_real_human_dataset(samples: Iterable[DiarizationBenchmarkSample]) -> None:
    required_tags = {"dataset", "dataset_version", "split", "license"}
    invalid: list[str] = []
    for sample in samples:
        reasons: list[str] = []
        if sample.audio_kind != "real" or sample.annotation_kind != "human":
            reasons.append("audio_kind=real and annotation_kind=human required")
        if sample.rttm is None:
            reasons.append("RTTM required")
        if sample.audio_sha256 is None or sample.rttm_sha256 is None:
            reasons.append("audio/RTTM SHA-256 required")
        missing_tags = sorted(required_tags - sample.tags.keys())
        if missing_tags:
            reasons.append(f"missing tags: {','.join(missing_tags)}")
        if reasons:
            invalid.append(f"{sample.id} ({'; '.join(reasons)})")
    if invalid:
        joined = ", ".join(invalid[:10])
        suffix = " ..." if len(invalid) > 10 else ""
        raise ValueError(
            "accuracy matrix requires verified real/human RTTM data with checksums "
            "and provenance tags; "
            f"invalid samples: {joined}{suffix}"
        )


def _assessment(
    name: str,
    report: DiarizationBenchmarkReport,
    thresholds: ProductionThresholds,
) -> ConfigurationAssessment:
    summary = report.summary
    failures: list[str] = []
    if summary.failed_count:
        failures.append(f"{summary.failed_count} samples failed")
    checks = (
        (summary.aggregate_der, thresholds.maximum_der, "DER", "maximum"),
        (summary.mean_jer, thresholds.maximum_jer, "JER", "maximum"),
        (
            summary.speaker_count_accuracy,
            thresholds.minimum_speaker_count_accuracy,
            "speaker count accuracy",
            "minimum",
        ),
        (
            summary.boundary_p95_ms,
            thresholds.maximum_boundary_p95_ms,
            "boundary P95",
            "maximum",
        ),
        (summary.mean_rtf, thresholds.maximum_mean_rtf, "mean RTF", "maximum"),
    )
    for actual, threshold, label, mode in checks:
        if actual is None:
            failures.append(f"{label} is unavailable")
        elif mode == "maximum" and actual > threshold:
            failures.append(f"{label} {actual:.4f} exceeds {threshold:.4f}")
        elif mode == "minimum" and actual < threshold:
            failures.append(f"{label} {actual:.4f} is below {threshold:.4f}")
    return ConfigurationAssessment(
        name=name,
        use_exclusive=report.use_exclusive,
        speaker_constraint=report.speaker_constraint,
        aggregate_der=summary.aggregate_der,
        mean_jer=summary.mean_jer,
        speaker_count_accuracy=summary.speaker_count_accuracy,
        boundary_p95_ms=summary.boundary_p95_ms,
        mean_rtf=summary.mean_rtf,
        failed_count=summary.failed_count,
        meets_metric_thresholds=not failures,
        failures=failures,
    )


def run_matrix(
    backend: DiarizationBackend,
    samples: list[DiarizationBenchmarkSample],
    *,
    manifest: str | Path,
    frame_ms: int = 10,
    collar_ms: int = 0,
    skip_overlap: bool = False,
    speaker_bound_padding: int = 1,
    boundary_tolerance_ms: int = 250,
    thresholds: ProductionThresholds | None = None,
) -> DiarizationBenchmarkMatrixReport:
    require_real_human_dataset(samples)
    thresholds = thresholds or ProductionThresholds()
    cached_backend = _TimelineCacheBackend(backend)
    reports: dict[str, DiarizationBenchmarkReport] = {}
    for use_exclusive in (False, True):
        timeline = "exclusive" if use_exclusive else "regular"
        for constraint in ("oracle_num", "bounded", "unconstrained"):
            name = f"{timeline}-{constraint}"
            reports[name] = DiarizationBenchmarkRunner(
                cached_backend,
                frame_ms=frame_ms,
                collar_ms=collar_ms,
                skip_overlap=skip_overlap,
                use_exclusive=use_exclusive,
                speaker_constraint=constraint,
                speaker_bound_padding=speaker_bound_padding,
                boundary_tolerance_ms=boundary_tolerance_ms,
            ).run(samples, manifest=manifest)
            if use_exclusive:
                reports[name] = _copy_shared_runtime(
                    reports[f"regular-{constraint}"], reports[name]
                )

    first_report = next(iter(reports.values()))
    total_duration = sum(sample.duration_seconds for sample in first_report.samples)
    dataset = DatasetAudit(
        sample_count=len(samples),
        total_duration_seconds=total_duration,
        real_audio_count=sum(sample.audio_kind == "real" for sample in samples),
        human_annotation_count=sum(
            sample.annotation_kind == "human" for sample in samples
        ),
        verified_real_human=True,
    )
    assessments = [
        _assessment(name, report, thresholds) for name, report in reports.items()
    ]
    production_assessments = [
        assessment
        for assessment in assessments
        if assessment.speaker_constraint == "unconstrained"
    ]
    passing = [
        assessment
        for assessment in production_assessments
        if assessment.meets_metric_thresholds
    ]
    ranked = sorted(
        passing or production_assessments,
        key=lambda assessment: (
            assessment.aggregate_der is None,
            assessment.aggregate_der or 0.0,
            assessment.mean_jer is None,
            assessment.mean_jer or 0.0,
            assessment.name,
        ),
    )
    candidate = ranked[0].name if ranked else None
    enough_data = (
        dataset.sample_count >= thresholds.minimum_samples
        and dataset.total_duration_seconds >= thresholds.minimum_duration_seconds
    )
    reasons: list[str] = []
    if dataset.sample_count < thresholds.minimum_samples:
        reasons.append(
            f"sample count {dataset.sample_count} is below {thresholds.minimum_samples}"
        )
    if dataset.total_duration_seconds < thresholds.minimum_duration_seconds:
        reasons.append(
            "duration "
            f"{dataset.total_duration_seconds:.1f}s is below "
            f"{thresholds.minimum_duration_seconds:.1f}s"
        )
    if not passing:
        reasons.append(
            "no production-eligible unconstrained configuration meets all metric "
            "thresholds"
        )
    if not enough_data:
        status = "insufficient_data"
    elif not passing:
        status = "rejected"
    else:
        status = "accepted"
        reasons.append(f"recommended configuration: {candidate}")
    recommendation = ProductionRecommendation(
        status=status,
        candidate_configuration=candidate,
        reasons=reasons,
        thresholds=thresholds,
    )
    return DiarizationBenchmarkMatrixReport(
        backend=backend.name,
        model=backend.model_name,
        manifest=str(Path(manifest).expanduser().resolve()),
        dataset=dataset,
        reports=reports,
        assessments=assessments,
        recommendation=recommendation,
    )


def render_matrix_markdown(report: DiarizationBenchmarkMatrixReport) -> str:
    lines = [
        "# Pyannote 说话人分离准确率矩阵",
        "",
        "> 本报告只接受真实音频和人工标注。`oracle_num` 与参考人数范围用于诊断上限，"
        "不能假设线上始终可得。",
        "",
        f"- 模型：`{report.model}`",
        f"- 样本：`{report.dataset.sample_count}`；总时长："
        f"`{report.dataset.total_duration_seconds:.1f}s`",
        f"- 数据审计：真实音频 `{report.dataset.real_audio_count}`；人工标注 "
        f"`{report.dataset.human_annotation_count}`",
        "",
        "## 配置比较",
        "",
        "| 配置 | 时间轴 | 人数约束 | DER | JER | Speaker 准确率 | 边界 P95 (ms) | RTF | 门槛 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for assessment in report.assessments:
        lines.append(
            "| "
            + " | ".join(
                [
                    assessment.name,
                    "exclusive" if assessment.use_exclusive else "regular",
                    assessment.speaker_constraint,
                    _value(assessment.aggregate_der),
                    _value(assessment.mean_jer),
                    _value(assessment.speaker_count_accuracy),
                    _value(assessment.boundary_p95_ms, 1),
                    _value(assessment.mean_rtf),
                    "通过" if assessment.meets_metric_thresholds else "未通过",
                ]
            )
            + " |"
        )
    recommendation = report.recommendation
    lines.extend(
        [
            "",
            "## 生产门槛建议",
            "",
            f"- 状态：`{recommendation.status}`",
            f"- 当前候选：`{recommendation.candidate_configuration or '无'}`",
        ]
    )
    for reason in recommendation.reasons:
        lines.append(f"- {reason}")
    lines.extend(["", "## 失败样本分析", ""])
    for name, benchmark in report.reports.items():
        failed = [sample for sample in benchmark.samples if not sample.success]
        mismatched = [
            sample
            for sample in benchmark.samples
            if sample.success and sample.speaker_count_correct is False
        ]
        worst = sorted(
            [sample for sample in benchmark.samples if sample.success],
            key=lambda sample: sample.der or 0.0,
            reverse=True,
        )[:5]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- 推理失败：{len(failed)}；speaker 数错误：{len(mismatched)}。"
        )
        if failed:
            for sample in failed[:10]:
                lines.append(f"- 失败 `{sample.id}`：{sample.error}")
        for sample in worst:
            lines.append(
                f"- 高误差 `{sample.id}`：DER={_value(sample.der)}，"
                f"JER={_value(sample.jer)}，speaker="
                f"{sample.reference_speakers}/{sample.hypothesis_speakers}，"
                f"boundary P95={_value(sample.boundary_p95_ms, 1)}ms"
            )
        lines.append("")
    return "\n".join(lines)


def write_matrix_report(
    report: DiarizationBenchmarkMatrixReport, output_directory: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, benchmark in report.reports.items():
        write_report(benchmark, destination / "runs" / name)
    json_path = destination / "matrix-report.json"
    markdown_path = destination / "matrix-report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_matrix_markdown(report), encoding="utf-8")
    return json_path, markdown_path
