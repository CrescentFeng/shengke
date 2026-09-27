from __future__ import annotations

import json
import platform
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from audio_core.contracts import (
    AudioAsset,
    AudioJob,
    AudioQualityMetrics,
    AudioReport,
    AudioSegment,
    DiarizationDocument,
    Finding,
    JobState,
    Operation,
    ToolRun,
    ToolRunStatus,
    TranscriptDocument,
    TranscriptSpan,
)
from audio_core.hashing import sha256_file
from audio_core.normalize import NormalizeSpec, normalize_audio
from audio_core.probe import probe_file
from audio_core.quality import (
    AudioQualityPolicy,
    QualityAnalysisSpec,
    analyze_quality,
    evaluate_quality,
)
from audio_core.segments import SegmentExportSpec, extract_audio_segment
from audio_core.vad import VadResult, VadSpec, detect_voice_activity

from .asr import AsrBackend, AsrRequestOptions
from .diarization import (
    DiarizationBackend,
    DiarizationSpec,
    assign_speakers,
)
from .s3_store import safe_filename


class ArtifactStorage(Protocol):
    def fetch_input(self, uri: str, destination: Path) -> Path: ...

    def upload_file(self, source: Path, key: str, content_type: str | None = None) -> str: ...


class MetadataStorage(Protocol):
    def add_asset(self, job_id: UUID, asset: AudioAsset, role: str) -> None: ...

    def add_tool_run(self, job_id: UUID, run: ToolRun) -> None: ...

    def add_segment(self, job_id: UUID, segment: AudioSegment) -> None: ...

    def save_transcript(self, transcript: TranscriptDocument, transcript_uri: str) -> None: ...

    def save_diarization(
        self, diarization: DiarizationDocument, diarization_uri: str
    ) -> None: ...

    def save_report(self, report: AudioReport, report_uri: str) -> None: ...

    def transition(
        self,
        job_id: UUID,
        target: JobState,
        *,
        message: str | None = None,
        progress: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> AudioJob: ...


class IngestProcessor:
    def __init__(
        self,
        metadata: MetadataStorage,
        artifacts: ArtifactStorage,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        asr: AsrBackend | None = None,
        asr_language: str | None = None,
        diarization: DiarizationBackend | None = None,
    ) -> None:
        self.metadata = metadata
        self.artifacts = artifacts
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.asr = asr
        self.asr_language = asr_language
        self.diarization = diarization

    def process(self, job: AudioJob) -> AudioReport:
        with tempfile.TemporaryDirectory(prefix=f"audio-ingest-{job.id}-") as temporary:
            root = Path(temporary)
            input_name = safe_filename(job.input_uri)
            local_input = self.artifacts.fetch_input(job.input_uri, root / input_name)
            input_hash = sha256_file(local_input)
            input_probe = probe_file(local_input, ffprobe=self.ffprobe)
            original_key = f"jobs/{job.id}/original/{input_hash}/{input_name}"
            original_uri = self.artifacts.upload_file(local_input, original_key)
            original = AudioAsset(
                id=uuid5(job.id, f"asset:original:{input_hash}"),
                uri=original_uri,
                filename=input_name,
                sha256=input_hash,
                size_bytes=local_input.stat().st_size,
                probe=input_probe,
            )
            self.metadata.add_asset(job.id, original, "original")
            runs = [self._record_probe(job, original)]
            assets = [original]
            findings: list[Finding] = []
            quality: AudioQualityMetrics | None = None
            segments: list[AudioSegment] = []
            vad_result: VadResult | None = None
            slice_paths: dict[UUID, Path] = {}
            transcript: TranscriptDocument | None = None
            transcript_uri: str | None = None
            diarization: DiarizationDocument | None = None
            diarization_uri: str | None = None

            if Operation.VAD in job.operations or Operation.TRANSCRIBE in job.operations:
                segments, segment_assets, slice_paths, run, vad_result = self._segment_speech(
                    job, original, local_input, root
                )
                assets.extend(segment_assets)
                runs.append(run)

            if Operation.DIARIZE in job.operations:
                segments, diarization, diarization_uri, run = self._diarize(
                    job, original, local_input, segments, root
                )
                runs.append(run)

            if Operation.TRANSCRIBE in job.operations:
                segments, transcript, transcript_uri, run = self._transcribe_segments(
                    job, segments, slice_paths, root
                )
                runs.append(run)

            if Operation.LOUDNESS in job.operations:
                quality, run, quality_findings = self._analyze_quality(job, original, local_input)
                runs.append(run)
                findings.extend(quality_findings)

            if Operation.NORMALIZE in job.operations:
                derived, run = self._normalize(job, original, local_input, root)
                assets.append(derived)
                runs.append(run)

            report = AudioReport(
                id=uuid5(job.id, "report:v1"),
                job_id=job.id,
                input_asset_id=original.id,
                processing_chain_id=self._processing_chain_id(job),
                assets=assets,
                segments=segments,
                tool_runs=runs,
                findings=findings,
                summary={
                    "input_uri": job.input_uri,
                    "asset_count": len(assets),
                    "operations": [operation.value for operation in job.operations],
                    "quality": quality.model_dump(mode="json") if quality else None,
                    "vad": (
                        vad_result.model_dump(mode="json", exclude={"segments"})
                        if vad_result
                        else None
                    ),
                    "transcript": (
                        {
                            "uri": transcript_uri,
                            "language": transcript.language,
                            "text": transcript.text,
                            "span_count": len(transcript.spans),
                        }
                        if transcript
                        else None
                    ),
                    "diarization": (
                        {
                            "uri": diarization_uri,
                            "backend": diarization.backend,
                            "model": diarization.model,
                            "exclusive": diarization.exclusive,
                            "speaker_count": diarization.speaker_count,
                            "turn_count": len(diarization.turns),
                        }
                        if diarization
                        else None
                    ),
                },
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            report_uri = self.artifacts.upload_file(
                report_path,
                f"jobs/{job.id}/reports/{report.id}.json",
                "application/json",
            )
            self.metadata.save_report(report, report_uri)
            self.metadata.transition(
                job.id,
                JobState.SUCCEEDED,
                message="ingest pipeline completed",
                progress=1.0,
            )
            return report

    def _record_probe(self, job: AudioJob, asset: AudioAsset) -> ToolRun:
        now = datetime.now(UTC)
        run = ToolRun(
            id=uuid5(job.id, f"tool:probe:{asset.id}"),
            tool_name="ffprobe",
            tool_version=self._tool_version(self.ffprobe),
            operation=Operation.PROBE,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[asset.id],
            output_asset_ids=[asset.id],
            metrics={
                "duration_seconds": asset.probe.duration_seconds if asset.probe else None,
                "sample_rate_hz": asset.probe.sample_rate_hz if asset.probe else None,
            },
            started_at=now,
            finished_at=now,
        )
        self.metadata.add_tool_run(job.id, run)
        return run

    def _segment_speech(
        self,
        job: AudioJob,
        source_asset: AudioAsset,
        local_input: Path,
        root: Path,
    ) -> tuple[
        list[AudioSegment],
        list[AudioAsset],
        dict[UUID, Path],
        ToolRun,
        VadResult,
    ]:
        started = datetime.now(UTC)
        spec = VadSpec.model_validate(job.parameters.get("vad", {}))
        result = detect_voice_activity(local_input, spec=spec, ffmpeg=self.ffmpeg)
        segments: list[AudioSegment] = []
        slice_assets: list[AudioAsset] = []
        slice_paths: dict[UUID, Path] = {}
        segment_root = root / "segments"

        for index, detected in enumerate(result.segments):
            segment_id = uuid5(
                job.id,
                f"segment:{source_asset.id}:{detected.start_ms}:{detected.end_ms}",
            )
            slice_path = extract_audio_segment(
                local_input,
                segment_root / f"speech-{index:04d}.wav",
                spec=SegmentExportSpec(
                    start_ms=detected.start_ms,
                    end_ms=detected.end_ms,
                    sample_rate_hz=spec.sample_rate_hz,
                    channels=1,
                ),
                ffmpeg=self.ffmpeg,
            )
            slice_hash = sha256_file(slice_path)
            slice_uri = self.artifacts.upload_file(
                slice_path,
                f"jobs/{job.id}/segments/{segment_id}/{slice_hash}.wav",
                "audio/wav",
            )
            slice_asset = AudioAsset(
                id=uuid5(job.id, f"asset:speech-segment:{segment_id}:{slice_hash}"),
                uri=slice_uri,
                filename=f"speech-{index:04d}.wav",
                sha256=slice_hash,
                size_bytes=slice_path.stat().st_size,
                probe=probe_file(slice_path, ffprobe=self.ffprobe),
                source_asset_id=source_asset.id,
            )
            segment = AudioSegment(
                id=segment_id,
                asset_id=source_asset.id,
                slice_asset_id=slice_asset.id,
                start_ms=detected.start_ms,
                end_ms=detected.end_ms,
                confidence=detected.confidence,
                labels={"speech": detected.confidence},
            )
            self.metadata.add_asset(job.id, slice_asset, "speech_segment")
            self.metadata.add_segment(job.id, segment)
            slice_assets.append(slice_asset)
            slice_paths[segment.id] = slice_path
            segments.append(segment)

        run = ToolRun(
            id=uuid5(job.id, f"tool:vad:{source_asset.id}"),
            tool_name=result.backend,
            tool_version="1",
            operation=Operation.VAD,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[source_asset.id],
            output_asset_ids=[asset.id for asset in slice_assets],
            parameters=spec.model_dump(),
            metrics={
                "segment_count": len(segments),
                "speech_duration_ms": result.speech_duration_ms,
                "speech_ratio": result.speech_ratio,
                "threshold_dbfs": result.threshold_dbfs,
            },
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self.metadata.add_tool_run(job.id, run)
        return segments, slice_assets, slice_paths, run, result

    def _diarize(
        self,
        job: AudioJob,
        source_asset: AudioAsset,
        local_input: Path,
        segments: list[AudioSegment],
        root: Path,
    ) -> tuple[list[AudioSegment], DiarizationDocument, str, ToolRun]:
        if self.diarization is None:
            raise RuntimeError(
                "diarize operation requested but AUDIO_PLATFORM_DIARIZATION_BACKEND is disabled"
            )
        started = datetime.now(UTC)
        spec = DiarizationSpec.model_validate(job.parameters.get("diarization", {}))
        result = self.diarization.diarize(local_input, spec=spec)
        updated_segments = assign_speakers(segments, result.turns)
        for segment in updated_segments:
            self.metadata.add_segment(job.id, segment)

        document = DiarizationDocument(
            id=uuid5(job.id, "diarization:v1"),
            job_id=job.id,
            asset_id=source_asset.id,
            backend=self.diarization.name,
            model=self.diarization.model_name,
            exclusive=result.exclusive,
            speaker_count=len({turn.speaker for turn in result.turns}),
            turns=result.turns,
            regular_turns=result.regular_turns,
            exclusive_turns=result.exclusive_turns,
        )
        document_path = root / "diarization.json"
        document_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        document_uri = self.artifacts.upload_file(
            document_path,
            f"jobs/{job.id}/diarizations/{document.id}.json",
            "application/json",
        )
        self.metadata.save_diarization(document, document_uri)
        run = ToolRun(
            id=uuid5(job.id, "tool:diarize:v1"),
            tool_name=self.diarization.name,
            tool_version=self.diarization.version,
            operation=Operation.DIARIZE,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[source_asset.id],
            parameters={"model": self.diarization.model_name, **spec.model_dump()},
            metrics={
                "speaker_count": document.speaker_count,
                "turn_count": len(document.turns),
                "regular_turn_count": len(document.regular_turns),
                "exclusive_turn_count": len(document.exclusive_turns),
                "exclusive": document.exclusive,
                "assigned_segment_count": sum(
                    segment.speaker is not None for segment in updated_segments
                ),
            },
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self.metadata.add_tool_run(job.id, run)
        return updated_segments, document, document_uri, run

    def _transcribe_segments(
        self,
        job: AudioJob,
        segments: list[AudioSegment],
        slice_paths: dict[UUID, Path],
        root: Path,
    ) -> tuple[list[AudioSegment], TranscriptDocument, str, ToolRun]:
        started = datetime.now(UTC)
        options = AsrRequestOptions.model_validate(job.parameters.get("asr", {}))
        language_hint = options.language or self.asr_language
        if segments and self.asr is None:
            raise RuntimeError(
                "transcribe operation requested but AUDIO_PLATFORM_ASR_BACKEND is disabled"
            )

        updated_segments: list[AudioSegment] = []
        spans: list[TranscriptSpan] = []
        texts: list[str] = []
        languages: list[str] = []
        language_confidences: list[float] = []

        for segment in segments:
            assert self.asr is not None
            result = self.asr.transcribe(
                slice_paths[segment.id],
                language=language_hint,
                hotwords=options.hotwords,
            )
            segment_text = result.text.strip()
            language = result.language or language_hint
            labels = dict(segment.labels)
            chunk_confidences = [
                chunk.confidence for chunk in result.chunks if chunk.confidence is not None
            ]
            if chunk_confidences:
                labels["transcript"] = sum(chunk_confidences) / len(chunk_confidences)
            updated = segment.model_copy(
                update={
                    "transcript": segment_text or None,
                    "language": language,
                    "labels": labels,
                }
            )
            self.metadata.add_segment(job.id, updated)
            updated_segments.append(updated)
            if segment_text:
                texts.append(segment_text)
            if language:
                languages.append(language)
            if result.language_confidence is not None:
                language_confidences.append(result.language_confidence)

            segment_duration = segment.end_ms - segment.start_ms
            for chunk in result.chunks:
                relative_start = min(segment_duration, max(0, chunk.start_ms))
                relative_end = min(segment_duration, max(0, chunk.end_ms))
                if relative_end <= relative_start or not chunk.text.strip():
                    continue
                spans.append(
                    TranscriptSpan(
                        segment_id=segment.id,
                        start_ms=segment.start_ms + relative_start,
                        end_ms=segment.start_ms + relative_end,
                        text=chunk.text.strip(),
                        confidence=chunk.confidence,
                        speaker=segment.speaker,
                    )
                )
            if segment_text and not result.chunks:
                spans.append(
                    TranscriptSpan(
                        segment_id=segment.id,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        text=segment_text,
                        speaker=segment.speaker,
                    )
                )

        language = Counter(languages).most_common(1)[0][0] if languages else language_hint
        language_confidence = (
            sum(language_confidences) / len(language_confidences) if language_confidences else None
        )
        backend_name = self.asr.name if self.asr else "no-speech"
        model_name = self.asr.model_name if self.asr else "none"
        backend_version = self.asr.version if self.asr else "1"
        transcript = TranscriptDocument(
            id=uuid5(job.id, "transcript:v1"),
            job_id=job.id,
            backend=backend_name,
            model=model_name,
            language=language,
            language_confidence=language_confidence,
            text=self._join_transcripts(texts),
            spans=spans,
        )
        transcript_path = root / "transcript.json"
        transcript_path.write_text(
            json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        transcript_uri = self.artifacts.upload_file(
            transcript_path,
            f"jobs/{job.id}/transcripts/{transcript.id}.json",
            "application/json",
        )
        self.metadata.save_transcript(transcript, transcript_uri)
        run = ToolRun(
            id=uuid5(job.id, "tool:transcribe:v1"),
            tool_name=backend_name,
            tool_version=backend_version,
            operation=Operation.TRANSCRIBE,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[
                segment.slice_asset_id
                for segment in updated_segments
                if segment.slice_asset_id is not None
            ],
            parameters={
                "model": model_name,
                "language_hint": language_hint,
                "hotwords": options.hotwords,
            },
            metrics={
                "segment_count": len(updated_segments),
                "span_count": len(spans),
                "character_count": len(transcript.text),
                "language": language,
                "hotword_count": len(options.hotwords),
            },
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self.metadata.add_tool_run(job.id, run)
        return updated_segments, transcript, transcript_uri, run

    @staticmethod
    def _join_transcripts(parts: list[str]) -> str:
        output = ""
        for part in parts:
            normalized = part.strip()
            if not normalized:
                continue
            if output and output[-1].isascii() and normalized[0].isascii():
                output += " "
            output += normalized
        return output

    def _analyze_quality(
        self,
        job: AudioJob,
        asset: AudioAsset,
        local_input: Path,
    ) -> tuple[AudioQualityMetrics, ToolRun, list[Finding]]:
        started = datetime.now(UTC)
        if asset.probe is None or not asset.probe.duration_seconds:
            raise ValueError("quality analysis requires a measurable audio duration")
        analysis_spec = QualityAnalysisSpec.model_validate(
            job.parameters.get("quality_analysis", {})
        )
        policy = AudioQualityPolicy.model_validate(job.parameters.get("quality_policy", {}))
        metrics = analyze_quality(
            local_input,
            duration_seconds=asset.probe.duration_seconds,
            spec=analysis_spec,
            ffmpeg=self.ffmpeg,
        )
        findings = evaluate_quality(asset.id, metrics, policy)
        run = ToolRun(
            id=uuid5(job.id, f"tool:quality:{asset.id}"),
            tool_name="ffmpeg-quality",
            tool_version=self._tool_version(self.ffmpeg),
            operation=Operation.LOUDNESS,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[asset.id],
            parameters={
                "analysis": analysis_spec.model_dump(),
                "policy": policy.model_dump(),
            },
            metrics=metrics.model_dump(),
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self.metadata.add_tool_run(job.id, run)
        return metrics, run, findings

    def _normalize(
        self,
        job: AudioJob,
        source_asset: AudioAsset,
        local_input: Path,
        root: Path,
    ) -> tuple[AudioAsset, ToolRun]:
        started = datetime.now(UTC)
        spec = NormalizeSpec(
            sample_rate_hz=int(job.parameters.get("sample_rate_hz", 16000)),
            channels=int(job.parameters.get("channels", 1)),
            sample_format=str(job.parameters.get("sample_format", "s16")),
        )
        output = normalize_audio(
            local_input,
            root / "normalized.wav",
            spec=spec,
            ffmpeg=self.ffmpeg,
        )
        output_hash = sha256_file(output)
        output_uri = self.artifacts.upload_file(
            output,
            f"jobs/{job.id}/derived/{output_hash}/normalized.wav",
            "audio/wav",
        )
        derived = AudioAsset(
            id=uuid5(job.id, f"asset:normalized:{output_hash}"),
            uri=output_uri,
            filename="normalized.wav",
            sha256=output_hash,
            size_bytes=output.stat().st_size,
            probe=probe_file(output, ffprobe=self.ffprobe),
            source_asset_id=source_asset.id,
        )
        self.metadata.add_asset(job.id, derived, "normalized")
        run = ToolRun(
            id=uuid5(job.id, f"tool:normalize:{source_asset.id}"),
            tool_name="ffmpeg",
            tool_version=self._tool_version(self.ffmpeg),
            operation=Operation.NORMALIZE,
            status=ToolRunStatus.SUCCEEDED,
            input_asset_ids=[source_asset.id],
            output_asset_ids=[derived.id],
            parameters=spec.model_dump(),
            metrics={"output_size_bytes": derived.size_bytes},
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self.metadata.add_tool_run(job.id, run)
        return derived, run

    @staticmethod
    def _tool_version(executable: str) -> str:
        resolved = shutil.which(executable)
        return f"{Path(resolved).name if resolved else executable}@runtime"

    @staticmethod
    def _processing_chain_id(job: AudioJob) -> str:
        operations = "+".join(operation.value for operation in job.operations)
        return f"audio-platform/0.9.0/{operations}/{platform.machine()}"
