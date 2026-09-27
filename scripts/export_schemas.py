from __future__ import annotations

import json
from pathlib import Path

from asr_eval.models import AsrBenchmarkReport
from audio_core.contracts import (
    AudioAsset,
    AudioJob,
    AudioQualityMetrics,
    AudioReport,
    AudioSegment,
    DiarizationDocument,
    SpeakerTurn,
    ToolRun,
    TranscriptDocument,
)
from diarization_eval.models import (
    DiarizationBenchmarkMatrixReport,
    DiarizationBenchmarkReport,
)
from vad_eval.models import VadBenchmarkReport


def main() -> None:
    schemas = {
        "audio_asset.schema.json": AudioAsset,
        "audio_job.schema.json": AudioJob,
        "audio_quality_metrics.schema.json": AudioQualityMetrics,
        "audio_report.schema.json": AudioReport,
        "audio_segment.schema.json": AudioSegment,
        "diarization_document.schema.json": DiarizationDocument,
        "diarization_benchmark_report.schema.json": DiarizationBenchmarkReport,
        "diarization_benchmark_matrix_report.schema.json": (
            DiarizationBenchmarkMatrixReport
        ),
        "asr_benchmark_report.schema.json": AsrBenchmarkReport,
        "tool_run.schema.json": ToolRun,
        "transcript_document.schema.json": TranscriptDocument,
        "speaker_turn.schema.json": SpeakerTurn,
        "vad_benchmark_report.schema.json": VadBenchmarkReport,
    }
    for filename, model in schemas.items():
        destination = Path("contracts") / filename
        destination.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
