"""Shared audio primitives and contracts."""

from .contracts import (
    AudioAsset,
    AudioJob,
    AudioProbe,
    AudioQualityMetrics,
    AudioReport,
    AudioReportEnvelope,
    AudioSegment,
    CreateAudioJob,
    DiarizationDocument,
    DiarizationEnvelope,
    Finding,
    FindingSeverity,
    JobEvent,
    JobState,
    MediaFormat,
    Operation,
    SpeakerTurn,
    StoredArtifact,
    ToolRun,
    ToolRunStatus,
    TranscriptDocument,
    TranscriptEnvelope,
    TranscriptSpan,
)
from .hashing import sha256_file
from .normalize import NormalizeSpec, normalize_audio
from .probe import probe_file
from .quality import AudioQualityPolicy, QualityAnalysisSpec, analyze_quality, evaluate_quality
from .segments import SegmentExportSpec, extract_audio_segment
from .vad import DetectedSpeechSegment, VadResult, VadSpec, detect_voice_activity

__all__ = [
    "AudioAsset",
    "AudioJob",
    "AudioProbe",
    "AudioQualityMetrics",
    "AudioReport",
    "AudioReportEnvelope",
    "AudioSegment",
    "CreateAudioJob",
    "DiarizationDocument",
    "DiarizationEnvelope",
    "Finding",
    "FindingSeverity",
    "JobEvent",
    "JobState",
    "MediaFormat",
    "NormalizeSpec",
    "Operation",
    "SpeakerTurn",
    "AudioQualityPolicy",
    "QualityAnalysisSpec",
    "DetectedSpeechSegment",
    "VadResult",
    "VadSpec",
    "SegmentExportSpec",
    "StoredArtifact",
    "ToolRun",
    "ToolRunStatus",
    "TranscriptDocument",
    "TranscriptEnvelope",
    "TranscriptSpan",
    "normalize_audio",
    "analyze_quality",
    "detect_voice_activity",
    "evaluate_quality",
    "extract_audio_segment",
    "probe_file",
    "sha256_file",
]

__version__ = "0.9.0"
