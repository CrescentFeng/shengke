from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from audio_core.contracts import AudioSegment, SpeakerTurn
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiarizationDependencyError(RuntimeError):
    pass


class DiarizationInferenceError(RuntimeError):
    pass


class DiarizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_speakers: int | None = Field(default=None, ge=1, le=100)
    min_speakers: int | None = Field(default=None, ge=1, le=100)
    max_speakers: int | None = Field(default=None, ge=1, le=100)
    use_exclusive: bool = True

    @model_validator(mode="after")
    def speaker_bounds_are_valid(self) -> DiarizationSpec:
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError("num_speakers cannot be combined with min_speakers/max_speakers")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers")
        return self


class DiarizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turns: list[SpeakerTurn] = Field(default_factory=list)
    regular_turns: list[SpeakerTurn] = Field(default_factory=list)
    exclusive_turns: list[SpeakerTurn] = Field(default_factory=list)
    exclusive: bool = True


class DiarizationBackend(Protocol):
    name: str
    model_name: str
    version: str

    def diarize(self, path: Path, *, spec: DiarizationSpec) -> DiarizationResult: ...


def _turns_from_annotation(annotation: Any) -> list[SpeakerTurn]:
    raw: list[tuple[Any, Any]] = []
    if hasattr(annotation, "itertracks"):
        raw = [(turn, speaker) for turn, _track, speaker in annotation.itertracks(yield_label=True)]
    else:
        for item in annotation:
            if isinstance(item, tuple) and len(item) >= 2:
                raw.append((item[0], item[-1]))

    turns: list[SpeakerTurn] = []
    for turn, speaker in raw:
        try:
            start_ms = max(0, round(float(turn.start) * 1000))
            end_ms = round(float(turn.end) * 1000)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DiarizationInferenceError("pyannote returned an invalid speaker turn") from exc
        if end_ms <= start_ms:
            continue
        turns.append(
            SpeakerTurn(
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=str(speaker),
            )
        )
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))


class PyannoteDiarizationBackend:
    name = "pyannote"

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
        *,
        token: str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import pyannote.audio
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationDependencyError(
                "pyannote backend requires the 'diarization-pyannote' optional dependencies"
            ) from exc
        self.model_name = model_name
        self.version = str(getattr(pyannote.audio, "__version__", "runtime"))
        try:
            self._pipeline = Pipeline.from_pretrained(model_name, token=token)
            if device != "cpu":
                import torch

                self._pipeline.to(torch.device(device))
        except Exception as exc:
            raise DiarizationDependencyError(
                f"unable to load pyannote diarization model '{model_name}': {exc}"
            ) from exc

    def diarize(self, path: Path, *, spec: DiarizationSpec) -> DiarizationResult:
        parameters = {
            name: value
            for name, value in {
                "num_speakers": spec.num_speakers,
                "min_speakers": spec.min_speakers,
                "max_speakers": spec.max_speakers,
            }.items()
            if value is not None
        }
        try:
            output = self._pipeline(str(path), **parameters)
            regular = getattr(output, "speaker_diarization", output)
            exclusive = getattr(output, "exclusive_speaker_diarization", None)
            use_exclusive = spec.use_exclusive and exclusive is not None
            regular_turns = _turns_from_annotation(regular)
            exclusive_turns = (
                _turns_from_annotation(exclusive) if exclusive is not None else []
            )
            return DiarizationResult(
                turns=exclusive_turns if use_exclusive else regular_turns,
                regular_turns=regular_turns,
                exclusive_turns=exclusive_turns,
                exclusive=use_exclusive,
            )
        except DiarizationInferenceError:
            raise
        except Exception as exc:
            raise DiarizationInferenceError(f"pyannote inference failed: {exc}") from exc


def build_diarization_backend(
    backend: str,
    *,
    model_name: str,
    token: str | None,
    device: str,
) -> DiarizationBackend | None:
    normalized = backend.strip().lower()
    if normalized in {"", "disabled", "none"}:
        return None
    if normalized == "pyannote":
        return PyannoteDiarizationBackend(
            model_name=model_name,
            token=token,
            device=device,
        )
    raise ValueError(f"unsupported diarization backend: {backend}")


def assign_speakers(
    segments: list[AudioSegment], turns: list[SpeakerTurn]
) -> list[AudioSegment]:
    assigned: list[AudioSegment] = []
    for segment in segments:
        overlap_by_speaker: dict[str, int] = {}
        for turn in turns:
            overlap = max(
                0,
                min(segment.end_ms, turn.end_ms) - max(segment.start_ms, turn.start_ms),
            )
            if overlap:
                overlap_by_speaker[turn.speaker] = (
                    overlap_by_speaker.get(turn.speaker, 0) + overlap
                )
        if not overlap_by_speaker:
            assigned.append(segment)
            continue
        speaker, overlap_ms = min(
            overlap_by_speaker.items(),
            key=lambda item: (-item[1], item[0]),
        )
        duration_ms = segment.end_ms - segment.start_ms
        labels = dict(segment.labels)
        labels["speaker_coverage"] = min(1.0, overlap_ms / duration_ms)
        assigned.append(
            segment.model_copy(update={"speaker": speaker, "labels": labels})
        )
    return assigned
