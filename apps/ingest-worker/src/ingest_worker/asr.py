from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AsrDependencyError(RuntimeError):
    pass


class AsrInferenceError(RuntimeError):
    pass


class AsrChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def interval_is_valid(self) -> AsrChunk:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class AsrResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    language: str | None = None
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    chunks: list[AsrChunk] = Field(default_factory=list)


class AsrRequestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str | None = Field(default=None, max_length=32)
    hotwords: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("hotwords")
    @classmethod
    def normalize_hotwords(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            term = value.strip()
            if not term:
                raise ValueError("hotwords must not contain blank terms")
            if len(term) > 128:
                raise ValueError("each hotword must contain at most 128 characters")
            key = term.casefold()
            if key not in seen:
                normalized.append(term)
                seen.add(key)
        if sum(len(term) for term in normalized) > 4096:
            raise ValueError("combined hotwords must contain at most 4096 characters")
        return normalized


class AsrBackend(Protocol):
    name: str
    model_name: str
    version: str

    def transcribe(
        self,
        path: Path,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> AsrResult: ...


def _confidence_from_log_probability(value: object) -> float | None:
    try:
        probability = math.exp(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return min(1.0, max(0.0, probability))


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        try:
            import faster_whisper
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrDependencyError(
                "faster-whisper backend requires the 'asr-whisper' optional dependencies"
            ) from exc
        self.model_name = model_name
        self.version = str(getattr(faster_whisper, "__version__", "runtime"))
        self.beam_size = beam_size
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(
        self,
        path: Path,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> AsrResult:
        try:
            generated, info = self._model.transcribe(
                str(path),
                language=language,
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
                hotwords=" ".join(hotwords) if hotwords else None,
            )
            materialized = list(generated)
        except Exception as exc:
            raise AsrInferenceError(f"faster-whisper inference failed: {exc}") from exc
        chunks: list[AsrChunk] = []
        for segment in materialized:
            text = segment.text.strip()
            if not text or segment.end <= segment.start:
                continue
            start_ms = max(0, round(segment.start * 1000))
            chunks.append(
                AsrChunk(
                    start_ms=start_ms,
                    end_ms=max(start_ms + 1, round(segment.end * 1000)),
                    text=text,
                    confidence=_confidence_from_log_probability(
                        getattr(segment, "avg_logprob", None)
                    ),
                )
            )
        return AsrResult(
            text="".join(segment.text for segment in materialized).strip(),
            language=getattr(info, "language", language),
            language_confidence=getattr(info, "language_probability", None),
            chunks=chunks,
        )


def _clean_funasr_text(value: object) -> str:
    return re.sub(r"<\|[^|]*\|>", "", str(value or "")).strip()


def _optional_probability(value: object) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if 0.0 <= probability <= 1.0 else None


class FunAsrBackend:
    name = "funasr"

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        try:
            import funasr
            from funasr import AutoModel
        except ImportError as exc:
            raise AsrDependencyError(
                "FunASR backend requires the 'asr-funasr' optional dependencies"
            ) from exc
        self.model_name = model_name
        self.version = str(getattr(funasr, "__version__", "runtime"))
        self._model = AutoModel(model=model_name, device=device, disable_update=True)

    @staticmethod
    def _chunks(result: dict[str, Any]) -> list[AsrChunk]:
        chunks: list[AsrChunk] = []
        for sentence in result.get("sentence_info") or []:
            text = _clean_funasr_text(sentence.get("text", sentence.get("sentence")))
            start = sentence.get("start")
            end = sentence.get("end")
            if text and start is not None and end is not None and int(end) > int(start):
                chunks.append(AsrChunk(start_ms=int(start), end_ms=int(end), text=text))
        if chunks:
            return chunks

        rich_timestamps = result.get("timestamps") or []
        for item in rich_timestamps:
            if not isinstance(item, dict):
                continue
            text = _clean_funasr_text(item.get("token", item.get("text")))
            start = item.get("start_time")
            end = item.get("end_time")
            if text and start is not None and end is not None and float(end) > float(start):
                start_ms = round(float(start) * 1000)
                chunks.append(
                    AsrChunk(
                        start_ms=start_ms,
                        end_ms=max(start_ms + 1, round(float(end) * 1000)),
                        text=text,
                    )
                )
        if chunks:
            return chunks

        timestamps = result.get("timestamp") or []
        pairs: list[tuple[float, float]] = []
        for pair in timestamps:
            if not isinstance(pair, list | tuple) or len(pair) < 2:
                continue
            try:
                start, end = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                continue
            if end > start:
                pairs.append((start, end))
        text = _clean_funasr_text(result.get("text"))
        if pairs and text:
            return [
                AsrChunk(
                    start_ms=int(pairs[0][0]),
                    end_ms=max(int(pairs[0][0]) + 1, int(pairs[-1][1])),
                    text=text,
                )
            ]
        return []

    def transcribe(
        self,
        path: Path,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> AsrResult:
        try:
            parameters: dict[str, Any] = {
                "input": str(path),
                "language": language or "auto",
                "sentence_timestamp": True,
            }
            if hotwords:
                parameters["hotword"] = " ".join(hotwords)
            results = self._model.generate(**parameters)
        except Exception as exc:
            raise AsrInferenceError(f"FunASR inference failed: {exc}") from exc
        if not results:
            return AsrResult(text="", language=language)
        result = results[0]
        if not isinstance(result, dict):
            raise AsrInferenceError("FunASR returned an unsupported result shape")
        return AsrResult(
            text=_clean_funasr_text(result.get("text")),
            language=result.get("language") or language,
            language_confidence=_optional_probability(
                result.get("language_confidence", result.get("language_probability"))
            ),
            chunks=self._chunks(result),
        )


def build_asr_backend(
    backend: str,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
) -> AsrBackend | None:
    normalized = backend.strip().lower()
    if normalized in {"", "disabled", "none"}:
        return None
    if normalized == "faster-whisper":
        return FasterWhisperBackend(
            model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
        )
    if normalized == "funasr":
        return FunAsrBackend(model_name, device=device)
    raise ValueError(f"unsupported ASR backend: {backend}")
