"""Model-independent contracts for Shengke; no inference framework dependencies."""

from __future__ import annotations

import base64
import re
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from .contracts import AudioAsset, AudioQualityMetrics, StrictModel

ModelId = Literal["cosyvoice2", "gpt-sovits-v2", "vits-csmsc", "transformer-csmsc"]
Emotion = Literal["neutral", "happy", "sad", "angry"]
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MODEL_IDS = ("cosyvoice2", "gpt-sovits-v2", "vits-csmsc", "transformer-csmsc")
EMOTION_PROMPTS = {
    "happy": "请用开心、轻快、自然的语气说这句话。",
    "sad": "请用克制、低沉、悲伤但不过度哭泣的语气说这句话。",
    "angry": "请用愤怒、坚定、有力度但不尖叫的语气说这句话。",
}


class TtsSynthesisSpec(StrictModel):
    model: ModelId = "cosyvoice2"
    text: str = Field(min_length=1, max_length=200)
    reference_text: str = Field(default="", max_length=500)
    language: Literal["zh"] = "zh"
    emotion: Emotion = "neutral"
    inference_profile: Literal[
        "baseline", "pilot_v2", "gpt_pause_v3", "cosy_happy_v4", "cosy_rhythm_v5", "gpt_rhythm_v6"
    ] = "baseline"
    voice_id: UUID | None = None

    @field_validator("text", "reference_text", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_capabilities(self):
        if self.inference_profile == "gpt_rhythm_v6" and not re.fullmatch(
            r"[\u3400-\u9fff\s，,。.!?！？;；:：、…]+", self.text
        ):
            raise ValueError("节奏试用暂支持纯中文及标点，请移除数字和外语后重试")
        if self.model in {"cosyvoice2", "gpt-sovits-v2"} and not self.reference_text:
            raise ValueError("参考音频的准确文本不能为空")
        if self.emotion != "neutral" and self.model != "cosyvoice2":
            raise ValueError("当前只有 CosyVoice2 支持情绪指令")
        if self.inference_profile != "baseline" and self.model not in MODEL_IDS[:2]:
            raise ValueError("实验推理配置仅支持克隆引擎")
        if (
            self.inference_profile in {"gpt_pause_v3", "gpt_rhythm_v6"}
            and self.model != "gpt-sovits-v2"
        ):
            raise ValueError("停顿实验仅支持 GPT-SoVITS")
        if self.inference_profile in {"cosy_happy_v4", "cosy_rhythm_v5"} and (
            self.model != "cosyvoice2" or self.emotion != "happy"
        ):
            raise ValueError("简短开心指令实验仅支持 CosyVoice2 开心模式")
        return self


class VoiceUpload(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    filename: str = Field(min_length=1, max_length=200)
    audio_base64: str = Field(min_length=1, max_length=14 * 1024 * 1024)
    reference_text: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=2000)
    license: str = Field(min_length=1, max_length=500)

    @field_validator("name", "reference_text", "source", "license", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class VoiceProfile(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    reference_text: str
    source: str
    license: str
    asset: AudioAsset
    quality: AudioQualityMetrics
    warnings: list[str] = Field(default_factory=list)


class EvaluationCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    job_ids: list[UUID] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_jobs(self):
        if len(set(self.job_ids)) != len(self.job_ids):
            raise ValueError("评测不能包含重复作业")
        return self


class ListeningScore(StrictModel):
    listener_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    sample_id: UUID
    naturalness: int = Field(ge=1, le=5, strict=True)
    similarity: int | None = Field(default=None, ge=1, le=5, strict=True)
    emotion_match: int | None = Field(default=None, ge=1, le=5, strict=True)
    comment: str = Field(default="", max_length=1000)


def decode_audio(encoded: str) -> bytes:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("音频编码无效") from exc
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise ValueError("参考音频必须非空且不超过 10 MiB")
    return data
