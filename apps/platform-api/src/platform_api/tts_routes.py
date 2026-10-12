from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import secrets
import statistics
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from audio_core.contracts import AudioAsset, JobState
from audio_core.hashing import sha256_file
from audio_core.tts import (
    MODEL_IDS,
    EvaluationCreate,
    ListeningScore,
    VoiceProfile,
    VoiceUpload,
    decode_audio,
)
from audio_core.tts_quality import inspect_reference
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from ingest_worker.tts import model_endpoints

from .dependencies import get_store
from .store import JobNotFound
from .tts_store import get_tts_artifacts, get_tts_store

TtsStoreDep = Annotated[Any, Depends(get_tts_store)]
ArtifactsDep = Annotated[Any, Depends(get_tts_artifacts)]
JobsDep = Annotated[Any, Depends(get_store)]

router = APIRouter(prefix="/v1/tts", tags=["shengke"])


INSTANCE_ID = os.getenv("SHENGKE_INSTANCE_ID") or str(uuid4())


@router.get("/info")
def info():
    from audio_core.shengke_release import BUILD_ID, RELEASE

    return {**RELEASE, "build_id": BUILD_ID, "instance_id": INSTANCE_ID}


@router.get("/jobs")
def history(jobs: JobsDep, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return jobs.list_tts_jobs(limit, offset)


@router.get("/models")
def models():
    endpoints = model_endpoints()

    def status(model):
        result = {
            "id": model,
            "cloning": model in MODEL_IDS[:2],
            "emotions": ["neutral", "happy", "sad", "angry"]
            if model == "cosyvoice2"
            else ["neutral"],
            "status": "not_configured",
            "runtime": None,
        }
        if endpoint := endpoints.get(model):
            try:
                with urllib.request.urlopen(endpoint.rstrip("/") + "/healthz", timeout=2) as r:
                    health = json.loads(r.read(65536))
                result["runtime"] = health
                result["status"] = (
                    "ready" if health.get("model") == model and health.get("ready") else "not_ready"
                )
            except (OSError, ValueError):
                result["status"] = "offline"
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(status, MODEL_IDS))


@router.post("/voices", response_model=VoiceProfile, status_code=201)
def upload_voice(body: VoiceUpload, store: TtsStoreDep, artifacts: ArtifactsDep):
    suffix = Path(body.filename).suffix.lower()
    if suffix not in {".wav", ".flac"}:
        raise HTTPException(422, "首版支持 WAV/FLAC 参考音频")
    with tempfile.TemporaryDirectory(prefix="shengke-upload-") as directory:
        path = Path(directory) / ("reference" + suffix)
        try:
            path.write_bytes(decode_audio(body.audio_base64))
            probe, quality, warnings = inspect_reference(path)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        digest = sha256_file(path)
        voice_id = uuid4()
        uri = artifacts.upload_file(path, f"tts/voices/{voice_id}/{digest}{suffix}")
        profile = VoiceProfile(
            id=voice_id,
            name=body.name,
            reference_text=body.reference_text,
            source=body.source,
            license=body.license,
            asset=AudioAsset(
                uri=uri,
                filename="reference" + suffix,
                sha256=digest,
                size_bytes=path.stat().st_size,
                probe=probe,
            ),
            quality=quality,
            warnings=warnings,
        )
        store.save_voice(profile)
        return profile


@router.get("/voices", response_model=list[VoiceProfile])
def list_voices(store: TtsStoreDep):
    return store.list_voices()


def audio_response(uri, artifacts):
    with tempfile.TemporaryDirectory(prefix="shengke-listen-") as directory:
        path = artifacts.fetch_input(uri, Path(directory) / "audio.wav")
        # All synthesized outputs are WAV; references may also be FLAC.
        data = path.read_bytes()
        media = "audio/flac" if data.startswith(b"fLaC") else "audio/wav"
        return Response(data, media_type=media, headers={"Cache-Control": "no-store"})


@router.get("/voices/{voice_id}/audio")
def voice_audio(voice_id: UUID, store: TtsStoreDep, artifacts: ArtifactsDep):
    try:
        profile = store.get_voice(voice_id)
    except KeyError as exc:
        raise HTTPException(404, "未找到参考音频") from exc
    return audio_response(profile["asset"]["uri"], artifacts)


@router.get("/jobs/{job_id}/audio")
def job_audio(job_id: UUID, jobs: JobsDep, artifacts: ArtifactsDep):
    try:
        if jobs.get(job_id).state != JobState.SUCCEEDED:
            raise HTTPException(409, "作业尚未完成")
        report = jobs.get_report(job_id).report
        output_id = report.summary.get("output_asset_id")
        asset = next(a for a in report.assets if str(a.id) == output_id)
        return audio_response(asset.uri, artifacts)
    except (JobNotFound, StopIteration) as exc:
        raise HTTPException(404, "未找到合成音频") from exc


@router.post("/evaluations", status_code=201)
def create_evaluation(body: EvaluationCreate, store: TtsStoreDep, jobs: JobsDep):
    samples = []
    for job_id in body.job_ids:
        try:
            job = jobs.get(job_id)
            if job.state != JobState.SUCCEEDED or "tts" not in job.parameters:
                raise HTTPException(422, "听评只能使用已成功的 TTS 作业")
            report = jobs.get_report(job_id).report
            output_id = report.summary["output_asset_id"]
            asset = next(a for a in report.assets if str(a.id) == output_id)
        except (JobNotFound, KeyError, StopIteration) as exc:
            raise HTTPException(422, f"作业 {job_id} 缺少合成结果") from exc
        spec = job.parameters["tts"]
        samples.append(
            {
                "id": str(uuid4()),
                "job_id": str(job_id),
                "uri": asset.uri,
                "reference_uri": job.input_uri,
                "text": spec["text"],
                "model": spec["model"],
                "emotion": spec["emotion"],
                "inference_profile": spec.get("inference_profile", "baseline"),
                "cloning": spec["model"] in MODEL_IDS[:2],
            }
        )
    evaluation = {
        "id": str(uuid4()),
        "name": body.name,
        "shuffle_seed": secrets.token_hex(16),
        "samples": samples,
    }
    store.save_evaluation(evaluation)
    return {"id": evaluation["id"], "name": body.name, "sample_count": len(samples)}


def get_evaluation(store, evaluation_id):
    try:
        return store.get_evaluation(evaluation_id)
    except KeyError as exc:
        raise HTTPException(404, "评测不存在") from exc


@router.get("/evaluations/{evaluation_id}")
def evaluation_samples(
    evaluation_id: UUID,
    store: TtsStoreDep,
    listener_id: str = Query(pattern=r"^[A-Za-z0-9_-]{1,64}$"),
):
    evaluation = get_evaluation(store, evaluation_id)
    samples = list(evaluation["samples"])
    seed = hashlib.sha256((evaluation["shuffle_seed"] + listener_id).encode()).hexdigest()
    random.Random(seed).shuffle(samples)
    return {
        "id": evaluation["id"],
        "name": evaluation["name"],
        "samples": [
            {
                "id": s["id"],
                "text": s["text"],
                "similarity_applicable": s["cloning"],
                "target_emotion": s["emotion"] if s["emotion"] != "neutral" else None,
            }
            for s in samples
        ],
    }


@router.get("/evaluations/{evaluation_id}/samples/{sample_id}/audio")
def evaluation_audio(
    evaluation_id: UUID,
    sample_id: UUID,
    store: TtsStoreDep,
    artifacts: ArtifactsDep,
    reference: bool = False,
):
    evaluation = get_evaluation(store, evaluation_id)
    sample = next((s for s in evaluation["samples"] if s["id"] == str(sample_id)), None)
    if not sample:
        raise HTTPException(404, "样本不存在")
    if reference and not sample["cloning"]:
        raise HTTPException(422, "固定音色基线不评估克隆相似度")
    return audio_response(sample["reference_uri"] if reference else sample["uri"], artifacts)


@router.post("/evaluations/{evaluation_id}/scores")
def save_score(evaluation_id: UUID, score: ListeningScore, store: TtsStoreDep):
    evaluation = get_evaluation(store, evaluation_id)
    sample = next((s for s in evaluation["samples"] if s["id"] == str(score.sample_id)), None)
    if not sample:
        raise HTTPException(422, "样本不属于该评测")
    if not sample["cloning"] and score.similarity is not None:
        raise HTTPException(422, "固定音色基线不评估克隆相似度")
    if sample["emotion"] == "neutral" and score.emotion_match is not None:
        raise HTTPException(422, "中性样本不评估目标情绪匹配度")
    store.save_score(evaluation_id, score)
    return {"saved": True}


@router.get("/evaluations/{evaluation_id}/progress")
def listening_progress(
    evaluation_id: UUID,
    store: TtsStoreDep,
    listener_id: str = Query(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$"),
):
    evaluation = get_evaluation(store, evaluation_id)
    sample_ids = {s["id"] for s in evaluation["samples"]}
    # Return only this listener's saved ratings, never model/variant identities.
    ratings = {
        s["sample_id"]: {
            k: s.get(k) for k in ("naturalness", "similarity", "emotion_match", "comment")
        }
        for s in store.list_scores(evaluation_id)
        if s["listener_id"] == listener_id and s["sample_id"] in sample_ids
    }
    return {"completed": len(ratings), "total": len(sample_ids), "ratings": ratings}


def summarize(evaluation, scores):
    listeners = {s["listener_id"] for s in scores}
    complete = sum(
        sum(s["listener_id"] == listener for s in scores) == len(evaluation["samples"])
        for listener in listeners
    )
    groups = []
    variants = sorted(
        {(s["model"], s.get("inference_profile", "baseline")) for s in evaluation["samples"]},
        key=lambda item: (MODEL_IDS.index(item[0]), item[1]),
    )
    for model, profile in variants:
        ids = {
            s["id"]
            for s in evaluation["samples"]
            if s["model"] == model and s.get("inference_profile", "baseline") == profile
        }
        rows = [s for s in scores if s["sample_id"] in ids]
        if not ids:
            continue
        means = {}
        for key in ("naturalness", "similarity", "emotion_match"):
            values = [s[key] for s in rows if s.get(key) is not None]
            means[key] = {"mean": statistics.mean(values) if values else None, "n": len(values)}
        groups.append(
            {
                "model": model,
                "inference_profile": profile,
                "ratings": len(rows),
                "scores": means,
            }
        )
    return {
        "listeners": len(listeners),
        "complete_listeners": complete,
        "status": (
            "not_performed"
            if not scores
            else "multi_listener"
            if complete >= 5
            else "pilot_not_formal_mos"
        ),
        "note": (
            "尚无真人评分，听评未执行；自动指标不能替代 MOS。"
            if not scores
            else "不足 5 位完整评分者时仅为预评测；自动指标不能替代 MOS。"
        ),
        "expected_ratings_per_listener": len(evaluation["samples"]),
        "groups": groups,
    }


@router.get("/evaluations/{evaluation_id}/report")
def evaluation_report(evaluation_id: UUID, store: TtsStoreDep):
    evaluation = get_evaluation(store, evaluation_id)
    return summarize(evaluation, store.list_scores(evaluation_id))


@router.get("/evaluations/{evaluation_id}/export")
def export_scores(evaluation_id: UUID, store: TtsStoreDep):
    evaluation = get_evaluation(store, evaluation_id)
    samples = {s["id"]: s for s in evaluation["samples"]}
    buffer = io.StringIO()
    fields = [
        "listener_id",
        "sample_id",
        "job_id",
        "model",
        "emotion",
        "inference_profile",
        "naturalness",
        "similarity",
        "emotion_match",
        "comment",
    ]
    writer = csv.DictWriter(buffer, fields)
    writer.writeheader()
    for score in store.list_scores(evaluation_id):
        sample = samples[score["sample_id"]]
        row = {**score, **{k: sample[k] for k in ("job_id", "model", "emotion")}}
        row["inference_profile"] = sample.get("inference_profile", "baseline")
        for key, value in row.items():
            if isinstance(value, str) and re.match(r"^[\s]*[=+@-]", value):
                row[key] = "'" + value
        writer.writerow(row)
    return Response(
        buffer.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="listening.csv"'},
    )
