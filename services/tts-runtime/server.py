"""One model per process. Runs in the upstream model's own Python environment.

No platform package imports: CosyVoice's Python 3.10 environment remains isolated
from the platform's Python 3.11+ dependencies. Bind to loopback and use SSH tunnels.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROMPTS = {
    "happy": "请用开心、轻快、自然的语气说这句话。",
    "sad": "请用克制、低沉、悲伤但不过度哭泣的语气说这句话。",
    "angry": "请用愤怒、坚定、有力度但不尖叫的语气说这句话。",
}
MODELS = ("cosyvoice2", "gpt-sovits-v2", "vits-csmsc", "transformer-csmsc")


def delivery_settings(spec):
    """Keep the original delivery reproducible; candidates require explicit opt-in."""
    profile = spec.get("inference_profile", "baseline")
    if profile not in {
        "baseline",
        "pilot_v2",
        "gpt_pause_v3",
        "cosy_happy_v4",
        "cosy_rhythm_v5",
        "gpt_rhythm_v6",
    }:
        raise ValueError("unknown inference profile")
    model = spec["model"]
    if profile == "gpt_rhythm_v6" and not re.fullmatch(
        r"[\u3400-\u9fff\s，,。.!?！？;；:：、…]+", spec.get("text", "")
    ):
        raise ValueError("节奏试用暂支持纯中文及标点，请移除数字和外语后重试")
    if profile != "baseline" and model not in MODELS[:2]:
        raise ValueError("experimental delivery requires a cloning model")
    if profile in {"gpt_pause_v3", "gpt_rhythm_v6"} and model != "gpt-sovits-v2":
        raise ValueError("pause trial requires GPT-SoVITS")
    emotion = spec.get("emotion", "neutral")
    instruction = PROMPTS.get(emotion) if model == "cosyvoice2" else None
    if profile in {"cosy_happy_v4", "cosy_rhythm_v5"}:
        if model != "cosyvoice2" or emotion != "happy":
            raise ValueError("short happy instruction requires CosyVoice happy mode")
        instruction = "用开心的语气说这句话。"
    speed = 1.0
    if profile == "pilot_v2":
        if model == "cosyvoice2" and emotion == "happy":
            instruction = "请用非常开心、兴奋的语气说这句话。"
        elif emotion == "neutral":
            speed = 0.92 if model == "cosyvoice2" else 0.9
    result = {
        "inference_profile": profile,
        "instruction": instruction + "<|endofprompt|>" if instruction else None,
        "speed": speed,
    }
    if profile == "cosy_rhythm_v5":
        result["postprocess_tempo"] = 0.85
    if model == "gpt-sovits-v2":
        result.update(text_split_method="cut0", fragment_interval=0.3)
        if profile in {"gpt_pause_v3", "gpt_rhythm_v6"}:
            # Earlier cut5 trials truncated tiny phrases. This conservative gate
            # is an experiment, not a guarantee of intelligibility: only split
            # Chinese clauses of >=8 characters, with no short trailing clause.
            text = spec["text"]
            punctuation = r"，,。.!?！？;：、…"
            clauses = [s.strip() for s in re.split(f"[{punctuation}]+", text) if s.strip()]
            eligible = bool(re.fullmatch(f"[\u3400-\u9fff{punctuation} ]+", text))
            eligible = (
                eligible
                and len(clauses) > 1
                and all(len(re.findall(r"[\u3400-\u9fff]", s)) >= 8 for s in clauses)
            )
            result["text_split_method"] = "cut5" if eligible else "cut0"
            result["segmentation_reason"] = (
                "eligible_long_chinese_clauses" if eligible else "retain_whole_utterance"
            )
    return result


def apply_gpt_rhythm(waveform, sample_rate, text):
    import numpy as np
    import soundfile as sf

    python = os.getenv("SHENGKE_RHYTHM_PYTHON")
    weights = os.getenv("SHENGKE_RHYTHM_WEIGHTS")
    if not python or not weights:
        raise RuntimeError("尚未配置节奏试用环境，请配置本地 Whisper 运行环境与权重")
    with tempfile.TemporaryDirectory(prefix="shengke-gpt-rhythm-") as directory:
        source, target, report = [Path(directory) / n for n in ("in.wav", "out.wav", "report.json")]
        sf.write(source, waveform, sample_rate, subtype="PCM_16")
        try:
            subprocess.run(
                [
                    python,
                    str(Path(__file__).with_name("gpt_rhythm.py")),
                    "--audio",
                    str(source),
                    "--text",
                    text,
                    "--weights",
                    weights,
                    "--output",
                    str(target),
                    "--report",
                    str(report),
                ],
                capture_output=True,
                check=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("节奏处理未完成，请检查本地 Whisper 和 FFmpeg 环境") from exc
        result, rate = sf.read(target, dtype="float32")
        if (
            rate != sample_rate
            or result.ndim != 1
            or not result.size
            or not np.isfinite(result).all()
        ):
            raise RuntimeError("invalid GPT rhythm output")
        return result, json.loads(report.read_text())


def stretch_rhythm(waveform, sample_rate, tempo):
    """Replay the listening trial's PCM16 -> atempo path; fail if unavailable."""
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="shengke-rhythm-") as directory:
        source, target = Path(directory) / "source.wav", Path(directory) / "result.wav"
        sf.write(source, waveform, sample_rate, subtype="PCM_16")
        try:
            version = (
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
                .stdout.decode()
                .splitlines()[0]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-i",
                    str(source),
                    "-af",
                    f"atempo={tempo}",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("舒缓节奏处理失败，请检查 FFmpeg；未返回未经处理的替代音频") from exc
        audio, rate = sf.read(target, dtype="float32")
        if rate != sample_rate or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise RuntimeError("invalid rhythm processing output")
        return audio, {
            "method": "ffmpeg_atempo",
            "tempo": tempo,
            "ffmpeg_version": version,
            "input_audio_duration_s": len(waveform) / sample_rate,
            "input_wav_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_wav_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "includes_postprocessing": True,
        }


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict
    reference_base64: str = Field(min_length=1, max_length=14 * 1024 * 1024)


def fingerprints(root, suffixes=None):
    suffixes = suffixes or {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".pkl"}
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix in suffixes and not path.name.startswith("._"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def validate_espnet_configuration(model_id, acoustic, vocoder=None):
    expected = {"vits-csmsc": "vits", "transformer-csmsc": "transformer"}[model_id]
    if acoustic.get("tts") != expected:
        raise ValueError(f"{model_id} requires the {expected} architecture")
    if acoustic.get("g2p") != "pypinyin_g2p_phone":
        raise ValueError("CSMSC baseline requires its original Mandarin phoneme frontend")
    if model_id == "transformer-csmsc":
        if not vocoder:
            raise ValueError("Transformer requires a pretrained matching vocoder")
        features = acoustic["feats_extract_conf"]
        for a, v in {
            "fs": "sampling_rate",
            "hop_length": "hop_size",
            "n_mels": "num_mels",
            "n_fft": "fft_size",
            "win_length": "win_length",
            "fmin": "fmin",
            "fmax": "fmax",
        }.items():
            if features.get(a) != vocoder.get(v):
                raise ValueError(f"Transformer/vocoder feature mismatch: {a} vs {v}")


def git_revision(repo):
    root = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if root.returncode or Path(root.stdout.strip()).resolve() != Path(repo).resolve():
        return "unverified"
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unverified"


class GptEarlyTerminationError(RuntimeError):
    """Reject invalid semantic spans before upstream's [-0:] includes the prompt."""


class Backend:
    def __init__(self):
        import torch

        self.torch = torch
        self.runtime_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.model_id = os.getenv("SHENGKE_MODEL", "cosyvoice2")
        if self.model_id not in MODELS:
            raise RuntimeError("unsupported SHENGKE_MODEL")
        self.device = os.getenv("SHENGKE_DEVICE", "cpu")
        if self.device not in {"cpu", "cuda"}:
            raise RuntimeError("SHENGKE_DEVICE must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.device == "cpu" and torch.cuda.is_available():
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for a CPU-only run")
        torch.set_num_threads(int(os.getenv("SHENGKE_CPU_THREADS", "4")))
        repo = Path(os.environ["SHENGKE_REPO_ROOT"]).resolve()
        model_dir = Path(os.environ["SHENGKE_MODEL_DIR"]).resolve()
        if not repo.is_dir() or not model_dir.is_dir():
            raise RuntimeError("model repository/weights directory is missing")
        self.revision = git_revision(repo)
        self.weights = fingerprints(model_dir)
        self.configurations = fingerprints(model_dir, {".yaml", ".yml", ".json", ".npz", ".h5"})
        if not self.weights:
            raise RuntimeError("No local model weights found; download the pinned model first")
        self.fp16 = os.getenv("SHENGKE_FP16", "0") == "1"
        if self.fp16 and self.device != "cuda":
            raise RuntimeError("FP16 requires CUDA")
        started = time.perf_counter()
        sys.path.insert(0, str(repo))
        if self.model_id == "cosyvoice2":
            sys.path.insert(0, str(repo / "third_party/Matcha-TTS"))
            from cosyvoice.cli.cosyvoice import CosyVoice2

            # We deliberately use text_frontend=False. Avoid constructing optional
            # normalizers, whose constructors otherwise download unpinned assets.
            with patch.dict(sys.modules, {"ttsfrd": None, "wetext": None}):
                self.model = CosyVoice2(
                    str(model_dir), load_jit=False, load_trt=False, load_vllm=False, fp16=self.fp16
                )
        elif self.model_id == "gpt-sovits-v2":
            import yaml

            config_path = Path(os.environ["SHENGKE_GPT_CONFIG"]).resolve()
            configuration = yaml.safe_load(config_path.read_text())
            configured = configuration.get("custom", {})
            for key in (
                "t2s_weights_path",
                "vits_weights_path",
                "bert_base_path",
                "cnhuhbert_base_path",
            ):
                candidate = Path(configured.get(key, ""))
                if not candidate.is_absolute() or not candidate.exists():
                    raise ValueError(f"GPT-SoVITS requires an existing absolute {key}; no fallback")
            os.chdir(repo)
            sys.path.insert(0, str(repo / "GPT_SoVITS"))
            from TTS_infer_pack.TTS import TTS, TTS_Config

            config = TTS_Config(str(config_path))
            config.device = torch.device(self.device)
            config.is_half = self.fp16
            config.version = "v2"
            self.model = TTS(config)
        else:
            from importlib.metadata import version

            # ESPnet imports its English frontend even for pypinyin models. Its
            # import-time NLTK downloads are unused by these Mandarin baselines.
            with patch("nltk.download", return_value=False):
                from espnet2.bin.tts_inference import Text2Speech

            self.revision = "espnet==" + version("espnet")
            # Released ESPnet configs contain model-relative normalization stats paths.
            # This process hosts exactly one model, so its own working directory is safe.
            os.chdir(model_dir)
            # Local files prevent a supposedly pinned run silently downloading newer weights.
            options = {
                "train_config": os.environ["SHENGKE_ESPNET_CONFIG"],
                "model_file": os.environ["SHENGKE_ESPNET_WEIGHTS"],
                "device": self.device,
            }
            if self.fp16:
                raise RuntimeError("ESPnet baselines use FP32 for reproducibility")
            if self.model_id == "transformer-csmsc":
                options["vocoder_config"] = os.environ["SHENGKE_VOCODER_CONFIG"]
                options["vocoder_file"] = os.environ["SHENGKE_VOCODER_WEIGHTS"]
            import yaml

            acoustic = yaml.safe_load(Path(options["train_config"]).read_text())
            vocoder = (
                yaml.safe_load(Path(options["vocoder_config"]).read_text())
                if "vocoder_config" in options
                else None
            )
            validate_espnet_configuration(self.model_id, acoustic, vocoder)
            self.model = Text2Speech(**options)
        self.load_seconds = time.perf_counter() - started

    def gpt_chunks(self, parameters):
        decoder = self.model.t2s_model.model
        original = decoder.infer_panel_naive_batched

        def checked(*args, **kwargs):
            predictions, lengths = original(*args, **kwargs)
            if not predictions or len(predictions) != len(lengths):
                raise GptEarlyTerminationError("invalid GPT semantic batch")
            for prediction, length in zip(predictions, lengths, strict=True):
                # Upstream idx==0 makes prediction[-idx:] select the entire
                # reference prompt. Reject before the vocoder ever sees it.
                if int(length) <= 0 or int(length) > len(prediction):
                    raise GptEarlyTerminationError(f"invalid GPT generated span: {int(length)}")
            return predictions, lengths

        # TTS.run binds infer_panel from this method on every invocation. Scope
        # the guard to each run: upstream may rebuild the decoder after errors.
        with patch.object(decoder, "infer_panel_naive_batched", checked):
            return list(self.model.run(parameters))

    def generate(self, spec, reference):
        import numpy as np

        delivery = delivery_settings(spec)
        self.generation_delivery = delivery
        if self.model_id == "cosyvoice2":
            if spec.get("emotion", "neutral") == "neutral":
                iterator = self.model.inference_zero_shot(
                    spec["text"],
                    spec["reference_text"],
                    str(reference),
                    stream=False,
                    text_frontend=False,
                    speed=delivery["speed"],
                )
            else:
                iterator = self.model.inference_instruct2(
                    spec["text"],
                    delivery["instruction"],
                    str(reference),
                    stream=False,
                    text_frontend=False,
                    speed=delivery["speed"],
                )
            chunks = [part["tts_speech"].detach().cpu().numpy().reshape(-1) for part in iterator]
            if not chunks:
                raise RuntimeError("empty model output")
            waveform = np.concatenate(chunks)
            if tempo := delivery.get("postprocess_tempo"):
                waveform, processing = stretch_rhythm(waveform, self.model.sample_rate, tempo)
                delivery["postprocessing"] = processing
            return self.model.sample_rate, waveform
        if self.model_id == "gpt-sovits-v2":
            parameters = {
                "text": spec["text"],
                "text_lang": "all_zh",
                "ref_audio_path": str(reference),
                "prompt_text": spec["reference_text"],
                "prompt_lang": "all_zh",
                # Baseline stays whole; the opt-in pause trial guards
                # against tiny chunks before enabling native splitting.
                "text_split_method": delivery["text_split_method"],
                "fragment_interval": delivery["fragment_interval"],
                "batch_size": 1,
                "split_bucket": False,
                "return_fragment": False,
                "seed": 42,
                "parallel_infer": False,
                "speed_factor": delivery["speed"],
            }
            try:
                chunks = self.gpt_chunks(parameters)
            except GptEarlyTerminationError:
                if parameters["text_split_method"] != "cut5":
                    raise
                # Discard all partial fragments, retry the same full text once.
                # TTS.run resets its seed; no voice/text/model substitution.
                parameters["text_split_method"] = "cut0"
                delivery.update(
                    requested_text_split_method="cut5",
                    text_split_method="cut0",
                    fallback_reason="early_semantic_termination",
                )
                chunks = self.gpt_chunks(parameters)
            if not chunks or len({item[0] for item in chunks}) != 1:
                raise RuntimeError("invalid GPT-SoVITS output")
            audio = np.concatenate([item[1] for item in chunks])
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            if spec.get("inference_profile") == "gpt_rhythm_v6":
                audio, processing = apply_gpt_rhythm(audio, chunks[0][0], spec["text"])
                delivery["postprocessing"] = processing
            return chunks[0][0], audio
        result = self.model(spec["text"])
        return self.model.fs, result["wav"].detach().cpu().numpy().reshape(-1)

    def synthesize(self, spec, raw):
        import numpy as np
        import psutil
        import soundfile as sf

        stop = threading.Event()
        process = psutil.Process()
        peak = [process.memory_info().rss]

        def sample_rss():
            while not stop.wait(0.02):
                peak[0] = max(peak[0], process.memory_info().rss)

        with tempfile.TemporaryDirectory(prefix="shengke-model-") as directory:
            reference = Path(directory) / "reference.wav"
            data, rate = sf.read(io.BytesIO(raw), always_2d=True)
            if not 3 <= len(data) / rate <= 10 or not np.isfinite(data).all():
                raise ValueError("reference must be 3–10 seconds of finite audio")
            if np.max(np.abs(data)) < 0.00001:
                raise ValueError("reference is silent")
            sf.write(reference, data.mean(axis=1), rate, subtype="PCM_16")
            cuda = self.device == "cuda"
            if cuda:
                self.torch.cuda.synchronize()
                self.torch.cuda.reset_peak_memory_stats()
            thread = threading.Thread(target=sample_rss, daemon=True)
            thread.start()
            started = time.perf_counter()
            try:
                random.seed(42)
                np.random.seed(42)
                self.torch.manual_seed(42)
                if cuda:
                    self.torch.cuda.manual_seed_all(42)
                with self.torch.inference_mode():
                    sample_rate, waveform = self.generate(spec, reference)
                if cuda:
                    self.torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            finally:
                stop.set()
                thread.join()
            if not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("model returned empty or non-finite audio")
            output = io.BytesIO()
            sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
            duration = len(waveform) / sample_rate
            metrics = {
                "model": self.model_id,
                "revision": self.revision,
                "runtime_sha256": self.runtime_sha256,
                "weights_sha256": self.weights,
                "configuration_sha256": self.configurations,
                "device": self.device,
                "device_name": self.torch.cuda.get_device_name() if cuda else platform.processor(),
                "precision": "fp16" if self.fp16 else "fp32",
                "torch": self.torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda_version": self.torch.version.cuda if cuda else None,
                "load_seconds": self.load_seconds,
                "inference_s": elapsed,
                "audio_duration_s": duration,
                "rtf": elapsed / duration,
                "peak_rss_bytes": peak[0],
                "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated()
                if cuda
                else None,
                "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved() if cuda else None,
                "sample_rate_hz": sample_rate,
                "includes_vocoder": True,
                "seed": 42,
                "delivery": self.generation_delivery,
                "text_split_method": self.generation_delivery.get("text_split_method"),
            }
            return {
                "audio_base64": base64.b64encode(output.getvalue()).decode(),
                "metrics": metrics,
            }


backend = None
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global backend
    backend = Backend()
    yield
    backend = None
    gc.collect()


app = FastAPI(title="Shengke isolated model runtime", lifespan=lifespan)


@app.get("/healthz")
def health():
    return {
        "model": os.getenv("SHENGKE_MODEL", "cosyvoice2"),
        "ready": backend is not None,
        "revision": backend.revision if backend else None,
        "device": backend.device if backend else None,
        "runtime_sha256": backend.runtime_sha256 if backend else None,
    }


@app.post("/synthesize")
def synthesize(request: Request, authorization: str | None = Header(default=None)):
    token = os.getenv("SHENGKE_MODEL_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(401, "invalid runtime token")
    if backend is None:
        raise HTTPException(503, "model not ready")
    spec = request.spec
    if spec.get("model") != backend.model_id:
        raise HTTPException(422, "requested model does not match this runtime")
    if not isinstance(spec.get("text"), str) or not 1 <= len(spec["text"].strip()) <= 200:
        raise HTTPException(422, "text must contain 1–200 characters")
    if spec.get("language", "zh") != "zh":
        raise HTTPException(422, "MVP supports Mandarin only")
    emotion = spec.get("emotion", "neutral")
    if emotion not in {"neutral", *PROMPTS} or (
        emotion != "neutral" and backend.model_id != "cosyvoice2"
    ):
        raise HTTPException(422, "emotion unsupported by this model")
    if backend.model_id in MODELS[:2] and not str(spec.get("reference_text", "")).strip():
        raise HTTPException(422, "reference transcript required")
    try:
        delivery_settings(spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        raw = base64.b64decode(request.reference_base64, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("reference exceeds 10 MiB")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not lock.acquire(blocking=False):
        raise HTTPException(429, "model busy; one request at a time")
    try:
        return backend.synthesize(spec, raw)
    except GptEarlyTerminationError as exc:
        raise HTTPException(503, "模型提前结束，未发布音频；请更换参考录音后重试") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        import logging

        logging.exception("Model inference failed")
        message = (
            "CUDA 显存不足"
            if "out of memory" in str(exc).lower()
            else "模型推理失败，请查看服务日志"
        )
        raise HTTPException(503, message) from exc
    finally:
        lock.release()
