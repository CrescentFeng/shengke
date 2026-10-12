from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from audio_core.tts import MODEL_IDS


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows):
    reports = []
    for model in MODEL_IDS:
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue
        ok = [r for r in subset if r["status"] == "succeeded"]
        metrics = {}
        for key in (
            "inference_s",
            "rtf",
            "request_elapsed_s",
            "peak_rss_bytes",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ):
            values = [r["metrics"][key] for r in ok if r["metrics"].get(key) is not None]
            metrics[key] = (
                {
                    "mean": statistics.mean(values),
                    "p50": percentile(values, 0.5),
                    "p95": percentile(values, 0.95),
                    "max": max(values),
                }
                if values
                else None
            )
        reports.append(
            {
                "model": model,
                "requests": len(subset),
                "succeeded": len(ok),
                "success_rate": len(ok) / len(subset),
                "metrics": metrics,
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty to preserve existing evidence")
    args.output.mkdir(parents=True, exist_ok=True)

    def api(path, body=None, binary=False):
        request = urllib.request.Request(
            args.api.rstrip("/") + path,
            json.dumps(body).encode() if body is not None else None,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        return raw if binary else json.loads(raw)

    voices = api("/v1/tts/voices")
    voice = next((v for v in voices if v["id"] == args.voice_id), None)
    if not voice:
        parser.error("voice-id was not found")
    items = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not items:
        parser.error("manifest is empty")
    manifest_ids = [x["id"] for x in items]
    if len(set(manifest_ids)) != len(manifest_ids):
        parser.error("duplicate sample IDs")
    rows, warmups = [], []
    for model in args.models:
        # Warm up each model once, exclude it from reported distributions.
        cases = [(items[0], 0)] + [(item, n) for item in items for n in range(1, args.repeats + 1)]
        for item, repeat in cases:
            row = {
                "model": model,
                "sample_id": item["id"],
                "repeat": repeat,
                "text": item["text"],
                "status": "failed",
            }
            job_id = None
            try:
                job = api(
                    "/v1/audio/jobs",
                    {
                        "input_uri": voice["asset"]["uri"],
                        "operations": ["synthesize"],
                        "idempotency_key": str(uuid4()),
                        "parameters": {
                            "tts": {
                                "model": model,
                                "text": item["text"],
                                "reference_text": voice["reference_text"],
                                "voice_id": voice["id"],
                                "emotion": item.get("emotion", "neutral"),
                                "language": "zh",
                            }
                        },
                    },
                )
                job_id = job["id"]
                row["job_id"] = job_id
                started = time.monotonic()
                while job["state"] not in {"failed", "succeeded", "cancelled"}:
                    if time.monotonic() - started > args.timeout:
                        api(f"/v1/audio/jobs/{job_id}/cancel", {})
                        raise TimeoutError("job timeout; cancellation requested")
                    time.sleep(1)
                    job = api(f"/v1/audio/jobs/{job_id}")
                row["status"] = job["state"]
                if job["state"] == "succeeded":
                    report = api(f"/v1/audio/jobs/{job_id}/report")["report"]
                    row["metrics"] = report["summary"]["tts"]
                    row["reference_sha256"] = voice["asset"]["sha256"]
                    audio = api(f"/v1/tts/jobs/{job_id}/audio", binary=True)
                    path = args.output / f"{job_id}.wav"
                    path.write_bytes(audio)
                    row["audio"] = path.name
                    row["quality"] = report["summary"]["quality"]
                else:
                    row["error"] = job.get("error_message")
            except Exception as exc:
                # A completed model job with missing downloaded evidence is a failed
                # benchmark case; never include it in performance distributions.
                row["status"] = "failed"
                row["error"] = str(exc)
            target = warmups if repeat == 0 else rows
            target.append(row)
            filename = "warmup.jsonl" if repeat == 0 else "runs.jsonl"
            with (args.output / filename).open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{model} {item['id']} repeat={repeat}: {row['status']}", flush=True)
    result = {
        "models": summarize(rows),
        "warmup_excluded": True,
        "listener_evaluation": "not_performed",
        "manifest": items,
        "note": "RTF 包含声码器；请求耗时包含传输。CPU RSS 不代表 CUDA 显存。",
    }
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = [
        "# 声刻 TTS 基准",
        "",
        result["note"],
        "",
        "| 模型 | 成功/总数 | 平均 RTF |",
        "|---|---:|---:|",
    ]
    for model in result["models"]:
        rtf = model["metrics"]["rtf"]
        lines.append(
            f"| {model['model']} | {model['succeeded']}/{model['requests']} | "
            f"{rtf['mean'] if rtf else '未测得'} |"
        )
    lines.extend(["", "人工 MOS：未执行。音频质量和跨模型音色差异需人工评估。"])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    (args.output / "listening-job-ids.txt").write_text(
        "\n".join(r["job_id"] for r in rows if r["status"] == "succeeded" and r["repeat"] == 1)
        + "\n"
    )
    if any(r["status"] != "succeeded" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
