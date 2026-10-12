"""Real end-to-end check; requires a running API, worker and model runtime."""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="cosyvoice2")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty to preserve existing evidence")
    args.output.mkdir(parents=True, exist_ok=True)

    def api(path, body=None, binary=False):
        req = urllib.request.Request(
            args.api + path,
            json.dumps(body).encode() if body is not None else None,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
        return raw if binary else json.loads(raw)

    voice = api(
        "/v1/tts/voices",
        {
            "name": "公开参考音色 · 冒烟测试",
            "filename": args.reference.name,
            "audio_base64": base64.b64encode(args.reference.read_bytes()).decode(),
            "reference_text": args.transcript,
            "source": args.source,
            "license": args.license,
        },
    )
    payload = {
        "input_uri": voice["asset"]["uri"],
        "operations": ["synthesize"],
        "idempotency_key": str(uuid4()),
        "parameters": {
            "tts": {
                "model": args.model,
                "text": "你好，欢迎来到声刻。让每一句话，都有自己的温度。",
                "reference_text": args.transcript,
                "voice_id": voice["id"],
                "emotion": args.emotion,
            }
        },
    }
    job = api("/v1/audio/jobs", payload)
    repeat = api("/v1/audio/jobs", payload)
    assert repeat["id"] == job["id"], "idempotency check failed"
    started = time.monotonic()
    while job["state"] not in {"succeeded", "failed", "cancelled"}:
        if time.monotonic() - started > 900:
            api(f"/v1/audio/jobs/{job['id']}/cancel", {})
            raise TimeoutError("synthesis timed out")
        time.sleep(2)
        job = api(f"/v1/audio/jobs/{job['id']}")
    (args.output / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
    (args.output / "voice.json").write_text(json.dumps(voice, ensure_ascii=False, indent=2))
    if job["state"] != "succeeded":
        raise RuntimeError(job["error_message"])
    report = api(f"/v1/audio/jobs/{job['id']}/report")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / "speech.wav").write_bytes(api(f"/v1/tts/jobs/{job['id']}/audio", binary=True))
    evaluation = api(
        "/v1/tts/evaluations", {"name": "首轮试听（尚未评分）", "job_ids": [job["id"]]}
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "job_id": job["id"],
                "voice_id": voice["id"],
                "evaluation_id": evaluation["id"],
                "metrics": report["report"]["summary"]["tts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
