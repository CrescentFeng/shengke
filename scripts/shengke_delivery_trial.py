"""Generate opt-in delivery candidates from completed jobs, without altering the baseline."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--jobs", nargs="+", required=True, help="Completed baseline job IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["pilot_v2", "gpt_pause_v3", "cosy_happy_v4"], default="pilot_v2"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty; preserve earlier experiment evidence")
    args.output.mkdir(parents=True, exist_ok=True)

    def api(path, body=None, binary=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(args.api + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
        return raw if binary else json.loads(raw)

    # Resolve and validate every source before submitting work.
    sources = []
    for job_id in args.jobs:
        job = api("/v1/audio/jobs/" + job_id)
        spec = job["parameters"]["tts"]
        if job["state"] != "succeeded" or spec.get("inference_profile", "baseline") != "baseline":
            parser.error("sources must be completed baseline jobs")
        if spec["model"] not in {"cosyvoice2", "gpt-sovits-v2"}:
            parser.error("delivery trial requires cloning jobs")
        if spec["emotion"] not in {"neutral", "happy"}:
            parser.error("pilot_v2 changes only neutral or happy delivery")
        if args.profile == "gpt_pause_v3" and spec["model"] != "gpt-sovits-v2":
            parser.error("pause trial requires GPT-SoVITS sources")
        if args.profile == "cosy_happy_v4" and (
            spec["model"] != "cosyvoice2" or spec["emotion"] != "happy"
        ):
            parser.error("short happy instruction requires CosyVoice happy sources")
        sources.append(job)
    pairs = []
    for source in sources:
        original = source["parameters"]["tts"]
        candidate = {**original, "inference_profile": args.profile}
        job = api(
            "/v1/audio/jobs",
            {
                "input_uri": source["input_uri"],
                "operations": ["synthesize"],
                "idempotency_key": str(uuid4()),
                "parameters": {"tts": candidate},
            },
        )
        job_id = job["id"]
        pair = {
            "baseline_job_id": source["id"],
            "candidate_job_id": job_id,
            "baseline_spec": original,
            "candidate_spec": candidate,
            "reference_uri": source["input_uri"],
            "state": job["state"],
        }
        pairs.append(pair)

        def persist():
            (args.output / "pairs.json").write_text(
                json.dumps(pairs, ensure_ascii=False, indent=2) + "\n"
            )

        persist()
        print("Submitted", job_id, candidate["model"], candidate["emotion"], flush=True)
        deadline = time.monotonic() + 900
        while job["state"] not in {"succeeded", "failed", "cancelled"}:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Job {job_id} may still be running; inspect it before retrying")
            time.sleep(1)
            job = api("/v1/audio/jobs/" + job_id)
        pair["state"] = job["state"]
        persist()
        if job["state"] != "succeeded":
            raise RuntimeError(f"Job {job_id}: {job.get('error_message')}")
        for label, jid in [("baseline", source["id"]), ("candidate", job_id)]:
            report = api(f"/v1/audio/jobs/{jid}/report")
            raw = api(f"/v1/tts/jobs/{jid}/audio", binary=True)
            (args.output / f"{jid}.wav").write_bytes(raw)
            (args.output / f"{jid}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            pair[label + "_audio_sha256"] = hashlib.sha256(raw).hexdigest()
            pair[label + "_metrics"] = report["report"]["summary"]["tts"]
        assert pair["candidate_metrics"]["delivery"]["inference_profile"] == args.profile
        persist()
        print(
            "Completed",
            job_id,
            "duration",
            pair["candidate_metrics"]["audio_duration_s"],
            flush=True,
        )
    print(
        "Candidates ready. Listening quality remains unverified; no scores were written.",
        flush=True,
    )


if __name__ == "__main__":
    main()
