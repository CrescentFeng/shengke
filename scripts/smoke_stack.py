from __future__ import annotations

import argparse
import json
import math
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from uuid import uuid4


def write_fixture(path: Path, *, seconds: float = 0.4, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        for index in range(frame_count):
            sample = int(0.25 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            stream.writeframesraw(struct.pack("<h", sample))


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    idempotency_key: str | None = None,
) -> dict | list:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="verify the persistent audio ingest stack")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--input-file",
        type=Path,
        help="use an existing audio file under data/input instead of generating a fixture",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="also request ASR; requires a configured worker ASR backend",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="also request speaker diarization; requires a configured worker backend",
    )
    speaker_group = parser.add_mutually_exclusive_group()
    speaker_group.add_argument("--num-speakers", type=int)
    speaker_group.add_argument(
        "--speaker-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--no-exclusive",
        action="store_true",
        help="align segments to the regular instead of exclusive speaker timeline",
    )
    parser.add_argument("--language", help="optional ASR language code, for example zh")
    parser.add_argument(
        "--hotword",
        action="append",
        default=[],
        help="ASR hotword; repeat this option to provide multiple terms",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_root = (project_root / "data" / "input").resolve()
    if args.input_file is None:
        fixture = input_root / "smoke.wav"
        write_fixture(fixture)
    else:
        fixture = args.input_file
        if not fixture.is_absolute():
            fixture = project_root / fixture
        fixture = fixture.resolve()
        if not fixture.is_file():
            parser.error(f"input file does not exist: {fixture}")
    try:
        container_relative_path = fixture.relative_to(input_root)
    except ValueError:
        parser.error("input file must be located under data/input")

    try:
        idempotency_key = f"smoke-{uuid4()}"
        operations = ["probe", "vad"]
        if args.diarize:
            operations.append("diarize")
        if args.transcribe:
            operations.append("transcribe")
        operations.extend(["loudness", "normalize"])
        parameters: dict[str, object] = {"sample_rate_hz": 16000, "channels": 1}
        if args.transcribe and (args.language or args.hotword):
            parameters["asr"] = {
                "language": args.language,
                "hotwords": args.hotword,
            }
        if args.diarize:
            diarization_parameters: dict[str, object] = {
                "use_exclusive": not args.no_exclusive
            }
            if args.num_speakers is not None:
                diarization_parameters["num_speakers"] = args.num_speakers
            elif args.speaker_range is not None:
                diarization_parameters["min_speakers"] = args.speaker_range[0]
                diarization_parameters["max_speakers"] = args.speaker_range[1]
            parameters["diarization"] = diarization_parameters
        job = request_json(
            f"{args.api}/v1/audio/jobs",
            method="POST",
            idempotency_key=idempotency_key,
            payload={
                "input_uri": f"file:///data/input/{container_relative_path.as_posix()}",
                "operations": operations,
                "idempotency_key": idempotency_key,
                "parameters": parameters,
            },
        )
        job_id = job["id"]
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            job = request_json(f"{args.api}/v1/audio/jobs/{job_id}")
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"job {job_id} did not finish within {args.timeout:g}s")

        if job["state"] != "succeeded":
            raise RuntimeError(f"job {job_id} ended in {job['state']}: {job.get('error_message')}")
        assets = request_json(f"{args.api}/v1/audio/jobs/{job_id}/assets")
        segments = request_json(f"{args.api}/v1/audio/jobs/{job_id}/segments")
        report = request_json(f"{args.api}/v1/audio/jobs/{job_id}/report")
        transcript = (
            request_json(f"{args.api}/v1/audio/jobs/{job_id}/transcript")
            if args.transcribe
            else None
        )
        diarization = (
            request_json(f"{args.api}/v1/audio/jobs/{job_id}/diarization")
            if args.diarize
            else None
        )
        if diarization and args.num_speakers is not None:
            actual_speakers = diarization["diarization"]["speaker_count"]
            if actual_speakers != args.num_speakers:
                raise RuntimeError(
                    f"expected {args.num_speakers} speakers, got {actual_speakers}"
                )
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "state": job["state"],
                    "asset_count": len(assets),
                    "segment_count": len(segments),
                    "finding_count": len(report["report"]["findings"]),
                    "report_uri": report["report_uri"],
                    "transcript_uri": transcript["transcript_uri"] if transcript else None,
                    "transcript_text": transcript["transcript"]["text"] if transcript else None,
                    "transcript_language": (
                        transcript["transcript"]["language"] if transcript else None
                    ),
                    "transcript_span_count": (
                        len(transcript["transcript"]["spans"]) if transcript else 0
                    ),
                    "transcript_speakers": (
                        sorted(
                            {
                                span["speaker"]
                                for span in transcript["transcript"]["spans"]
                                if span["speaker"] is not None
                            }
                        )
                        if transcript
                        else []
                    ),
                    "segment_speakers": sorted(
                        {segment["speaker"] for segment in segments if segment["speaker"]}
                    ),
                    "diarization_uri": (
                        diarization["diarization_uri"] if diarization else None
                    ),
                    "speaker_count": (
                        diarization["diarization"]["speaker_count"]
                        if diarization
                        else 0
                    ),
                    "turn_count": (
                        len(diarization["diarization"]["turns"])
                        if diarization
                        else 0
                    ),
                    "regular_turn_count": (
                        len(diarization["diarization"]["regular_turns"])
                        if diarization
                        else 0
                    ),
                    "exclusive_turn_count": (
                        len(diarization["diarization"]["exclusive_turns"])
                        if diarization
                        else 0
                    ),
                    "exclusive": (
                        diarization["diarization"]["exclusive"]
                        if diarization
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
