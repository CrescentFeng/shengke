"""Offline ASR diagnostic for TTS outputs; never a replacement for listening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    """Ignore case, punctuation and spacing; preserve lexical differences."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def edit_distance(expected: str, observed: str) -> int:
    previous = list(range(len(observed) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def compare(expected: str, observed: str, threshold: float) -> dict:
    target, transcript = normalize(expected), normalize(observed)
    if not target:
        raise ValueError("expected text must contain letters or numbers")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("warning threshold must be in (0, 1]")
    distance = edit_distance(target, transcript)
    ratio = distance / len(target)
    return {
        "normalized_expected": target,
        "normalized_transcript": transcript,
        "edit_distance": distance,
        "raw_character_error_ratio": ratio,
        "warning_threshold": threshold,
        "status": "review_required"
        if not transcript or ratio >= threshold
        else "no_gross_mismatch_detected",
        "human_acceptance": "not_assessed",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--expected-text", required=True, help="Compared only AFTER ASR; never used as a prompt"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Existing local OpenAI Whisper .pt; no downloads",
    )
    parser.add_argument("--weights-sha256", required=True, help="Trusted expected model digest")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON report; existing evidence is never overwritten",
    )
    parser.add_argument("--warning-threshold", type=float, default=0.6)
    args = parser.parse_args()
    compare(args.expected_text, "", args.warning_threshold)
    if args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    for path in [args.weights, *args.audio]:
        if not path.is_file():
            parser.error(f"missing local file: {path}")
    weights_digest = sha256(args.weights)
    if weights_digest != args.weights_sha256.lower():
        parser.error("weights SHA-256 mismatch")

    import torch
    import whisper

    torch.set_num_threads(4)
    model = whisper.load_model(str(args.weights.resolve()), device="cpu")
    report = {
        "purpose": "independent_local_asr_diagnostic",
        "model": "OpenAI Whisper",
        "model_version": whisper.__version__,
        "weights_sha256": weights_digest,
        "device": "cpu",
        "asr_parameters": {
            "language": "zh",
            "temperature": 0,
            "beam_size": 5,
            "fp16": False,
            "condition_on_previous_text": False,
            "initial_prompt": None,
        },
        "expected_text": args.expected_text,
        "limitations": [
            "ASR and synthesis errors are not distinguished; word accuracy is not certified.",
            "Traditional/simplified characters, homophones and number spellings are not collapsed.",
            "The warning threshold is a diagnostic heuristic, not a calibrated quality gate.",
            "Naturalness, similarity and emotion are not assessed; listening remains required.",
        ],
        "files": [],
    }
    for path in args.audio:
        # Snapshot bytes before recognition and verify unchanged afterwards.
        digest = sha256(path)
        entry = {"file": str(path.resolve()), "sha256": digest}
        try:
            result = model.transcribe(str(path.resolve()), **report["asr_parameters"])
            if sha256(path) != digest:
                raise RuntimeError("audio changed during transcription")
            entry.update(
                transcript=result["text"],
                segments=[{k: s[k] for k in ("start", "end", "text")} for s in result["segments"]],
                comparison=compare(args.expected_text, result["text"], args.warning_threshold),
            )
        except Exception as exc:
            entry.update(status="diagnostic_failed", error=f"{type(exc).__name__}: {exc}")
        report["files"].append(entry)
        print(path.name, entry.get("comparison", {}).get("status", entry.get("status")), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return 2 if any(item.get("status") == "diagnostic_failed" for item in report["files"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
