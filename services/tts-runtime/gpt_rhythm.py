"""Conservative Mandarin punctuation timing for the opt-in GPT rhythm profile.

ASR is unprompted. Edit alignment and timestamps are heuristic, not proof of content accuracy.
Runs in a separate, explicitly configured Whisper environment; never downloads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

WEIGHTS_SHA256 = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
MARKS = "，,。.!?！？;；:：、…"


def chinese(text):
    return "".join(c for c in text if "\u3400" <= c <= "\u9fff")


def boundaries(text):
    if not re.fullmatch(r"[\u3400-\u9fff\s，,。.!?！？;；:：、…]+", text):
        raise ValueError("节奏试用暂支持纯中文及标点，请移除数字和外语后重试")
    count, result = 0, []
    total = len(chinese(text))
    for c in text:
        if chinese(c):
            count += 1
        elif c in MARKS and 0 < count < total and count not in result:
            result.append(count)
    if len(result) > 20:
        raise ValueError("节奏试用最多支持二十处句内标点")
    return result


def align(expected, observed):
    """Levenshtein mapping; substitutions stay visible and consume error budget."""
    n, m = len(expected), len(observed)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]),
            )
    mapping, i, j = {}, n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]):
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapping, d[n][m] / max(n, 1)


def plan(text, words, duration):
    points, expected = boundaries(text), chinese(text)
    chars, timing = [], []
    previous_start = -1.0
    for index, word in enumerate(words):
        start, end, probability = (float(word[k]) for k in ("start", "end", "probability"))
        if (
            not all(math.isfinite(v) for v in (start, end, probability))
            or not 0 <= start <= end <= duration + 0.05
            or start < previous_start
        ):
            raise ValueError("invalid ASR word timing")
        previous_start = start
        for c in chinese(word["word"]):
            chars.append(c)
            timing.append({"token": index, "start": start, "end": end, "probability": probability})
    observed = "".join(chars)
    mapping, error = align(expected, observed)
    decisions = []
    for boundary in points:
        item = {"character_offset": boundary, "action": "skip"}
        decisions.append(item)
        if not observed or error > 0.25:
            item["reason"] = "transcript_mismatch"
            continue
        context = list(range(max(0, boundary - 2), min(len(expected), boundary + 2)))
        indices = [mapping.get(i) for i in context]
        if any(i is None for i in indices) or indices != list(
            range(indices[0], indices[0] + len(indices))
        ):
            item["reason"] = "ambiguous_local_alignment"
            continue
        if sum(expected[i] != observed[mapping[i]] for i in context) > 1:
            item["reason"] = "uncertain_boundary_text"
            continue
        left, right = timing[mapping[boundary - 1]], timing[mapping[boundary]]
        if (
            left["token"] == right["token"]
            or min(left["probability"], right["probability"]) < 0.5
            or left["end"] <= left["start"]
            or right["end"] <= right["start"]
            or left["end"] > right["start"] + 0.08
        ):
            item["reason"] = "uncertain_boundary_timing"
            continue
        item.update(
            action="locate",
            reason="estimated_word_boundary",
            estimate_s=(left["end"] + right["start"]) / 2,
            left_start_s=left["start"],
            right_end_s=right["end"],
            recognized_gap_s=max(0, right["start"] - left["end"]),
        )
    return {"raw_character_error_ratio": error, "transcript": observed, "boundaries": decisions}


def insert_pauses(audio, rate, decisions):
    import numpy as np

    pieces, cursor = [], 0
    for item in decisions:
        if item["action"] != "locate":
            continue
        estimate = round(item["estimate_s"] * rate)
        # Keep existing acoustic silence; never trim or shorten source speech.
        a, b = max(0, estimate - round(0.18 * rate)), min(len(audio), estimate + round(0.18 * rate))
        step = max(1, round(0.005 * rate))
        quiet = [
            np.sqrt(np.mean(audio[k : min(k + step, b)] ** 2)) < 0.0032 for k in range(a, b, step)
        ]
        longest, run = 0, 0
        for value in quiet:
            run = run + 1 if value else 0
            longest = max(longest, run)
        if longest * 0.005 >= 0.12:
            item.update(action="keep", reason="existing_acoustic_pause")
            continue
        lo = max(
            cursor + round(0.005 * rate),
            estimate - round(0.025 * rate),
            round((item["left_start_s"] + 0.02) * rate),
        )
        hi = min(
            len(audio) - round(0.005 * rate),
            estimate + round(0.025 * rate),
            round((item["right_end_s"] - 0.02) * rate),
        )
        if hi <= lo:
            item.update(action="skip", reason="invalid_splice_window")
            continue
        half = max(1, round(0.003 * rate))
        point = min(range(lo, hi), key=lambda k: float(np.mean(audio[k - half : k + half] ** 2)))
        head = audio[cursor:point].copy()
        fade = min(round(0.002 * rate), len(head))
        head[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        # Fade the next edge without removing frames.
        audio = audio.copy()
        audio[point : point + fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        silence = np.zeros(round(0.198 * rate), dtype=np.float32)
        pieces.extend([head, silence])
        cursor = point
        item.update(action="insert", splice_source_s=point / rate, inserted_s=len(silence) / rate)
    pieces.append(audio[cursor:])
    return np.concatenate(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("refusing to overwrite evidence")
    points = boundaries(args.text)
    import numpy as np
    import soundfile as sf
    from server import stretch_rhythm

    audio, rate = sf.read(args.audio, dtype="float32")
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("invalid source waveform")
    result = {"boundaries": [], "asr_status": "not_needed_no_internal_punctuation"}
    if points:
        if hashlib.sha256(args.weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Whisper weights SHA-256 mismatch")
        import torch
        import whisper

        torch.set_num_threads(4)
        model = whisper.load_model(str(args.weights.resolve()), device="cpu")
        decoded = model.transcribe(
            str(args.audio),
            language="zh",
            temperature=0,
            beam_size=5,
            fp16=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
        )
        words = [w for s in decoded["segments"] for w in s["words"]]
        result = plan(args.text, words, len(audio) / rate)
        result.update(asr_status="completed_unprompted", asr_weights_sha256=WEIGHTS_SHA256)
    processed = insert_pauses(audio, rate, result["boundaries"])
    processed, stretching = stretch_rhythm(processed, rate, 0.9)
    sf.write(args.output, processed, rate, subtype="PCM_16")
    result.update(
        tempo=0.9,
        method="estimated_punctuation_and_atempo",
        input_wav_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        output_wav_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stretching=stretching,
        human_acceptance="not_assessed",
    )
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
