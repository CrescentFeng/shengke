from __future__ import annotations

import math
from pathlib import Path

from audio_core.contracts import SpeakerTurn


def load_rttm(path: str | Path, *, recording_id: str) -> list[SpeakerTurn]:
    rttm_path = Path(path).expanduser().resolve()
    if not rttm_path.is_file():
        raise ValueError(f"RTTM is not a regular file: {rttm_path}")
    turns: list[SpeakerTurn] = []
    for line_number, raw_line in enumerate(
        rttm_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"invalid RTTM row at {rttm_path}:{line_number}")
        if fields[1] != recording_id:
            continue
        try:
            start_seconds = float(fields[3])
            duration_seconds = float(fields[4])
        except ValueError as exc:
            raise ValueError(
                f"invalid RTTM timing at {rttm_path}:{line_number}"
            ) from exc
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(duration_seconds)
            or start_seconds < 0
            or duration_seconds <= 0
        ):
            raise ValueError(f"invalid RTTM timing at {rttm_path}:{line_number}")
        start_ms = round(start_seconds * 1000)
        end_ms = round((start_seconds + duration_seconds) * 1000)
        if end_ms <= start_ms:
            raise ValueError(f"empty RTTM turn at {rttm_path}:{line_number}")
        turns.append(
            SpeakerTurn(start_ms=start_ms, end_ms=end_ms, speaker=fields[7])
        )
    if not turns:
        raise ValueError(f"RTTM contains no SPEAKER rows for {recording_id}: {rttm_path}")
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))
