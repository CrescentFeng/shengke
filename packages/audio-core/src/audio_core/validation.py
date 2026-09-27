from __future__ import annotations

from pathlib import Path

from .errors import InvalidAudioInput


def require_regular_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise InvalidAudioInput(f"input does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise InvalidAudioInput(f"input is not a regular file: {resolved}")
    return resolved


def require_distinct_output(input_path: Path, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve(strict=False)
    if output == input_path:
        raise InvalidAudioInput("output must not overwrite the original audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
