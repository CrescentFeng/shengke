from __future__ import annotations

import hashlib
from pathlib import Path

from .validation import require_regular_file


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = require_regular_file(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
