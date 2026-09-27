from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


@dataclass(frozen=True)
class PlatformSettings:
    database_url: str | None

    @classmethod
    def from_env(cls) -> PlatformSettings:
        return cls(database_url=_optional_env("AUDIO_PLATFORM_DATABASE_URL"))
