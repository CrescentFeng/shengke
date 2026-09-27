from __future__ import annotations

from functools import lru_cache

from .settings import PlatformSettings
from .store import InMemoryJobStore, JobStore


@lru_cache(maxsize=1)
def get_store() -> JobStore:
    settings = PlatformSettings.from_env()
    if settings.database_url:
        from .postgres_store import PostgresJobStore

        return PostgresJobStore(settings.database_url)
    return InMemoryJobStore()
