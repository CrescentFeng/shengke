from __future__ import annotations

import json
from functools import lru_cache

from fastapi import HTTPException

from .settings import PlatformSettings


class PostgresTtsStore:
    def __init__(self, dsn):
        from .postgres_store import PostgresJobStore

        self.db = PostgresJobStore(dsn)

    def save_voice(self, profile):
        with self.db._connection() as conn:
            conn.execute(
                "INSERT INTO tts_voices(id,payload) VALUES (%s,%s::jsonb)",
                (profile.id, profile.model_dump_json()),
            )

    def list_voices(self):
        with self.db._connection() as conn:
            return [
                row["payload"]
                for row in conn.execute(
                    "SELECT payload FROM tts_voices ORDER BY created_at DESC"
                ).fetchall()
            ]

    def get_voice(self, voice_id):
        with self.db._connection() as conn:
            row = conn.execute("SELECT payload FROM tts_voices WHERE id=%s", (voice_id,)).fetchone()
            if not row:
                raise KeyError(str(voice_id))
            return row["payload"]

    def save_evaluation(self, evaluation):
        with self.db._connection() as conn:
            conn.execute(
                "INSERT INTO tts_evaluations(id,payload) VALUES (%s,%s::jsonb)",
                (evaluation["id"], json.dumps(evaluation)),
            )

    def get_evaluation(self, evaluation_id):
        with self.db._connection() as conn:
            row = conn.execute(
                "SELECT payload FROM tts_evaluations WHERE id=%s", (evaluation_id,)
            ).fetchone()
            if not row:
                raise KeyError(str(evaluation_id))
            return row["payload"]

    def save_score(self, evaluation_id, score):
        with self.db._connection() as conn:
            conn.execute(
                """INSERT INTO tts_scores(evaluation_id,listener_id,sample_id,payload)
                VALUES (%s,%s,%s,%s::jsonb)
                ON CONFLICT(evaluation_id,listener_id,sample_id)
                DO UPDATE SET payload=EXCLUDED.payload,updated_at=now()""",
                (evaluation_id, score.listener_id, score.sample_id, score.model_dump_json()),
            )

    def list_scores(self, evaluation_id):
        with self.db._connection() as conn:
            return [
                row["payload"]
                for row in conn.execute(
                    "SELECT payload FROM tts_scores WHERE evaluation_id=%s "
                    "ORDER BY listener_id,sample_id",
                    (evaluation_id,),
                ).fetchall()
            ]


@lru_cache(maxsize=1)
def get_tts_store():
    dsn = PlatformSettings.from_env().database_url
    if not dsn:
        raise HTTPException(503, "声刻需要 PostgreSQL 持久化存储；请启动本地基础服务")
    return PostgresTtsStore(dsn)


@lru_cache(maxsize=1)
def get_tts_artifacts():
    from ingest_worker.s3_store import S3ArtifactStore
    from ingest_worker.settings import WorkerSettings

    try:
        return S3ArtifactStore(WorkerSettings.from_env())
    except RuntimeError as exc:
        raise HTTPException(503, "声刻的制品存储尚未配置") from exc
