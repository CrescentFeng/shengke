from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from .settings import WorkerSettings


def safe_filename(value: str) -> str:
    parsed = urlparse(value)
    candidate = parsed.path if parsed.scheme else value
    name = Path(unquote(candidate)).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "audio.bin"


class S3ArtifactStore:
    def __init__(self, settings: WorkerSettings) -> None:
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3/MinIO storage") from exc
        self.settings = settings
        self._client_error = ClientError
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            use_ssl=settings.s3_use_ssl,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def ensure_buckets(self) -> None:
        self._ensure_bucket(self.settings.input_bucket)
        self._ensure_bucket(self.settings.artifact_bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
            return
        except self._client_error as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
        parameters = {"Bucket": bucket}
        if self.settings.s3_region != "us-east-1":
            parameters["CreateBucketConfiguration"] = {
                "LocationConstraint": self.settings.s3_region
            }
        try:
            self.client.create_bucket(**parameters)
        except self._client_error as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                raise

    def fetch_input(self, uri: str, destination: Path) -> Path:
        parsed = urlparse(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if parsed.scheme == "s3":
            if not parsed.netloc or not parsed.path.lstrip("/"):
                raise ValueError("invalid s3 input URI")
            self.client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))
            return destination
        if parsed.scheme == "file":
            source = Path(unquote(parsed.path)).resolve(strict=True)
            allowed = self.settings.allowed_input_root.resolve(strict=True)
            if not source.is_relative_to(allowed) or not source.is_file():
                raise ValueError("file input is outside the allowed input root")
            shutil.copyfile(source, destination)
            return destination
        raise ValueError("input URI must use s3:// or file://")

    def upload_file(self, source: Path, key: str, content_type: str | None = None) -> str:
        normalized_key = key.lstrip("/")
        if not normalized_key or ".." in Path(normalized_key).parts:
            raise ValueError("invalid artifact key")
        content_type = content_type or mimetypes.guess_type(source.name)[0]
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            self.client.upload_file(
                str(source),
                self.settings.artifact_bucket,
                normalized_key,
                ExtraArgs=extra_args,
            )
        else:
            self.client.upload_file(str(source), self.settings.artifact_bucket, normalized_key)
        return f"s3://{self.settings.artifact_bucket}/{normalized_key}"
