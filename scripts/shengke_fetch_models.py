"""Download only pinned public model files, verify Git/LFS hashes and keep a lock manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            git.update(block)
    return sha256.hexdigest(), git.hexdigest()


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest, git = hashes(path)
    return digest == entry["lfs"]["sha256"] if "lfs" in entry else git == entry["blobId"]


def fetch(url, destination, expected_size=None):
    transport = os.getenv("SHENGKE_DOWNLOAD_TRANSPORT", "urllib")
    if transport not in {"urllib", "curl"}:
        raise ValueError("SHENGKE_DOWNLOAD_TRANSPORT must be urllib or curl")
    if transport == "curl":
        if expected_size and destination.exists() and destination.stat().st_size == expected_size:
            return  # The caller still verifies the complete file's pinned hash.
        command = [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--max-time",
            "1800",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--output",
            str(destination),
        ]
        if expected_size:
            command += ["--continue-at", "-"]
        subprocess.run([*command, url], check=True)
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RuntimeError("curl download size mismatch")
        return
    failures = 0
    while True:
        offset = destination.stat().st_size if expected_size and destination.exists() else 0
        if expected_size and offset == expected_size:
            return
        try:
            headers = {}
            if expected_size:
                end = min(offset + 8 * 1024 * 1024, expected_size) - 1
                headers["Range"] = f"bytes={offset}-{end}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                partial = response.status == 206
                if partial and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise OSError("incorrect download range")
                with destination.open("ab" if partial else "wb") as out:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        out.write(block)
            size = destination.stat().st_size
            if expected_size is None or size == expected_size:
                return
            if size <= offset or size > expected_size:
                raise OSError("download size did not advance within expected bounds")
            failures = 0  # Retry budget is per failed range, not for the whole large file.
            print(f"Downloaded {size}/{expected_size} bytes", flush=True)
        except (OSError, http.client.HTTPException) as exc:
            failures += 1
            print(f"Download retry {failures}/3: {type(exc).__name__}", flush=True)
            if failures == 3:
                raise RuntimeError(f"Download failed: {url}") from exc
            time.sleep(2)


def main():
    catalog = json.loads((ROOT / "configs/shengke/models.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", choices=[key for key, item in catalog.items() if "weights_files" in item]
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models/shengke")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--source", choices=["huggingface", "zenodo"], default="huggingface")
    args = parser.parse_args()
    config = catalog[args.model]
    repo, revision = config["weights"], config["weights_revision"]
    destination = args.output / args.model
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "upstream-metadata.json"
    fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata["sha"] != revision:
        raise RuntimeError("upstream returned a different revision")
    selected = [
        entry
        for entry in metadata["siblings"]
        if any(
            fnmatch.fnmatchcase(entry["rfilename"], pattern) for pattern in config["weights_files"]
        )
    ]
    for pattern in config["weights_files"]:
        if not any(fnmatch.fnmatchcase(entry["rfilename"], pattern) for entry in selected):
            raise RuntimeError(f"required model file is missing: {pattern}")
    lock = {"model": args.model, "repository": repo, "revision": revision, "files": selected}
    (destination / "download-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(
        f"{args.model}: {len(selected)} files, {sum(e['size'] for e in selected) / 1e6:.1f} MB",
        flush=True,
    )
    if args.metadata_only:
        return
    archive = None
    if args.source == "zenodo":
        source = config.get("original_archive")
        if not source:
            parser.error("this model has no verified original Zenodo archive")
        archive_path = destination / "upstream-zenodo.zip"
        if not archive_path.exists():
            partial_archive = archive_path.with_suffix(".zip.partial")
            fetch(source["url"], partial_archive, source["size"])
            if hashlib.md5(partial_archive.read_bytes()).hexdigest() != source["md5"]:
                raise RuntimeError("original archive checksum mismatch")
            partial_archive.replace(archive_path)
        if hashlib.md5(archive_path.read_bytes()).hexdigest() != source["md5"]:
            raise RuntimeError("cached archive checksum mismatch")
        archive = zipfile.ZipFile(archive_path)
    for entry in selected:
        relative = PurePosixPath(entry["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe upstream filename")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if verified(path, entry):
            print(f"Verified cached: {relative}", flush=True)
            continue
        partial = path.with_name(path.name + ".partial")
        route = "resolve" if "lfs" in entry else "raw"
        members = (
            []
            if archive is None
            else [
                name
                for name in archive.namelist()
                if name == str(relative) or name.endswith("/" + str(relative))
            ]
        )
        if len(members) > 1:
            raise RuntimeError(f"ambiguous archive entry: {relative}")
        if members:
            with archive.open(members[0]) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
        else:
            fetch(
                f"https://huggingface.co/{repo}/{route}/{revision}/{relative}",
                partial,
                entry["size"] if "lfs" in entry else None,
            )
        if not verified(partial, entry):
            raise RuntimeError(f"size or hash mismatch: {relative}")
        partial.replace(path)
        print(f"Verified downloaded: {relative}", flush=True)
    if archive is not None:
        archive.close()
    print(
        "Pinned model files verified. Runtime inference is still required for acceptance.",
        flush=True,
    )


if __name__ == "__main__":
    main()
