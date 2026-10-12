"""Download/verify the fixed CSMSC Parallel WaveGAN archive and extract only its three files."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare(archive, output, specification):
    if digest(archive) != specification["archive_sha256"]:
        raise ValueError("vocoder archive SHA-256 mismatch")
    expected = {**specification["weights_sha256"], **specification["configuration_sha256"]}
    with tarfile.open(archive, "r:gz") as bundle:
        for name, sha in expected.items():
            if Path(name).name != name:
                raise ValueError("unsafe vocoder filename")
            candidates = [
                member
                for member in bundle.getmembers()
                if member.name == name or member.name.endswith("/" + name)
            ]
            if len(candidates) != 1 or not candidates[0].isfile():
                raise ValueError(f"missing, duplicate or non-file vocoder member: {name}")
            output.mkdir(parents=True, exist_ok=True)
            target = output / name
            if target.exists() and digest(target) == sha:
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            with bundle.extractfile(candidates[0]) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if digest(partial) != sha:
                raise ValueError(f"vocoder file SHA-256 mismatch: {name}")
            partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ROOT / "models/shengke/transformer-csmsc"
    parser.add_argument("--archive", type=Path, default=model / "csmsc_parallel_wavegan.v1.tar.gz")
    parser.add_argument(
        "--output", type=Path, default=model / "vocoder/train_nodev_csmsc_parallel_wavegan.v1"
    )
    args = parser.parse_args()
    specification = json.loads((ROOT / "configs/shengke/csmsc-vocoder.json").read_text())
    if not args.archive.exists():
        import gdown

        args.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = args.archive.with_suffix(args.archive.suffix + ".partial")
        gdown.download(specification["download_url"], str(partial), quiet=False, resume=True)
        if not partial.exists() or digest(partial) != specification["archive_sha256"]:
            raise ValueError("downloaded vocoder archive SHA-256 mismatch")
        partial.replace(args.archive)
    prepare(args.archive, args.output, specification)
    print(f"Verified CSMSC vocoder: {args.output}")


if __name__ == "__main__":
    main()
