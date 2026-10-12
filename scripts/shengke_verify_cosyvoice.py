"""Verify an existing CosyVoice2 installation against the CPU-validated file lock."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/shengke/cosyvoice2-weights-lock.json").read_text())
    for entry in lock["files"]:
        path = args.weights / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise SystemExit(f"Missing or wrong-sized model file: {entry['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise SystemExit(f"Model checksum mismatch: {entry['path']}")
    print(f"Verified {len(lock['files'])} CosyVoice2 files against the local validated lock")


if __name__ == "__main__":
    main()
