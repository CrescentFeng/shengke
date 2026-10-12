"""Fetch pinned public WAVs and crop six aligned 3/5/10-second references."""

import hashlib
import json
import wave
from pathlib import Path

from shengke_fetch_models import fetch, verified

ROOT = Path(__file__).resolve().parents[1]


def main():
    recipe = json.loads((ROOT / "data/tts-eval/reference-crops.json").read_text())
    raw = ROOT / "data/raw/shengke/thchs30"
    output = ROOT / "data/input/shengke/public-references"
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    repo, revision = recipe["audio_repository"], recipe["audio_revision"]
    metadata = raw / "download-metadata.json"
    fetch(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true", metadata)
    upstream = json.loads(metadata.read_text())
    if upstream["sha"] != revision:
        raise ValueError("unexpected public audio revision")
    entries = {e["rfilename"]: e for e in upstream["siblings"]}
    manifest = []
    for item in recipe["items"]:
        name = f"audio/{item['source_id']}.wav"
        entry = entries[name]
        source = raw / name
        source.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        if not verified(source, entry):
            partial = source.with_suffix(".wav.partial")
            fetch(url, partial, entry["size"])
            if not verified(partial, entry):
                raise ValueError(f"public reference checksum mismatch: {name}")
            partial.replace(source)
        path = output / f"{item['id']}.wav"
        with wave.open(str(source), "rb") as wav:
            params = wav.getparams()
            start = round(item["start_s"] * params.framerate)
            count = round(item["duration_s"] * params.framerate)
            if start + count > params.nframes:
                raise ValueError("crop extends past the original recording")
            wav.setpos(start)
            frames = wav.readframes(count)
        with wave.open(str(path), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(frames)
        manifest.append(
            {
                **{k: v for k, v in item.items() if k != "aligned_words"},
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source": recipe["source"],
                "mirror_url": url,
                "license": recipe["license"],
                "alignment_revision": recipe["alignment_revision"],
                "human_transcript_verification": (
                    item.get("human_transcript_verification", "pending")
                    if item.get("human_review", {}).get("audio_sha256")
                    == hashlib.sha256(path.read_bytes()).hexdigest()
                    else "pending"
                ),
                "processing": "PCM frame crop only; original sample rate; no added silence",
            }
        )
        print(f"Prepared {item['id']}: {item['duration_s']} seconds", flush=True)
    target = ROOT / "data/tts-eval/public-references-3-5-10.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {target}. Human review remains required.")


if __name__ == "__main__":
    main()
