"""Import checksum-verified public references into a running Shengke workbench."""

import argparse
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/tts-eval/public-references.json"
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def api(path, body=None):
        request = urllib.request.Request(
            args.api.rstrip("/") + path,
            json.dumps(body).encode() if body is not None else None,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    profiles = api("/v1/tts/voices")
    imported = []
    for item in json.loads(args.manifest.read_text()):
        path = ROOT / item["path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"reference checksum mismatch: {item['id']}")
        profile = next(
            (
                p
                for p in profiles
                if p["asset"]["sha256"] == digest and p["reference_text"] == item["reference_text"]
            ),
            None,
        )
        if profile is None:
            profile = api(
                "/v1/tts/voices",
                {
                    "name": "THCHS-30 · " + item["id"],
                    "filename": path.name,
                    "audio_base64": base64.b64encode(raw).decode(),
                    "reference_text": item["reference_text"],
                    "source": item["mirror_url"],
                    "license": item["license"],
                },
            )
        imported.append({"reference": item, "voice": profile})
        print(f"{item['id']}: {profile['id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(imported, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
