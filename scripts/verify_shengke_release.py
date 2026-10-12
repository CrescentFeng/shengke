"""Read-only live MVP gate; writes a local evidence report, never submits jobs or scores."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from audio_core.shengke_release import BUILD_ID, RELEASE, ROOT

OUTPUT = ROOT / "artifacts/shengke/release-v1"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read(path, binary=False):
    with OPENER.open("http://127.0.0.1:8000" + path, timeout=20) as response:
        raw = response.read()
    return raw if binary else json.loads(raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    info = read("/v1/tts/info")
    assert info["version"] == RELEASE["version"] == "1.0.0"
    assert info["build_id"] == BUILD_ID, "运行代码与当前源码不一致，需要重启"
    models = read("/v1/tts/models")
    assert len(models) == 4 and all(m["status"] == "ready" for m in models)
    assert read("/readyz")
    with OPENER.open("http://127.0.0.1:7860/", timeout=10) as response:
        assert response.status == 200
    before = json.loads((OUTPUT / "ratings-before.json").read_text())
    ratings = {}
    for identity, checksum in before.items():
        actual = hashlib.sha256(read(f"/v1/tts/evaluations/{identity}/export", True)).hexdigest()
        assert actual == checksum, "原始评分发生变化"
        ratings[identity] = read(f"/v1/tts/evaluations/{identity}/progress?listener_id=listener_01")
    assert sum(v["completed"] for v in ratings.values()) == 21
    lifecycle = json.loads((OUTPUT / "lifecycle.json").read_text())
    assert all(lifecycle["ports_closed"].values())
    assert lifecycle["before"]["instance_id"] == lifecycle["after_reuse"]["instance_id"]
    assert lifecycle["before"]["instance_id"] != info["instance_id"]
    samples = []
    for row in lifecycle["jobs_before_restart"]["items"]:
        job = read(f"/v1/audio/jobs/{row['id']}")
        assert job["state"] == row["state"]
        if job["state"] == "succeeded":
            raw = read(f"/v1/tts/jobs/{row['id']}/audio", True)
            report = read(f"/v1/audio/jobs/{row['id']}/report")
            path = OUTPUT / "jobs" / row["id"]
            path.mkdir(parents=True, exist_ok=True)
            (path / "speech.wav").write_bytes(raw)
            (path / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
            (path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            samples.append(
                {"id": row["id"], "model": row["model"], "sha256": hashlib.sha256(raw).hexdigest()}
            )
    page = read("/v1/tts/jobs?limit=2&offset=0")
    assert len(page["items"]) == 2 and page["total"] >= 2
    assert not read(f"/v1/tts/jobs?offset={page['total']}")["items"]
    source_files = []
    for directory in [
        "apps",
        "packages",
        "scripts",
        "configs",
        "contracts",
        "infra",
        "docs",
        "tests",
    ]:
        source_files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and not any(
                part in {"__pycache__", ".pytest_cache", "node_modules"} for part in p.parts
            )
            and p.suffix
            in {
                ".py",
                ".json",
                ".toml",
                ".yaml",
                ".yml",
                ".md",
                ".sql",
                ".command",
                ".txt",
                ".html",
            }
        )
    source_files += [ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "Makefile"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(source_files))}
    tests = (OUTPUT / "tests-full.log").read_text()
    assert "126 passed" in tests and "failed" not in tests
    assert "\nOK\n" in (OUTPUT / "tests-ui.log").read_text()
    result = {
        "product": RELEASE,
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "engineering_mvp_passed",
        "runtime": info,
        "models_ready": [m["id"] for m in models],
        "ratings_preserved": True,
        "ratings_count": 21,
        "historical_job_count": page["total"],
        "persistent_samples": samples,
        "source_sha256": hashes,
        "limitations": [
            "N=1 自听，不是多人 MOS",
            "情绪与韵律为试用能力",
            "新节奏配置仅 CPU 验证，不沿用旧 GPU 性能结论",
            "本机单用户，不提供公网生成",
            "电脑睡眠或退出 Docker 会影响服务",
        ],
        "evidence_sha256": {
            str(p.relative_to(OUTPUT)): digest(p)
            for p in sorted(OUTPUT.rglob("*"))
            if p.is_file() and p.name not in {"final-report.json", "SHA256SUMS"}
        },
    }
    (OUTPUT / "final-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("声刻 1.0.0 工程发布检查通过；21 份评分保持原样，四模型就绪。")


if __name__ == "__main__":
    main()
