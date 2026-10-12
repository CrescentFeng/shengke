"""Start the four local CPU models and workbench; supervise only owned processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"cosyvoice2": 9881, "gpt-sovits-v2": 9882, "vits-csmsc": 9883, "transformer-csmsc": 9884}


def resolve(value):
    p = Path(value).expanduser()
    # Do not resolve Python symlinks: the virtual environment prefix is significant.
    return p if p.is_absolute() else ROOT / p


def commands(config):
    if set(config["models"]) != set(PORTS):
        raise ValueError("启动配置必须包含四个指定模型")
    paths = {
        k: resolve(config[k])
        for k in ("platform_python", "ui_python", "rhythm_python", "rhythm_weights")
    }
    for key, p in paths.items():
        if not p.is_file():
            raise ValueError(f"{key} 不存在：{p}")
    if not (resolve(config["ffmpeg_directory"]) / "ffmpeg").is_file():
        raise ValueError("FFmpeg 不存在，请检查 ffmpeg_directory")
    result = []
    for name, port in PORTS.items():
        model = {k: resolve(config["models"][name][k]) for k in ("python", "repo", "weights")}
        if not model["python"].is_file() or not all(model[k].is_dir() for k in ("repo", "weights")):
            raise ValueError(f"{name} 的 Python、源码或权重目录缺失")
        cmd = [
            str(paths["platform_python"]),
            str(ROOT / "scripts/shengke_model.py"),
            name,
            "--python",
            str(model["python"]),
            "--repo",
            str(model["repo"]),
            "--weights",
            str(model["weights"]),
            "--device",
            "cpu",
        ]
        if name == "gpt-sovits-v2":
            cmd += [
                "--rhythm-python",
                str(paths["rhythm_python"]),
                "--rhythm-weights",
                str(paths["rhythm_weights"]),
            ]
        result.append((name, cmd, f"http://127.0.0.1:{port}/healthz"))
    result.append(
        (
            "workbench",
            [
                str(paths["platform_python"]),
                str(ROOT / "scripts/shengke_local.py"),
                "--ui-python",
                str(paths["ui_python"]),
            ],
            "http://127.0.0.1:8000/readyz",
        )
    )
    return [result[-1], *result[:-1]]


def busy_ports():
    busy = []
    for port in [*PORTS.values(), 8000, 7860]:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
    return busy


class Supervisor:
    def __init__(self, env, logs):
        self.env, self.logs, self.children = env, logs, []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self, name, command):
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / f"{name}.log").open("ab") as output:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children.append((name, child))

    def check_alive(self):
        for name, child in self.children:
            if child.poll() is not None:
                raise RuntimeError(f"{name} 已退出，请查看 {self.logs / name}.log")

    def wait_ready(self, url, expected_model=None, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check_alive()
            try:
                with self.opener.open(url, timeout=2) as response:
                    if expected_model:
                        state = json.load(response)
                        if state.get("model") == expected_model and state.get("ready"):
                            return
                    elif response.status == 200:
                        return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"服务未在 {timeout} 秒内就绪：{url}")

    def stop(self):
        # Every entry was created here with a new session; never signal discovered PIDs.
        for _, child in reversed(self.children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, child in reversed(self.children):
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
        self.children.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/shengke/local-macos.json")
    parser.add_argument(
        "--check", action="store_true", help="Validate local paths without starting services"
    )
    parser.add_argument(
        "--start-infra", action="store_true", help="Start existing PostgreSQL/MinIO Docker services"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open workbench in default browser after ready"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    jobs = commands(config)
    if args.check:
        print("本地四模型、工作台、Whisper 与 FFmpeg 路径检查通过。")
        return
    busy = busy_ports()
    if busy:
        raise RuntimeError(
            f"端口已被占用：{busy}。若声刻已启动，请直接打开 http://127.0.0.1:7860；本命令不会关闭已有服务。"
        )
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        SHENGKE_MODEL_ENDPOINTS=json.dumps(
            {name: f"http://127.0.0.1:{port}" for name, port in PORTS.items()}
        ),
    )
    env["PATH"] = str(resolve(config["ffmpeg_directory"])) + os.pathsep + env.get("PATH", "")
    if args.start_infra:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(ROOT / "infra/compose.yaml"),
                "up",
                "-d",
                "postgres",
                "minio",
            ],
            env=env,
            check=True,
            timeout=120,
        )
    manager = Supervisor(env, ROOT / "artifacts/shengke/logs/stack")

    def interrupt(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for name, command, url in jobs:
            print(f"正在启动 {name}…", flush=True)
            manager.start(name, command)
            manager.wait_ready(url, name if name in PORTS else None)
            print(f"{name} 已就绪", flush=True)
        manager.wait_ready("http://127.0.0.1:7860", timeout=45)
        print(
            "声刻已就绪：http://127.0.0.1:7860/?view=synthesis\n"
            "保留此窗口；按 Ctrl+C 停止本次启动的服务。",
            flush=True,
        )
        if args.open:
            webbrowser.open("http://127.0.0.1:7860/?view=synthesis")
        while True:
            manager.check_alive()
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止本次启动的模型与工作台；保留数据库和已保存音频。", flush=True)
    finally:
        manager.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"声刻启动未完成：{exc}") from exc
