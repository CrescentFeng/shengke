"""On-demand local service. No login item; control only our own supervisor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from uuid import uuid4

from shengke_stack import ROOT, busy_ports, commands, resolve

RUNTIME = ROOT / "artifacts/shengke/service"
SOCKET = Path(
    f"/tmp/shengke-{os.getuid()}-{hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]}.sock"
)
URL = "http://127.0.0.1:7860/?view=synthesis"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def control(operation):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(25)
        client.connect(str(SOCKET))
        client.sendall(json.dumps({"operation": operation, "root": str(ROOT)}).encode() + b"\n")
        data = client.recv(65536)
        if not data:
            raise ConnectionResetError("控制通道已关闭")
        result = json.loads(data)
        if result.get("root") != str(ROOT):
            raise RuntimeError("控制通道身份不符，不操作该进程")
        return result


def current():
    try:
        return control("status")
    except (FileNotFoundError, ConnectionRefusedError):
        return {"stage": "stopped", "root": str(ROOT)}
    except (ConnectionResetError, BrokenPipeError):
        return {"stage": "stopping" if SOCKET.exists() else "stopped", "root": str(ROOT)}


def healthy(instance):
    try:
        with OPENER.open("http://127.0.0.1:8000/v1/tts/info", timeout=2) as r:
            info = json.load(r)
        if info.get("instance_id") != instance:
            return False
        for port, path in [(8000, "/readyz"), (7860, "/")]:
            with OPENER.open(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                if r.status != 200:
                    return False
        with OPENER.open("http://127.0.0.1:8000/v1/tts/models", timeout=12) as r:
            models = json.load(r)
        return len(models) == 4 and all(m["status"] == "ready" for m in models)
    except (OSError, ValueError):
        return False


def ensure_docker(env, timeout=90):
    docker = shutil.which("docker", path=env["PATH"])
    if not docker:
        raise RuntimeError("未找到 Docker；请安装并打开 Docker Desktop")

    def available():
        try:
            return (
                subprocess.run(
                    [docker, "info"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).returncode
                == 0
            )
        except subprocess.TimeoutExpired:
            return False

    if available():
        return
    if sys.platform != "darwin":
        raise RuntimeError("Docker 未运行；请先启动 Docker")
    subprocess.run(["/usr/bin/open", "-a", "Docker"], check=True, timeout=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if available():
            return
        time.sleep(2)
    raise RuntimeError("Docker 未就绪；请在 Docker Desktop 处理启动提示后重试")


def daemon(config_path):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "instance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The lifetime lock proves no other managed daemon owns this socket.
        SOCKET.unlink(missing_ok=True)
        state = {
            "root": str(ROOT),
            "instance_id": str(uuid4()),
            "pid": os.getpid(),
            "stage": "starting",
            "url": URL,
        }
        child = None
        stopping = False

        def stop_signal(_sig, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGINT, stop_signal)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(SOCKET))
            SOCKET.chmod(0o600)
            server.listen(4)
            try:
                env = os.environ.copy()
                env["SHENGKE_INSTANCE_ID"] = state["instance_id"]
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/shengke_stack.py"),
                        "--config",
                        str(config_path),
                        "--start-infra",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 900
                last_check = 0
                while not stopping:
                    if child.poll() is not None:
                        raise RuntimeError(f"服务进程退出：{child.returncode}；请查看 stack 日志")
                    now = time.monotonic()
                    if now - last_check > 5:
                        last_check = now
                        ready = healthy(state["instance_id"])
                        state["stage"] = (
                            "ready" if ready else "starting" if now < deadline else "unhealthy"
                        )
                        if now >= deadline and not ready:
                            raise RuntimeError("服务健康检查未通过，请查看日志后重新启动")
                        (RUNTIME / "latest.json").write_text(
                            json.dumps(state, ensure_ascii=False, indent=2)
                        )
                    if select.select([server], [], [], 0.3)[0]:
                        with server.accept()[0] as client:
                            client.settimeout(2)
                            try:
                                request = json.loads(client.recv(4096))
                                if request.get("root") != str(ROOT):
                                    client.sendall(b'{"error":"identity mismatch"}')
                                    continue
                                if request.get("operation") == "stop":
                                    state["stage"] = "stopping"
                                    stopping = True
                                client.sendall(json.dumps(state).encode())
                            except (OSError, ValueError):
                                continue
            except Exception as exc:
                state.update(stage="failed", error=str(exc))
                raise
            finally:
                if child is not None and child.poll() is None:
                    # Signal the stack parent; it cleans up its own model process groups.
                    child.terminate()
                    try:
                        child.wait(timeout=75)
                    except subprocess.TimeoutExpired:
                        state.update(stage="failed", error="退出超时；保留诊断，不强杀未知子进程")
                if state["stage"] != "failed":
                    state["stage"] = "stopped"
                (RUNTIME / "latest.json").write_text(
                    json.dumps(state, ensure_ascii=False, indent=2)
                )
                SOCKET.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status", "daemon"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/shengke/local-macos.json")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.action == "daemon":
        daemon(args.config.resolve())
        return
    if args.action == "status":
        state = current()
        if state["stage"] == "ready" and not healthy(state["instance_id"]):
            state["stage"] = "unhealthy"
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "action.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一个启动或停止操作正在进行，请稍后重试") from exc
        state = current()
        if args.action == "stop":
            if state["stage"] == "stopped":
                print("没有受此入口管理的声刻实例；没有关闭其他进程。")
                return
            control("stop")
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if current()["stage"] == "stopped":
                    print("声刻已停止；音频、评分、数据库及其他项目服务保留。")
                    return
                time.sleep(0.5)
            raise RuntimeError("停止尚未完成，请查看 service/latest.json 和日志")
        if state["stage"] == "ready" and healthy(state["instance_id"]):
            print("声刻已在后台运行，复用当前实例。")
            if args.open:
                webbrowser.open(URL)
            return
        if state["stage"] != "stopped":
            raise RuntimeError("已有实例正在启动或异常；请查看状态，必要时先停止再启动")
        config = json.loads(args.config.read_text())
        commands(config)
        if ports := busy_ports():
            raise RuntimeError(
                f"端口 {ports} 被其他服务或旧前台实例占用，请先从原启动窗口退出；不会强制关闭。"
            )
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [
                str(Path.home() / ".docker/bin"),
                str(resolve(config["ffmpeg_directory"])),
                env.get("PATH", ""),
            ]
        )
        ensure_docker(env)
        with (RUNTIME / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "daemon",
                    "--config",
                    str(args.config.resolve()),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print("正在后台启动四模型与工作台，首次加载请稍候…", flush=True)
        deadline = time.monotonic() + 930
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"启动失败，请查看 {RUNTIME / 'supervisor.log'}")
            state = current()
            if state["stage"] == "ready":
                print(f"声刻已就绪：{URL}；可关闭此终端。", flush=True)
                if args.open:
                    webbrowser.open(URL)
                return
            time.sleep(1)
        raise RuntimeError("启动尚未完成，请查看运行状态；不会重复启动")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"声刻操作未完成：{exc}") from exc
