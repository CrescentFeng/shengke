from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import threading
from contextlib import AbstractContextManager


def peak_process_memory_mb() -> float:
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return maximum_rss / divisor


class GpuMemorySampler(AbstractContextManager["GpuMemorySampler"]):
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executable = shutil.which("nvidia-smi")

    def _sample(self) -> None:
        if not self._executable:
            return
        try:
            completed = subprocess.run(
                [
                    self._executable,
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if completed.returncode != 0:
            return
        process_id = str(os.getpid())
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or fields[0] != process_id:
                continue
            try:
                used_mb = float(fields[1])
            except ValueError:
                continue
            self.peak_mb = used_mb if self.peak_mb is None else max(self.peak_mb, used_mb)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> GpuMemorySampler:
        if self._executable:
            self._sample()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
