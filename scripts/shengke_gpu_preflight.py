"""Check actual CUDA execution before loading a Shengke model; save honest evidence."""

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def inspect():
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - model Python 3.10
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ready": False,
    }
    try:
        import torch

        result.update(torch=torch.__version__, cuda_version=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this environment; no GPU test performed")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result.update(
            device_name=properties.name,
            total_vram_bytes=properties.total_memory,
            cuda_device=device,
        )
        # A driver listing alone does not prove the installed Torch can execute CUDA.
        x = torch.ones((64, 64), device="cuda")
        value = float((x @ x).mean().item())
        torch.cuda.synchronize()
        if value != 64:
            raise RuntimeError("CUDA matrix operation returned an unexpected result")
        listing = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["nvidia_smi"] = list(csv.reader(io.StringIO(listing.stdout)))
        result["ready"] = True
        result["scope"] = "CUDA environment only; model loading and benchmarks still required"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence")
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
