"""Launch one pinned Shengke model service in its own environment (loopback only)."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"cosyvoice2": 9881, "gpt-sovits-v2": 9882, "vits-csmsc": 9883, "transformer-csmsc": 9884}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=PORTS)
    parser.add_argument("--python", type=Path, help="Python in the upstream model's environment")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--rhythm-python", type=Path, help="Optional local Whisper Python")
    parser.add_argument("--rhythm-weights", type=Path, help="Pinned local Whisper base.pt")
    args = parser.parse_args()
    weights = (args.weights or ROOT / "models/shengke" / args.model).resolve()
    environment = "espnet" if args.model.endswith("csmsc") else args.model.replace("-v2", "")
    python = (args.python or ROOT / "models/environments" / environment / "bin/python").absolute()
    if not python.is_file():
        parser.error(f"model environment is missing: {python}; set --python")
    if not weights.is_dir():
        parser.error(f"model directory is missing: {weights}; download pinned weights first")
    repo = args.repo
    if repo is None:
        repo = (
            weights
            if args.model.endswith("csmsc")
            else ROOT
            / "models/repos"
            / {"cosyvoice2": "CosyVoice", "gpt-sovits-v2": "GPT-SoVITS"}[args.model]
        )
    repo = repo.resolve()
    if not repo.is_dir():
        parser.error(f"upstream repository is missing: {repo}; set --repo")
    env = os.environ.copy()
    if bool(args.rhythm_python) != bool(args.rhythm_weights):
        parser.error("provide both --rhythm-python and --rhythm-weights")
    if args.rhythm_python:
        if not args.rhythm_python.is_file() or not args.rhythm_weights.is_file():
            parser.error("rhythm Python or weights missing")
        env.update(
            SHENGKE_RHYTHM_PYTHON=str(args.rhythm_python.absolute()),
            SHENGKE_RHYTHM_WEIGHTS=str(args.rhythm_weights.resolve()),
        )
    cache = ROOT / "models/cache" / args.model
    cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_CACHE_HOME", str(cache))
    env.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.update(
        SHENGKE_MODEL=args.model,
        SHENGKE_MODEL_DIR=str(weights),
        SHENGKE_REPO_ROOT=str(repo),
        SHENGKE_DEVICE=args.device,
        SHENGKE_FP16="1" if args.fp16 else "0",
    )
    required = []
    if args.model == "gpt-sovits-v2":
        custom = {
            "device": args.device,
            "is_half": args.fp16,
            "version": "v2",
            "t2s_weights_path": str(
                weights / "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
            ),
            "vits_weights_path": str(weights / "gsv-v2final-pretrained/s2G2333k.pth"),
            "bert_base_path": str(weights / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(weights / "chinese-hubert-base"),
        }
        required = [Path(custom[key]) for key in ["t2s_weights_path", "vits_weights_path"]]
        required += [
            Path(custom[key]) / "pytorch_model.bin"
            for key in ["bert_base_path", "cnhuhbert_base_path"]
        ]
        config = weights / "shengke-runtime.yaml"
        # JSON is valid YAML; this generated file is reserved for this launcher.
        configuration = json.dumps({"version": "v2", "custom": custom}, indent=2)
        env["SHENGKE_GPT_CONFIG"] = str(config)
    elif args.model.endswith("csmsc"):
        architecture = args.model.split("-")[0]
        folder = weights / f"exp/tts_train_{architecture}_raw_phn_pypinyin_g2p_phone"
        checkpoint = (
            "train.total_count.ave_10best.pth"
            if architecture == "vits"
            else "train.loss.ave_5best.pth"
        )
        env.update(
            SHENGKE_ESPNET_CONFIG=str(folder / "config.yaml"),
            SHENGKE_ESPNET_WEIGHTS=str(folder / checkpoint),
        )
        required = [folder / "config.yaml", folder / checkpoint]
        if architecture == "transformer":
            vocoder = weights / "vocoder/train_nodev_csmsc_parallel_wavegan.v1"
            env.update(
                SHENGKE_VOCODER_CONFIG=str(vocoder / "config.yml"),
                SHENGKE_VOCODER_WEIGHTS=str(vocoder / "checkpoint-400000steps.pkl"),
            )
            required += [
                vocoder / name for name in ["config.yml", "checkpoint-400000steps.pkl", "stats.h5"]
            ]
            required.append(
                weights / "exp/tts_stats_raw_phn_pypinyin_g2p_phone/train/feats_stats.npz"
            )
    else:
        required = [weights / name for name in ["cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt"]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("missing model files:\n" + "\n".join(missing))
    if args.model == "gpt-sovits-v2":
        config.write_text(configuration + "\n")
    print(f"Starting {args.model} on 127.0.0.1:{args.port or PORTS[args.model]}", flush=True)
    os.execve(
        str(python),
        [
            str(python),
            "-m",
            "uvicorn",
            "server:app",
            "--app-dir",
            str(ROOT / "services/tts-runtime"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port or PORTS[args.model]),
        ],
        env,
    )


if __name__ == "__main__":
    sys.exit(main())
