"""Download the MiniMax H3 W4A8 stack into the D: tree that extra_model_paths.yaml already maps.

`ComfyUI/extra_model_paths.yaml` points at `D:/ComfyUI-Models/` but that tree does not exist, so
the mapping is currently dead. Creating it here makes the existing config valid and keeps 34 GB off
F:, which has only ~52 GB free.

Repo layout matches ComfyUI's folder names for the Comfy-Org files, so `text_encoders/...` and
`vae/...` land where they belong on their own. The starsfriday checkpoint has no subfolder and is
placed explicitly.

Nothing is overwritten: a file already present at the right size is skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path("D:/ComfyUI-Models")

# (repo_id, filename in repo, local_dir, expected bytes)
FILES = [
    ("starsfriday/MiniMax-H3-w4a8",
     "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
     ROOT / "diffusion_models", 12_540_858_008),
    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     ROOT, 15_687_142_551),
    ("Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors",
     ROOT, 5_207_808_496),
    ("Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_audio_vae_fp32.safetensors",
     ROOT, 605_254_808),
]


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError


def main() -> int:
    for folder in ("diffusion_models", "text_encoders", "vae", "unet", "loras", "checkpoints"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    total = sum(size for *_, size in FILES)
    print(f"{len(FILES)} files, {human(total)} total, into {ROOT}\n", flush=True)

    for index, (repo, filename, local_dir, expected) in enumerate(FILES, 1):
        destination = local_dir / filename
        if destination.is_file() and destination.stat().st_size == expected:
            print(f"[{index}/{len(FILES)}] present, skipping  {destination.name}", flush=True)
            continue
        print(f"[{index}/{len(FILES)}] {repo}  {filename}  ({human(expected)})", flush=True)
        started = time.perf_counter()
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(local_dir))
        elapsed = time.perf_counter() - started
        size = Path(path).stat().st_size
        rate = size / elapsed / 1024**2 if elapsed else 0
        status = "OK" if size == expected else f"SIZE MISMATCH: got {size}, expected {expected}"
        print(f"          -> {path}\n          {human(size)} in {elapsed / 60:.1f} min "
              f"({rate:.1f} MiB/s)  {status}\n", flush=True)
        if size != expected:
            return 1

    print("all files present")
    for *_, local_dir, _ in FILES:
        pass
    for repo, filename, local_dir, expected in FILES:
        p = local_dir / filename
        print(f"  {human(p.stat().st_size):>10}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
