"""Download the whole Lightricks/LTX-2.5 repo into the tree extra_model_paths.yaml maps.

Ordered by usefulness rather than by folder, so the set becomes runnable long before the 200 GB
finishes: the int8-convrot transformer and text encoder first, then the VAEs and patches every
workflow needs, then the reference and quantization-source checkpoints.

Why int8-convrot before nvfp4 on this host: `comfy/ops.py:pick_operations` disables a format when
the GPU cannot compute it, and on SM 8.6 `supports_nvfp4_compute` is False, so `nvfp4` is added to
the disabled set and dequantized to bf16 for every matmul. `int8_tensorwise` + convrot is not in
that set and runs natively. The nvfp4 file is still worth having as the reference to compare a
locally-produced conversion against.

The bf16 checkpoints are the quantization sources -- 4-bit conversion needs high-precision weights,
so they are not optional if the point is to quantize LTX-2.5 rather than only run it.

Repo folder names match ComfyUI's, so each file lands where it belongs on its own. Nothing is
overwritten: a file already present at the expected size is skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from hf_parallel_get import download as parallel_download

REPO = "Lightricks/LTX-2.5"
ROOT = Path("D:/ComfyUI-Models")

# (filename in repo, expected bytes, why it is in this position)
FILES = [
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
     21_504_034_224, "runs natively on SM 8.6"),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
     15_372_971_786, "runs natively on SM 8.6"),
    ("vae/ltx-2.5-video-vae-bf16.safetensors", 1_472_223_346, "required by any workflow"),
    ("vae/ltx-2.5-audio-vae-bf16.safetensors", 364_866_540, "required for audio"),
    ("model_patches/ltx-2.5-duration-head-bf16.safetensors", 3_843_690, "tiny, required"),
    ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
     995_778_752, "upscale chain"),
    ("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
     261_944_000, "upscale chain"),
    ("vae/ltx-2.5-video-vae-conv-bf16.safetensors", 1_452_269_922, "alternative video VAE"),
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
     18_721_548_408, "emulated here, but the reference NVFP4 conversion to compare against"),
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
     42_018_190_584, "quantization source"),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
     26_263_860_594, "quantization source"),
    ("diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
     21_504_034_224, "dev variant"),
    ("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
     42_018_190_584, "dev variant, quantization source"),
    ("loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors", 8_899_889_568, "lora"),
]


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError


def main() -> int:
    # `--only SUBSTRING` (repeatable) restricts the run. A skipped file keeps its `.incomplete`
    # in the HF cache, so a later run without the filter resumes it rather than restarting.
    argv = sys.argv[1:]
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only" and i + 1 < len(argv)]

    for folder in ("diffusion_models", "text_encoders", "vae", "loras",
                   "latent_upscale_models", "model_patches"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    selected = [f for f in FILES if not only or any(o in f[0] for o in only)]
    if only and not selected:
        raise SystemExit(f"--only {only} matched nothing")
    total = sum(size for _, size, _ in selected)
    done = sum(size for name, size, _ in selected
               if (ROOT / name).is_file() and (ROOT / name).stat().st_size == size)
    scope = f" (filtered by {only})" if only else ""
    print(f"{len(selected)} files{scope}, {human(total)} total, "
          f"{human(done)} already present\n", flush=True)

    failures = []
    for index, (filename, expected, why) in enumerate(selected, 1):
        destination = ROOT / filename
        if destination.is_file() and destination.stat().st_size == expected:
            print(f"[{index}/{len(selected)}] present, skipping  {destination.name}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] {filename}  ({human(expected)})  -- {why}", flush=True)
        started = time.perf_counter()
        try:
            # download() raises SystemExit (not Exception) on a size mismatch, and returns 130
            # rather than raising when the user hits Ctrl-C mid-chunk -- both handled below,
            # since letting either fall through as a per-file "FAILED" would either miss the
            # SystemExit or silently roll on to the next file after an interrupt.
            rc = parallel_download(REPO, filename, ROOT, expected_size=expected)
        except (SystemExit, Exception) as error:
            print(f"          FAILED: {type(error).__name__}: {str(error)[:160]}\n", flush=True)
            failures.append(filename)
            continue
        if rc == 130:
            print("\ninterrupted; rerun the same command to resume")
            return 130
        elapsed = time.perf_counter() - started
        size = (ROOT / filename).stat().st_size if (ROOT / filename).is_file() else 0
        rate = size / elapsed / 1024**2 if elapsed else 0
        status = "OK" if size == expected else f"SIZE MISMATCH: got {size}, expected {expected}"
        if size != expected:
            failures.append(filename)
        print(f"          {human(size)} in {elapsed / 60:.1f} min ({rate:.1f} MiB/s)  {status}\n",
              flush=True)

    print(f"\n{'=' * 60}")
    for filename, expected, _ in FILES:
        path = ROOT / filename
        mark = "OK " if path.is_file() and path.stat().st_size == expected else "MISSING"
        actual = human(path.stat().st_size) if path.is_file() else "-"
        print(f"  {mark:<8}{actual:>12}  {filename}")
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
