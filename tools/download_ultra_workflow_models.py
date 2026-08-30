"""Download only the missing models used by the ULTRA_IMAGE_VIDEO pack.

The Hugging Face cache and ComfyUI model folders are both on F:, so completed
artifacts are hard-linked into place instead of consuming disk space twice.
Existing targets are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".scratch" / "hf_cache"
MODELS = ROOT / "ComfyUI" / "models"

DOWNLOADS = (
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        MODELS / "diffusion_models" / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        MODELS / "loras" / "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        MODELS / "loras" / "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        MODELS / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    ),
    (
        "YarvixPA/FLUX.1-Fill-dev-GGUF",
        "flux1-fill-dev-Q8_0.gguf",
        MODELS / "unet" / "flux1-fill-dev-Q8_0.gguf",
    ),
)


def install_from_cache(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Refusing to overwrite mismatched target: {target} "
                f"({target.stat().st_size} != {source.stat().st_size})"
            )
        print(f"already installed: {target} ({target.stat().st_size / 1e9:.2f} GB)")
        return
    os.link(source, target)
    print(f"installed: {target} ({target.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    for repo_id, filename, target in DOWNLOADS:
        print(f"fetching: {repo_id}/{filename}")
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=CACHE,
            )
        )
        install_from_cache(cached, target)


if __name__ == "__main__":
    main()
