---
license: apache-2.0
base_model: lodestones/MOSS-Image-SFT-8-steps-dmd
base_model_relation: quantized
library_name: diffusion-single-file
pipeline_tag: text-to-image
tags:
  - comfyui
  - text-to-image
  - quantized
  - fp8
  - nvfp4
  - int4
  - w4a16
language:
  - en
---

# MOSS-Image SFT (8-step DMD) — FP8, NVFP4 and W4A16 for ComfyUI

Three quantized builds of `lodestones/MOSS-Image-SFT-8-steps-dmd` (the 8-step distillation of `OpenMOSS-Team/MOSS-Image-SFT`), in
ComfyUI's own quantization formats. No custom node: the stock `UNETLoader` reads the per-layer metadata. The text encoder
(`qwen3.5_9b_int8_convrot`) and the VAE (`flux2-vae`) are the official ones and are not part of this repo.

| | BF16 | INT8 ConvRot (official) | FP8 | NVFP4 | **W4A16** |
|---|---|---|---|---|---|
| file | 9.5 GiB | 5.0 GiB | 5.04 GiB | 3.10 GiB | **3.38 GiB** |
| peak VRAM, 1024², 8 steps | 20.3 GB | 15.8 GB | 15.8 GB | 13.9 GB | 14.4 GB |
| time per image, RTX 3090 | 5.26 s | 3.46 s | 5.34 s | 5.45 s | 7.34 s (+40 %) |
| PSNR vs BF16 (6 images) | — | 11.1–15.9 dB | 17.3–25.9 dB | 12.3–16.2 dB | 15.4–17.3 dB |

Files: `moss_image_sft_dmd8_fp8.safetensors`, `moss_image_sft_dmd8_nvfp4.safetensors`, `moss_image_sft_dmd8_w4a16.safetensors`, each
with its `.quant.json` quantizer log. 168 linear layers are quantized (qkv, o, ffn up/gate, ffn down of the 42 blocks); 211 tensors are
kept as in the source.

## Read this before choosing

- **These save memory, not time.** On an RTX 3090 (sm86, no native fp8/fp4) FP8 and NVFP4 are emulated by dequantizing, and W4A16
  dequantizes the weights to BF16: none is faster than BF16, and W4A16 is 40 % slower. The only faster one is the official INT8 (not here).
- **FP8 and NVFP4 are not weight-only.** In ComfyUI both formats also quantize the activations, with a dynamic scale when the file has no
  `input_scale` (these have none). They are W8A8-fp8 and W4A4-fp4 with dynamic activation scale. Only W4A16 keeps bf16 activations.
- W4A16 carries `full_precision_matrix_mult` in each layer's metadata, which it needs.
- My review (one reviewer, six images per arm): INT8 and FP8 are good; **W4A16 looks very good**, on a par with INT8; I have no verdict
  on NVFP4. One W4A16 render has hair passing through a hat that the BF16 render of the same seed does not, so W4A16 is not lossless.

## What they look like

Three prompts, seed 1030 (a fisherman on a pier at dawn; a neon "OPEN 24/7" ramen sign in rain; a Mediterranean alley with a red bicycle),
8 steps, euler/simple, cfg 1.0. Columns: BF16, INT8 (official), FP8, NVFP4, W4A16.

![](images/moss_comparison_all.jpg)

## What I measured, and what I did not

- 6 renders per arm (the first also warms the server, so timings use the other five), RTX 3090 (sm86), one server per arm, ComfyUI v0.43.2,
  comfy-kitchen 0.2.40.
- **PSNR is not quality here**: an 8-step distilled model reroutes the sampler under any perturbation and arrives at a different but good
  image, so the ordering by PSNR does not say which is better (the official INT8 has the lowest PSNR of the group).
- No calibration: FP8 per-tensor scale, NVFP4 and Q4_1 (group 32) by min/max. No AWQ, no GPTQ. A calibrated build could be better.
- Not tested: other GPUs (on Ada/Hopper/Blackwell the fp8/nvfp4 speed picture changes), other text encoders, resolutions other than 1024².

Tools and the measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench** (`tools/quant_fp.py`, profile `moss`).

License: Apache-2.0, as the source model.
