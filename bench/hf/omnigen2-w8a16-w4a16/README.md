---
license: apache-2.0
base_model: OmniGen2/OmniGen2
base_model_relation: quantized
library_name: diffusion-single-file
pipeline_tag: text-to-image
tags:
  - comfyui
  - text-to-image
  - quantized
  - int8
  - int4
  - w8a16
  - w4a16
  - weight-only
language:
  - en
---

# OmniGen2 DiT — W8A16 and W4A16 for ComfyUI

Weight-only builds of the OmniGen2 transformer (ComfyUI's fp16 repack of `OmniGen2/OmniGen2`). Activations stay FP16.
No custom node: the stock `UNETLoader` reads the per-layer metadata.

| | FP16 (source) | **W8A16** | **W4A16 (group 40)** |
|---|---|---|---|
| file | 7.4 GiB | **4.05 GiB** | **2.71 GiB** |
| peak VRAM, 1024², 20 steps | 16.8 GB | **13.4 GB** | **12.0 GB** |
| seconds per step, RTX 3090 | 1.42 | **1.43** | 1.77 (+25 %) |
| PSNR vs FP16 (6 images) | — | 24.5–30.3 dB | 16.1–24.2 dB |

**W8A16 is the one to use**: same speed as FP16, 3.4 GB less VRAM, 45 % smaller file. W4A16 saves another 1.4 GB of
VRAM but is 25 % slower and drifts further from the FP16 image. In my own review OmniGen2 looks dated next to newer
models and I do not use it; the files are published for those who do.

## Why a quantizer of its own

The model width is 2520, and 2520 is not a multiple of 256 (ConvRot), of 32 (the Q4_1 group of the gguf format) or
of 16 (INT8 GEMM). The ready-made ConvRot/INT8 formats do not apply to most layers. So:

- **W8A16**: ComfyUI's `int8_tensorwise` with `full_precision_matrix_mult` (INT8 weight, scale per row, FP16 matmul).
- **W4A16**: ComfyUI's `awq_w4a16` (Q4_1) with **group 40**, the gcd of 2520 and 10240, so every linear layer fits.

266 linear layers are quantized (attention q/k/v/out and the three feed-forward matrices of the main, noise-refiner,
context-refiner and reference-image-refiner blocks); 316 tensors are kept as in the source. Mean relative weight error
of W4A16: 0.0834 (max 0.0893).

Pitfall found on the way: with the model in FP16, scales and zeros must be FP16. BF16 scales promote the product to
FP32 and the linear layer fails with a dtype mismatch.

## Files

| file | what |
|---|---|
| `omnigen2_fp16_w8a16.safetensors` | W8A16 |
| `omnigen2_fp16_w4a16_g40.safetensors` | W4A16, group 40 |
| `*.quant.json` | quantizer log |

Tested on ComfyUI v0.43.2, comfy-kitchen 0.2.40, with `qwen_2.5_vl_fp16` as text encoder (`CLIPLoader`, type `omnigen2`)
and DualCFG guidance (cfg 5.0 / 2.0), euler, simple scheduler, 20 steps. Put the files in `models/diffusion_models`.

## What they look like

Three prompts, seed 1030, same text encoder and VAE in every column.

![](images/omnigen2_comparison.jpg)

The prompts, top to bottom: a cat with a crown on a velvet throne; a red vintage bicycle against a stone wall in a
Mediterranean alley; a lighthouse in a storm.

## What I measured, and what I did not

- 6 renders per arm (the first also warms the server, so timings use the other five), RTX 3090, one server per arm, FP16 as reference.
- PSNR/SSIM against FP16 of the same prompt and seed (SSIM 0.89–0.97 for W8A16, 0.55–0.91 for W4A16). PSNR is not
  quality; it orders the arms. Between runs of different quantizations there is a floor of about 28 dB from sampler chaos.
- Text-to-image only. The image-editing and multi-image modes were not tested.
- No calibration (min/max per row or per group).

Method and measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench** (`tools/quant_omnigen2.py`).

License: Apache-2.0, as the source model.
