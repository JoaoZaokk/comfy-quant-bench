---
license: other
license_name: flux-kommunity-license
license_link: LICENSE.md
base_model: black-forest-labs/flux-3-image-klein-comfy
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
  - w4a4
  - convrot
---

# FLUX.3 klein — FP8, NVFP4, W4A16 and W4A4 for ComfyUI

Four quantized builds of the FLUX.3 klein transformer (`black-forest-labs/flux-3-image-klein-comfy`, `flux3-image-klein-bf16`),
in ComfyUI's own quantization formats. No custom node: the stock `UNETLoader` reads the per-layer metadata.
The text encoder and the VAE are **not** in this repo; use Black Forest Labs' files.

> **License: FLUX Kommunity License (non-commercial).** These files are Derivatives of the FLUX Model, **modified by
> JoaoZaokk**, not an official product of Black Forest Labs and not endorsed by them. Read `LICENSE.md` and `NOTICE.md`
> before use. Any rights to use the files are granted to you directly by Black Forest Labs under that license.

| file | format | size | peak VRAM | time per image, RTX 3090 | PSNR vs BF16 (6 images) |
|---|---|---|---|---|---|
| BF16 (BFL, reference) | bf16 | 10.3 GiB | 16.5 GB | 7.62 s | — |
| INT8 ConvRot (BFL, not in this repo) | int8 | 5.4 GiB | 11.5 GB | 5.23 s | 20.5–31.3 dB |
| NVFP4 (BFL, not in this repo) | nvfp4, static act scale | 4.9 GiB | 11.1 GB | 7.96 s | 16.3–24.9 dB |
| `flux3-image-klein_fp8.safetensors` | fp8 e4m3, per-tensor | **5.42 GiB** | 11.5 GB | 7.95 s | 24.4–30.7 dB |
| `flux3-image-klein_nvfp4.safetensors` | nvfp4, dynamic act scale | **3.27 GiB** | 9.4 GB | 8.12 s | 15.5–23.2 dB |
| `flux3-image-klein_w4a16.safetensors` | Q4_1 group 32, bf16 activations | **3.58 GiB** | 9.9 GB | 11.85 s (+55 %) | 18.6–22.2 dB |
| `flux3-image-klein_w4a4.safetensors` | ConvRot W4A4 | **2.97 GiB** | 9.0 GB | **4.16 s (−45 %)** | 10.6–18.9 dB |

1024², 16 steps, euler/simple, cfg 1.0, same INT8 text encoder and VAE in every run.

## Read this before choosing

- **Only INT8 and W4A4 are faster than BF16 on an RTX 3090.** FP8 and NVFP4 have no native kernel on Ampere (sm86): ComfyUI
  emulates them by dequantizing, so they save memory but not time (+4 to +6 %). W4A16 is slower still (+55 %).
- **FP8 and NVFP4 here are not weight-only.** In ComfyUI both formats also quantize the activations; with no `input_scale`
  in the file the scale is dynamic. So these are W8A8-fp8 and W4A4-fp4 with dynamic activation scale. The official BFL NVFP4
  uses a static, calibrated `input_scale` and keeps some blocks in bf16 (hence its larger file). Only W4A16 keeps bf16 activations.
- **W4A16 needs `full_precision_matrix_mult`**, already set in each layer's metadata. Without it the quantized path of
  comfy-kitchen 0.2.40 with group 32 returned NaN (black images) in my tests.
- **W4A4 is the only one that breaks the picture.** My own review: hands and structure fall apart, the same pattern I see in
  every W4A4 I have built; it may only be usable for landscape-like content. It is published for completeness and for the speed.
- **NVFP4 (this repo) vs the official NVFP4:** in my review the official one has an "artificial" look, while this one follows
  the BF16 style. One reviewer, six images: take it as an observation, not a measurement.

## What they look like

Three prompts, seed 1030 (a fisherman on a pier at dawn; a neon "OPEN 24/7" ramen sign in rain; a Mediterranean alley with
a red bicycle). BF16 on the left in both sheets.

![](images/flux3_comparison_a.jpg)

![](images/flux3_comparison_b.jpg)

## What I measured, and what I did not

- 6 renders per arm (the first also warms the server, so timings use the other five), RTX 3090 (sm86), one server per arm.
- PSNR against the BF16 render of the same prompt and seed. **PSNR is not quality**: it orders the arms, and different
  quantizations sit on a floor of about 28 dB from sampler chaos.
- No calibration of any kind in the files of this repo: fp8 per-tensor scale, nvfp4 and Q4_1 by min/max, ConvRot W4A4 as in
  my other releases. All 258 linear layers of the transformer are quantized (the BFL NVFP4 keeps some in bf16, so file sizes
  are not like for like).
- Not tested: other GPUs (on Ada/Hopper/Blackwell the fp8/nvfp4 speed picture changes), image editing, other resolutions.

Tools and the measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench** (`tools/quant_fp.py`,
profile `flux3`).
