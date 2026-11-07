---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: PrunaAI/P-H3-Image-Comfy
base_model_relation: quantized
library_name: diffusion-single-file
pipeline_tag: text-to-image
tags:
  - comfyui
  - text-to-image
  - image-editing
  - quantized
  - int4
  - w4a16
---

# P-H3-Image — W4A16 for ComfyUI

The 4-bit weights of Pruna AI's `p-h3-image_w4a8` with the activations switched back to 16 bits
(`full_precision_matrix_mult`). Same weights, byte for byte; only the per-layer configuration inside the file changes.
Needs **ComfyUI v0.44.0 or newer** (P-H3 support, `PH3ImageConditioning`, `MiniMaxH3SigmaShift`). The text encoder, VAE and workflows
are Pruna AI's (`PrunaAI/P-H3-Image-Comfy`); only the diffusion model is here.

> **Territory restriction — read `LICENSE`.** The MiniMax H3 Community License Agreement grants rights **only outside the
> European Union, the United Kingdom, the Republic of Korea and the United States of America**. If you are in one of them, this
> repository is not licensed to you. The Acceptable Use Policy (Exhibit A) and the commercial terms (Section IV) also apply.
> This file was **modified by JoaoZaokk** and is not an official product of MiniMax or of Pruna AI. See `NOTICE`.

| | INT8 ConvRot (Pruna AI) | W4A8 (Pruna AI) | **W4A16 (this repo)** |
|---|---|---|---|
| file | 19.8 GiB | 12.44 GiB | **12.44 GiB** |
| time per step, 4 MP (2048²) | 1.26 s | 1.31 s | 2.12 s (**+62 %**) |
| time per step, 8 MP | 3.26 s | 3.31 s | 4.61 s (**+39 %**) |
| peak VRAM | 22.7 GB | 22.0 GB | 22.1 GB |
| PSNR vs INT8 (6 images) | — | 15.1–17.6 dB | 14.5–17.2 dB |

RTX 3090, 8 steps, no CFG, INT8 text encoder and VAE, prompt upsampler bypassed.

## Read this before downloading

**This build is not better than the W4A8 it comes from.** Same 4-bit weights and the same memory use; it is 39–62 % slower on an RTX 3090
(the weights are dequantized and multiplied in 16 bits) and its PSNR against INT8 is no closer. It exists to answer one
question — does keeping activations in 16 bits help a 4-bit P-H3? — and the answer in these six images is no. It
runs; I did not find a reason to prefer it. My own review: the model is fine, its style is not mine.
The BF16 file (40 GB) does not fit in 24 GB, so the reference here is the INT8 build, not BF16.

## What they look like

Three prompts, 4 MP (2048²), seed 1030 (a fisherman on a pier at dawn; a neon "OPEN 24/7" ramen sign in rain; a Mediterranean
alley with a red bicycle). INT8 on the left.

![](images/ph3_comparison_4mp.jpg)

## What I measured, and what I did not

- 6 renders per arm (4 MP and 8 MP; the first also warms the server), RTX 3090 (sm86), one server per arm, ComfyUI v0.44.1.
- PSNR is not quality; it orders the arms. Visual review: one person, six images.
- Not tested: BF16 (does not fit), image editing and multi-image modes, the LoRA, other GPUs.

Tools and the measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench** (`tools/quant_weight_only.py`, which now
also handles the inline `.comfy_quant` layer configs this model uses).
