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
  - int4
  - w4a16
  - weight-only
language:
  - en
---

# MOSS-Image SFT (8-step DMD) — W4A16 for ComfyUI

Weight-only 4-bit build of `lodestones/MOSS-Image-SFT-8-steps-dmd` (itself the 8-step distillation of
`OpenMOSS-Team/MOSS-Image-SFT`), in ComfyUI's own `awq_w4a16` format. Weights are 4-bit (Q4_1, group 32,
asymmetric); **activations stay BF16**. No custom node: the stock `UNETLoader` reads the per-layer metadata.

| | BF16 | INT8 ConvRot (official) | **W4A16 (this repo)** |
|---|---|---|---|
| file | 9.5 GiB | 5.0 GiB | **3.4 GiB** (3.62 GB) |
| peak VRAM, 1024², 8 steps | 20.3 GB | 15.8 GB | **14.4 GB** |
| time per image, RTX 3090 | 5.26 s | 3.46 s | 7.34 s |

**Read this before downloading: it saves memory, not time.** On Ampere (RTX 3090) the 4-bit weights are
dequantized to BF16 per layer and multiplied in BF16, so W4A16 is **+40 % slower** than BF16 and 2.1× slower than
INT8. Pick it when VRAM is what you lack; pick INT8 when speed is.

## File

| file | what |
|---|---|
| `moss_image_sft_dmd8_w4a16.safetensors` | 168 linear layers in `awq_w4a16` (qkv, o, ffn up/gate, ffn down of every block), 211 tensors kept as in the source |
| `moss_image_sft_dmd8_w4a16.quant.json` | quantizer log: relative weight error per layer (mean 0.0798, max 0.0838) |

`full_precision_matrix_mult` is already set in each layer's metadata. It is required: without it the
quantized path of comfy-kitchen 0.2.40 returned NaN (black images) with group 32 in my tests on other models.
Tested on ComfyUI v0.43.2 with comfy-kitchen 0.2.40, with the official `qwen3.5_9b_int8_convrot` text encoder
and the `flux2-vae` (neither is part of this repo). Put the file in `models/diffusion_models`.

## What they look like

Three prompts, seed 1030, 8 steps, euler/simple, cfg 1.0, 1024², same text encoder and VAE in every column.

![](images/moss_comparison.jpg)

The prompts, top to bottom: a fisherman mending a net on a pier at dawn; a neon "OPEN 24/7" ramen sign in rain;
a Mediterranean alley with a red bicycle.

## What I measured, and what I did not

- 6 renders per arm (3 prompts at seed 1030, plus a seed-41 render that also warms the server and two extra seeds; timings use the other five), RTX 3090 (sm86), one server per arm.
- PSNR against the BF16 render of the same prompt and seed: W4A16 15.4–17.3 dB, INT8 11.1–15.9 dB. **PSNR is not a
  quality measure here**: an 8-step distilled model reroutes the sampler under any perturbation and arrives at a
  different but good image, so the ordering by PSNR does not say which is better.
- The visual review was mine alone, on the six images per arm. My verdict: the W4A16 renders look very good, on a par
  with INT8. One render shows an artefact (hair passing through a hat) that is not in the BF16 render
  of the same seed, so W4A16 is not lossless. Six images do not rule out other artefacts.
- No calibration: Q4_1 min/max per group of 32, no AWQ, no GPTQ. A calibrated build could be better.
- Not tested: other GPUs, other text encoders, resolutions other than 1024².

Method and measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench** (`tools/quant_fp.py`, format `w4a16`,
profile `moss`).

License: Apache-2.0, as the source model.
