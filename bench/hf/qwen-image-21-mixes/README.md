---
license: other
license_name: qwen-research
license_link: LICENSE_QWEN_RESEARCH.txt
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-image
  - quantized
  - int4
  - int8
  - convrot
  - mixed-precision
language:
  - en
---

# W4A4 / INT8 ConvRot checkpoints for Qwen-Image 2.1, ranked by a blind review

A full INT8 ConvRot build and four mixed-precision builds of the Qwen-Image 2.1 diffusion transformer for ComfyUI. Each
mix keeps a share of its 192 linear layers in ConvRot W4A4 (4-bit weights and activations) and promotes the layers with
the largest 4-bit error to INT8 ConvRot; the mixes differ only in the error threshold, and so in how many layers stay in
4 bits.

**Non-commercial use only** (Qwen Research License, see below). These files are modified, quantized versions of the
original transformer weights; they are not the original model.

## Which one

| file | W4A4 / INT8 layers | size | bad scenes (blind, of 24) | KSampler, Sage | KSampler, FA2 |
|---|---|---:|---:|---:|---:|
| plain W4A4 (not published here) | 192 / 0 | 3.77 GB | 18 | 8.41 s | 9.72 s |
| `qwen_image_2.1_bf16_mixed_p015_int8.safetensors` | 120 / 72 | 4.61 GB | 6 | 9.25 s | 10.56 s |
| `qwen_image_2.1_bf16_mixed_p012_int8.safetensors` | 88 / 104 | 5.81 GB | 6 | 10.92 s | 12.07 s |
| **`qwen_image_2.1_bf16_mixed_p010_int8.safetensors`** | 68 / 124 | 6.55 GB | **0** | 11.79 s | 12.89 s |
| `qwen_image_2.1_bf16_mixed_p008_int8.safetensors` | 35 / 157 | 6.93 GB | **0** | 12.96 s | 13.80 s |
| **`qwen_image_2.1_bf16_int8_convrot.safetensors`** (full INT8) | 0 / 192 | 7.26 GB | never worst | 12.70 s | 14.19 s |

RTX 3090 at 280 W, 1024², 25 steps, euler/simple, median of 24 renders. "Sage" = SageAttention 2 with an fp16 PV
accumulator; "FA2" = `--use-flash-attention`.

- **For quality, use full INT8 ConvRot** (the build here is the one that was reviewed;
  [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) publishes its own `qwen_image_2.1_int8_convrot`, not
  measured here) — in a blind review it was never picked as the worst version and was repeatedly
  called "nearly identical" to BF16. With SageAttention it is as fast as p008.
- **p010 is the cheapest mix that never looked bad** in the review: about 7 % faster than full INT8 under Sage, 9 %
  under FA2. On the hardest scenes it can look slightly forced, as can p008.
- **p012 and p015 fail the same hard scenes** (motion blur, crowded interiors) in 6 of 24 scenes. p015 is the fast end if
  that is acceptable.

![Hard scenes, five versions](images/hard_scenes_five_versions.jpg)

## How they were ranked

Three blind rounds of 24 scenes (12 prompts × 2 seeds), one rater. Every version of a scene was shown side by side in a
per-scene random order with no labels, with a full-size flicker comparison; each image was graded good / OK / bad. The
round that compared these mixes:

| version | best | good | OK | bad |
|---|---:|---:|---:|---:|
| W4A4 | 0 | 0 | 6 | 18 |
| p015 | 0 | 11 | 7 | 6 |
| p012 | 0 | 13 | 5 | 6 |
| p010 | 0 | 17 | 6 | 0 |
| p008 | 3 | 16 | 5 | 0 |

p010 and p008 are not separable at n = 24 (Wilcoxon p 0.16); p010 beats p012 and p015 (p < 0.01).

**A warning about PSNR.** In the same study a step schedule (INT8 for the first 5 steps, W4A4 for the rest) scored 2 dB
*higher* PSNR against BF16 than p008, and was the worst version in 24 of 24 blind scenes. PSNR against the BF16 output
measures whether the composition stays the same, not whether the image is clean. These checkpoints are ranked by the
blind review, not by PSNR.

## How they were made

Mixes: `tools/quant_mixed.py --promote-format int8 --promote-error <0.08 | 0.10 | 0.12 | 0.15>`; full INT8:
`tools/quant_int8.py --convrot` (both from
[comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench)), on the BF16 transformer repackaged by Comfy-Org;
the mixes rank layers by a per-layer W4A4 error measured on calibration activations from real sampling runs. Each file has a `.quant.json`
sidecar with the source hash, the threshold and the format of every layer. Write-up:
[docs/kernel-optimization.md](https://github.com/JoaoZaokk/comfy-quant-bench/blob/main/docs/kernel-optimization.md).

## Use

Put the file in `ComfyUI/models/diffusion_models/` and load it with the stock **Load Diffusion Model** node, with the
Qwen-Image 2.1 text encoder and VAE from [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Measured with
comfy-kitchen's CUDA backend for ConvRot on an RTX 3090 (sm86); other GPUs were not tested.

## Licence

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co.,
Ltd. All Rights Reserved. A copy is in `LICENSE_QWEN_RESEARCH.txt` and the attribution notice in `NOTICE.txt`.
Non-commercial use only; commercial use needs a separate licence from the original authors. Built with Qwen.
