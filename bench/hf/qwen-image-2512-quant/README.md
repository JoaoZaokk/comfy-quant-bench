---
license: apache-2.0
base_model: Qwen/Qwen-Image
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-image
  - quantized
  - int4
  - convrot
  - w4a8
language:
  - en
---

# Qwen-Image 2512 — 4-bit weights at 3.53x, and the third family where 4-bit *activations* fail

`asym_w4a8_int8` build of `qwen_image_2512`, measured against the BF16 original and against a pure
W4A4 build of the same source.

**38.05 GiB → 10.79 GiB, 3.53x lighter and 3.65x faster per step**, with output this bench cannot
separate from the original on a contact sheet. The W4A4 build is published **as a measured
negative**, not as a file to use.

This is the **non-edit** half of the Qwen-Image family. The editor lives in its own repo:
[Qwen-Image-Edit-2511-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot).

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The measurement

12 runs (6 prompts × 2 seeds), 1024 px, 20 steps, cfg 2.5, euler/simple, one RTX 3090. **The
transformer is the only thing that changes** — same encoder (`qwen_2.5_vl_7b`), same VAE, same
prompts, same seeds.

| arm | GiB | s/step | divergence from BF16 | min–max | picture |
|---|---|---|---|---|---|
| BF16 original | 38.05 | 5.722 | — | — | sharp |
| **this file, W4A8** | **10.79** | **1.566** | **0.3875** | 0.2023–0.7235 | **indistinguishable on a contact sheet** |
| W4A4, same source | 9.60 | 1.038 | 1.3369 | 1.1018–1.5449 | **buried in coloured speckle** |

![three arms](images/tres_bracos.png)

Four prompts, one seed. Left BF16, middle this build, right W4A4. The middle column tracks the left
one — apple, face, OPEN sign, night market. The right column still shows the subject and is
unusable.

The BF16 arm's 5.722 s/step is **not a fair kernel comparison**: 38.05 GiB does not fit a 24 GB
card, so it streams weights every step. The two quantized arms are resident. What the column does
say honestly is the thing a user feels — the smaller file finishes 3.65x sooner on this hardware.

## Third family, third failure mode, one axis

W4A4 and W4A8 carry the **same 4-bit weights**. Only the activation path differs. Across three
families on this bench:

| model | W4A4 result |
|---|---|
| Qwen-Image-Edit 2511 | pure static, divergence 1.7440 |
| Wan 2.2 TI2V 5B | blur, divergence 0.3847 |
| **Qwen-Image 2512 (this)** | **coloured speckle, subject still visible, divergence 1.3369** |

Three different-looking failures, one cause. **Weights survive four bits in every family measured
here; activations do not survive in any of them.** That is the transferable result, and it is
worth more than any single file in this repo.

Note also what the divergence numbers do *not* do: Wan's blur scored **0.3847**, lower than this
build's perfectly good **0.3875**. The metric is biased toward soft failure — blur is a small
latent change — so it cannot be used alone to accept a build. Only a render decides.

## The files

| file | bytes | GiB | layout |
|---|---|---|---|
| `qwen_image_2512_w4a8.safetensors` | 11,581,151,800 | **10.79** | 840 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256 |

Source `qwen_image_2512_bf16.safetensors`, 40,861,031,488 B, sha256 beginning
`d8c57b76f262e1ef27f5eca1`. The W4A4 build carries the same source hash, so the two arms are
comparable by construction.

Calibrated on this checkpoint, **not** on the Edit's analysis. The two models differ by 72 bytes in
file size and share the MMDiT architecture, so reusing would have saved an hour of GPU — and
`quant_mixed` refuses foreign analyses by default, correctly: different weights give different
per-layer errors, and reusing would pick formats from measurements of another model.

Backend recorded in the sidecar: `comfy_kitchen.backends.cuda` on all four ops.

## What is NOT covered

- **12 runs, 6 prompts, 2 seeds, one resolution, one sampler, one card.**
- **"Indistinguishable on a contact sheet" is a human verdict**, not a perceptual metric. No LPIPS,
  no FID, no user study. Judge the sheet yourself.
- **The text encoder is not in this repo** and is shared with the Edit model; a `convrot_w4a4`
  build of `qwen_2.5_vl_7b` exists separately on this account.
- **The VAE is not quantized and will not be** — 254 MiB against a 38 GiB transformer.
- `convrot_groupsize` **256 only**; no mixed build was tried. The choice here was the clean single
  axis, all-W4A4 against all-W4A8, not a search for an optimum between them.
- **The W4A4 weights are not published.** Its contact sheet is, so the claim is checkable.

## Credits

- **[Qwen](https://huggingface.co/Qwen) (Alibaba)** — Qwen-Image itself. Everything here is a
  re-encoding of their weights.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)** — the repackaged BF16
  single-file weights this was converted from, and the `qwen_image` layer profile this bench uses,
  which was derived from their own `int8_convrot` build of the Edit model and matches it 840/840
  with no slack either way.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format and the CUDA kernels that run it.
