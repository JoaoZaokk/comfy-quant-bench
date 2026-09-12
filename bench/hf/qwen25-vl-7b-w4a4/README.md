---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
  - comfyui
  - quantized
  - w4a4
  - convrot
  - text-encoder
library_name: comfyui
---

# Qwen 2.5-VL 7B — ConvRot W4A4 for ComfyUI

4-bit weights and a 4-bit activation path (Hadamard-rotated), in ComfyUI's native per-layer
format. **15.45 GiB → 6.34 GiB, 2.44x lighter.**

This is the text encoder of the **Qwen-Image / Qwen-Image-Edit** chain. The quantized transformer
that pairs with it is measured separately.

**Read the fidelity number before you use this.** Quantizing the weight costs **3.3471e-1**
against the BF16 original — on the same bench, on the same day, a W4A8 Gemma 3 12B measured
**1.1753e-1**. This file is **2.8x less faithful** than that one. It is published because a
measured negative is worth more than an unpublished one, not because it is recommended.

    source          qwen_2.5_vl_7b.safetensors        16,584,415,576 B
    this file       qwen_2.5_vl_7b_w4a4_convrot        6,802,084,504 B
    quantized       196 layers, convrot_w4a4, convrot_groupsize 256
    preserved       533 tensors, byte-identical
    excluded        embed_tokens, norms, lm_head, and the whole vision tower

Converted with `tools/quant_w4a4.py --profile qwen` on an RTX 3090 (sm86), comfy-kitchen 0.2.31,
torch 2.13.0+cu130, backend `comfy_kitchen.backends.cuda`. 19.9 s.

## What it costs, measured against the BF16 original

**A** = BF16 original, **B** = this file as ComfyUI loads it, **C** = this file with the two
text-encoder locks released.

    quantizing the WEIGHT already costs      3.3471e-1   (B against A)
    releasing the locks ADDS                 3.7293e-1   (C against B)
    total for whoever releases them          5.0087e-1   (C against A)
    cosine with BF16, prompt 0               0.9557 locked / 0.8911 released

## Time

    encode, prompt 0, median of 5      BF16 108.1 ms   locked 107.3 ms   released 57.4 ms

Locked costs the same as BF16, so 9.1 GiB are saved for free. The 1.87x needs the locks released.
Stock ComfyUI keeps them shut, and it takes both: `comfy/sd.py` and `comfy/sd1_clip.py` each
independently route a text encoder's math to the dequantized path.

The prompts here are short (18-22 tokens), which is the regime where the 4-bit kernel's per-layer
fixed cost is worst amortised; on another encoder on this bench the crossover sat between 75 and
199 tokens, and above it the quantized encoder beats the BF16 original outright.

> **Correction, 2026-09-12.** The first version of this card said the released path was
> **"reachable only by a post-load monkeypatch"** and that **"ComfyUI offers no way to do that"**.
> True when written, false now. Both locks are released at the source by a patch kept in this
> project (`patches/comfyui_text_encoder_quantized_math.patch`), behind a new
> `--disable-quantized-text-encoder` flag that restores the old behaviour, and applied only to a
> checkpoint that actually carries quantized layers — a non-quantized encoder still gets the
> float32 upcast, verified by a control arm that must come out unchanged.
>
> **The numbers above were measured through the monkeypatch, not through the flag**, and are left
> as they were rather than silently restated: a number carries the path it was measured on. A
> re-measurement through the released build is pending and will replace this block.
>
> One trap worth repeating, because it cost a run here: writing the attribute on the modules does
> not survive, since `model_patcher.py` rewrites `comfy_force_cast_weights` on **every** load to
> GPU. The same command line produced 350 kernel calls on one run and 0 on the next, depending on
> whether the model happened to be resident already.

## Not covered

One model, one card (sm86), short prompts, no perceptual metric, **no image generated** — this
measures the CONDITIONING, not the result. And the accuracy cost is not a property of the format:
two W4A4 encoders measured on this same bench differ by 6x in what releasing adds, so "W4A4 costs
X" cannot be quoted without naming the checkpoint.
