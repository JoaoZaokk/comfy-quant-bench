---
license: gemma
base_model: DreamFast/gemma-3-12b-it-heretic
tags:
  - comfyui
  - quantized
  - w4a8
  - text-encoder
  - convrot
library_name: comfyui
---

# Gemma 3 12B it Heretic — W4A8 (asym_w4a8_int8) for ComfyUI

4-bit weights with an 8-bit activation path, in ComfyUI's native per-layer quantization format.
**21.93 GiB → 7.53 GiB, 2.91x lighter**, and on the path ComfyUI takes by default it costs **no
extra time at all**.

    source          gemma_3_12B_it_heretic.safetensors     23,545,681,250 B
    this file       gemma_3_12B_it_heretic_w4a8             8,089,619,138 B
    quantized       336 layers, asym_w4a8_int8, group_size 16
    preserved       293 tensors, byte-identical to the source

Converted with `tools/quant_w4a8.py` on an RTX 3090 (sm86), comfy-kitchen 0.2.31,
torch 2.13.0+cu130, backend `comfy_kitchen.backends.cuda`. 25.5 s.

## What it costs, measured against the BF16 original

Three arms on the same prompts, same card, same day. **A** = BF16 original, **B** = this file
loaded the way ComfyUI loads it, **C** = this file with ComfyUI's two text-encoder locks
released.

    quantizing the WEIGHT already costs      1.1753e-1   (B against A)
    releasing the locks ADDS                 1.8393e-1   (C against B)
    total for whoever releases them          2.0684e-1   (C against A)
    cosine with BF16, prompt 0               0.9967 locked / 0.9780 released

## The memory is free; the speed needs the locks released

    encode, prompt 0, median of 5      BF16 1876.8 ms   locked 1835.7 ms   released 490.8 ms

**Locked is the same time as BF16** (1835.7 against 1876.8), so 13.8 GiB of VRAM are saved at no
cost in seconds. The 3.74x needs the locks released.

Stock ComfyUI keeps them shut, and it takes both: `comfy/sd.py` calls
`set_model_compute_dtype(torch.float32)` for every CLIP object, which turns on
`comfy_force_cast_weights`, and `comfy/sd1_clip.py` hardcodes `full_precision_mm=True` for every
text encoder. Either one alone sends the math down the dequantized path. Counted on a real forward
through the normal loader: **0 quantized forwards, 336 `dequantize` calls.** Memory saved, kernel
never reached.

That is a property of being a *text encoder* in ComfyUI, not of this file or of W4A8 — the same
count comes out of W4A4 encoders from other authors.

> **Correction, 2026-09-12.** The first version of this card said the released path was
> **"reachable only by a post-load monkeypatch"** and that **"ComfyUI offers no way to do that"**.
> That was true when it was written and is no longer true. Both locks are now released at the
> source by a patch kept in this project
> (`patches/comfyui_text_encoder_quantized_math.patch`), gated behind a new
> `--disable-quantized-text-encoder` flag so the old behaviour stays one flag away, and applied
> only to a checkpoint that actually carries quantized layers — a non-quantized encoder still gets
> the float32 upcast, verified by a control arm. The release fires the kernel where the monkeypatch
> did: the same two conditions end up false.
>
> **The numbers above were measured through the monkeypatch, not through the flag**, and are left
> unchanged rather than silently restated, because a number acquires the path it was measured on. A
> re-measurement through the released build is pending; when it lands it replaces this block, not
> the table.
>
> Two earlier attempts at this release were wrong and were caught by a forward counter rather than
> by reading: writing `model_options["quantization_metadata"]` never reaches `sd.py` because every
> text-encoder family copies `model_options` first, and walking the tree for `_quant_config` finds
> nothing because that attribute lives on the outer class, not on the nested `Linear`. Releasing
> only one of the two locks leaves the count at exactly zero, which looks identical to changing
> nothing.

## Not covered

One model, one card (sm86), one prompt set. **No perceptual metric and no image was generated**:
these numbers measure the CONDITIONING, not the picture. Nothing here says whether releasing the
locks is safe for output quality — it buys speed and costs fidelity, and which side of that trade
is right depends on the workflow.
