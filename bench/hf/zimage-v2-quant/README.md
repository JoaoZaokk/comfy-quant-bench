---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-image
  - quantized
  - int4
  - int8
  - convrot
  - w4a4
language:
  - en
---

# Beyond Reality Z-Image v2 — ConvRot quantized, and the model where 4 bits actually works

Quantized builds of the `Beyond_Reality Z-Image v2` fine-tune in ComfyUI's native `convrot_w4a4` and
`asym_w4a8_int8` formats. **Both files here are usable** — which makes this repo the counterpart to
the two next to it, where the same format and the same converter destroyed the output.

**11.46 GiB → 3.06 GiB, 3.74x lighter**, and *faster* per step than the BF16 source.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The files

| file | median effective error | format | GiB | latent divergence | s/step | verdict |
|---|---|---|---|---|---|---|
| `zimage-v2-w4a4.safetensors` | — | 170 × `convrot_w4a4` | 3.06 | 0.7854 | 0.585 | good |
| `zimage-v2-mixed.safetensors` | **0.1241** | 115 × 4-bit / 55 × 8-bit | 3.18 | 0.7216 | 0.637 | good |

Source: 11.46 GiB BF16, 8 steps, 1024×1024, two seeds. "Effective error" is, per layer, the measured
relative error of the format that layer *actually* received, on the real activations it saw during
sampling.

Pure W4A4 is 4% smaller and 8% faster per step; mixed sits closer to the reference latent. **Neither
of those facts tells you which picture you prefer** — see the warning below.

### What they look like

| seed | BF16 reference | pure W4A4 | mixed |
|---|---|---|---|
| 1 | ![](images/seed1_bf16_referencia.png) | ![](images/seed1_w4a4_puro.png) | ![](images/seed1_misto.png) |
| 2 | ![](images/seed2_bf16_referencia.png) | ![](images/seed2_w4a4_puro.png) | ![](images/seed2_misto.png) |

---

## Speed

Re-measured over three seeds after an earlier single-run number was found to have the sign inverted:

**ConvRot W4A4 is 1.83x–1.93x faster per step than the BF16 source** on an RTX 3090 at 1024×1024.

That is the opposite of what the same format, the same converter and the same kernel do to
HunyuanVideo 1.5, where W4A4 is slower *and* destroys the picture. The format is not the variable.
The model is.

---

## Why this model is the good case

Measured across three architecture families, the per-layer error a model tolerates before its output
breaks is **not a property of the format**:

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| **Z-Image v2** | **~6 B** | **0.1241** | ***not measured*** |
| HunyuanVideo 1.5 family | ~13 B | 0.1837 | 0.2147, and 0.2163 on `capybara_v0.1` |

Monotone in the tolerated column. Three families make that a hypothesis, not a law.

**Z-Image's upper bound is honestly empty**, and that gap is worth stating rather than papering
over: what is known is that 0.1241 works and that pure W4A4 works. Where it stops working has not
been measured on this model. An earlier version of this table filled that cell with 0.2163, credited
to `capybara_v0.1` as a Z-Image checkpoint. Read from the file, `capybara_v0.1` carries 1364 tensors
and 54 `double_blocks` — HunyuanVideo 1.5's architecture — against Z-Image's 453 tensors and zero.
The evidence was real; it was in the wrong row.

### Do not use a generated image to compare two quantizations of this model

At 8 steps a tiny perturbation reroutes the sampler, and the destination is still a good image. Three
seeds, comparing pure W4A4 against a mixed build free-running: **2–1**, with both arms sitting
0.3–0.5 from BF16. That comparison measures chaos, not fidelity.

What does work is a **matched-input** comparison: record every `(x, timestep)` the BF16 arm was
called with, then replay exactly those into the quantized arm, so trajectory divergence cannot exist
by construction. Under that instrument the error is concentrated at high sigma — 6.17e-1 at
sigma 1.000 falling to 5.31e-2 at sigma 0.300 — so quantization damage lands hardest where structure
is decided and decays into texture.

Latent divergence is a distance, not a verdict. On HunyuanVideo 0.8255 was destroyed and 0.7173 was
fine: a 15% gap separating unusable from shippable, so no cut on that axis decides anything.

---

## Verified, not assumed

The converter hard-refuses to run unless both `quantize_convrot_w4a4_weight` and
`convrot_w4a4_linear` resolve to `comfy_kitchen.backends.cuda`, because the eager backend declares
the same capabilities and would silently produce numbers describing dequantized math. Dispatch is
separately counted on a real load and a real forward: **340 quantized modules, 0 `dequantize`
calls** on this model.

Built and verified with:

```
comfy-kitchen  0.2.31        ComfyUI  c1739380 (0.33.0)
torch          2.13.0+cu130  CUDA     13.0
GPU            RTX 3090 (sm_86)
```

Native INT4 MMA requires `major == 8` (Ampere / Ada). Hopper and Blackwell are routed to an INT8
branch deliberately.

**These files are remapped to ComfyUI's own module names** and are not diffusers-named. That step is
not optional for Z-Image: ComfyUI fuses `attention.to_{q,k,v}` into `attention.qkv` at load and only
`.weight` is in that map, so `weight_scale` and `comfy_quant` would pass through unrenamed and the
layer would load with no scale **and no error**. The remap is derived from ComfyUI's own
`z_image_to_diffusers` table and its output is bit-identical in the latent.

---

## Format

Standard safetensors, ComfyUI-native, mixed precision in one file — the format's own behaviour, not
a trick played on it. Per quantized layer, `<layer>.weight` as an INT8 container holding packed
signed INT4 plus `<layer>.weight_scale` as FP32, and a per-layer entry in
`__metadata__._quantization_metadata` that ComfyUI turns into a `<layer>.comfy_quant` tensor at load
and dispatches on individually. `convrot_groupsize` 256. Every non-quantized tensor is preserved
byte for byte from the source and verified as such. The `.quant.json` sidecars carry full conversion
provenance.

Pair it with the quantized text encoder if you are tight on VRAM:
[Qwen3-4B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen3-4B-W4A4-ConvRot).

---

## Not covered

One prompt, two seeds for the image ladder, 1024×1024, 8 steps, one scheduler, one card. No
perceptual metric — "good" is the judgement of someone who looked. No SASS. Only
`convrot_groupsize` 256 here. BF16 is the reference, not ground truth: it was never itself validated
against float32. And the upper bound of what this model tolerates is unmeasured, as the table above
says.

---

## Credits

- **Nurburgring** — author of the [`Beyond Reality`](https://civitai.com/models/1090420/beyond-reality)
  Z-Image fine-tune these files are derived from, released under Apache 2.0, and mirrored at
  [Nurburgring/BEYOND_REALITY_Z_IMAGE](https://huggingface.co/Nurburgring/BEYOND_REALITY_Z_IMAGE).
  This repo is a quantization of their work and would not exist without it.
- **Tongyi-MAI / Alibaba** — [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
  Apache 2.0, the base model underneath the fine-tune.
- **tonera** — [Beyond_Reality_Zimage_v2_svdq](https://huggingface.co/tonera/Beyond_Reality_Zimage_v2_svdq),
  an independent SVDQuant/Nunchaku quantization of the same fine-tune. Different method, same model;
  worth comparing against what is here.
- **Comfy-Org / comfyanonymous and the ComfyUI contributors** — the `QuantizedTensor` / `Layout` /
  `MixedPrecisionOps` model this format plugs into, and the `z_image_to_diffusers` table the remap is
  derived from.
- **comfy-kitchen** — the ConvRot W4A4 and W4A8 CUDA kernels.
- Quantized by [JoaoZaokk](https://huggingface.co/JoaoZaokk) with
  [comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench).
