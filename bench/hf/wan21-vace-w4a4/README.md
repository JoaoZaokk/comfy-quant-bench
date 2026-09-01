---
license: apache-2.0
base_model: Wan-AI/Wan2.1-VACE-1.3B
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-video
  - quantized
  - int4
  - int8
  - convrot
language:
  - en
  - zh
---

# Wan 2.1 VACE 1.3B — ConvRot quantized, and the one that proved the threshold is per model

Three quantized builds of Wan 2.1 VACE 1.3B in ComfyUI's native `convrot_w4a4` / `asym_w4a8_int8`
formats. **One of them is good. Two of them are here on purpose as examples of what not to ship.**

This repo exists because this model refuted a rule the bench had published. Everything below was
**executed**, not inferred, and the prediction was written down *before* the measurement — and was
wrong.

Method, tools and the full log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The three files

| file | median effective error | 4-bit / 8-bit layers | GiB | verdict |
|---|---|---|---|---|
| `wan21-vace-13b-misto005.safetensors` | **0.0546** | 2 / 298 | 2.15 | **use this one** |
| `wan21-vace-13b-misto015.safetensors` | 0.0793 | 134 / 166 | 2.11 | structure returns, everything smeared — **unusable** |
| `wan21-vace-13b-w4a4.safetensors` | 0.1602 | 300 / 0 | 2.07 | **DESTROYED — AN EXAMPLE OF WHAT NOT TO DO** |

Source: 4.01 GiB FP16. The good build is **1.87x lighter**; the destroyed one saves another 3.7%
and costs you the picture. That is the whole point of publishing all three.

"Effective error" is, per layer, the measured relative error of the format that layer *actually
received*, on the real activations the layer saw during sampling.

### What they look like

Same prompt, same seed, same sampler, `vace_strength 0`, 25 steps, 33 frames, 480x480:

| | |
|---|---|
| FP16 reference | ![](images/seed1_fp16_referencia.png) |
| `misto005` — good | ![](images/seed1_misto005_bom.png) |
| `misto015` — unusable | ![](images/seed1_misto015_borrado.png) |
| pure W4A4 — destroyed | ![](images/seed1_w4a4_puro_DESTRUIDO.png) |

Second seed, reference against the good build:

| | |
|---|---|
| FP16 reference | ![](images/seed3_fp16_referencia.png) |
| `misto005` | ![](images/seed3_misto005_bom.png) |

---

## Why this model matters more than its size suggests

The bench had derived a pre-flight rule from two model families: median per-layer W4A4 error above
0.21 breaks, below 0.15 works, in between is correct-but-grainy. Wan 2.1 measures **0.1602** — the
middle band, which nothing had ever occupied — and the render is destroyed: no subject, no bench,
across three seeds. Not grainy. Gone.

Bracketing with mixed builds puts this model's line **between 0.0546 and 0.0793**:

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| **Wan 2.1 VACE** | **1.3 B** | **0.0546** | **0.0793** |
| Z-Image v2 | ~6 B | 0.1241 | *not measured* |
| HunyuanVideo 1.5 family | ~13 B | 0.1837 | 0.2147, and 0.2163 on `capybara_v0.1` |

**There is no threshold of the format. There is one per model**, and across these three families the
tolerated error grows monotonically with model size. The practical consequence is blunt:
`--promote-error 0.15`, a default chosen on a 6B model and carried everywhere since, writes a file
here that loads cleanly, dispatches natively, passes every structural check, and renders a smear.

Three points make that a hypothesis, not a law. The next model may break it the way this one broke
the last.

> **Correction, 2026-09-01.** An earlier version of this table credited the 0.2163 break to Z-Image,
> naming `capybara_v0.1` as a checkpoint in that family. It is not one. Read from the file,
> `capybara_v0.1` has 1364 tensors and 54 `double_blocks` — HunyuanVideo 1.5's architecture, not
> Z-Image's 453 tensors and zero. So that break belongs to the ~13B row, where it agrees with the
> 0.2147 measured on Hunyuan itself, and **Z-Image's upper bound has never been measured**: all that
> is known there is that 0.1241 works. The monotonicity in the tolerated column survives; one cell
> of evidence moved rows and one cell became honestly empty.

---

## The trap that cost four renders, and will cost you the same

**A Wan VACE checkpoint sampled without a VACE control node is destroyed by ComfyUI's own defaults
— with no error and no warning.** The FP16 reference, unquantized, came out as woven fabric in
every configuration tried (6 steps/1 frame, 25/33, cfg 6 and cfg 1, with and without
`ModelSamplingSD3 shift 8`).

| `vace_strength` | latent norm | result |
|---|---|---|
| **1.0** — ComfyUI's default when no VACE node is present | 607.6 | ![](images/armadilha_vace_strength_1.0.png) |
| **0.0** | 1543.2 | ![](images/armadilha_vace_strength_0.0.png) |

`WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) fills `vace_frames` with **zeros** when
no VACE conditioning is supplied, runs each block through `process_latent_in` — which subtracts the
latent format's mean, so **zero becomes non-zero** — concatenates an **all-ones** mask, and applies
the result at **full strength**. That is not "no control". It is a constant control at maximum.

If you use these files for plain text-to-video, you must drive `vace_strength` to 0, or use a real
VACE control node. This is a property of the base model's integration, not of the quantization: the
unquantized FP16 original fails identically.

---

## Speed, and why the sign flips

Same file, same card, only the batch size varied:

| frames | FP16 | quantized | |
|---|---|---|---|
| 1 | 0.204 s/step | 0.363 s/step | 1.78x **slower** |
| 33 | 0.870 s/step | 0.653 s/step | 1.33x faster |

The 4-bit kernel carries a per-layer fixed cost that a tiny batch cannot amortise. At a realistic
video workload it wins; at one frame it loses. Do not quote a single speed ratio for this format
without naming the batch size it was measured at.

---

## Verified, not assumed

`probe_quant_dispatch.py --forward-only` counts, on a real load and a real forward: **300 quantized
modules, 12/12 forwards running quantized math, 0 calls to `dequantize`**, `linear_dtype int4`,
implementation resolving to `comfy_kitchen.backends.cuda`, all quantized weights on `cuda:0`.

One counting trap worth knowing: loading these prints
`WARNING: unet unexpected [...comfy_quant]` for all 300 layers. It is **cosmetic** — the tensors are
consumed before that check runs and complained about afterwards. It is not a failed load.

Built and verified with:

```
comfy-kitchen  0.2.31        ComfyUI  c1739380 (0.33.0)
torch          2.13.0+cu130  CUDA     13.0
GPU            RTX 3090 (sm_86)
```

Native INT4 MMA needs `major == 8` (Ampere / Ada). Hopper and Blackwell are routed to an INT8
branch deliberately. The converter hard-refuses to run unless the kernels resolve to
comfy-kitchen's CUDA backend, because the eager backend declares the same capabilities and would
silently produce numbers describing dequantized math.

---

## Format

Standard safetensors, ComfyUI-native, mixed precision in one file — which is the format's own
behaviour, not a trick played on it. Per quantized layer, `<layer>.weight` plus `<layer>.weight_scale`,
and a per-layer entry in `__metadata__._quantization_metadata` that ComfyUI turns into a
`<layer>.comfy_quant` tensor at load and dispatches on individually. `convrot_groupsize` 256. Every
non-quantized tensor is preserved byte for byte from the source and verified as such. The
`.quant.json` sidecars record full conversion provenance.

The `vace_blocks` branch is **excluded from quantization by design** and stays FP16: it only runs
when a control input is present, so a calibration without control records nothing for it and the
converter would be guessing. Patch/VACE embeddings, `norm_q`/`norm_k`, `time_embedding`,
`text_embedding` and the head are excluded as always.

---

## Not covered

One prompt, three seeds (two for the `misto005` arm), 480x480, 33 frames, one scheduler, one card.
No perceptual metric — "good", "smeared" and "destroyed" are the judgement of someone who looked at
them. The checkpoint is **FP16**, and it is a **VACE variant driven as plain T2V**, which is not the
mode it was trained for: the tolerance measured here may belong to the mode rather than to the
model. The calibration sampled one frame; the renders used 33. No SASS. And the monotonicity in
model size rests on three points.

---

## Credits

- **Wan-AI / Alibaba** — [Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B),
  Apache 2.0. This is a derivative of their weights.
- **Comfy-Org / comfyanonymous and the ComfyUI contributors** — the `QuantizedTensor` / `Layout` /
  `MixedPrecisionOps` model this format plugs into, and `WAN21_Vace` itself.
- **comfy-kitchen** — the ConvRot W4A4 and W4A8 CUDA kernels.
- Quantized by [JoaoZaokk](https://huggingface.co/JoaoZaokk) with
  [comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench).
