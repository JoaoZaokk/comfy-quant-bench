---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-encoder
  - quantized
  - int4
  - convrot
  - w4a4
language:
  - en
  - zh
---

# Qwen3-4B — ConvRot W4A4 (ComfyUI text encoder)

4-bit weight / 4-bit activation quantization of the Qwen3-4B text encoder as ComfyUI loads it
(the Lumina2 / Z-Image conditioning path), in ComfyUI's native `convrot_w4a4` format.

**7.49 GiB → 2.42 GiB, 3.09x lighter.** 252 of 398 tensors quantized.

Produced on a bench that tries to refute its own results. Every number below was **executed**, not
inferred, and each one carries the condition it was measured under. What was not measured is listed
at the bottom, on purpose.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## Read this before you download

**Out of the box in stock ComfyUI, this file saves memory and saves no time.** That is not a defect
of the quantization — it is how ComfyUI treats *every* text encoder, whatever the format.

Two independent locks force dequantized math on the CLIP path:

| lock | where it comes from |
|---|---|
| `full_precision_mm=True` | `comfy/sd1_clip.py:114`, hardcoded for every text encoder |
| `comfy_force_cast_weights=True` | `comfy/sd.py:269` → `set_model_compute_dtype(torch.float32)` for every CLIP object |

Counted during a real encode on a comparable public W4A4 encoder: **0 calls to the 4-bit kernel,
350 calls to `QuantizedTensor.dequantize`.** The weight stays 4-bit in VRAM; the GEMM runs in BF16.
Releasing only one of the two locks changes nothing — both have to go.

So the honest pitch for this file is: **it fits where the BF16 original does not.** If your card
cannot hold 7.5 GiB of text encoder alongside a diffusion model, this is what you want. If you came
for speed, read the next section first.

---

## What it costs in fidelity

Conditioning tensors compared against the BF16 original, same prompts, same tokenizer, on an
RTX 3090:

| prompt | shape | rel-RMSE | cosine |
|---|---|---|---|
| `a red apple on a weathered wooden table, soft window light` | `[1, 21, 2560]` | 0.1442 | **0.98957** |
| `portrait of an elderly fisherman, weathered face, golden hour` | `[1, 21, 2560]` | 0.1412 | **0.99001** |
| third prompt, 23 tokens | `[1, 23, 2560]` | 0.1435 | **0.98968** |

Cosine **0.9896–0.9900** against BF16. The absolute maximum barely moves (13753.5 → 13570.4) and the
standard deviation is nearly untouched (63.57 → 63.24), so this is not a scale distortion — it is
distributed noise.

---

## What it costs in speed, and why the sign depends on your prompt

Measured on an RTX 3080 Ti, median of 3, same file, **only the prompt length varied**. "locked" is
what you get in stock ComfyUI; "released" is with both locks patched off:

| tokens | locked | released | |
|---|---|---|---|
| 22 | 80.7 ms | 120.1 ms | 1.49x **slower** |
| 75 | 100.1 ms | 105.8 ms | 1.06x **slower** |
| 199 | 151.9 ms | 95.2 ms | 1.60x faster |
| 424 | 245.2 ms | 102.0 ms | 2.40x faster |
| 850 | 456.1 ms | 124.3 ms | 3.67x faster |
| 1496 | 824.5 ms | 249.0 ms | 3.31x faster |

**The crossover sits between 75 and 199 tokens.** The released path is nearly flat from 22 to 424
tokens (120 → 102 ms) while the locked path climbs with the sequence: the 4-bit kernel carries a
per-layer fixed cost a short prompt cannot amortise, and the dequantized path pays a BF16 GEMM that
grows with the token count. At 850 tokens the released 4-bit path also beats the **BF16 original**
(372.1 ms at ~700 tokens against 124.3 ms).

An earlier version of this table claimed the cost scaled with weight size. That was a mechanism
argued rather than measured, and varying one axis refused it. The sequence length is what sets the
sign.

Releasing the locks is not something any ComfyUI node exposes today. It needs
`clip.patcher.force_cast_weights = False` (on the patcher, **not** on the modules — `model_patcher.py`
rewrites the module attribute on every load to GPU, so a module-level patch survives only by
accident of VRAM state) plus popping `manual_cast_dtype`. **Accuracy costs more when you do:** on
this file the 4-bit weight alone costs 1.44e-1 against its BF16 twin, and releasing takes it to
6.09e-1 — 4.23x.

---

## Hardware

Native INT4 MMA (`m16n8k64 s4`) is selected when `major == 8` — **Ampere and Ada (sm_86, sm_89)**.
Hopper and Blackwell are routed to an INT8 branch deliberately by comfy-kitchen, not by accident.

On this bench the INT8 branch is consistently **more faithful**, measured three independent ways:
1.49x on real Z-Image activations (24/24 layers), 1.33x on per-step epsilon with matched inputs
(8/8 steps), and 1.40x on a different model family quantized by a different author (60/60). Native
INT4 wins on large-batch throughput (1.41x–1.67x at M=1024) and loses at M=1 (1.3x–1.74x slower).
Neither is strictly better; the trade is real.

Built and verified with:

```
comfy-kitchen  0.2.31        ComfyUI  c1739380 (0.33.0)
torch          2.13.0+cu130  CUDA     13.0
GPU            RTX 3090 (sm_86)
```

The converter **hard-refuses to run** unless both `quantize_convrot_w4a4_weight` and
`convrot_w4a4_linear` resolve to `comfy_kitchen.backends.cuda`, because the eager backend declares
the same capabilities and would silently produce numbers describing dequantized math.

---

## Format

Standard safetensors, ComfyUI-native. Per quantized layer: `<layer>.weight` as `I8` of shape
`[rows, cols/2]` (INT8 container holding signed INT4) plus `<layer>.weight_scale` as `F32` of shape
`[rows]`. Every other tensor is preserved **byte for byte** from the source and verified as such.
`__metadata__` carries `_quantization_metadata` with a per-layer entry; ComfyUI turns each into a
`<layer>.comfy_quant` tensor at load and dispatches on it.

`convrot_groupsize` 256. Excluded from quantization by design: `embed_tokens`, all norms, `lm_head`,
and any `visual` / `vision` tower. The `.quant.json` sidecar in this repo records the full
conversion provenance.

A note on loading: ComfyUI prints `WARNING: unet unexpected [...comfy_quant]` for these tensors. It
is **cosmetic** — they are consumed before that check runs and complained about afterwards.

---

## This is an encoder, not a chat model

Driven as a text generator it does not terminate: it loops on its own output. That is expected —
what is published here is the conditioning path, and the measurement that matters is the cosine
against BF16 above, not a generation sample. Qwen3-4B ties its word embeddings and ComfyUI's
encoder wrapper carries no generation head.

---

## Not covered

One card for the fidelity numbers (3090) and one for the timing table (3080 Ti). Three prompts for
the conditioning comparison, all short. No SASS inspection. No perceptual evaluation of images
generated through this encoder — the conditioning distance is measured, the *visual* consequence is
not. The lock-release timings come from a post-load monkeypatch, not from anything ComfyUI offers a
user today. Only `convrot_groupsize` 256 was built. And the fidelity cost of releasing the locks is
**not** a property of the W4A4 format: two different public W4A4 encoders differ by 6x in that
number, so it cannot be quoted without naming the file.

---

## Credits

- **Qwen team, Alibaba** — [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), Apache 2.0. This
  is a derivative of their weights and nothing here would exist without them.
- **Comfy-Org / comfyanonymous and the ComfyUI contributors** — the `QuantizedTensor` / `Layout` /
  `MixedPrecisionOps` model this format plugs into, and the repackaged encoder this was converted
  from.
- **comfy-kitchen** — the ConvRot W4A4 CUDA kernels that make the format executable rather than
  merely storable.
- Quantized by [JoaoZaokk](https://huggingface.co/JoaoZaokk) with
  [comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench).
