---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/JoaoZaokk/HunyuanVideo-1.5-720p-T2V-Quantized/blob/main/LICENSE_TENCENT_HUNYUANVIDEO15.txt
base_model: tencent/HunyuanVideo-1.5
base_model_relation: quantized
library_name: diffusion-single-file
extra_gated_prompt: >-
  These are MODIFIED files derived from Tencent HunyuanVideo 1.5, redistributed
  under the Tencent Hunyuan Community License Agreement. That Agreement DOES NOT
  APPLY IN THE EUROPEAN UNION, THE UNITED KINGDOM AND SOUTH KOREA. Read
  LICENSE_TENCENT_HUNYUANVIDEO15.txt and NOTICE.txt before downloading.
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

# HunyuanVideo 1.5 720p T2V — quantized, with the failures kept

Quantized builds of HunyuanVideo 1.5 720p T2V in ComfyUI's native
`asym_w4a8_int8` and `convrot_w4a4` formats. **One is usable. Two are here on purpose as measured
negative results.**

> ### ⚠️ Licence — read this first
>
> These are **modified files** redistributed under the **Tencent Hunyuan Community License
> Agreement**, not a permissive licence. The full Agreement is in this repository as
> [`LICENSE_TENCENT_HUNYUANVIDEO15.txt`](LICENSE_TENCENT_HUNYUANVIDEO15.txt) and the required notice,
> statement of modifications and non-affiliation statement are in
> [`NOTICE.txt`](NOTICE.txt).
>
> **The Agreement does not apply in the European Union, the United Kingdom, or South Korea**, and
> the rights it grants are limited to a Territory that excludes all three. If your monthly active
> users exceed 100 million you must request a separate licence from Tencent. There is an Acceptable
> Use Policy in Exhibit A of the Agreement. Read it.
>
> Tencent is **not** affiliated with, sponsoring, or endorsing this repository or anything measured
> in it.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The files

| file | median effective error | format | GiB | verdict |
|---|---|---|---|---|
| `hv15_w4a8.safetensors` | — | 432 × `asym_w4a8_int8` | 8.24 | **use this one** |
| `hunyuan15-misto-t025.safetensors` | **0.1837** | 282 × 4-bit / 150 × 8-bit | 8.03 | correct, visibly grainy |
| `hunyuan15-misto-t040.safetensors` | **0.2147** | 402 × 4-bit / 30 × 8-bit | 7.95 | **DESTROYED — AN EXAMPLE OF WHAT NOT TO DO** |

Source: 15.51 GiB FP16. The usable build is **1.88x lighter**.

"Effective error" is, per layer, the measured relative error of the format that layer *actually*
received, on the real activations that layer saw during sampling.

### What they look like

Same prompt, same seed (12345), same steps, resolution, sampler and scheduler:

| | |
|---|---|
| FP16 reference | ![](images/fp16_referencia.png) |
| `misto-t025` — 0.1837, correct but grainy | ![](images/misto_t025_0.1837_correta_granulada.png) |
| `misto-t040` — 0.2147, destroyed | ![](images/misto_t040_0.2147_DESTRUIDA.png) |
| pure ConvRot W4A4 — destroyed | ![](images/w4a4_puro_DESTRUIDO.png) |

`images/` carries the **whole** ladder that was measured — `t015`, `t021`, `t022` and the pure
ConvRot W4A4 build as well. Their weights are **not** published here: they sit between or beyond the
three files above and add tens of gigabytes without adding a finding. The pictures are the evidence;
the three checkpoints are what is worth downloading.

---

## Why this model is the interesting one

**A 15% gap in effective error separates "ship it" from "unusable".** 0.1837 renders a correct if
grainy picture; 0.2147 renders nothing recognizable. Between those two builds sits 0.08 GiB of disk.

That line is **not** a property of the format. Measured across three architecture families:

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| Z-Image v2 | ~6 B | 0.1241 | 0.2163 |
| **HunyuanVideo 1.5** | **~13 B** | **0.1837** | **0.2147** |

Monotone in model size, 2.4x to 3.4x between the ends — so a threshold chosen on one model is not
transferable to another. Three points make that a hypothesis, not a law.

**Latent divergence does not decide it either.** 0.8255 (destroyed) against 0.7173 (fine) is a 15%
gap on that axis too, so no cut on latent distance separates usable from unusable. Only a render
does. A checkpoint that converts cleanly, resolves the CUDA backend, and passes every structural
check can still produce garbage.

---

## Speed

Pure ConvRot W4A4 against FP16, same card, re-measured over three seeds after an earlier
single-run measurement was found to have inverted the sign:

```
FP16          0.990 s/step
ConvRot W4A4  0.536 s/step      1.85x faster
```

An earlier version of this bench published "1.055x slower", from one run per arm with
`1 runs is a small sample` printed on the screen. It agreed with a previous result, so agreement
with expectation was mistaken for confirmation by measurement. Three seeds inverted it. The
correction is recorded rather than quietly edited.

---

## Verified, not assumed

The converter hard-refuses to run unless both `quantize_convrot_w4a4_weight` and
`convrot_w4a4_linear` resolve to `comfy_kitchen.backends.cuda` — the eager backend declares the same
capabilities and would silently produce numbers describing dequantized math. Dispatch is separately
counted on a real load and a real forward: quantized modules, quantized forwards, zero
`dequantize` calls.

Built and verified with:

```
comfy-kitchen  0.2.31        ComfyUI  c1739380 (0.33.0)
torch          2.13.0+cu130  CUDA     13.0
GPU            RTX 3090 (sm_86)
```

Native INT4 MMA requires `major == 8` (Ampere / Ada). Hopper and Blackwell are routed to an INT8
branch deliberately.

---

## Format

Standard safetensors, ComfyUI-native, mixed precision in one file — the format's own behaviour, not
a trick played on it. Per quantized layer, `<layer>.weight` as an INT8 container plus
`<layer>.weight_scale` as FP32, and a per-layer entry in `__metadata__._quantization_metadata` that
ComfyUI turns into a `<layer>.comfy_quant` tensor at load and dispatches on individually.
`convrot_groupsize` 256. Every non-quantized tensor is preserved byte for byte from the source and
verified as such. The `.quant.json` sidecars record full conversion provenance.

---

## Not covered

One prompt, one seed for the image ladder (three for the timing), 480x480, one frame, one scheduler,
one card. No perceptual metric — "correct", "grainy" and "destroyed" are the judgement of someone
who looked at them. No SASS. Only `convrot_groupsize` 256 in the ladder. The monotonicity across
model sizes rests on three points.

---

## Credits

- **Tencent** — [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5). These
  files are derivatives of their model, redistributed under their Community License Agreement,
  a full copy of which is included here. All trademark rights in "Tencent Hunyuan" are theirs.
  Tencent does not endorse this work.
- **Comfy-Org / comfyanonymous and the ComfyUI contributors** — the `QuantizedTensor` / `Layout` /
  `MixedPrecisionOps` model this format plugs into.
- **comfy-kitchen** — the ConvRot W4A4 and W4A8 CUDA kernels.
- Quantized by [JoaoZaokk](https://huggingface.co/JoaoZaokk) with
  [comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench).
