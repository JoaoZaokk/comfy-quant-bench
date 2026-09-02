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

Six seeds, one prompt, `uni_pc` / `simple`, 25 steps, **cfg 6.0**, **`ModelSamplingSD3` shift 8**, a
real negative prompt, `vace_strength 0`, 33 frames at 480x480. Middle frame of each clip.

Rows top to bottom: **FP16 reference · `misto005` · `misto015` · pure W4A4**.

![](images/ladder_6_sementes_regime_certo.png)

Two seeds at full size, so the degradation is visible without squinting:

| | seed 4 | seed 5 |
|---|---|---|
| FP16 reference | ![](images/s4_fp16_referencia.png) | ![](images/s5_fp16_referencia.png) |
| `misto005` — **use this one** | ![](images/s4_misto005_bom.png) | ![](images/s5_misto005_bom.png) |
| `misto015` — smeared | ![](images/s4_misto015_borrado.png) | ![](images/s5_misto015_borrado.png) |
| pure W4A4 — destroyed | ![](images/s4_w4a4_puro_DESTRUIDO.png) | ![](images/s5_w4a4_puro_DESTRUIDO.png) |

`misto005` is not "close enough". Across all six seeds it is a sharp watchmaker at a bench, at the
same level as the unquantized reference, from a file **1.87x smaller**. `misto015` keeps the subject
and loses everything else to a painterly smear. Pure W4A4 is blocky coloured wreckage in all six.

---

## The images on this page were wrong until 2026-09-01, and here is what was wrong with them

Every image previously published here was rendered at **cfg 1.0, no shift, no negative prompt**.
That is the operating point of a *distilled* model — it is what this bench uses for Z-Image Turbo.
**Wan 2.1 is not distilled.** Without guidance it drifts, and the reference drifted along with
everything else: melted faces, smeared arms, washed-out tables, in the FP16 arm that is supposed to
be the standard.

Same file, same seeds, **only the regime changed**:

![](images/o_regime_errado.png)

Top row is what was published. The two rows below are the same unquantized FP16 weights at
`cfg 6 + shift 8 + negative`, and with `uni_pc` on top of that.

This matters to you as a downloader more than it matters as an apology: **if you run these files at
cfg 1 with no shift, you will get the top row and blame the quantization.** The settings in the
section above are not decoration, they are the operating point.

What the correction did *not* change: the three verdicts. `misto005` good, `misto015` smeared, pure
W4A4 destroyed — all three survived re-rendering in the correct regime across six seeds, and the
tolerated / not-tolerated band below was re-judged against the new images rather than the old ones.

One number that did move, and that is worth knowing: `misto005`'s mean divergence from the reference
**rose** from 0.2936 to 0.4750 when the regime was fixed — while the picture got dramatically
better. Higher guidance separates trajectories, so that distance measures trajectory, not fidelity.
**Do not compare divergence numbers across sampler settings, and do not read them as quality.**

The owner of this bench caught this by looking at the pictures and asking what the reference was.
No automated check on this bench would have. That is worth stating plainly on a page that otherwise
argues for measurement.

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
— with no error and no warning.** At `vace_strength 1.0` the unquantized FP16 reference comes out as
woven fabric.

> **Corrected 2026-09-01.** This paragraph used to add *"in every configuration tried (6 steps/1
> frame, 25/33, cfg 6 and cfg 1, with and without `ModelSamplingSD3 shift 8`)"*, which read as *no
> setting saves it*. That is false and the images above refute it: at `vace_strength 0` **with**
> cfg 6, shift 8 and a negative prompt, the FP16 reference is excellent. `vace_strength 0` is
> **necessary and not sufficient** — the sampler settings are the other half, and conflating the two
> axes is what made this page publish broken references for two weeks.

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
