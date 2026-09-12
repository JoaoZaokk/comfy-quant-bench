---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
base_model_relation: quantized
library_name: diffusion-single-file
pipeline_tag: text-to-image
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

# Z-Image Turbo — ConvRot 4-bit for ComfyUI, 3.75x lighter and 2.65x faster per step

Native `convrot_w4a4` and mixed 4/8-bit builds of the official `Tongyi-MAI/Z-Image-Turbo`, in
ComfyUI's own quantization format. No custom node: the stock `UNETLoader` reads the per-layer
metadata and dispatches on it.

**11.46 GiB → 3.06 GiB**, and **0.903 → 0.341 s/step** on an RTX 3090 at 1024×1024.
Thirty-six renders across six prompts and two seeds, none broken.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The files

| file | format | GiB | median per-layer error | latent divergence | s/step |
|---|---|---|---|---|---|
| `zimage_turbo_w4a4.safetensors` | 170 × `convrot_w4a4` | **3.06** | **0.1228** | 0.6589 | **0.341** |
| `zimage_turbo_mixed.safetensors` | 113 × 4-bit + 57 × 8-bit | 3.18 | — | 0.5500 | 0.391 |

Source: 11.46 GiB BF16, 6.15 B parameters. Eight steps, 1024×1024, euler/simple, cfg 1.0, two
seeds, six prompts.

"Per-layer error" is, for each layer, the measured relative error of the format that layer
*actually* received, on the activations it saw during real sampling — captured with forward hooks
and reservoir-sampled across the steps, not on random input. The floor for comparison is
`err_bf16 = 0.0019`: that much error is not quantization at all.

```
err_bf16   median 0.0019
err_w4a4   median 0.1228
err_w4a8   median 0.0389      <- 3.16x cheaper, at 8 bits
```

**Dispatch was counted, not assumed**, with the weights on the card: 170 quantized modules,
8/8 quantized forwards, **0 `dequantize`**, `convrot_linear_dtype=int4` on
`comfy_kitchen.backends.cuda` — the native INT4 MMA path, not the INT8 fallback branch.

---

## What they look like

Six prompts, seed 1. Reference on top.

![](images/grid_seed1.png)

The hardest cell is the enamel sign, because text is where quantization damage shows first:

| BF16 reference | W4A4 | mixed |
|---|---|---|
| ![](images/sign_bf16.png) | ![](images/sign_w4a4.png) | ![](images/sign_mixed.png) |

---

## How to read the divergence column, and why it is not "damage"

`0.6589` is the mean distance between this build's latent and the BF16 latent on the same prompt
and seed. It sounds alarming and it is not: at 8 steps a small perturbation reroutes the sampler
and it arrives at a *different good image*. All 36 renders are usable.

**The between-run spread inside a single arm is 0.54**, which is close to the gap between the two
arms' means — so the mean comparison is weak evidence here and the paired comparison is the real
one:

```
paired against zimage_turbo_w4a4, run by run
zimage_turbo_mixed    mean delta -0.1089    wins 11/12
```

**The mixed build is measurably closer to the BF16 latent — it wins 11 of 12 paired runs** — for
0.12 GiB and 15% more time per step. If you want the smallest and fastest, take W4A4; if you want
the one that tracks the original trajectory, take mixed.

That result does **not** carry across the family. The same converter at the same threshold, run on
the de-distilled sibling (`ostris/Z-Image-De-Turbo`), gives the mixed build only **7 of 12** — a
split decision. Whether promoting 57 layers to 8 bits buys anything depends on the checkpoint, not
on the recipe.

---

## Where Z-Image sits

The per-layer error a model survives before its output breaks is **not a property of the format**.
Measured across four architecture families on this bench:

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| **Z-Image** | **~6 B** | **0.1421** | **0.1848** |
| Krea 2 Turbo | 12.82 B | 0.1377 | not reached on the only axis available |
| HunyuanVideo 1.5 | ~13 B | 0.1837 | 0.2147 |

Z-Image's upper bound was filled by pushing `convrot_groupsize` down: cg 256 (0.1312) good, cg 64
(0.1421) good, **cg 16 (0.1848) destroyed in 3 of 3 renders**. At 0.1228 these builds sit well
inside the working band.

The mechanism written down before that run was backwards, and the pre-written control caught it: a
Hadamard rotation of size N spreads each outlier across N channels, so a *larger* group mixes more
and flattens outliers harder. "Coarser rotation, more error" was the intuition of a per-group
quantizer, and the rotation is not that.

---

## Using them

Drop into `ComfyUI/models/diffusion_models/` and load with `UNETLoader`.

```
text encoder   qwen_3_4b.safetensors
VAE            ae.safetensors
sampler        euler / simple, 8 steps, cfg 1.0
```

A quantized `qwen_3_4b` text encoder is published separately at
[JoaoZaokk/Qwen3-4B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen3-4B-W4A4-ConvRot) —
7.49 GiB → 2.4 GiB. Note that stock ComfyUI runs *every* text encoder with dequantized math behind
two independent locks, so a quantized encoder saves VRAM and no time unless those are released.

---

## What this does not cover

- **Six prompts, two seeds, one card, one resolution, no perceptual metric.** "36 of 36 usable" is
  the judgement of someone who looked at the contact sheets.
- **No image-space metric.** Latent divergence measures trajectory, not fidelity — a free-running
  generated image cannot separate two quantizations of the same model, only a matched-input probe
  can, and that probe was not run on these builds.
- **The calibration used one concatenated prompt block**, not six separate ones: the tool that
  captures activations read a whole prompt file as a single prompt, a defect found and fixed the
  same day these were built. The published `beyond-reality-zimage-v2` numbers this family is
  compared against were produced the same way, so the comparison is internally consistent, and a
  fresh calibration would shift the medians by a few percent — the calibration seed alone already
  moves them 2-6%.
- **Nothing here was tested with LoRAs, ControlNet, or img2img.**
