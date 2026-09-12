---
license: other
license_name: krea-2-community-license
license_link: https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf
base_model: krea/Krea-2-Turbo
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

# Krea 2 Turbo — ConvRot W4A4, 3.26x lighter, 1.47x faster per step, and the ceiling we could not reach

Native `convrot_w4a4` and mixed 4/8-bit builds of `krea/Krea-2-Turbo` for ComfyUI.

**24.48 GiB → 7.50 GiB**, and **1.47x faster per step** than the int8 build everybody already uses.
Forty renders, none broken.

This card also carries the result that goes the other way, because a repo that only prints its wins
is not evidence: **Comfy-Org's own int8 build is more faithful than this one in 10 of 10 paired
runs.** Read §3 before you download 7.5 GiB.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## 1. The files

| file | format | GiB | median per-layer error | s/step | verdict |
|---|---|---|---|---|---|
| `krea2_turbo_w4a4.safetensors` | 224 × `convrot_w4a4` | **7.50** | **0.1199** | 0.839 | good |
| `krea2_turbo_mixed.safetensors` | 173 × 4-bit + 51 × 8-bit | **7.70** | — | 1.000 | good |

Source: 24.48 GiB BF16, **12,820,073,036 parameters** summed from the safetensors header, not
estimated from the file size.

"Per-layer error" is, for each layer, the measured relative error of the format that layer
*actually* received, on the real activations it saw during sampling — captured with forward hooks
and reservoir-sampled across every step, not on random input.

Krea 2 is a `SingleStreamDiT`: 28 blocks × 8 Linears = the 224 layers quantized here. Modulation is
an `nn.Parameter`, not a Linear, so it sits outside by construction. The `txtfusion` sub-network
(32 Linears) and the model's ends stay in their original precision.

### Where the error sits

```
err_bf16    median 0.0019   max 0.0025      <- the floor. This is not quantization.
err_w4a4    median 0.1199   p25 0.0822   p75 0.1475   max 0.2942
err_w4a8    median 0.0373   p25 0.0264   p75 0.0467   max 0.0634
```

8-bit is cheaper than 4-bit in **224 of 224** layers, median ratio 3.12x. By family:

| family | n | median | | family | n | median |
|---|---|---|---|---|---|---|
| `attn.wo` | 28 | **0.2181** | | `mlp.gate` | 28 | 0.1181 |
| `mlp.down` | 28 | **0.2029** | | `attn.gate` | 28 | 0.0979 |
| `mlp.up` | 28 | 0.1370 | | `attn.wq` | 28 | 0.0813 |
| `attn.wv` | 28 | 0.1283 | | `attn.wk` | 28 | 0.0749 |

The two worst families are both **output projections** — the layers that write back into the
residual stream. The mixed build promotes 51 layers to 8-bit and they are overwhelmingly those two.

---

## 2. What they look like

Two of the five prompts, seed 1. Full sheets for all five are in the repo's bench directory.

| | BF16 reference | int8 (Comfy-Org) | our W4A4 | our mixed |
|---|---|---|---|---|
| portrait | ![](images/p1_bf16.png) | ![](images/p1_int8_convrot.png) | ![](images/p1_w4a4.png) | ![](images/p1_mixed.png) |
| `"OPEN"` sign | ![](images/p2_bf16.png) | ![](images/p2_int8_convrot.png) | ![](images/p2_w4a4.png) | ![](images/p2_mixed.png) |

---

## 3. The honest comparison, and it does not favour this repo

40 renders: 5 prompts × 2 seeds × 4 checkpoints, 10 steps, 1024×1024, euler/simple, cfg 1.0.

| arm | latent divergence vs BF16 | between-run spread | s/step | GiB |
|---|---|---|---|---|
| BF16 | — | — | 2.239 | 24.48 |
| int8 (Comfy-Org) | **0.2435** | 0.6512 | 1.233 | 12.57 |
| our W4A4 | 0.5843 | 0.7005 | **0.839** | **7.50** |
| our mixed | 0.4972 | 0.7257 | 1.000 | 7.70 |

**Do not read the means.** The spread *inside* one arm (0.65–0.73) is larger than the gaps
*between* the arm means, so a mean comparison here is noise with a decimal point on it. The paired
comparison — same prompt, same seed, different arm — is the evidence:

```
vs int8 (Comfy-Org)     mean delta    wins
our W4A4                  +0.3408     0/10
our mixed                 +0.2537     0/10
```

**int8 is 2.40x more faithful to the BF16 latent and wins every single paired run.** That is the
fourth independent measurement on this bench pointing the same way, now in a fourth model family.

What this repo offers instead: **1.68x smaller and 1.47x faster per step.** That is a trade, and
whether it is a good one is your call. What cannot be said is that 4-bit is more faithful.

One caveat on that divergence column: 0.5843 sounds alarming and is not. At 10 steps a small
perturbation reroutes the sampler and it arrives at a *different good image*. A free-running
generated image cannot compare two quantizations of the same model — only a matched-input probe
can. All 40 renders are usable; nothing here is broken.

**The reference arm was checked before any of this was believed.** On the unquantized BF16 file,
`d_prompt / d_seed = 1.7686` — the model responds to its own conditioning, with the seed distance
as the negative control. Prediction written beforehand: `> 0.8`.

**Dispatch was counted, not assumed.** With the weights on the card: 224 quantized modules, 8/8
quantized forwards, 0 `dequantize`, `convrot_linear_dtype=int4` on `comfy_kitchen.backends.cuda` —
native INT4 MMA, not the INT8 fallback branch.

---

## 4. The ceiling: we went looking, and the format's only axis does not reach it

The tolerance a model survives before its output breaks is **not a property of the format** — this
bench has measured a different line for every architecture. Krea 2's upper bound was empty, so we
tried to fill it.

`convrot_groupsize` is the only lever the format exposes that raises per-layer error, legal at
16 / 64 / 256 / 1024, and *smaller* is worse. (A Hadamard rotation of size N spreads each outlier
across N channels, so a bigger N flattens outliers harder. The opposite intuition was written down
first and measured to be backwards.)

| cg | median | max | vs 256 | render, 5 prompts × 2 seeds |
|---|---|---|---|---|
| 256 (shipped) | 0.1199 | 0.2942 | 1.000x | 10/10 good |
| 64 | 0.1238 | 0.3397 | 1.033x | 10/10 good |
| 16 | **0.1377** | 0.4331 | 1.149x | **10/10 good** |

![](images/ceiling_groupsize_seed1.png)

Monotone in the median and in 215 of 224 layers individually, measured over the intersection of
layers all three accept. **At the smallest legal groupsize the model does not break** — the
`"OPEN"` sign stays legible in both cg-16 cells above.

The easy and wrong reading — *the picture survived because the 4-bit kernel never ran* — was ruled
out: both ceiling builds count 224 quantized modules, 8/8 quantized forwards, 0 dequantize, int4.

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| Z-Image v2 | ~6 B | 0.1421 | 0.1848 |
| **Krea 2 Turbo** | **12.82 B** | **0.1377** | **not reached on the only axis available** |
| HunyuanVideo 1.5 | ~13 B | 0.1837 | 0.2147 |

**Two transfer hypotheses died on this checkpoint in one day.** Size monotonicity predicted a
12.8 B model would land where the ~13 B HunyuanVideo does (0.15–0.22); it measured 0.1199, *below*
the ~6 B Z-Image. And Z-Image's groupsize ratios (1.155x, 1.468x) predicted Krea 2's; it measures
1.033x and 1.149x, three times less sensitive. Tolerance and groupsize sensitivity both belong to
the model, not to the format.

---

## 5. Image editing works on the 4-bit weights, with one measured deviation

Krea 2's identity-preserving edit LoRA (`krea2_identity_edit_v1_2`) loads and takes effect over
quantized weights. Four instruction categories — clothing, time of day, object insertion,
background replacement — **4 of 4 changed only what was asked**, on both the int8 build and ours,
with identity, pose and drawing style preserved.

![](images/edit_int8_vs_w4a4.png)

**The deviation, stated plainly because it is real:** asked to *"Add a small brown dog sitting on
the grass next to her"*, our W4A4 build puts **two** dogs at seed 1002. int8 and our mixed build put
one. It reproduces exactly on rerun.

Nine seeds per arm:

| arm | original instruction | with `"exactly one"` |
|---|---|---|
| int8 (Comfy-Org) | 0 of 9 duplicated | 0 of 8 |
| **our W4A4** | **1 of 9 duplicated** | 0 of 8 |
| our mixed | 0 of 9 duplicated | 0 of 8 |

Naming the count in the instruction fixes it. The mixed build — which promotes the 51
highest-error layers to 8-bit — never duplicates, which is consistent with the deviation tracking
per-layer error, but that is two measurements agreeing, not a demonstrated cause.

An honest note on how this was nearly reported wrong: the first sweep was 48 renders across eight
seeds, every one with a single dog, which reads as "does not reproduce". None of those 48 included
seed 1002. **Measuring around a point is not measuring the point.** And 1 in 9 does not pin the
frequency — it establishes that the event exists and is arm-specific.

---

## 6. Using them

Drop into `ComfyUI/models/diffusion_models/` and load with the normal `UNETLoader`. No custom node,
no extra flag: ComfyUI reads the per-layer quantization metadata and dispatches on it natively.

Text encoder `qwen3vl_4b`, VAE `qwen_image_vae`, `shift 1.15`, 10 steps, cfg 1.0, euler/simple.

**One trap worth knowing, and it is not about these files.** The BF16 source mixes two floating
dtypes among its 2-D weights — the blocks are BF16 and the model's ends are F32, which is Krea 2's
own precision policy. With ComfyUI's dynamic VRAM on, the lazy `Linear` load assigns a weight
straight from the file and never casts it to the module's dtype, so the BF16 source dies after
reading 24.5 GiB with

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

pointing at the first layer of the model and saying nothing about the mechanism. **The quantized
files here do not hit it** — they load with zero divergent dtypes.

---

## 7. What none of this covers

- **Five prompts, two seeds, one card, one resolution, no perceptual metric.** "40 of 40 usable" is
  the judgement of someone who looked at two contact sheets.
- **`cg 1024` was not measured** — it moves toward *less* error, and the ceiling run was looking up.
- **The profile's exclusion list is untested.** 39 excluded 2-D tensors would pass the divisibility
  filter, including the whole `txtfusion` sub-network. Quantizing them would not move the median;
  it changes *which* layers degrade.
- **The matched-input epsilon comparison never ran on Krea 2**, which is the only instrument that
  can separate two quantizations of one model. It is blocked by an import-order problem in
  ComfyUI's dynamic-VRAM host buffer, documented in the bench repo.
- **Editing was tested on one source image**, a flat children's drawing, with one identity LoRA.
- **`krea2_raw`** — the non-turbo base — was never quantized. Nothing here speaks for it.
