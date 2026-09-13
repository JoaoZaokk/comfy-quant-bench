---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-video
  - image-to-video
  - video
  - quantized
  - int4
  - convrot
  - w4a8
language:
  - en
---

# Wan 2.2 TI2V 5B — 4-bit weights at 3.39x, and the second model family where 4-bit *activations* fail

`asym_w4a8_int8` build of `wan2.2_ti2v_5B`, measured against the FP16 original and against a pure
W4A4 build of the same source.

**9.31 GiB → 2.75 GiB, 3.39x lighter and 1.59x faster per step**, with output this bench cannot
tell apart from the original at a glance. The W4A4 build is published **as a measured negative**,
not as a file to use.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The measurement

6 runs (6 prompts × 1 seed), 33 frames at 480 px, 20 steps, euler/simple, one RTX 3090. **All
three arms are resident on the card — none is offloaded, none is block-split**, so the s/step
column is a real comparison and not a placement artifact.

| arm | GiB | s/step | divergence from FP16 | min–max | picture |
|---|---|---|---|---|---|
| FP16 original | 9.31 | 1.579 | — | — | sharp |
| **this file, W4A8** | **2.75** | **0.994** | **0.2115** | 0.1185–0.3275 | **indistinguishable at a glance** |
| W4A4, same source | 2.46 | 0.753 | 0.3847 | 0.2431–0.5748 | **blurred, mushy** |

![three arms](images/tres_bracos.png)

Four prompts, one seed. Left FP16, middle this build, right W4A4. The middle column tracks the
left one. The right column smears every face, every brick, every hand.

## The two findings that outlive this file

**1. Four-bit activations fail here too, and this is the second family.** W4A4 and W4A8 carry the
same 4-bit *weights*; only the activation path differs. On Qwen-Image-Edit 2511 the W4A4 build
rendered pure static; here it renders blur. Different failure mode, same axis. Weights survive 4
bits in both families; activations do not.

**2. Latent divergence has a direction bias, and this is the counterexample that shows it.**
0.3847 sits *inside* the band that has always worked on this bench — below Z-Image v2's 0.7854 and
Krea 2 Turbo's 0.5843, both of which render fine. And the picture is degraded.

Until now this bench only had the opposite failure: *high* divergence with a good picture, because
a tiny perturbation reroutes the sampler somewhere else that is also good. This is the mirror, and
the mechanism is plain once seen: **blur is a small latent change.** Smoothing moves you less far
from the reference than going somewhere sharp and different does. So the metric *rewards*
degradation, as long as the degradation is smooth.

That is stronger than "there is no threshold". It means divergence is not merely noisy — it is
**biased toward soft failure**, and a build that blurs will always flatter itself on this number.

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `wan2.2_ti2v_5B_w4a8.safetensors` | 2,950,547,712 | **2.75** | 300 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256 |

Source `wan2.2_ti2v_5B_fp16.safetensors`, 9,999,658,848 B, sha256
`0a09fde7bc7f6a0d456f87acc914d3380bef60bc0f47f8ff0a94e2ae6a95ef76`. The W4A4 build carries the
same source hash, so the two are comparable by construction. Backend recorded in the sidecar:
`comfy_kitchen.backends.cuda`.

Calibrated on real activations — 6 sampling runs with forward hooks on all 300 Linear layers,
reservoir-sampled across steps, not just step 0.

### The profile, checked against someone else's work

The layer set comes from a `wan_2_2` profile that was validated **against Comfy-Org's own
`wan2.2_animate_14B_int8_convrot`**: 480 layers matched, 480 they quantize, zero either way. The
2.1 profile this bench already had missed 80 of them — `cross_attn.k_img` and `.v_img`, the image
cross-attention projections, which do not exist in a pure T2V model. On this TI2V 5B the same
pattern degrades to 300 by itself, because this model has no `k_img`/`v_img` either.

## Running it

Needs a **48-channel VAE**: `wan2.2_vae.safetensors`. The `wan_2.1_vae` that ships beside it in
many installs has 16 channels and will decode this model's latents into a plausible, silently
wrong picture. That is not a hypothetical — it was caught here by counting channels before
decoding, not after.

## What is NOT covered

- **Six runs, one seed, 33 frames, 480 px, one sampler, one card.** The ladder's own output says
  six runs is a small sample for a quantity this noisy, and the spread proves it: W4A8's
  divergence spans 0.1185–0.3275 around a mean of 0.2115.
- **"Indistinguishable at a glance" is a human verdict on a contact sheet**, not a perceptual
  metric. No LPIPS, no FVD, no user study.
- **33 frames is 1.3 seconds.** Nothing here says how either build behaves over a long generation.
- **The audio path does not exist in this model**, so nothing about audio is claimed.
- `convrot_groupsize` **256 only**, and no mixed build was tried — the choice here was the clean
  single axis (all-W4A4 against all-W4A8), not a search.
- **The W4A4 weights are published as evidence, not as a usable model.** Use the W4A8.

## Credits

- **[Wan-AI](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)** — Wan 2.2 TI2V 5B itself.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)** — the repackaged
  single-file FP16 weights this was converted from, the 48-channel VAE, and the
  `animate_14B_int8_convrot` build whose layer list was used to validate the profile. That
  validation is why 80 layers are not silently missing from this file.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format and the CUDA kernels that run it.
