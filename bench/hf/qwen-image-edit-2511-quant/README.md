---
license: apache-2.0
base_model: Qwen/Qwen-Image-Edit-2511
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - image-to-image
  - image-editing
  - quantized
  - int4
  - int8
  - convrot
  - w4a4
language:
  - en
---

# Qwen-Image-Edit 2511 — ConvRot quantized, and the lowest per-layer error on this bench

Quantized builds of **Qwen-Image-Edit-2511** in ComfyUI's native `convrot_w4a4` and
`asym_w4a8_int8` formats, converted from the BF16 weights Comfy-Org repackages.

**38.05 GiB → 9.60 GiB, 3.96x lighter.** At 20.4 billion parameters this is the largest model this
bench has converted, and it measures the *lowest* median per-layer error of any checkpoint here.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The files

| file | bytes | GiB | layout | `convrot_groupsize` |
|---|---|---|---|---|
| `qwen_image_edit_2511_w4a4.safetensors` | 10,306,850,912 | **9.60** | 840 × `convrot_w4a4` | 256 |
| `qwen_image_edit_2511_mixed.safetensors` | 10,603,610,352 | **9.88** | 607 × 4-bit / 233 × 8-bit | 256 |

Both carry the same `source_identity_sha256`
(`0ae1768041ecad9c09ca8f989d2f9253148fe75baa3b923112181bc4949ef248`), so they are comparable to
each other by construction — same source bytes, same converter, same calibration, one axis varied.

Sizes are `stat()` on the files, and the parameter counts below are summed from the Safetensors
header, not estimated from the file size:

| build | bytes | tensors | parameters | dtypes |
|---|---|---|---|---|
| BF16 source | 40,861,031,560 | 1934 | 20,430,401,088 | 1933 BF16 + 1 F32 |
| `int8_convrot` (Comfy-Org) | 20,499,083,824 | 3614 | 20,435,991,168 | 840 I8 + 840 U8 + 1093 BF16 + 841 F32 |
| `w4a4` (this repo) | 10,306,850,912 | 2774 | 10,243,771,968 | 840 I8 + 1093 BF16 + 841 F32 |
| `mixed` (this repo) | 10,603,610,352 | 3240 | 10,540,457,168 | 840 I8 + 233 U8 + 1093 BF16 + 1074 F32 |

The `U8` column is the asymmetric zero-point: Comfy-Org's int8 build carries one per quantized
layer, our pure W4A4 carries none (it is symmetric), and our mixed build carries exactly **233** —
the count of layers promoted to `asym_w4a8_int8`. The header agrees with the sidecar.

## Per-layer error, measured on real activations

840 Linear layers, **all 840 calibrated** — no layer took a format chosen without a measurement
behind it. Each error is the relative error of that format against a float32 reference, on the
activation rows the model actually produced during sampling (reservoir-sampled across steps, stored
BF16):

| format | median | p25 | p75 | max |
|---|---|---|---|---|
| `err_bf16` (the floor) | 0.0020 | 0.0019 | 0.0022 | 0.0028 |
| `err_w4a4` | **0.1080** | 0.0794 | 0.1571 | 0.2953 |
| `err_w4a8` | **0.0358** | 0.0269 | 0.0493 | 0.0804 |

Two things worth reading off that table.

**0.1080 is the lowest median `err_w4a4` on this bench**, across Z-Image (~6 B), Krea 2 Turbo
(12.8 B), HunyuanVideo 1.5 (~13 B) and Wan 2.1 VACE (1.3 B). A 20.4 B model tolerating 4 bits
*better* than a 1.3 B one is the second checkpoint in a row to refute size-monotonicity on this
bench; the first was Krea 2. **Per-layer tolerance is a property of the model, not of the
parameter count**, and nothing in this repo should be read as a rule that transfers.

**`err_bf16` is 0.0020, so the reference arm is not free either.** Every number above is a distance
from float32, and BF16 itself sits 0.0020 away. W4A8 is 18x the BF16 floor; W4A4 is 54x.

W4A8 is **3.02x more faithful per layer** than W4A4 here (0.1080 / 0.0358) for **0.28 GiB** more on
disk in the mixed build. Whether that buys a better picture is a different question, answered below
— and on this bench per-layer error has never predicted the free-running image.

### Which layers are expensive, and where the mixed build spent its budget

60 transformer blocks × 14 Linear families = the 840 layers. Median per family:

| family | shape | `err_w4a4` | `err_w4a8` | ratio | promoted to 8-bit |
|---|---|---|---|---|---|
| `attn.to_out.0` | [3072, 3072] | **0.2101** | 0.0589 | 3.57x | 50/60 (83%) |
| `txt_mlp.net.2` | [3072, 12288] | 0.1953 | 0.0583 | 3.35x | 44/60 (73%) |
| `attn.to_add_out` | [3072, 3072] | 0.1924 | 0.0486 | 3.96x | 48/60 (80%) |
| `img_mlp.net.2` | [3072, 12288] | 0.1689 | 0.0517 | 3.27x | 42/60 (70%) |
| `attn.to_v` | [3072, 3072] | 0.1506 | 0.0501 | 3.01x | 30/60 (50%) |
| `attn.add_v_proj` | [3072, 3072] | 0.1346 | 0.0454 | 2.96x | 11/60 (18%) |
| `attn.add_q_proj` | [3072, 3072] | 0.1118 | 0.0379 | 2.95x | 4/60 (7%) |
| `img_mlp.net.0.proj` | [12288, 3072] | 0.1078 | 0.0367 | 2.94x | — |
| `attn.to_q` | [3072, 3072] | 0.0961 | 0.0326 | 2.95x | — |
| `attn.to_k` | [3072, 3072] | 0.0926 | 0.0315 | 2.94x | — |
| `txt_mlp.net.0.proj` | [12288, 3072] | 0.0918 | 0.0310 | 2.96x | 4/60 (7%) |
| `attn.add_k_proj` | [3072, 3072] | 0.0841 | 0.0285 | 2.96x | — |
| `img_mod.1` | [18432, 3072] | 0.0281 | 0.0068 | 4.11x | — |
| `txt_mod.1` | [18432, 3072] | **0.0248** | 0.0060 | 4.14x | — |

Two things in that table are worth more than the summary median.

**The modulation layers are the cheapest in the whole block** — `txt_mod.1` at 0.0248 against
`attn.to_out.0` at 0.2101, a factor of **8.5**. That is the fourth architecture on this bench where
modulation turns out to be the *safest* place to spend 4 bits, against a widely repeated instinct
that modulation is too sensitive to quantize. The instinct has never been accompanied by a
measurement here.

**Output projections are expensive and input projections are cheap**, and that ordering *transfers*:
Krea 2 Turbo measured `attn.wo` worst (0.2181) and `attn.wk` cheapest (0.0749) — a different
architecture, a different parameter count, the same shape of answer. Worth flagging because two
other transfer hypotheses died on this bench (size-monotonicity, and the groupsize ratio), so a
pattern that does carry across families is the exception, not the rule.

The mixed build's promotion budget landed exactly on the expensive end — 83% / 80% / 73% / 70% of
the four worst families, nothing at all on the two cheapest — which is the selection criterion
doing what it is supposed to do, shown rather than asserted.

## Quality: MEASUREMENT IN PROGRESS, NOT YET PUBLISHED

**This section is deliberately empty and this repo is not published until it is filled.**

A 48-render quality ladder — BF16 reference, our two builds, and Comfy-Org's `int8_convrot`, over 6
prompts × 2 seeds at 20 steps, 1024 px, cfg 2.5, euler/simple — is running at the time of writing.
Until it lands, this repo states no claim about the images, the latent divergence, or the speed.

That restraint is not decoration. This bench has measured a checkpoint that converts cleanly,
resolves the CUDA backend, dispatches genuinely 4-bit math, passes every structural check, and
renders a smear (Wan 2.1 VACE, per-layer 0.1602). **Structure never proves quality; only a render
does.** A per-layer median of 0.1080 is the most encouraging number this bench has produced and it
still is not a picture.

## What is NOT covered

- **No image has been rendered from these files yet** at the time of writing. See above.
- **The text encoder is not in this repo.** Qwen-Image-Edit runs on `qwen_2.5_vl_7b`; a
  `convrot_w4a4` build of it exists on this bench but was measured *before* the ComfyUI text-encoder
  lock was released, so its published numbers describe the dequantized path and are being redone.
- **The VAE is not quantized and will not be.** It is ~254 MiB against a 38 GiB transformer: the
  saving is noise and the risk is not.
- **No perceptual metric.** Latent divergence measures where the sampler went, not whether the
  picture is better — this bench has measured a 2.40x more faithful latent next to an image nobody
  preferred.
- **One card (RTX 3090, sm86), one sampler, one scheduler, one resolution.** `convrot_groupsize` 256
  only; 16 / 64 / 1024 are legal for W4A4 and untested on this model.
- **Nunchaku's SVDQuant INT4 build of the same model is on disk and is not compared here.** It is a
  different format needing a different loader, so it cannot share this ladder.

## Credits

- **[Qwen](https://huggingface.co/Qwen) (Alibaba)** — the Qwen-Image-Edit-2511 model itself.
  Everything here is a re-encoding of their weights; the model is theirs.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI)** — the repackaged BF16
  single-file weights these builds were converted from, and the `int8_convrot` build used as the
  comparison arm. Their int8 has beaten this project's W4A4 on fidelity in four model families in a
  row; it is downloaded and measured here precisely because it is the arm that can win.
- **ComfyUI / comfy-kitchen** — the `convrot_w4a4` and `asym_w4a8_int8` formats, the
  `MixedPrecisionOps` dispatch, and the CUDA kernels that execute them.

## Reproducing

```bash
python tools/quant_mixed.py --input qwen_image_edit_2511_bf16.safetensors \
  --profile qwen_image --analysis calib/qwen_image_edit_2511.analysis.json
python tools/quality_ladder.py --reference <bf16> --models <w4a4> <mixed> <int8_convrot> \
  --clip qwen_2.5_vl_7b.safetensors --clip-type qwen_image --vae qwen_image_vae.safetensors \
  --seeds 1 2 --steps 20 --size 1024 --cfg 2.5
```

Backend recorded in both sidecars, on every op the converter resolved:
`quantize_convrot_w4a4_weight`, `convrot_w4a4_linear`, `quantize_w4a8_int8_weight` and
`w4a8_int8_linear` all resolve to `comfy_kitchen.backends.cuda`.
