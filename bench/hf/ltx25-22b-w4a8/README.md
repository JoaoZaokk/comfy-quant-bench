---
license: other
license_name: ltx-2.x-community-license
license_link: https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot/blob/main/LICENSE_LTX_2x_COMMUNITY.txt
base_model: Lightricks/LTX-2.5
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-video
  - video
  - quantized
  - int4
  - convrot
  - w4a8
language:
  - en
---

# LTX 2.5 22B distilled — 4-bit weights, a real 10-second video, and the arm that beats it

A `asym_w4a8_int8` build of `ltx-2.5-22b-distilled-transformer`, measured on a **complete
10-second video** against the BF16 original and against Lightricks' own INT8.

**39.13 GiB → 11.66 GiB, 3.36x lighter**, and **1.95x faster** than the original on the same
249-frame render. It is also **1.90x less faithful than Lightricks' own INT8 build** — that number
is here because it is the one that decides whether you want this file.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The measurement: one 10-second video, three transformers

249 frames at 25 fps (**9.96 s** — LTX requires `8n+1`, so 250 is illegal and 249 is the legal
neighbour), 512×512, 3 steps, cfg 1.0, euler, manual sigmas `0.909375, 0.725, 0.421875, 0.0`,
seed 1234, one RTX 3090.

**The transformer is the only thing that changes.** All three arms use the same encoder
(`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot`), the same VAE, the same prompt, the same seed
and the same sigmas.

| arm | GiB | 249 frames | s/frame | MAE vs BF16 | PSNR vs BF16 |
|---|---|---|---|---|---|
| BF16 original | 39.13 | 780.7 s | 3.14 | — | — |
| `comfy-int8-convrot` (Lightricks) | 20.03 | 481.5 s | 1.93 | **4.10** | **29.71 dB** |
| **this file, W4A8** | **11.66** | **400.9 s** | **1.61** | 7.81 | 25.39 dB |

MAE is the mean absolute per-pixel difference on the 0–255 scale, averaged over **all 249 frames**;
the spread is tight (ours 7.16–8.84, theirs 3.61–4.61), so the ordering is not a one-frame
accident.

![three arms](comparacao_10s_tres_bracos.png)

Frames 1 / 63 / 125 / 187 / 249. The middle row tracks the top row closely — same thin white
lighthouse, same rock, same framing. **The bottom row is a different composition**: a larger,
closer, brick-coloured tower with dark smoke where the original has a light beam.

All three are good videos with coherent motion across the full ten seconds. **Ours is a good video
that is further from the original.** If you want the original's picture, take Lightricks' INT8 and
pay 8.4 GiB and 20% more time for it. If VRAM is what binds you, this file gets a real ten seconds
out of a 24 GB card faster than anything else here.

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-w4a8.safetensors` | 12,520,267,816 | **11.66** | 1440 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256 |

Source: `ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 42,018,190,584 B, byte-for-byte the
file `Lightricks/LTX-2.5` publishes. Backend recorded in the sidecar:
`comfy_kitchen.backends.cuda`.

## Running it

Needs `--disable-dynamic-vram`. At 11.66 GiB it is resident on a 24 GB card and needs **no**
block splitting — which is most of why it is faster than the other two arms, both of which have to
move weights every step.

For contrast, and because it is a measured limit rather than a guess: **the BF16 original does not
render this video without help.** Four attempts:

```
BF16, no DisTorch2,   49 frames   -> CUDA error: out of memory (from mem_get_info; the card is exhausted)
BF16, DisTorch2 6 GiB from cuda:1, 49 frames  -> works, 717.8 s
BF16, DisTorch2 6 GiB from cuda:1, 249 frames -> Windows fatal exception: access violation, process dead
BF16, DisTorch2 40 GB on cpu,      249 frames -> works, 780.7 s  (28.5% cuda:0 / 71.5% cpu)
```

The frame count alone is not the problem — 249 frames run fine on this W4A8 build with no
splitting at all. The combination of a 39 GiB model spread across two cards and a 249-frame latent
is what kills the process.

## What is NOT covered

- **One prompt, one seed, one resolution (512), one frame rate, one card.** Three arms, not a
  sweep.
- **MAE and PSNR are pixel metrics, not perceptual ones.** They say this build lands further from
  the reference; they do not say a viewer prefers the reference. Judge the contact sheet yourself.
- **The audio branch was never decoded.** LTX 2.5 generates audio alongside video and the graph
  concatenates both latents; only the video branch is decoded here. Nothing in this card describes
  the audio.
- **The text encoder is quantized in every arm** (`comfy-int8-convrot`), held fixed so the axis
  stays clean. A BF16-encoder run would measure a different thing and is not included.
- **No per-layer error analysis for this checkpoint.** Unlike the image models on this bench, this
  build predates the calibration record, so its sidecar has no `source_identity_sha256` and there
  is no `err_w4a8` distribution to quote.
- `convrot_groupsize` **256 only**.

## License and changes

LTX 2.5 is distributed under the **LTX-2.x Community License Agreement**, and this derivative is
distributed **exclusively under those same terms**, as Section 3.2 requires. The complete
agreement, including the use restrictions in Section 4 and Attachment A, ships beside these weights
as `LICENSE_LTX_2x_COMMUNITY.txt` — read it before use; its acceptable-use restrictions bind you
and anyone you pass this on to.

**Statement of changes, per Section 3.3:** the 1440 Linear layers of the transformer were
re-encoded from BF16 into ComfyUI's `asym_w4a8_int8` format (4-bit weights, 8-bit activation path,
Hadamard rotation at group size 256). No weights were fine-tuned, no architecture was altered, and
every non-Linear tensor is preserved from the source.

## Credits

- **[Lightricks](https://huggingface.co/Lightricks/LTX-2.5)** — LTX 2.5 itself, the BF16 weights
  this was converted from, and the `comfy-int8-convrot` build used as the comparison arm. That arm
  is more faithful than this one, and it is theirs.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format, the `MixedPrecisionOps` dispatch and
  the CUDA kernels that execute it.
- **ComfyUI-MultiGPU (DisTorch2)** — without its block splitting the BF16 reference arm could not
  have been rendered at all on this hardware, and there would have been nothing to compare against.
