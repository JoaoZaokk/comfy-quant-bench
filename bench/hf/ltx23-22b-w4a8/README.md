---
license: other
license_name: ltx-2-community-license-agreement
license_link: https://huggingface.co/JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot/blob/main/LICENSE_LTX_2_COMMUNITY.txt
base_model: Lightricks/LTX-2.3
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-video
  - video
  - audio
  - quantized
  - int4
  - convrot
  - w4a8
language:
  - en
---

# LTX 2.3 22B distilled 1.1 — 4-bit weights in the single-file checkpoint, measured on 10 seconds of video **and audio**

A `asym_w4a8_int8` build of `ltx-2.3-22b-distilled-1.1`, the single-file checkpoint (transformer +
video VAE + audio VAE + vocoder + text projection), measured on a **complete 10-second video with
its soundtrack** against the BF16 original, against a pure W4A4 build of the same source, and
against a third-party 6-bit GGUF.

**42.98 GiB → 15.51 GiB, 2.77x lighter for the whole checkpoint** (the transformer is what shrinks;
the VAEs, vocoder and projection are preserved byte-for-byte, 1503 tensors, verified by hash).

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**
The 2.5 sibling, measured on the same protocol: [LTX-2.5-22B-distilled-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot).

---

## The measurement: one 10-second video, four transformers, picture and sound

249 frames at 25 fps (**9.96 s** — LTX requires `8n+1`), 512×512, **8 steps** (Lightricks' README:
*"distilled v1.1, 8 steps, CFG=1"*; the sigmas are the first stage of ComfyUI's own
`video_ltx2_3_t2v` template: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`),
cfg 1.0, euler, seed 1234, the same prompt as the 2.5 measurement, one RTX 3090.

**The transformer is the only thing that changes.** Every arm uses the factory text encoder
(`gemma_3_12B_it`, Comfy-Org's BF16 repack, **not** an abliterated variant), the same video VAE,
audio VAE, vocoder and text projection — all read from files whose tensors are byte-identical to
the BF16 checkpoint's.

TBD_MEASUREMENT_TABLE

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1_w4a8.safetensors` | 16,651,422,654 | **15.51** | 1440 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256; 4507 tensors preserved |

Source: `ltx-2.3-22b-distilled-1.1.safetensors`, 46,149,345,334 B, sha256
`b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`, byte-for-byte the file
`Lightricks/LTX-2.3` publishes. Backend recorded in the sidecar: `comfy_kitchen.backends.cuda`.
Converted on an RTX 3080 Ti in 1296.5 s.

**Same layers as the 2.5 build, by construction.** LTX 2.3 and 2.5 share 76 identical 2-D weight
families with identical shapes (read from both headers), and the same profile selects the same
1440 Linear layers on both: every attention and feed-forward projection in the 48 transformer
blocks, video and audio streams and the cross-modal attentions included, plus the audio/video
embeddings connectors. Modulation (`adaln_single`, `scale_shift_table`), norms, patch/caption
projections and `to_gate_logits` are left in BF16.

## Running it

Load it with `CheckpointLoaderSimple` like the factory checkpoint — the file carries the VAEs and
the projection, and ComfyUI reads the quantization metadata the same way it reads the factory
`ltx-2.3-22b-dev-fp8`. Needs `--disable-dynamic-vram`. Text encoder: any Gemma 3 12B in ComfyUI's
single-file format via `LTXAVTextEncoderLoader`, pointed at this file for the projection.

## What is NOT covered

- **One prompt, one seed, one resolution (512), one frame rate, one card.** Four arms, not a sweep.
- **MAE, PSNR, SSIM and log-mel distance are distances from the reference, not judgements of
  quality**; no audio metric here was validated against human judgement, and audio-video synchrony
  is not measured. Watch and listen to the files.
- **The text encoder is BF16 in every arm.** A W4A8 build of the factory Gemma 3 12B exists on this
  bench (8.31 GiB, 336 layers); stock ComfyUI routes every text encoder's math through the
  dequantized path, so it saves memory and not time, and it is not part of this measurement.
- **No per-layer error analysis for this checkpoint** — no activation calibration was run on it.
- `convrot_groupsize` **256 only**; no mixed build.

## License and changes

LTX 2.3 is distributed under the **LTX-2 Community License Agreement** (January 5, 2026), and this
derivative is distributed **exclusively under those same terms**, as Section 3(b) requires. The
complete agreement ships beside these weights as `LICENSE_LTX_2_COMMUNITY.txt` — read it before use.
Two things in it bind you and anyone you pass this on to: **the use-based restrictions of Section 4
and Attachment A**, and **Section 2**, under which entities with annual revenues of at least
US$10,000,000 need a paid commercial licence from Lightricks to use LTX-2 or any derivative of it,
this file included.

**Statement of changes, per Section 3(c):** the 1440 Linear layers of the transformer were
re-encoded from BF16 into ComfyUI's `asym_w4a8_int8` format (4-bit weights, 8-bit activation path,
Hadamard rotation at group size 256). No weights were fine-tuned, no architecture was altered, and
every other tensor — video VAE, audio VAE, vocoder, text projection, modulation, norms, embeddings —
is preserved from the source byte-for-byte. All copyright and attribution notices of the source are
retained (Section 3(d)).

## Credits

- **[Lightricks](https://huggingface.co/Lightricks/LTX-2.3)** — LTX 2.3 itself and the BF16 weights
  this was converted from.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/ltx-2)** — the single-file Gemma 3 12B text encoder
  used in every arm, and the `ltx-2.3` templates the graph and sigmas were taken from.
- **[city96 / ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — the loader for the
  third-party GGUF arm.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format, the `MixedPrecisionOps` dispatch and
  the CUDA kernels that execute it.
- **ComfyUI-MultiGPU (DisTorch2)** — the block splitting that let the 43 GiB BF16 reference arm
  render on a 24 GB card.
