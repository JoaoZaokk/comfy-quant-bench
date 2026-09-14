---
license: gemma
base_model: google/gemma-3-12b-it
tags:
  - comfyui
  - quantized
  - w4a8
  - text-encoder
  - ltx-2
  - convrot
library_name: comfyui
---

# Gemma 3 12B it — W4A8 (asym_w4a8_int8) for ComfyUI, measured as the LTX 2.3 text encoder

The factory Gemma 3 12B instruct — Comfy-Org's single-file BF16 repack that LTX-2.x workflows load
through `LTXAVTextEncoderLoader` — with 4-bit weights and an 8-bit activation path, in ComfyUI's
native per-layer quantization format. **22.70 GiB → 8.31 GiB, 2.73x lighter.** Unlike the
[heretic build](https://huggingface.co/JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8) published earlier,
this one was **measured at the output**: the prompt it encodes was rendered into a 10-second video
with sound by the LTX 2.3 22B transformer and compared, frame by frame and in the spectrogram, with
the same render on the BF16 encoder's conditioning.

    source          gemma_3_12B_it.safetensors       24,379,468,890 B   (Comfy-Org/ltx-2, text_encoders/)
    this file       gemma_3_12B_it_w4a8.safetensors   8,923,405,898 B
    quantized       336 layers, asym_w4a8_int8, group_size 16 (q/k/v/o and gate/up/down of all 48 blocks)
    preserved       730 tensors, byte-identical to the source (embeddings, norms, vision tower)

Converted with `tools/quant_w4a8.py` on an RTX 3080 Ti (sm86), comfy-kitchen 0.2.31,
torch 2.13.0+cu130, backend `comfy_kitchen.backends.cuda`, 334.3 s. Method repo:
**https://github.com/JoaoZaokk/comfy-convrot-w4a4** (`bench/criterio_fechamento_2026-09-14.md`,
criterion A, written before the numbers).

## What it costs, measured at the conditioning and at the output

One prompt (*a lone lighthouse on a rocky cliff at dusk, waves breaking against the rocks, the beam
sweeping across low clouds, seabirds circling*; negative *blurry, out of focus, low contrast,
washed out*), 27 + 13 tokens, encoded with the LTX 2.3 text projection into the `[1, 27, 6144]`
context the transformer reads. The BF16 reference conditioning was encoded on the RTX 3080 Ti
(ComfyUI's partial load, 22.7 GB streamed), this file on the RTX 3090, both by the same bare-process
encoder (`tools/ltx_encode_lowcommit.py`, every encoder option saved beside the tensor).

**Two paths, because ComfyUI has two.** Stock ComfyUI keeps two locks shut on every text encoder
(`comfy/sd.py` sets `comfy_force_cast_weights` through `set_model_compute_dtype(torch.float32)`;
`comfy/sd1_clip.py` hardcodes `full_precision_mm=True`), so a quantized encoder saves memory and
runs **dequantized** math. The method repo carries a patch
(`patches/comfyui_text_encoder_quantized_math.patch`) that releases both at the source for a
checkpoint that actually carries quantized layers, with `--disable-quantized-text-encoder` to get
the stock behaviour back. Both were encoded; **the released path is what was rendered.**

### Conditioning against the BF16 encoder

| path | rel-L2 (pos / neg) | cosine (pos / neg) | max abs diff | bit-equal in bf16 |
|---|---|---|---|---|
| released (quantized math, this project's ComfyUI) | 0.0429 / 0.0403 | 0.99907 / 0.99919 | 7 | 4.2 % / 4.9 % |
| locked (dequantized math, stock ComfyUI) | 0.0420 / 0.0396 | 0.99911 / 0.99921 | 7 | 4.4 % / 5.0 % |
| *scale: the abliterated ("heretic") Gemma 3 12B in BF16, same prompt* | 0.0997 / 0.0929 | 0.99506 / 0.99568 | 12 | 1.4 % / 1.4 % |
| *floor: the same BF16 encoder on the other card* | 1.0e-3 | — | 1.0 | 87 % |

The 4-bit weight costs 0.04 of relative L2 on the conditioning, on either path; **the 8-bit
activation path adds nothing measurable** — released and locked differ by 0.001, in both
directions across the two prompts. For scale, swapping the factory model for its abliterated twin,
both in BF16, moves the conditioning 2.3x further than this quantization does. (The criterion had
predicted 0.05–0.30 for the weight cost; 0.043 fell just under the band, on the side that matters.)
One thing this does not settle: the heretic card's 2026-08-31 measurement, taken through a
post-load monkeypatch on the encoder's raw output, said releasing the locks on W4A8 added 0.18 of
relative error. Different tensor, different instrument, opposite conclusion; it is recorded on that
card as an open discrepancy rather than overwritten.

### The render: LTX 2.3 22B W4A8, 249 frames + audio, 8 steps, same seed

Reference: the same transformer, same seed, same sigmas, on the BF16 encoder's conditioning
(`JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot`, its `av/` set).

| arm | MAE vs BF16-encoder render | PSNR | SSIM | log-mel L1 | waveform SNR | lag | level |
|---|---|---|---|---|---|---|---|
| this file as encoder (released path) | **6.75** [6.40–7.15] | 24.67 dB | 0.894 | **0.087** | 4.3 dB | 0 ms | −23.8 dBFS |
| *scale: the transformer quantized instead (W4A8 vs BF16 transformer, same conditioning)* | 10.39 | 21.89 dB | 0.829 | 0.163 | −1 dB | 0 ms | −24.0 dBFS |
| *floor: saved conditioning vs the live encoder, nothing quantized* | 1.74 | 35.7 dB | 0.977 | 0.074 | 4.5 dB | 0 ms | −24.0 dBFS |
| *control: silence* | — | — | — | 8.38 | 0 dB | — | — |
| *control: white noise at the reference's RMS* | — | — | — | 1.73 | −3.0 dB | — | — |

Same scene, same dusk light, same lighthouse and rocks (`ltx23_encoder_test/render_contato_av.png`).
The render moved **6.75 MAE** from the BF16-encoder render — **less than the transformer's own
quantization moves it (10.39)** and four times the floor of saved-against-live conditioning (1.74);
the sound moved 0.087 in log-mel, half of what the transformer quantization costs (0.163), level
unchanged, no lag. Movement 1.90 against the reference's 2.02: nothing froze. The criterion (A2,
written before the render) asked for the same scene with MAE between 3 and 12 and log-mel between
0.1 and 0.4: the picture landed inside the band, the sound just under it.

Proofs in this repo: `ltx23_encoder_test/conditioning_vs_bf16.json` (the conditioning table),
`ltx23_encoder_test/render_comparacao_av.json` and `render_contato_av.png` (the render, with the
controls).

## Speed

Not measured cleanly on this file. The one number this bench has for this architecture in this
format is on the heretic build — same 336 layers, same `asym_w4a8_int8`, same card family: locked
**1835.7 ms** against BF16 **1876.8 ms** (memory free, time unchanged) and released **490.8 ms**
(3.74x), median of 5 on one prompt, measured through a post-load monkeypatch before the patch
existed. Read it on that card; it is not restated here as this file's.

**On a 12 GB card it did not fit where the BF16 did.** With about 8 GB free on the RTX 3080 Ti,
loading this file for a single encode died with `CUDA error: out of memory` four times out of four
(this build and three 4-bit heretic builds), while the 22.7 GB BF16 passed on the same card through
ComfyUI's partial load. The quantized encodes were done on the RTX 3090. Not isolated: whether the
partial-load path simply does not cover quantized layers, or the dequantized activation is what
overflows.

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `gemma_3_12B_it_w4a8.safetensors` | 8,923,405,898 | **8.31** | 336 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256, codebook |
| `gemma_3_12B_it_w4a8.quant.json` | — | — | the sidecar the converter wrote (source, sizes, backend, versions, seconds) |

Load it with `LTXAVTextEncoderLoader` in place of `gemma_3_12B_it.safetensors`, with the LTX 2.3
(or 2.5) text projection file as usual. Stock ComfyUI runs it dequantized (memory saved, time not);
the method repo's patch runs the 8-bit path.

## What is NOT covered

- **One prompt, one seed, one resolution, one transformer (LTX 2.3 22B W4A8).** The BF16
  transformer was not rendered on this conditioning.
- **The render is on the released path only.** The locked path was measured at the conditioning
  and not rendered.
- **Distances, not judgements**: MAE, SSIM and log-mel say how far the render moved from the
  BF16-encoder render; no metric here was validated against a viewer or a listener; audio-video
  synchrony is not measured. Look at the contact sheet.
- **The reference conditioning and this file's were encoded on different cards** (3080 Ti / 3090);
  the same BF16 encoder on the two cards differs by rel-L2 1.0e-3, which is the floor row.
- No speed measured on this file; no per-layer error analysis (no activation calibration was run
  on this encoder).

## License and changes

Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
This file is a quantized derivative of `google/gemma-3-12b-it` as repacked by Comfy-Org for
ComfyUI: the 336 projection weights of the 48 transformer blocks were re-encoded from BF16 into
ComfyUI's `asym_w4a8_int8` format; every other tensor is byte-identical to the source. The Gemma
Prohibited Use Policy travels with it.

## Credits

- **Google DeepMind** — Gemma 3 12B it.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/ltx-2)** — the single-file BF16 repack this was
  converted from.
- **[Lightricks](https://huggingface.co/Lightricks/LTX-2.3)** — LTX 2.3, whose text projection and
  transformer this file was measured with.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format and the `MixedPrecisionOps` dispatch.
