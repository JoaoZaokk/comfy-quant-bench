---
license: gemma
base_model: DreamFast/gemma-3-12b-it-heretic
tags:
  - comfyui
  - quantized
  - w4a8
  - w4a4
  - smoothquant
  - text-encoder
  - ltx-2
  - convrot
library_name: comfyui
---

# Gemma 3 12B it Heretic — W4A8 (asym_w4a8_int8) for ComfyUI

4-bit weights with an 8-bit activation path, in ComfyUI's native per-layer quantization format.
**21.93 GiB → 7.53 GiB, 2.91x lighter**, and on the path ComfyUI takes by default it costs **no
extra time at all**.

    source          gemma_3_12B_it_heretic.safetensors     23,545,681,250 B
    this file       gemma_3_12B_it_heretic_w4a8             8,089,619,138 B
    quantized       336 layers, asym_w4a8_int8, group_size 16
    preserved       293 tensors, byte-identical to the source

Converted with `tools/quant_w4a8.py` on an RTX 3090 (sm86), comfy-kitchen 0.2.31,
torch 2.13.0+cu130, backend `comfy_kitchen.backends.cuda`. 25.5 s.

## What it costs, measured against the BF16 original

Three arms on the same prompts, same card, same day. **A** = BF16 original, **B** = this file
loaded the way ComfyUI loads it, **C** = this file with ComfyUI's two text-encoder locks
released.

    quantizing the WEIGHT already costs      1.1753e-1   (B against A)
    releasing the locks ADDS                 1.8393e-1   (C against B)
    total for whoever releases them          2.0684e-1   (C against A)
    cosine with BF16, prompt 0               0.9967 locked / 0.9780 released

## The memory is free; the speed needs the locks released

    encode, prompt 0, median of 5      BF16 1876.8 ms   locked 1835.7 ms   released 490.8 ms

**Locked is the same time as BF16** (1835.7 against 1876.8), so 13.8 GiB of VRAM are saved at no
cost in seconds. The 3.74x needs the locks released.

Stock ComfyUI keeps them shut, and it takes both: `comfy/sd.py` calls
`set_model_compute_dtype(torch.float32)` for every CLIP object, which turns on
`comfy_force_cast_weights`, and `comfy/sd1_clip.py` hardcodes `full_precision_mm=True` for every
text encoder. Either one alone sends the math down the dequantized path. Counted on a real forward
through the normal loader: **0 quantized forwards, 336 `dequantize` calls.** Memory saved, kernel
never reached.

That is a property of being a *text encoder* in ComfyUI, not of this file or of W4A8 — the same
count comes out of W4A4 encoders from other authors.

> **Re-measured through the flag on 2026-09-14 — this replaces the correction block that stood
> here.** The first version of this card said the released path was reachable only by a post-load
> monkeypatch and that ComfyUI offered no way to do it. Both locks are now released at the source
> by a patch kept in the method repo (`patches/comfyui_text_encoder_quantized_math.patch`, opt-out
> `--disable-quantized-text-encoder`), applied only to a checkpoint that carries quantized layers.
> Measured on the LTX 2.3 conditioning this encoder exists for — one prompt, 27 + 13 tokens, the
> LTX 2.3 text projection applied, relative L2 (positive / negative prompt) against the **heretic
> BF16** encoded the same way, on the same RTX 3090:
>
> | build | released (quantized math) | locked (stock ComfyUI, dequantized) |
> |---|---|---|
> | this file, W4A8 | 0.0421 / 0.0410 | 0.0427 / 0.0402 |
> | W4A4 ConvRot (see below) | 0.2207 / 0.2453 | 0.1111 / 0.0924 |
> | W4A4 SmoothQuant (see below) | 0.1583 / 0.1480 | 0.0818 / 0.0740 |
>
> **On W4A8 the release costs nothing measurable** — 0.001, and not in a consistent direction — so
> the 3.7x in the table above comes free. **On the two W4A4 builds it costs 2x**: the 4-bit
> activation is the term that hurts; the 4-bit weight alone (the locked column) already costs 2–2.6x
> what W4A8's does. The monkeypatch numbers above said releasing W4A8 added 1.84e-1 on the encoder's
> own output; the flag measures 0.001 on the projected conditioning. Same file, different tensor,
> different instrument, and they do not agree — left here as an open discrepancy rather than
> silently replaced.

## The two W4A4 builds, measured at the output on 2026-09-14: they change the scene

Two more quantizations of the same heretic BF16 were converted on this bench — `ConvRot W4A4`
(336 layers, 6.91 GiB, `convrot_groupsize` 256) and the same with SmoothQuant channel smoothing
applied first (α 0.5, six calibration prompts, activation channel outlier ratio 82.5 → 9.6). Both
were measured the way this card's W4A8 was, on the LTX 2.3 conditioning and at the output, against
the heretic BF16 encoder; criterion D in the method repo's `bench/criterio_fechamento_2026-09-14.md`
was written before the numbers.

| encoder build | conditioning rel-L2 vs heretic BF16, released / locked | render vs the BF16-encoder render: MAE | SSIM | log-mel | what the frames show |
|---|---|---|---|---|---|
| W4A8 (this file) | 0.042 / 0.043 | **5.45** [4.94–5.99] | 0.915 | 0.080 | the same dusk silhouette |
| W4A4 SmoothQuant | 0.158 / 0.082 | 29.82 [29.19–30.63] | 0.645 | 0.220 | **a different scene**: daylight, blue sky, white clouds |
| W4A4 ConvRot | 0.221 / 0.111 | 38.12 [35.89–40.41] | 0.567 | 0.193 | **a different scene**, the same daylight reading |
| *scale: the factory Gemma BF16 instead of the heretic* | 0.100 | 9.34 | 0.832 | 0.131 | the same scene |

Reference: the LTX 2.3 22B W4A8 transformer, 249 frames + audio, 8 steps, seed 1234, on the heretic
BF16 encoder's conditioning; the quantized arms differ only in the encoder. Every render on the
quantized-math (released) path; the sampler ran 2.30 s/it in all four, because the encoder is not
in the process. Proofs: `ltx23_encoder_test/render_contato_av.png`, `render_comparacao_av.json`,
`conditioning_vs_bf16.json`; the heretic-against-factory pair in
`heretic_vs_factory_contato_av.png` / `_comparacao_av.json`.

- **The order the criterion predicted holds — W4A8 < SmoothQuant < ConvRot — on the conditioning,
  on both math paths, and at the output.** Smoothing the channels before the 4-bit rotation buys
  1.4x on the conditioning (0.158 against 0.221) and 1.3x at the output (29.8 against 38.1). That
  answers a question this bench left open on 2026-09-01: channel smoothing is a real term, and it
  is not large enough to rescue W4A4 here.
- **Both W4A4 builds render a coherent, well-lit, *wrong* picture.** Not noise: a lighthouse on a
  rocky island under a blue sky with white clouds, where the prompt said *dusk* and the BF16, the
  W4A8 and the factory encoders all drew the dusk silhouette. The conditioning drifted enough
  (cosine 0.977–0.988) to lose a word. The abliterated BF16 sits 0.10 from the factory BF16 and
  keeps the scene; the flip is somewhere between 0.10 and 0.16 of relative L2 on this prompt — one
  prompt, one seed, so a range, not a line.
- **The W4A4 weights are therefore not in this repo.** Their conditioning numbers, the contact
  sheet and the comparison JSON are, so the negative is on the record with its proof — and so
  are their sidecars (`w4a4_not_published/`), so the exact format of what was measured is on
  record too.
- **On stock ComfyUI's locked path (4-bit weight, dequantized math) both W4A4 builds keep the
  scene** — rendered afterwards on the same transformer and seed: ConvRot **16.38** MAE [15.43–17.37],
  SSIM 0.786, log-mel 0.111; SmoothQuant **27.41** [26.33–28.46], SSIM 0.699, log-mel 0.160; dusk
  silhouette in both (`ltx23_encoder_test/locked_render_contato_av.png`, `_comparacao_av.json`, with
  the released renders on the same sheet). So the scene flips somewhere between relative L2 0.11
  (locked ConvRot, scene kept) and 0.16 (released SmoothQuant, scene lost). Two things this does
  not make them: the render order on the locked path (ConvRot closer than SmoothQuant) inverts the
  conditioning order (SmoothQuant 0.082 closer than ConvRot 0.111) — one seed, noted, not
  explained — and at 3–5x this file's distance, for 8 % less memory than this file and no speed
  on that path, nothing recommends them. The weights stay off the Hub.

## Not covered

One model, one card (sm86). The 2026-08-31 table: one prompt set, no image. The 2026-09-14
numbers: one prompt (27 + 13 tokens) at the conditioning, one prompt, one seed, 512 px and one
transformer at the output. MAE, SSIM and log-mel are distances from the BF16-encoder render, not
judgements of quality; "a different scene" is what a person saw on the contact sheet. The two
measurements disagree on what releasing the locks costs W4A8 (0.18 on the encoder's raw output
through a monkeypatch; 0.001 on the projected conditioning through the patch), and that is left
open rather than resolved by deleting one of them. Whether releasing the locks is safe for output
quality is now answered for this file (W4A8: yes, at 5.45 MAE with the scene intact) and for the
two W4A4 builds on the released path (no); the W4A4 builds on the locked path are covered by the
sentence above, and nothing else.
