---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
base_model_relation: quantized
tags:
  - comfyui
  - quantized
  - w4a8
  - w4a4
  - text-encoder
  - convrot
  - krea-2
library_name: comfyui
language:
  - en
---

# Qwen3-VL 4B as a ComfyUI text encoder — W4A8 (and a W4A4 you should not take)

The text encoder of the **Krea 2** pipeline in ComfyUI (`qwen3vl_4b`, Comfy-Org's single-file
repack of Qwen3-VL-4B-Instruct), re-encoded into ComfyUI's native per-layer quantization formats.
After the Krea 2 DiT drops to 7.5 GiB, this 8.27 GiB encoder is the largest file in the chain —
which is why it was the next thing measured.

| file | GiB | layers | format | verdict |
|---|---|---|---|---|
| `qwen3vl_4b_w4a8.safetensors` | **3.41** | 252 × `asym_w4a8_int8` (group 16, `convrot_groupsize` 256) | **take this one** |
| `qwen3vl_4b_w4a4_convrot.safetensors` | 3.19 | 252 × `convrot_w4a4` (`convrot_groupsize` 256) | measured, 2.5x less faithful, published as the comparison |

Source: `qwen3vl_4b_bf16.safetensors`, 8,875,719,384 B. The 252 layers are exactly 36 decoder
blocks × 7 Linears (`model.language_model.layers.N.{q,k,v,o}_proj`, `{gate,up,down}_proj`); the
vision tower, embeddings, norms and `lm_head` are preserved byte-for-byte. Backend recorded in the
sidecars: `comfy_kitchen.backends.cuda`. Method, tools and the full log:
**https://github.com/JoaoZaokk/comfy-quant-bench** (`bench/krea2_suite.md`, section 8).

## What it costs, measured against the BF16 original

Conditioning error (relative RMSE of the encoder's output against the BF16 encoder on the same
prompt) and encode time, **median of 5 encodes per cell, each arm in its own process**, RTX 3090.
Two prompt lengths, because the answer depends on it.

| encoder | GiB | 76 tokens: rel-RMSE / ms | 1220 tokens: rel-RMSE / ms |
|---|---|---|---|
| `qwen3vl_4b_bf16` (reference) | 8.27 | — / 96.4 | — / ~702 |
| **`qwen3vl_4b_w4a8`, quantized math** | **3.41** | **0.1438** / **87.2** | **0.5839** / **258.0** |
| `qwen3vl_4b_w4a8`, dequantized math (stock ComfyUI) | 3.41 | 0.1438 / 182.3 | 0.5848 / 787.2 |
| `qwen3vl_4b_w4a4_convrot`, quantized math | 3.19 | 0.5016 / 77.2 | 0.7393 / 215.5 |
| `qwen3vl_4b_w4a4_convrot`, dequantized math | 3.19 | 0.3637 / 97.2 | 0.7017 / 712.2 |

- **W4A8 is 2.53x more faithful than W4A4** at the short prompt (0.1438 against 0.3637) for
  0.22 GiB more, and running its math quantized costs it nothing measurable (0.14381 against
  0.14382) while making it **2.09x faster**. W4A4's quantized math is a real trade: 1.26x faster
  for 1.38x worse. Cosine with BF16 at 76 tokens, W4A8: 0.9896.
- **The error grows with the prompt** — 0.1438 at 76 tokens, 0.5839 at 1220 (cosine 0.8456) — and
  so does the speed win (2.09x → 3.05x against dequantized, 2.71x against BF16). The format is
  cheapest exactly where it is least accurate. The 1220-token prompt is synthetic and repetitive
  (nine clauses repeated six times), so the short-prompt column is the better founded of the two.
- **"Quantized math" needs a patched ComfyUI.** Stock ComfyUI runs every text encoder through the
  dequantized path behind two locks (`comfy/sd.py` forces float32 compute on every CLIP object and
  `comfy/sd1_clip.py` hardcodes `full_precision_mm=True`), so on stock ComfyUI these files **save
  memory and not time** — the "dequantized math" rows above are what you get. The method repo
  carries the patch that releases both locks only for checkpoints that really carry quantized
  layers (`patches/comfyui_text_encoder_quantized_math.patch`, opt-out flag
  `--disable-quantized-text-encoder`); a non-quantized encoder is untouched by it (control arm:
  0 quantized layers, `force_cast_weights` True in both arms).
- **Speed measurements here were noisy on the first pass and were re-done**: the first encode of
  any arm is 2.8–3.5x its median (warm-up), and the BF16 reference measured 1.54x apart between two
  single invocations. Everything above is a median of five with the reference repeated like the
  arms.

## Using it

`CLIPLoader` (or the Krea 2 template's loader) with `clip_type` for Krea 2, pointing at the file in
`ComfyUI/models/text_encoders/`. Needs the converter's `qwen3vl` profile lineage only at conversion
time; at load time ComfyUI reads the per-layer metadata itself. Krea 2 chain measured end to end
with this encoder in `bench/krea2_suite.md`.

## What is NOT covered

- Conditioning distance is not image quality: no image was rendered *for this card*; the Krea 2 DiT
  measurements in the sibling repo used the BF16 encoder.
- One card (RTX 3090), two prompts, one seed-free measurement (encoders are deterministic).
- No text-encoder LoRA, no vision input (the vision tower is preserved but was not exercised).

## Credits

- **[Qwen](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** — Qwen3-VL-4B-Instruct, Apache 2.0.
- **[Comfy-Org](https://huggingface.co/Comfy-Org)** — the single-file repack ComfyUI loads.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` and `convrot_w4a4` formats and the CUDA
  kernels that execute them.
