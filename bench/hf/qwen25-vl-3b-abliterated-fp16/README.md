---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated
library_name: diffusion-single-file
tags:
  - comfyui
  - text-encoder
  - omnigen2
  - abliterated
  - uncensored
  - experiment
---

# Qwen2.5-VL-3B abliterated — fp16 text encoder for OmniGen2 in ComfyUI

`huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated` repackaged with the tensor names, shapes and dtype (fp16) of ComfyUI's
`qwen_2.5_vl_fp16.safetensors`, so it drops into the OmniGen2 workflow in place of the original text encoder. **The
abliteration is huihui-ai's work; I only converted bf16 to fp16 and checked the names.**

> **License: Qwen RESEARCH LICENSE (non-commercial: research or evaluation only).** Modified files; see `LICENSE` and `NOTICE`.
> This is an uncensored model: refusals were removed from the text part, not the image part. You are responsible for what you do with it.

## What changed in the file

- 824 tensors, same names and shapes as ComfyUI's fp16 encoder; 71 of them differ in value: `embed_tokens`, 35 `o_proj` and 35 `down_proj`
  (the layers abliteration touches). Largest absolute difference 0.146. No inf/nan after the bf16 → fp16 cast.
- Put it in `models/text_encoders`, load it with `CLIPLoader` type `omnigen2`.

## What it does to OmniGen2

The conditioning changes for **every** prompt, not only the refused ones, and by different amounts. The same seed with the
original and the abliterated encoder (three neutral prompts, DiT fp16, 20 steps):

![](images/omnigen2_te_comparison.jpg)

PSNR between the two encoders on the same prompt and seed: 12.0–19.0 dB over six images. Some images change a lot, others little, and I
have not found out why; in my review it changes the result in a way I had not seen before. Treat it as an experiment. Time per step is unchanged (1.44 s against 1.42 s).

Credit: abliteration by huihui-ai; model by the Qwen team (Alibaba Cloud).
Tools and the measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**.
