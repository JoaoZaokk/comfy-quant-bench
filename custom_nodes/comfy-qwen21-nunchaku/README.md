# comfy-qwen21-nunchaku

One ComfyUI node, **Qwen-Image 2.1 Nunchaku INT4 Loader** (`Qwen21NunchakuLoader`). It loads Qwen-Image-2.1
SVDQuant INT4 checkpoints in the `qwen21-nunchaku-svdq-int4-v1` format, for example
`qwen-image-2.1-int4-r128.safetensors` from
[mesmertech/Mesmer-Image-21-Nunchaku](https://huggingface.co/mesmertech/Mesmer-Image-21-Nunchaku).

That checkpoint packs the 224 block linears (`attn.to_q/k/v/to_out.0`, `img_mlp.proj/out/gate_layer`) for
Nunchaku's `SVDQW4A4Linear` and keeps everything else in BF16. Its author ships a custom runtime, a Docker
image. This node runs the same contract **inside ComfyUI's native Qwen-Image-2.1 model**
(`comfy.ldm.qwen_image21`, ComfyUI ≥ 0.37), which uses the same layer names:

1. The model is built on `meta`.
2. The 224 linears are swapped for `SVDQW4A4Linear` (int4, rank taken from the manifest).
3. The file is loaded with `strict=True`, and any tensor left without a value is refused.

Everything else is stock ComfyUI: attention, prefix KV cache, text encoder, VAE and sampler.

## Measured (RTX 3090, ComfyUI 0.37.4, Nunchaku 1.2.1, 1024², 25 steps, one seed)

| DiT | on the GPU | speed |
|---|---|---|
| `qwen_image_2.1_int8_convrot` (Comfy-Org) | 6.9 GB | 2.35 it/s |
| `qwen-image-2.1-int4-r128` via this node | ~4.6 GB | 2.44 it/s |

In one prompt and seed, the image keeps the int8 composition but shows 4-bit artefacts on fine bright edges
(doubled neon outlines). No metric against BF16 is published here.

## Use

- Put the file in `models/diffusion_models`. Use this node in place of `UNETLoader` in the official Qwen-Image
  2.1 workflow, with the `TextEncodeQwenImage21` / `CLIPLoader (qwen_image)` / VAE nodes unchanged.
- Requires [Nunchaku](https://github.com/nunchaku-tech/nunchaku) with `SVDQW4A4Linear`. INT4 runs on
  Ampere/Ada. The FP4 file of the same repo is Blackwell-only and is refused.
- The whole model moves to the GPU at once: the Nunchaku linears have no ComfyUI per-layer offload. LoRA is not
  supported.

The node code is GPL-3.0, like ComfyUI. The model weights keep their own license: Qwen Research License,
research and evaluation only.
