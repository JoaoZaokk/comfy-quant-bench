---
license: apache-2.0
base_model: ostris/Z-Image-De-Turbo
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

# Z-Image De-Turbo — ConvRot 4-bit for ComfyUI, 3.75x lighter and 2.62x faster per step

Native `convrot_w4a4` and mixed 4/8-bit builds of `ostris/Z-Image-De-Turbo`, the de-distilled
fine-tune of Z-Image Turbo that people train LoRAs on. ComfyUI's own quantization format — stock
`UNETLoader`, no custom node.

**11.46 GiB → 3.06 GiB**, and **0.911 → 0.348 s/step** on an RTX 3090 at 1024×1024.
Thirty-six renders across six prompts and two seeds, none broken.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## The files

| file | format | GiB | median per-layer error | latent divergence | s/step |
|---|---|---|---|---|---|
| `zimage_deturbo_w4a4.safetensors` | 170 × `convrot_w4a4` | **3.06** | **0.1211** | 0.4126 | **0.348** |
| `zimage_deturbo_mixed.safetensors` | 116 × 4-bit + 54 × 8-bit | 3.17 | — | 0.4325 | 0.398 |

Source: 11.46 GiB BF16, 6.15 B parameters. Eight steps, 1024×1024, euler/simple, cfg 1.0.

```
err_bf16   median 0.0019     <- the floor. This much is not quantization.
err_w4a4   median 0.1211
err_w4a8   median 0.0384
```

**Dispatch counted with the weights on the card:** 170 quantized modules, 8/8 quantized forwards,
**0 `dequantize`**, `convrot_linear_dtype=int4` on `comfy_kitchen.backends.cuda`.

![](images/grid_seed1.png)

| BF16 reference | W4A4 | mixed |
|---|---|---|
| ![](images/portrait_bf16.png) | ![](images/portrait_w4a4.png) | ![](images/portrait_mixed.png) |

---

## Take W4A4 here. The mixed build does not earn its extra bits on this checkpoint.

```
paired against zimage_deturbo_w4a4, run by run
zimage_deturbo_mixed    mean delta +0.0199    wins 7/12
```

Seven of twelve is a coin flip, and the tool that produced it prints `Split decisions, i.e. not
separated at 12 run(s)` rather than a winner. So the mixed build costs 0.11 GiB and 14% more time
per step for no measurable gain **here**.

**That is the interesting part, because the same recipe does earn its keep on the parent model.**
Run on the official `Tongyi-MAI/Z-Image-Turbo` at the identical threshold, the mixed build wins
**11 of 12** paired runs. Same converter, same `--promote-error 0.15`, same six prompts, same two
seeds, same card — a model and its own de-distilled fine-tune disagree about whether promoting
~55 layers to 8 bits buys anything.

Nothing about the per-layer error predicted that: the two checkpoints measure 0.1228 and 0.1211,
within 1.4% of each other. Whatever separates them is not in that number.

One more reason not to read too much into the means: the between-run spread inside a single arm is
0.37–0.42 on this checkpoint, which is **larger than the 0.02 gap between the arm means**. Always
read the paired row.

---

## Using them

Drop into `ComfyUI/models/diffusion_models/` and load with `UNETLoader`.

```
text encoder   qwen_3_4b.safetensors
VAE            ae.safetensors
sampler        euler / simple, cfg 1.0
```

De-Turbo is **not** distilled — it exists precisely to undo the turbo distillation so LoRA training
behaves. The 8-step / cfg 1.0 settings above are what these measurements used, for comparability
with the Turbo builds; for actual generation this model wants more steps and real cfg.

A quantized `qwen_3_4b` text encoder is published separately at
[JoaoZaokk/Qwen3-4B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen3-4B-W4A4-ConvRot).

---

## What this does not cover

- **Six prompts, two seeds, one card, one resolution, no perceptual metric.** "36 of 36 usable" is
  a human judgement over contact sheets.
- **LoRA training is this model's whole reason to exist and was not tested.** Nothing here says
  whether a LoRA trains, or trains as well, on 4-bit weights.
- **Sampled at 8 steps / cfg 1.0**, which is the Turbo regime and not this model's own.
- **No image-space metric.** Latent divergence measures trajectory, not fidelity.
- **The calibration used one concatenated prompt block**, not six separate ones — a defect in the
  activation-capture tool, found and fixed the same day these were built. It is the same method the
  Turbo builds and the published `beyond-reality-zimage-v2` used, so the family stays internally
  comparable; a fresh calibration would move the medians a few percent.
