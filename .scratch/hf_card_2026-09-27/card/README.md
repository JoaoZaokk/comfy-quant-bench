---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - comfyui
  - diffusion-single-file
  - quantization
  - int8
  - w4a8
  - w4a4
  - qat
  - convrot
  - gguf
---

# Qwen-Image-2.1 DiT: quantizations for ComfyUI (W8A8, W8A16, W4A8, W4A16, W4A4)

These are quantized versions of the Qwen-Image-2.1 diffusion transformer (DiT only; the text encoder and VAE are
unchanged). The repo has eleven builds:

| Build | Format | Size | Speed vs BF16 (1024²) | Fidelity to BF16 | Pick it when |
|---|---|---|---|---|---|
| **int8 ConvRot** | W8A8 | 6.76 GiB | **2.3×** | ≈ BF16 (MS-SSIM 0.995) | You have ~8 GB for the DiT. Best choice on a 24 GB card. |
| **W4A8** | W4A8 | 3.91 GiB | 1.9× | close (0.932), clean text and neon | You need about 4 GB. |
| **mixed 0.10** | W4A4 on Q/K + W4A8 elsewhere | 3.83 GiB | 2.1× | same visual quality as W4A8 (0.917) | You need about 4 GB and want ~10% more speed than W4A8. |
| **W4A4 QAT** | W4A4 | 3.51 GiB | **3.1×** (same kernels as RTN) | lower (0.843); skin OK, text and neon damaged | Fastest option. Not for images with lettering. |
| W4A4 RTN | W4A4 | 3.51 GiB | 3.1× | lowest (0.821) | Reference only: the W4A4 build before QAT. |
| **W8A16** (native) | 8-bit weights, BF16 activations | 6.76 GiB | ≈ 1× (1.01 vs 1.05 it/s) | ≈ BF16 (0.993) | You want near-BF16 output in half the VRAM, with no activation quantization at all. |
| **W4A16 Q4_1** (native) | 4-bit weights (Q4_1 codes), BF16 activations | 4.32 GiB | ≈ 1× (1.02 it/s) ¶ | 0.913, same images as the GGUF Q4_1 | You liked the GGUF Q4_1: same output, 23% faster, no custom node. |
| **W4A16** (native) | 4-bit weights, BF16 activations | 3.91 GiB | ≈ 0.95× (0.99 it/s) § | like W4A8 (0.933), clean text and neon | You need about 4 GB and want activations untouched. |
| W8A16 Q8_0 (GGUF) | weight-only 8-bit | 7.16 GiB | 0.9× | ≈ BF16 (0.987) | You already use ComfyUI-GGUF. The native W8A16 is faster and slightly closer. |
| W4A16 Q4_1 (GGUF) | weight-only 4-bit | 4.32 GiB | 0.8× | 0.912, clean text and neon | You already use ComfyUI-GGUF. The native Q4_1 gives the same images, faster. |
| W8A8 rowwise | W8A8, no rotation | 6.76 GiB | 2.2× | 0.976 | Reference only: shows what the ConvRot rotation adds. |

The quality numbers are measured against BF16 renders with the same seed and prompt. The full protocol and every
number are below.

§ The native W4A16 speed needs a small comfy-kitchen fix (see [How to use](#how-to-use)). Without it, it runs at
0.84 it/s. ¶ The native Q4_1 needs a local ComfyUI patch and a comfy-kitchen patch (see [How to use](#how-to-use)).

**A16 vs A8/A4.** The W8A16 and W4A16 builds keep the activations in BF16. The weight is stored quantized and turned
back into BF16 right before each matmul, so they save VRAM but cannot be faster than BF16: the math is BF16. The
A8/A4 builds also quantize the activations and run on int8/int4 tensor cores, which is what makes them 2–3× faster.

![Overview, seed 42](images/overview_seed42.jpg)
*Same prompt and seed across all builds, 1024², 25 steps. Columns marked \* were rendered on the RTX 3080 Ti, the
rest on the RTX 3090 (see [Hardware](#evaluation-protocol)).*

<details><summary>Overview, seed 7</summary>

![Overview, seed 7](images/overview_seed7.jpg)
</details>

## Files

| File | Linear layers (192 in total) | Size |
|---|---|---|
| `qwen_image_2.1_int8_convrot.safetensors` | 192 × int8 ConvRot (W8A8) | 6.76 GiB |
| `qwen_image_2.1_w4a8.safetensors` | 192 × asym W4A8 (4-bit codebook weights, int8 activations) | 3.91 GiB |
| `qwen_image_2.1_mixed_p010.safetensors` | 68 × ConvRot W4A4 (Q/K) + 124 × W4A8 | 3.83 GiB |
| `qwen_image_2.1_w4a4_qat.safetensors` | 192 × ConvRot W4A4, block-wise QAT | 3.51 GiB |
| `qwen_image_2.1_w4a4_convrot_rtn.safetensors` | 192 × ConvRot W4A4, round-to-nearest (before QAT) | 3.51 GiB |
| `qwen_image_2.1_w8a16.safetensors` | 192 × int8 ConvRot weights, `full_precision_matrix_mult` (BF16 activations) | 6.76 GiB |
| `qwen_image_2.1_w4a16_q4_1.safetensors` | 192 × Q4_1 codes in comfy-kitchen's AWQ W4A16 layout (format `awq_w4a16`, group 32) | 4.32 GiB |
| `qwen_image_2.1_w4a16.safetensors` | 192 × W4A8 weights, `full_precision_matrix_mult` (BF16 activations) | 3.91 GiB |
| `qwen_image_2.1_w8a16_Q8_0.gguf` | 192 × GGUF Q8_0 (int8, one fp16 scale per 32 weights) | 7.16 GiB |
| `qwen_image_2.1_w4a16_Q4_1.gguf` | 192 × GGUF Q4_1 (uint4, fp16 scale and min per 32 weights) | 4.32 GiB |
| `qwen_image_2.1_w8a8_rowwise.safetensors` | 192 × int8 per row, weights and activations, no rotation | 6.76 GiB |
| `*.quant.json` | Build sidecars: source hash, per-layer choices, calibration settings | — |
| `qat.log`, `relatorio.json` | QAT training log and per-block validation numbers | — |

The other 73 weight tensors (norms, embeddings, modulation, final layer) stay in BF16. The safetensors builds use
ComfyUI's native quantized layout and need no custom node. The `.gguf` builds need
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

## How to use

1. Put the file in `ComfyUI/models/diffusion_models/`.
2. Load it with **Load Diffusion Model** (`UNETLoader`, weight dtype `default`). For the `.gguf` files, use
   **Unet Loader (GGUF)** from ComfyUI-GGUF instead (tested at commit `6ea2651`).
3. Use the standard Qwen-Image-2.1 text encoder and VAE from
   [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

**Native W8A16 / W4A16.** These need no ComfyUI patch. ComfyUI already honors a per-layer
`"full_precision_matrix_mult": true` in the quantization metadata:
- the weight stays quantized in VRAM;
- `comfy/ops.py` dequantizes it with comfy-kitchen's CUDA kernel;
- the matmul then runs in BF16.

The files carry exactly the same tensors as `int8_convrot` and `w4a8`; only that flag differs.

In comfy-kitchen 0.2.35 the W4A8 dequantization finishes with fp32 PyTorch ops over the whole matrix: 3.0 ms on the
largest layer. That makes the native W4A16 run at 0.84 it/s. The fix: the decoded weight already has the int8
ConvRot layout, so hand it to the existing fused kernel (0.64 ms, output identical to the fp32 reference rounded to
BF16). Add this in `comfy_kitchen/backends/cuda/__init__.py`, inside `dequantize_w4a8_int8_weight`, right after the
int4→int8 decode:

```python
    if correction is None and _should_use_convrot_dequant_kernel(int8_weight, int8_weight.shape[-1], convrot_groupsize):
        return dequantize_int8_convrot_weight_dtype(int8_weight, s_channel, convrot_groupsize, DTYPE_TO_CODE[output_dtype])
```

**Native Q4_1 (`awq_w4a16`).** This build stores the Q4_1 codes from gguf-py, the same codes as the GGUF file, in
comfy-kitchen's existing AWQ W4A16 layout:
- `weight`: int8 [N, K/2], two uint4 per byte
- `weight_scale`: bf16 [K/32, N], equal to d
- `weight_zeros`: bf16 [K/32, N], equal to m + 8d

So W = (q − 8)·scale + zeros is exactly Q4_1's q·d + m. Two local patches are needed; both are in
[comfy-quant-bench/patches](https://github.com/JoaoZaokk/comfy-quant-bench/tree/main/patches):
- `comfyui_awq_w4a16_format.patch`: ComfyUI 0.37.4 does not register that layout as a loadable format. This patch
  adds `awq_w4a16` to `comfy/quant_ops.py` and its loader branch to `comfy/ops.py`, about 20 lines.
- `comfy_kitchen_awq_w4a16_triton.patch`: at DiT batch sizes (M > 256), comfy-kitchen dequantizes this layout with a
  chain of PyTorch ops (5.7 ms on the largest layer). This patch adds a fused Triton kernel (0.35 ms) and keeps the
  cuBLAS matmul. Without it, the build runs but at GGUF-like speed.

Tested with:
- ComfyUI 0.37.4
- comfy-kitchen 0.2.35
- torch 2.13.0+cu130
- RTX 3090 and RTX 3080 Ti (Ampere)

The W4A4 and mixed builds need comfy-kitchen's native ConvRot W4A4 CUDA kernel. Other GPU generations were not
tested.

Settings used for every image on this page:
- `euler` / `simple`, 25 steps, cfg 1.0, 1024×1024
- `TextEncodeQwenImage21` with an empty negative prompt
- the model's default sampling shift

> **Known ComfyUI issue: large models on a network drive with dynamic VRAM.** With ComfyUI 0.37.4 and
> comfy-aimdo 0.5.5 in dynamic VRAM mode, loading the 13 GB BF16 from a network share aborted the process.
> The pinned host-buffer copy (`hostbuf_read_file_slice`) failed with an out-of-memory error. The quantized builds
> in this repo loaded fine from the same share, only slower on the first image. To avoid it:
> - `--disable-pinned-memory` or `--disable-dynamic-vram`, or
> - keep the model on a local disk.
>
> Details: [Comfy-Org/ComfyUI#16223](https://github.com/Comfy-Org/ComfyUI/issues/16223).

## Results

### Speed and fidelity

![Speed vs fidelity](images/speed_vs_fidelity.png)

Protocol for this table:
- RTX 3090, `--disable-dynamic-vram`.
- 12 images per build: 6 prompts × seeds 42 and 7.
- Metrics are computed against BF16 renders with the same prompt and seed. MS-SSIM shows the mean, with the worst
  image in brackets.

| Build | it/s 1024² | s/image (warm) | s/it 2048² | MS-SSIM (min) | SSIM | PSNR |
|---|---|---|---|---|---|---|
| BF16 (not in this repo) | 1.05 | 25.8 | 4.50 | 1 | 1 | ∞ |
| int8 ConvRot | 2.40 | 11.7 | 2.66 † | 0.995 (0.987) | 0.989 | 38.8 dB |
| W4A8 | 2.01 | 13.5 | 2.86 | 0.932 (0.842) | 0.919 | 25.0 dB |
| mixed 0.10 | 2.21 | 12.5 | 2.78 | 0.917 (0.854) | 0.894 | 24.1 dB |
| W4A4 RTN | **3.26** | **8.7** | **2.13** | 0.821 (0.695) | 0.725 | 20.2 dB |
| W4A4 QAT ‡ | same kernels as RTN | — | — | 0.843 (0.740) | 0.791 | 20.2 dB |
| W8A16 native | 1.01 | 26.6 | — | 0.993 (0.973) | 0.989 | 38.1 dB |
| W4A16 native § | 0.99 | 27.0 | — | 0.933 (0.840) | 0.919 | 25.1 dB |
| W4A16 Q4_1 native ¶ | 1.02 | 26.2 | — | 0.913 (0.822) | 0.899 | 23.7 dB |
| W8A16 Q8_0 (GGUF) | 0.94 | 28.4 | — | 0.987 (0.896) | 0.985 | 38.4 dB |
| W4A16 Q4_1 (GGUF) | 0.83 | 31.1 | — | 0.912 (0.817) | 0.899 | 23.6 dB |
| W8A8 rowwise | 2.35 | 11.9 | — | 0.976 (0.927) | 0.960 | 30.3 dB |

Notes on the table:
- **†** The 2048² speed for int8 was measured on Comfy-Org's int8 file, which uses the same codes as this one
  (99.99995% identical).
- **‡** QAT fidelity comes from renders on the RTX 3080 Ti. As a control, the RTN build on the same card scored
  MS-SSIM 0.818, against 0.821 on the 3090. The QAT build uses the same format and kernels as RTN; its speed on
  the 3090 was not measured separately.
- **§** With the small comfy-kitchen fix described in [How to use](#how-to-use); 0.84 it/s without it.
- At 2048², the unquantized attention takes a larger share of the step time. Speed-ups shrink there: int8 goes
  from 2.3× to 1.7×.
- **Noise floor:** diffusion trajectories are chaotic, so a 1-ulp difference in weights moves the mean MS-SSIM by up
  to ~0.006 (single images can drop to 0.955). For example, the int8 build here and Comfy-Org's int8 give visibly different apples on one prompt.
  Differences smaller than that are not quality differences.

### Blind judge (MiMo V2.6 Pro)

Each image was scored 0–10 by a vision LLM:
- File names and metadata were stripped.
- The judge sees BF16 as just another anonymous image.

The judge's own noise is 0.75 points: the same image scored twice differs by that much on average. Scores are
comparable only **within one session**.

| Session | Build | overall | artifacts | text |
|---|---|---|---|---|
| A | BF16 | 8.08 | 8.50 | 8.5 |
| A | W4A8 | 8.00 | 8.33 | 8.5 |
| A | int8 ConvRot | 7.92 | 8.50 | 9.25 |
| A | W4A4 RTN | 6.08 | 6.17 | 5.5 |
| B | BF16 | 8.00 | — | 9.5 |
| B | W4A8 | 7.83 | — | 8.5 |
| B | mixed 0.10 | 7.83 | — | 8.25 |
| C | W4A4 RTN ‡ | 6.50 | 6.67 | 6.5 |
| C | W4A4 QAT ‡ | 6.67 | **7.67** | **4.5** |

What the scores say:
- int8, W4A8 and mixed 0.10 are within the judge's noise of BF16.
- W4A4 is clearly below. QAT removes artifacts but writes text worse.

## Visual comparisons

Each image below shows the same crop from every build. The top row is seed 42 and the bottom row is seed 7.

**Neon lettering.** Every build that quantizes the MLP activations to 4 bits draws a doubled neon stroke. Builds
with 8-bit activations do not. QAT does not fix this: seed 7 improves and seed 42 gets worse.
![Neon detail](images/detail_neon.jpg)

**Skin texture.** W4A4 RTN turns skin into crackled leather. QAT brings it back close to BF16.
![Skin detail](images/detail_skin.jpg)

**Typography.**
![Poster detail](images/detail_poster.jpg)

**Glass and fine highlights.**
![Glass detail](images/detail_glass.jpg)

**Counting and hands.** The prompt asks for three apples, two pears and a hand.
![Fruit detail](images/detail_fruit.jpg)

## Weight-only (W8A16, W4A16) and W8A8 without rotation

Each image shows the same seed across builds; all of them were rendered on the RTX 3090. For each prompt, the top
row is seed 42 and the bottom row is seed 7.

![Weight-only and W8A8 detail](images/weightonly_detail.jpg)

![Weight-only overview, seed 42](images/weightonly_overview_seed42.jpg)

<details><summary>Overview, seed 7</summary>

![Weight-only overview, seed 7](images/weightonly_overview_seed7.jpg)
</details>

| Build | it/s | MS-SSIM vs BF16 (min) | PSNR | VRAM (weights) |
|---|---|---|---|---|
| BF16 | 1.05 | 1 | — | 13.6 GB |
| **W8A16 native** | **1.01** | **0.993 (0.973)** | 38.1 dB | 6.9 GB |
| W8A16 GGUF Q8_0 | 0.94 | 0.987 (0.896) | 38.4 dB | 7.5 GB |
| **W4A16 native** (with the comfy-kitchen fix) | **0.99** | **0.933 (0.840)** | 25.1 dB | 4.0 GB |
| W4A16 native (comfy-kitchen 0.2.35 as shipped) | 0.84 | 0.935 (0.839) | 25.2 dB | 4.0 GB |
| **W4A16 Q4_1 native** (both patches) | **1.02** | 0.913 (0.822) | 23.7 dB | 4.3 GB |
| W4A16 GGUF Q4_1 | 0.83 | 0.912 (0.817) | 23.6 dB | 4.5 GB |

What this adds:
- **Native beats GGUF on both axes.** W8A16 native runs at 96% of BF16 speed, against 90% for Q8_0. W4A16 native
  runs at 94% of BF16 speed, against 79% for Q4_1. Both native builds are also closer to BF16.
  - The difference is the kernel. ComfyUI-GGUF dequantizes with a chain of PyTorch ops. comfy-kitchen dequantizes
    in one fused CUDA kernel, which costs about 4–6% of a BF16 matmul on the largest layer.
- **Q4_1 native = GGUF Q4_1, faster.** Same codes, so the images are nearly the same: MS-SSIM 0.9925 between the
  native and GGUF renders, with the rest coming from the scales stored in BF16 instead of FP16. The native build
  runs at 1.02 it/s against 0.83.
- **No doubled neon stroke in any A16 build.** The weights are 4-bit like W4A8 and W4A4, but the activations stay
  BF16. This confirms that the neon artifact comes from 4-bit *activations*, not 4-bit weights.
- **W4A16 native looks like W4A8.** They share the weights, and 8-bit activations were already nearly lossless.
  Compared with W4A8, it keeps activations untouched at half the speed.
  - Q4_1 has different weights and drifts differently: a different face on the portrait with seed 42. It shows no
    visible artifacts, but it is farther from BF16 (0.912).
- **W8A16 native ≈ W8A8 ConvRot in fidelity** (0.993 against 0.995, within the noise floor). The native build's
  worst image is 0.973, against Q8_0's one drifted portrait at 0.896.
- **The ConvRot rotation is worth it.** W8A8 without rotation runs at the same speed (2.35 against 2.40 it/s)
  but lands at 0.976 / 30.3 dB against 0.995 / 38.8 dB. Without rotation, a few activation channels with large
  outliers set the int8 scale for every token.
- **The GGUF builds are slower than BF16 here:** 0.94 and 0.83 it/s against 1.05. They pay a dequantization on
  every matmul and gain no tensor-core speed. Their benefit is VRAM: 7.5 GB and 4.5 GB resident against 13.6 GB.

## QAT: before and after

For each block, the W4A4 weights were fine-tuned so the quantized block reproduces the BF16 block's output.
Both W4A4 columns were rendered on the same GPU (RTX 3080 Ti) with the same seeds; BF16 is shown as the reference.

**Skin: fixed.**
![QAT skin](images/qat_skin.jpg)

**Glass: more faithful.**
![QAT glass](images/qat_glass.jpg)

**Neon: not fixed** (mixed across seeds).
![QAT neon](images/qat_neon.jpg)

**Typography: seed 42 gets back BF16's thin serif weight; seed 7 misspells it as "MORINNGS".**
![QAT poster](images/qat_poster.jpg)

QAT numbers against RTN:

| | W4A4 RTN | W4A4 QAT |
|---|---|---|
| Validation error per block | RTN | **−8.5% on average** (32/32 blocks improved, −1.9% to −51.5%) |
| End-to-end latent error, 6 held-out prompts | 0.3109 | 0.3071 (−1.2%; 3 better, 3 worse) |
| MS-SSIM vs BF16, mean (min) | 0.818 (0.690) | **0.843 (0.740)** |
| SSIM vs BF16 | 0.722 | **0.791** |
| Images where QAT is closer to BF16 | | 7 / 12 |
| Blind judge: artifacts / text | 6.67 / 6.5 | **7.67** / 4.5 |

**Verdict:** a real but partial gain. Skin, textures and composition get much closer to BF16. Lettering and
neon do not.

For images with text, use mixed 0.10 or W4A8. The neon artifact comes from 4-bit activations in the MLP, and
those builds keep the MLP at 8-bit activations.

## Method

**ConvRot quantization.** Every linear layer's input dimension is rotated with a grouped Hadamard transform
(group size 256), which spreads outliers across channels. Then:
- **Weights:** per-row absmax int4 (or int8).
- **Activations:** quantized per token at runtime, int4 or int8, in the rotated space.

Codes are computed from the **FP32** weights. An earlier version of our converter rotated in BF16. That changed
~8% of int8 codes by ±1–2 and cost ~5% of per-layer error.

With the FP32 fix, our converters reproduce existing public files code for code:
- Comfy-Org's int8: 99.99995% of codes identical.
- NidAll's W4A8: 99.9998% identical, with the same codebook.

**Per-layer output error.** Relative RMSE of each layer's output, measured with real kernels on real calibration
activations from 6 held-out prompts:

| Build | mean | worst | to_q | to_k | to_v | to_out | gate_up | mlp.out |
|---|---|---|---|---|---|---|---|---|
| int8 ConvRot | 0.0083 | 0.019 | 0.005 | 0.005 | 0.011 | 0.013 | 0.007 | 0.008 |
| W4A8 | 0.0424 | 0.087 | 0.027 | 0.024 | 0.062 | 0.061 | 0.040 | 0.042 |
| W4A4 RTN | 0.1398 | 0.341 | 0.081 | 0.072 | 0.193 | 0.230 | 0.122 | 0.141 |

W4A8 and W4A4 share the same 4-bit weights. The 3× error gap between them comes entirely from the activations.

**mixed 0.10.** Layers are chosen by calibrated error:
- A layer stays W4A4 only if its W4A4 output error is at most 0.10. Otherwise it is promoted to W4A8.
- Result: 68 layers stay W4A4, almost all `to_q` and `to_k`. `to_v`, `to_out`, `gate_up` and `mlp.out` mostly
  become W4A8.

A looser threshold (0.15, which also left the MLP `gate_up` in W4A4) brought back the doubled neon stroke. That
is how the artifact was traced to the MLP.

**GGUF builds.** The same 192 layers are quantized from FP32 with gguf-py 0.19, the reference NumPy
implementation of llama.cpp's formats. All other tensors are kept in BF16, and the GGUF architecture tag is
`qwen_image`. The model itself is detected from its keys. Measured weight error (relative RMSE):
- Q8_0: 0.0057 on average
- Q4_1: 0.081 on average

K-quants such as Q4_K_M are not implemented in gguf-py, so they were not produced.

**Block-wise QAT** (BRECQ / OmniQuant-style reconstruction):
- **Teacher:** the BF16 DiT. **Student:** the same block with simulated ConvRot W4A4 on both weights and activations
  (straight-through estimator, FP32 master weights).
- Blocks are trained in order. Each block's input comes from the already-quantized blocks before it.
- **Data:** 48 training prompts (text/neon, skin, hands, landscapes, products, illustration). None of the evaluation
  prompts are among them.
  - Teacher trajectories: 25 steps, 4 timesteps per trajectory, 192 samples in total (16 for validation).
- **Optimizer:** AdamW, lr 1e-5, 300 steps per block.
- A block keeps its trained weights only if validation error drops. Otherwise it reverts to RTN. All 32 improved.
- Export uses the native `convrot_w4a4` layout: same tensors, same size and same kernels as the RTN build.
- Compute: Colab G4 (RTX PRO 6000), ~45 min.

## Evaluation protocol

- **Prompts:** 6 prompts, none used for calibration or training:
  - a neon sign reading "QWEN IMAGE 2.1"
  - an elderly fisherman portrait
  - a "SLOW MORNINGS" poster
  - an aerial river view
  - a glass perfume bottle
  - apples and pears with a hand
- **Seeds and settings:** seeds 42 and 7, 1024², 25 steps.
- **Reference:** BF16 renders with the same seed, the same graph and the same runtime. The runtime is
  deterministic: two BF16 files with differently fused MLP weights gave pixel-identical images 12/12.
- **Hardware:**
  - RTX 3090 for all builds except the QAT comparison.
  - QAT and its RTN control both ran on the RTX 3080 Ti.
- **Metrics:** MS-SSIM, SSIM and PSNR against BF16, plus the blind LLM judge above.
- **Visual review:** every image was also inspected side by side (not blind).

## Limitations

- One QAT run with one data shuffle and only 192 samples. More steps per block and more text-heavy samples may help.
  QAT on top of mixed 0.10 was not tried.
- The blind judge cannot reliably rank more than about four similar images. Only its scores are reported, not its
  rankings.
- Metrics against BF16 measure faithfulness, not beauty. A build can differ from BF16 and still look fine, as
  W4A8 often does.
- Only Ampere GPUs were tested.
- The A16 builds and W8A8 rowwise were not scored by the blind judge, and their speed was not measured at 2048².
- The native W4A16 speed depends on a local comfy-kitchen fix that is not upstream yet.

## License

Derived from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). The base model's
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) applies.
