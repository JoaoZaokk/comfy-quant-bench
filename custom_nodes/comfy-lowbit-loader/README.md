# comfy-lowbit-loader

A native ComfyUI loader for ternary (1.58-bit, stored in 2-bit slots) and binary (1-bit) diffusion models,
built for [Bonsai Image](https://huggingface.co/prism-ml) (FLUX.2-klein-4B trained to ternary/binary weights).

One node, **Load Diffusion Model (ternary / binary)**, figures out what it is given from the file contents:

| input | what it is | how it is read |
|---|---|---|
| gemlite pack | `transformer-gemlite-int2/state_dict.pt`, `...-int1/state_dict.pt` | `torch.load(weights_only=True)`; codes transposed to the canonical layout, fp32 scales kept |
| MLX pack | `transformer-packed-mflux/diffusion_pytorch_model.safetensors` | uint32 words are the canonical bytes, zero repacking |
| unpacked | BF16 `transformer/diffusion_pytorch_model.safetensors` | every group of 128 checked; layers that are exactly ternary/binary get packed, bit for bit |
| any BF16 ternary checkpoint | e.g. a BFL-named file from a ternary QAT run | same lossless packing |
| saved from this format | safetensors with `comfy_quant = lowbit_affine` | loads directly |

Diffusers-named FLUX.2 files are renamed to the BFL names ComfyUI uses (q/k/v fused, final modulation
halves swapped), so no conversion step or extra file is needed. Text encoder and VAE are the stock
klein ones (`qwen_3_4b` with type `flux2`, `flux2_klein_vae`).

## What you get

The node returns an ordinary `MODEL`. The quantized linears are a new comfy-kitchen layout
(`LowBitAffineLayout`, `W = code * scale + zero` per group of 128 along K) registered at import, so
every model-management feature applies without special casing:

- packed weights in VRAM **and** in the offload copy: ComfyUI's automatic offload / dynamic VRAM moves
  1-2 bit data, not BF16;
- `device`: run on any CUDA GPU or the CPU; `offload_device`: park the idle copy in RAM (default) or in
  a second GPU's VRAM;
- works with the core multi-GPU nodes and model saving (the saved file reloads through this node);
- fused dequantization: Triton on CUDA, a torch path everywhere else (CPU, or `LOWBIT_KERNEL=torch`).

Matmuls run in the compute dtype after dequantization (weight-only, "A16"). **Speed is at best that of
the BF16 model**; the gain is memory and offload bandwidth. A kernel with int8 activations would be the
way to go faster, and is not implemented.

## Measured (RTX 3090, klein-4B Bonsai, 1024², 4 steps, 2026-09-27)

- All 100 quantized layers of all six Bonsai files (ternary/binary × gemlite/MLX/unpacked) dequantize
  bit for bit to the unpacked release. Ternary renders are byte-identical to the same model converted to BF16.
- DiT in VRAM: ternary 1.36 GB, binary 0.92 GB, BF16 7.39 GB.
- it/s: fully resident BF16 2.52 vs ternary 2.42; ComfyUI defaults (dynamic VRAM, text encoder on the same
  card) BF16 2.01 vs ternary 2.47; `--novram` BF16 0.93 vs ternary 1.59 vs binary 1.65.

## Not supported

- LoRA on the packed layers.
- Requantizing a float model to ternary: this only loads models that already are.
- Group sizes other than what the file says, or bit widths other than 1, 2, 4.

## Files

- `layout.py`: the layout, its registration, and the reader `comfy.ops._load_quantized_module` calls through
  `QUANT_ALGOS["lowbit_affine"]["params_from_state_dict"]` (hook added by the local ComfyUI patch
  `patches/comfyui_awq_w4a16_format.patch`; without it lowbit files fail to load and the boot log says so).
- `kernel.py`: Triton and torch dequantization, code packing.
- `formats.py`: detection, the four readers, lossless packing, diffusers→BFL renaming.
- `test_lowbit.py`: `python_embeded\python.exe -s custom_nodes\comfy-lowbit-loader\test_lowbit.py`
  (the GPU test runs only with `LOWBIT_TEST_CUDA=1` and an explicit `LOWBIT_TEST_DEVICE=cuda:N`).
