# ConvRot W4A4 Handoff

## Scope and safety rules

This is the live ComfyUI Portable installation at `F:\COMFY_PORTABLE`. Always use `F:\COMFY_PORTABLE\python_embeded\python.exe`; never use global Python. Do not delete, overwrite, move, or requantize original models. Do not mass-upgrade dependencies. W4A4 must execute through the native ConvRot CUDA backend, not eager/dequantized BF16 GEMM. WSL currently hosts unrelated Qwen/DeepSeek work and must not be stopped or modified.

## Current environment

- Python 3.13.12
- Torch 2.13.0+cu130, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130
- CUDA reported by Torch: 13.0
- nunchaku 1.2.1 (built for torch 2.11, running under 2.13), spas_sage_attn 0.1.0 (SpargeAttn)
- No `pytest` in the embedded interpreter. The pytest commands in CLAUDE.md do not run as
  written; test files under `tools/` and in the WaveSpeed fork carry their own runner.
- comfy-kitchen 0.2.23
- ComfyUI version `0.33.0` (`v0.33.0-19-gc1739380`). It was `0.29.0`/`42d2aa55` earlier in this
  project's life; anything in this file that assumes 0.29 behaviour is suspect.
- GPUs: RTX 3090 24 GB (`cuda:0`) and RTX 3080 Ti 12 GB (`cuda:1`)
- Only the Torch/vision/audio trio was replaced, using embedded pip, same versions, `--no-deps`. Pre-change freeze: `_pip_freeze_before_w4a4_cu130_20260816.txt`.
- `pip check` passes.

The existing SageAttention wheel initially failed because it needs `cudart64_12.dll`. No Sage update was required. A pre-existing local CUDA 12.6 runtime DLL was copied without overwrite to:

`python_embeded\Lib\site-packages\torch\lib\cudart64_12.dll`

Source: `venvs\ultravox311\Lib\site-packages\torch\lib\cudart64_12.dll`; size 556,544 bytes; SHA-256 `D954CA542B3B6BCF03CC2B798A7D00051501CF734CA751050E986AF505CF9DAD`. Sage import, a real RTX 3090 kernel comparison, both FlashVSR nodes, and full `--use-sage-attention` server startup passed.

## Audit and tools

- Inventory: `quantization_inventory.json` and `quantization_inventory.md` (157 files, 608.56 GiB before the new output).
- Continuous report: `W4A4_PROGRESS.md`.
- Tools:
  - `tools\quant_audit.py`
  - `tools\quant_w4a4.py`
  - `tools\verify_w4a4.py`
- Converter currently supports strict `gemma` and `qwen` profiles.

### Mixed precision, calibrated on real activations (2026-08-18)

Three tools, used in this order. Full measurements in `W4A4_PROGRESS.md` part 9.

```powershell
.\python_embeded\python.exe -s .\tools\to_native.py --input <src.safetensors> --arch zimage --output <native.safetensors>
.\python_embeded\python.exe -s .\tools\calibrate_activations.py --model <native.safetensors> --profile zimage --clip qwen_3_4b.safetensors --clip-type lumina2 --prompt-file tools\benchmark_prompt.txt --seeds 1234 5678 --steps 8 --out calib\<name>.calib.pt
.\python_embeded\python.exe -s .\tools\quant_mixed.py --input <native.safetensors> --calibration calib\<name>.calib.pt --save-analysis calib\<name>.analysis.json --promote-error 0.15 --dry-run
```

Drop `--dry-run` to write; pass `--analysis <json>` afterwards to re-decide at a different
threshold or `--budget` without remeasuring.

Three things a future session must not rediscover the hard way:

1. **`to_native.py` is not optional for Z-Image.** A published checkpoint is in diffusers naming
   and ComfyUI fuses `attention.to_{q,k,v}` into `attention.qkv` at load. Only `.weight` is in
   that map, so `weight_scale` / `weight_s_rel` / `comfy_quant` pass through unrenamed and the
   layer loads **with no scale and no error message**. `quant_mixed.py` refuses a diffusers-named
   input for this reason. The remap is verified against ComfyUI's own `convert_diffusers_mmdit`
   and produces a bit-identical latent.
2. **Crest factor does not predict W4A4 error** (Spearman +0.10 over 170 layers). It is collected
   as a diagnostic only. What predicts it is the W4A8 error (Spearman +0.978).
3. **Calibration samples are stored in bfloat16, not fp16, on purpose.** Real Z-Image activations
   reach 344064, which overflows fp16 to `inf`; that made every error `nan`, and `nan > threshold`
   is False, so the worst layer in the model was silently given the cheapest format.

Done on `beyond-reality-zimage-v2`: 115 layers `convrot_w4a4` + 55 `asym_w4a8_int8`, verified
loading with the correct `quant_format` on all 170 modules and `_full_precision_mm` false on every
one. 11.46 GiB -> 3.18 GiB (**3.61x lighter**), 1.290 -> 0.598 s/step (**2.16x less**).
- Converter uses streaming Safetensors output and direct tensor reads by header offset. Do not revert to `safe_open` for huge source tensors on this Windows host: mapping the 21.93 GiB source failed with `os error 1455` and twice caused `torch_cpu.dll` access violations (`0xc0000005`).

## First completed model

Source, unchanged:

`ComfyUI\models\text_encoders\gemma_3_12B_it_heretic.safetensors`

- Size: 23,545,681,250 bytes (21.93 GiB)
- Real dtype: BF16

Output:

`ComfyUI\models\text_encoders\gemma_3_12B_it_heretic_w4a4_convrot.safetensors`

- Size: 7,417,110,666 bytes (6.91 GiB), about 68.5% smaller
- Conversion time: 30.209 seconds
- 336 attention/MLP Linear weights quantized
- 293 tensors preserved byte-for-byte
- Packed weight dtype: INT8 containing signed INT4
- Scales: FP32 per output row
- Layout metadata: `convrot_w4a4`, group size 256
- Sidecar: `ComfyUI\models\text_encoders\gemma_3_12B_it_heretic_w4a4_convrot.quant.json`

Verification command already passed:

```powershell
.\python_embeded\python.exe .\tools\verify_w4a4.py `
  ".\ComfyUI\models\text_encoders\gemma_3_12B_it_heretic_w4a4_convrot.safetensors" `
  --source ".\ComfyUI\models\text_encoders\gemma_3_12B_it_heretic.safetensors" `
  --kernel-smoke
```

Results: structural PASS for 336 layers; every preserved tensor byte comparison PASS; normal ComfyUI backend `comfy_kitchen.backends.cuda`; real layer executed via `comfy_kitchen.backends.cuda.convrot_w4a4_linear` with BF16 output. Random-input smoke relative RMSE was 0.2511; this is not a quality benchmark.

The native `CLIPLoader` with type `ltxv` also returned `comfy.sd.CLIP`, instantiated `Gemma3_12BModel_`, and reported exactly 336 modules with `quant_format='convrot_w4a4'`. It printed many missing vision-tower warnings because this text-encoder file contains no vision tower; compare against loading the BF16 source before deciding whether they are expected. On short interpreter shutdown, `ModelPatcher.__del__` printed an `ON_DETACH` AttributeError; loading itself exited code 0.

## Immediate next work

Two separate tracks. Everything in the first needs the GPU; the second does not.

### Needs the GPU (blocked while it is lent out)

1. **`--promote-error 0.15` was picked, not derived.** Sweep it (and `--budget`) against real
   images at several seeds to find where quality actually breaks. `--analysis` re-decides from a
   saved measurement in seconds, so a sweep costs only the generation time. Right now the only
   evidence that mixing helps is a 11.7% drop in latent divergence and a 3.18x drop in measured
   per-layer error; **the images do not visibly separate**, and saying otherwise would be
   overclaiming.
2. Extend `quant_mixed.py` past `zimage`. `ltx_2_5` and `hunyuan_video_15` profiles already exist
   in `calibrate_activations.py`, but each needs its file naming checked against its ComfyUI
   module naming first — the Z-Image trap above is not Z-Image-specific. Lightricks' shipped
   `comfy-int8-convrot` checkpoint quantizes LTX's file names directly and loads, which is
   evidence LTX needs no remap, but that is evidence and not a test.
3. Open and queue the 6 `*.nunchaku.json` workflows produced by `tools/swap_to_nunchaku.py`.
   They have never been opened. Still the only outstanding item from the SVDQuant evaluation.
4. Decide what to keep. `beyond-reality-zimage-v2_native.safetensors` (11.46 GiB on `F:`, which
   has ~45 GiB free) is only an intermediate, but regenerating it costs 13 seconds.
   `beyond-reality-recovered-bf16.safetensors` (11.46 GiB on `D:`) is the W4-vs-A4 ablation
   instrument. Both are the user's call.

### Aviso sobre numeros de latente

O caminho INT4 do **nunchaku** nao e deterministico entre processos: duas execucoes identicas
divergem em relL2 0,298. Qualquer `relL2`/`cosine` de INT4 do nunchaku neste projeto e uma
amostra, nao uma medida. Comparacoes de qualidade precisam de N execucoes. Disco, VRAM, velocidade
e erro de peso nao sao afetados.

Isso **nao** vale para os kernels do `comfy_kitchen`: `convrot_w4a4` e `asym_w4a8_int8` foram
medidos deterministicos entre processos (latente bit-identico, mesmo arquivo, mesma seed,
2026-08-18). Numeros de latente vindos do caminho ComfyUI sao medidas.

### Does not need the GPU

5. `svdq_to_bf16` on a whole model will produce a file in the tens of GiB. Check free space
   before starting: `D:` is a network share and had 684 GiB free on 2026-08-18.
6. The tonera `svdq-int4_r32-qwen-image-edit-2511-lightning.safetensors` (13.71 GiB) is
   confirmed unloadable and is now superseded. Deleting it is the user's call, not the tool's.

## Upstream contributions

| PR | repo | state |
|---|---|---|
| [#1](https://github.com/yannickcruz/Comfy-WaveSpeed-Fixed/pull/1) | yannickcruz/Comfy-WaveSpeed-Fixed | 7 FBCache bugs + 6 test suites |
| [#121](https://github.com/thu-ml/SpargeAttn/pull/121) | thu-ml/SpargeAttn | `UnboundLocalError` when `smooth_k=False` |
| [#124](https://github.com/nunchaku-ai/deepcompressor/pull/124) | nunchaku/deepcompressor | two fixes to make the package importable on Windows |
| [#149](https://github.com/chengzeyi/Comfy-WaveSpeed/pull/149) | chengzeyi/Comfy-WaveSpeed | first-block residual is always zero except on LTXVModel |
| [#949](https://github.com/nunchaku-ai/nunchaku/pull/949) | nunchaku-tech/nunchaku | `from_linear` dereferences `linear.weight` before honouring `torch_dtype` |
| [#828](https://github.com/nunchaku-ai/ComfyUI-nunchaku/pull/828) | nunchaku-tech/ComfyUI-nunchaku | same eager-default in `fuse_linears` |

The last two are siblings and **neither alone fixes the crash**: the traceback lands in
`nunchaku/linear.py:152` but `ComfyUI-nunchaku/models/zimage.py:62` carries the same pattern on
another path. Each PR points at the other. Until they land, this installation needs
`--disable-dynamic-vram` for any SVDQuant workflow.

Local modification not covered by any PR: `python_embeded/Lib/site-packages/spas_sage_attn/core.py`
carries the same `km = None` fix as SpargeAttn#121, applied to the 3 occurrences the woct0rdho
wheel has (upstream has 5). A reinstall of that wheel silently reverts it.

Do not start by installing packages. Do not touch the WSL jobs. No original model has been
modified.
