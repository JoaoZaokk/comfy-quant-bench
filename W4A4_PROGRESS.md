# ConvRot W4A4 Progress

Last updated: 2026-08-16 (America/Sao_Paulo)

## Inventory

- Portable root: `F:\COMFY_PORTABLE`
- Embedded Python: `F:\COMFY_PORTABLE\python_embeded\python.exe`
- Models root: `F:\COMFY_PORTABLE\ComfyUI\models`
- Inventoried: 158 model files after the first W4A4 output (157 before it)
- Detailed outputs: `quantization_inventory.json` and `quantization_inventory.md`
- Hardware: RTX 3090 24 GB, compute capability 8.6
- Stack: Python 3.13.12, Torch 2.12.1+cu130, CUDA 13.0, comfy-kitchen 0.2.23
- Free capacity at audit: 120.09 GiB disk; 63.15 GiB total RAM (4.56 GiB free at that moment)

## Published

The converter, the verifier and the text-encoder node are public at
**https://github.com/JoaoZaokk/ComfyUI-ConvRot-W4A4** (MIT).

The repo copies of `quant_w4a4.py` and `verify_w4a4.py` take a `--comfy-root` argument and
auto-detect a ComfyUI checkout, so they work from `custom_nodes/` as well as from a portable
root. `.gitignore` blocks every weight extension; the published tree is 7 files and no model data.
The working copy of the node in `ComfyUI/custom_nodes/comfy_convrot_native/` is unchanged and still
functional — replace it with a clone of the repo whenever you want them kept in sync.

## Second HunyuanVideo 1.5 model converted

`diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors` matched the `hunyuan_video_15`
profile with no changes: 1361 tensors all FP16, same `txt_in.individual_token_refiner` signature,
zero single blocks, 432/432 pattern matches all satisfying `shape[1] % 256 == 0`.

| Item | Source | W4A4 |
| --- | --- | --- |
| Size | 15.51 GiB | **7.92 GiB** |
| Tensors | 1361 FP16 | 432 quantized + 929 preserved |

Verification: structural **PASS** (432 layers), source comparison **PASS** (every preserved tensor
byte-identical), backend `comfy_kitchen.backends.cuda`, real kernel through
`comfy_kitchen.backends.cuda.convrot_w4a4_linear` with FP16 output, relative RMSE 0.2226, max abs
error 0.7109.

End-to-end native execution confirmed on the loaded model, same as capybara:

```
[12] quantized modules after load: 432; first double_blocks.0.img_attn.qkv
[15] forward ok: (1, 64, 6144); native=1 dequant=0
     impls=['comfy_kitchen.backends.cuda.convrot_w4a4_linear']
```

Two HunyuanVideo 1.5 models now run real ConvRot W4A4 through the stock ComfyUI diffusion path
with no core change and no helper node.

Note the auditor previously labelled `capybara_v0.1` as `flux` because its `double_blocks` keys matched the flux needle. `tools/quant_audit.py` now runs a structural HunyuanVideo check first, mirroring `comfy.model_detection`.

## Candidate ranking

1. `text_encoders/gemma_3_12B_it_heretic.safetensors` — 21.93 GiB, BF16, Gemma 3 12B text encoder. First safe source candidate; preserve embeddings/norms and quantize architecture-specific attention/MLP Linear weights.
2. `diffusion_models/capybara_v0.1.safetensors` — 15.51 GiB, BF16, Flux-style DiT keys. Recipe needs confirmation against its loader/model config.
3. `diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors` — 15.51 GiB, FP16, HunyuanVideo 1.5.
4. `text_encoders/qwen_2.5_vl_7b.safetensors` — 15.45 GiB, BF16, Qwen2.5-VL. Preserve embeddings, LM head, and vision tower initially.
5. `SEEDVR2/seedvr2_ema_7b_fp16.safetensors` — 15.35 GiB, FP16, SEEDVR2 transformer.
6. `SEEDVR2/seedvr2_ema_7b_sharp_fp16.safetensors` — 15.35 GiB, FP16, SEEDVR2 transformer.
7. `checkpoints/hidream_o1_image_bf16.safetensors` — 15.24 GiB, BF16, mixed checkpoint; component boundaries must be respected.
8. `diffusion_models/z_image_de_turbo_v1_bf16.safetensors` — 11.46 GiB, BF16, Z-Image.
9. `diffusion_models/z_image_turbo_bf16.safetensors` — 11.46 GiB, BF16, Z-Image.
10. `diffusion_models/void_pass2.safetensors` / `unet/void_pass1.safetensors` — 10.38 GiB each, BF16; architecture recipe not yet established.

## Completed

- Created the recursive, metadata-only auditor at `tools/quant_audit.py`.
- Created `tools/quant_w4a4.py` with strict Gemma/Qwen profiles, automatic output naming, sidecar manifests, atomic writes, disk/RAM checks, and a hard refusal when normal ComfyUI does not select the CUDA ConvRot backend.
- Created `tools/verify_w4a4.py` for metadata/layout/source-shape validation and optional native-kernel smoke comparison.
- Parsed Safetensors headers and `_quantization_metadata`; used the installed GGUF reader for GGUF tensor types; inspected PyTorch checkpoints with safe/meta loading where supported.
- Verified the installed ConvRot symbols and ComfyUI `convrot_w4a4` checkpoint format (`TensorCoreConvRotW4A4Layout`, INT8-packed W4 weights, group size 64).
- Direct comfy-kitchen CUDA probe on the RTX 3090 selected `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, produced INT8-packed `[256,128]` weights from `[256,256]`, and returned BF16 output. This proves the installed native kernel can execute in isolation.
- Saved the complete pre-change package snapshot as `_pip_freeze_before_w4a4_cu130_20260816.txt`.
- Replaced only Torch 2.12.1, torchvision 0.27.1, and torchaudio 2.11.0 with their same-version `cu130` builds through the embedded Python and `--no-deps`. `pip check`, torchvision NMS, torchaudio, and flash-attn imports pass.
- Repeated the ConvRot probe through normal `ComfyUI/comfy/quant_ops.py`: it now selects `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, stores packed weights as INT8, accepts BF16 activations, and returns BF16 output on the RTX 3090. Native-backend preflight: **PASS**.
- Diagnosed the SageAttention import failure at the PE dependency level: its `_fused.pyd` and `_qattn_sm80.pyd` require `cudart64_12.dll`, while the cu130 Torch wheel supplies only `cudart64_13.dll`.
- Added the already-installed local CUDA 12.6 runtime DLL to the embedded Torch library directory without overwriting anything: `python_embeded/Lib/site-packages/torch/lib/cudart64_12.dll`, 556,544 bytes, SHA-256 `D954CA542B3B6BCF03CC2B798A7D00051501CF734CA751050E986AF505CF9DAD`. Source: `venvs/ultravox311/Lib/site-packages/torch/lib/cudart64_12.dll`.
- SageAttention import and an actual RTX 3090 kernel comparison against PyTorch SDPA passed (`mean_abs` about 0.0011, finite FP16 output). The existing Sage launcher was preserved unchanged.
- Full `--use-sage-attention` startup smoke test reached `http://127.0.0.1:8191`; both `ComfyUI-FlashVSR` variants loaded, QwenVL detected Sage, SeedVR2 reported Sage/Flash/Triton available, and normal ComfyUI selected the CUDA ConvRot backend.
- Gemma dry-run selected 336 attention/MLP Linear weights and estimated a 6.91 GiB output. No output file was created.
- Existing workflow references found for the Gemma source (`ComfyUI/user/default/workflows/Video-LTX2_MultiGPU.app.json`), SEEDVR2, and several Z-Image workflows; these can become matched benchmark fixtures after native loading is enabled.

### gemma_3_12B_it_heretic → ConvRot W4A4

Source (unchanged): `ComfyUI/models/text_encoders/gemma_3_12B_it_heretic.safetensors`
Output: `ComfyUI/models/text_encoders/gemma_3_12B_it_heretic_w4a4_convrot.safetensors`
Sidecar: `gemma_3_12B_it_heretic_w4a4_convrot.quant.json`

| Item | Source | W4A4 |
| --- | --- | --- |
| Size | 23,545,681,250 B (21.93 GiB) | 7,417,110,666 B (6.91 GiB) |
| Weight dtype | BF16 | INT8 container holding signed INT4 |
| Scales | — | FP32, per output row |

- Reduction: 68.5%. Conversion time: 30.209 s on the RTX 3090.
- 336 attention/MLP Linear weights quantized (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`); 293 tensors preserved byte-for-byte.
- Layout metadata `convrot_w4a4`, `convrot_groupsize` 256, `quant_group_size` 64.
- `tools/verify_w4a4.py --kernel-smoke` passed: structural PASS for all 336 layers, byte-identical comparison PASS for every preserved tensor, normal ComfyUI resolved `comfy_kitchen.backends.cuda`, and a real layer executed through `comfy_kitchen.backends.cuda.convrot_w4a4_linear` returning BF16. Random-input relative RMSE 0.2511 — a liveness signal, not a quality metric.

Loader path confirmed by reading the installed code (not assumed):

- `comfy.sd.load_clip` reads the Safetensors metadata and `comfy.utils.convert_old_quants` turns `_quantization_metadata` into per-layer `<layer>.comfy_quant` tensors.
- `comfy.sd.llama_detect` → `comfy.utils.detect_layer_quantization` sees those keys and returns `{"mixed_ops": True}`, selecting `MixedPrecisionOps`.
- `comfy/ops.py` handles `quant_format == "convrot_w4a4"`: requires `weight_scale`, reads `convrot_groupsize` (default 256), hardcodes `quant_group_size = 64`, defaults `linear_dtype` to `int4`, and builds `TensorCoreConvRotW4A4Layout.Params`. The converter's constants match this exactly.
- `comfy_kitchen/tensor/convrot_w4a4.py` dispatches `aten.linear` to the native path only when `params.transposed` is False; a transposed weight silently falls back to `weight.dequantize()` + `F.linear`. Any future benchmark must count that fallback, not just check that the file loads.
- The real workflow node is `LTXAVTextEncoderLoader` (`comfy_extras/nodes_lt_audio.py`), which builds `LTXAVTEModel_` from two files — the Gemma encoder plus the LTX-2.3 checkpoint that supplies `text_embedding_projection`. The earlier single-file `CLIPLoader(type='ltxv')` probe instantiated a different class (`Gemma3_12BModel_`) and is therefore not the path to validate against.

Load result through that real path (`tools/te_smoke.py`, one long-lived process):

- `model_class` `LTXAVTEModel_`, `text_projection_type` `dual_linear`.
- 336 modules with `quant_format='convrot_w4a4'`, all with `layout_type='TensorCoreConvRotW4A4Layout'`, all with `convrot_groupsize=256`, all 336 weights materialized as `QuantizedTensor`.
- `loaded completely; 9276.55 MB loaded, full load: True` on `cuda:0`, load 23.277 s, encode 2.611 s, encode peak VRAM 11.34 GB.
- Conditioning produced normally: shape `[1, 27, 6144]`, float32, mean 0.0049, std 5.118, absmax 270.0.

Loader warnings resolved — **not** caused by quantization:

- 163 warnings total: 162 `Missing weight for layer vision_model.*` plus one `clip missing: ['vision_model...']`.
- Zero language-model keys are missing or unexpected.
- The BF16 source header contains **no** `vision_model.*` tensors at all (629 tensors: 626 `model.*`, 2 `multi_modal_projector.*`, 1 `spiece_model`). The W4A4 output has the same namespaces and 965 tensors — exactly 629 + 336 added `weight_scale`. The vision tower is simply absent from this text-encoder-only checkpoint, so these warnings are inherent to the source file.

### Native execution result: the text-encoder path does NOT run the W4A4 kernel

This is the decisive finding so far, and it is negative.

With the model loaded through the real `CLIPType.LTXV` path and `comfy_kitchen.tensor.convrot_w4a4` instrumented, one prompt encode produced:

| Metric | Value |
| --- | --- |
| `convrot_w4a4_linear` calls | **0** |
| `TensorCoreConvRotW4A4Layout.dequantize` calls | **336** (exactly one per quantized layer) |
| Backends actually used | none — no ConvRot kernel ran |

So the checkpoint is correct but the runtime behaviour is `W4 storage → dequantize → BF16 GEMM`, the outcome the project explicitly forbids.

Forward-time gate values captured from a real quantized Linear:

| Gate | Value |
| --- | --- |
| `layout_type` | `TensorCoreConvRotW4A4Layout` |
| `weight_is_quantized_tensor` | `True` |
| `weight.dtype` / `orig_dtype` | `torch.bfloat16` |
| `input.dtype` | **`torch.float32`** |
| `_full_precision_mm` | `True` |
| `comfy_force_cast_weights` | `True` |
| `weight_function` / `bias_function` | 0 / 0 |
| `transposed` | `False` |

Root cause, isolated to one variable with `tools/convrot_ops_probe.py` (builds a real `MixedPrecisionOps.Linear`, loads a real quantized layer, varies one setting at a time):

| Case | compute dtype | activation dtype | `full_precision_mm` | `force_cast` | native calls | dequant calls |
| --- | --- | --- | --- | --- | --- | --- |
| text-encoder settings | fp32 | fp32 | True | True | 1 | 0 |
| text-encoder settings, force_cast cleared | fp32 | fp32 | True | False | 1 | 0 |
| diffusion settings | bf16 | bf16 | False | False | 1 | 0 |
| **observed LTXAV Gemma** | **bf16** | **fp32** | True | True | **0** | **1** |
| same, only activation dtype changed | bf16 | bf16 | True | True | 1 | 0 |

`_full_precision_mm` and `comfy_force_cast_weights` are **not** the blockers. Because `QUANT_ALGOS["convrot_w4a4"]["quantize_input"]` is `False`, the native path is reached by `F.linear` dispatching on the `QuantizedTensor`, not through the `_use_quantized` branch. The single blocker is the dtype mismatch at `comfy/ops.py:390-393`:

```python
if weight_has_function or weight.dtype != dtype:
    weight = weight.to(dtype=dtype)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
```

Where the two dtypes come from:

- `comfy/ops.py:1138` sets the quantized weight's `orig_dtype` to the module's **compute dtype**, and for this model that is `dtype_llama` = the dtype of `model.norm.weight` = **BF16** (`comfy/text_encoders/hunyuan_video.py:11` `llama_detect`, applied in `comfy/text_encoders/lt.py:242`).
- Activations arrive as **FP32** because `comfy/sd.py:261-262` does `self.patcher.set_model_compute_dtype(torch.float32)` with the comment "Match torch.float32 hardcode upcast in TE implemention".

Consequence: on ComfyUI `42d2aa55` / 0.29.0, **no quantized text encoder of any format can execute its native kernel through the stock path** — fp8_scaled, int8_tensorwise and convrot_w4a4 all dequantize for compute. Quantized text encoders are a disk/VRAM-storage feature here, not a compute feature.

### Resolved without touching ComfyUI core

Does ComfyUI actually need the FP32 upcast? **Yes.** It is deliberate, not incidental:

- `comfy/sd1_clip.py:213` requests the input embeddings with `out_dtype=torch.float32`.
- `comfy/sd1_clip.py:279` passes `dtype=torch.float32` into the transformer.
- `comfy/sd1_clip.py:282-284` returns `.float()` outputs.
- `comfy/sd.py:262` only *matches* that with `set_model_compute_dtype(torch.float32)`.

So removing the FP32 text-encoder policy would change numerics for every encoder in the install (T5, CLIP-L/G, Llama, Gemma, Qwen). That is not a safe one-line change and was not made.

The FP32 policy is not the problem — the **mismatch** is. And `comfy_kitchen/tensor/base.py:417-436` shows `QuantizedTensor.to(dtype=...)` only rewrites `params.orig_dtype`; it never touches the packed data. Retyping the quantized weights to FP32 after load therefore removes the mismatch at zero cost, keeps ComfyUI's FP32 encoder policy intact, and lets `F.linear` dispatch to the ConvRot layout handler.

Measured on the real model, same load, two encodes:

| Metric | Stock (emulated) | After retype (native) |
| --- | --- | --- |
| `convrot_w4a4_linear` calls | 0 | **336** |
| Weight dequantizations | 336 | **0** |
| Backend | none | `comfy_kitchen.backends.cuda.convrot_w4a4_linear` |
| Encode time | 2.354 s | **0.539 s** (4.37x faster) |
| Encode peak VRAM | 11.340 GB | 11.372 GB |
| Relative RMSE vs emulated | — | 0.2011 (max abs 33.0) |

Delivered as a new custom node, `ComfyUI/custom_nodes/comfy_convrot_native/` → **ConvRot W4A4 Native (Text Encoder)**. Drop it between the loader and `CLIPTextEncode`. No core file is modified, so ComfyUI updates cannot clobber it. The node skips any layer carrying weight patches and logs how many it skipped, because a LoRA patch needs a dense weight and will keep dequantizing.

Caveats that still need resolving before calling this a win:

- Relative RMSE 0.2011 on the conditioning is a real quality cost, not noise. A matched-parameter visual A/B is required before recommending W4A4 for this encoder.
- `CLIP.clone()` shares `cond_stage_model`, so the retype is visible to every clone of that CLIP. Harmless (it changes only the declared logical dtype) but worth knowing.
- The TE LoRA used in some LTX templates (`gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors`) forces the dequantized path for the layers it patches.

For diffusion models no node is needed: `pick_operations` builds the ops with `full_precision_mm=False` and matching bf16 activations, and `convrot_w4a4` is never added to the `disabled` set, so the probe dispatches natively out of the box.

### Core-patch tooling (prepared, unused)

`tools/core_patch.py` provides `backup` / `status` / `diff` / `revert` for ComfyUI core files, keyed by SHA-256 and timestamp in `core_patches/ledger.json`, so a later ComfyUI update that rewrites a tracked file is detected instead of silently reverted over. Current state: `No ComfyUI core file is tracked. Core is untouched.`

### capybara_v0.1 — architecture identified before any conversion

The inventory guessed "Flux-style DiT keys". That was **wrong**, and checking it was the point.

Asked ComfyUI itself via the new `tools/inspect_diffusion_arch.py` (builds a meta-device state dict from the header, runs `comfy.model_detection`, instantiates the model on `meta`, enumerates every Linear):

- `model_config` = **`HunyuanVideo15`**, `image_model` = `hunyuan_video`.
- `hidden_size` 2048, `num_heads` 16, `depth` 54, `depth_single_blocks` **0**, `in_channels` 65, `context_in_dim` 3584, `byt5` True, `use_cond_type_embedding` True, `vision_in_dim` 1152, `meanflow_sum` True.
- 1364 tensors, all BF16, no `__metadata__` at all.
- So it is the same family as `diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors`; one profile covers both.

Note `comfy.model_detection.unet_prefix_from_state_dict` returns `'model.'` for this file, which detects nothing. `load_diffusion_model_state_dict` uses `""`, and so does the inspector.

Key-naming trap, resolved: the checkpoint's keys are **not** the ComfyUI module names.

| ComfyUI module | Checkpoint key |
| --- | --- |
| `double_blocks.N.img_attn.qkv` | `double_blocks.N.img_attn_qkv` |
| `double_blocks.N.img_attn.proj` | `double_blocks.N.img_attn_proj` |
| `double_blocks.N.img_mlp.0` / `.2` | `double_blocks.N.img_mlp.fc1` / `.fc2` |
| `double_blocks.N.img_mod.lin` | `double_blocks.N.img_mod.linear` |
| `txt_in.c_embedder.in_layer` | `txt_in.c_embedder.linear_1` |

`HunyuanVideo.process_unet_state_dict` performs these as plain substring replacements, and it runs **after** `comfy.utils.convert_old_quants` has injected the `<layer>.comfy_quant` keys. So the `_quantization_metadata` layer names must use the **checkpoint's** convention: the same replacements then carry `.comfy_quant` and `.weight_scale` along with the weight. Verified that `.weight_scale` is not caught by that function's `endswith(".scale")` rule.

Profile `hunyuan_video_15` added to `tools/quant_w4a4.py`, auto-detected structurally (presence of `txt_in.individual_token_refiner.blocks.0.norm1.weight` and `double_blocks.0.img_attn_qkv.weight`, mirroring ComfyUI's own HunyuanVideo branch) rather than from the file name — the file name says nothing here.

Quantized, 8 per block x 54 blocks = 432 tensors: `{img,txt}_attn_qkv`, `{img,txt}_attn_proj`, `{img,txt}_mlp.fc1`, `{img,txt}_mlp.fc2`. All satisfy the ConvRot `shape[1] % 256 == 0` constraint.

Preserved: `{img,txt}_mod.linear` (adaLN modulation drives each block's conditioning), every norm, `img_in`, `txt_in.*` including the token refiner, `byt5_in`, `vision_in`, `time_in`, `final_layer`, all embeddings and `task_bias`, and every bias.

Dry run: 432 layers selected, 15.51 GiB → estimated 7.92 GiB (about 49% smaller).

Conversion completed:

| Item | Source | W4A4 |
| --- | --- | --- |
| File | `capybara_v0.1.safetensors` | `capybara_v0.1_w4a4_convrot.safetensors` |
| Size | 16,653,435,264 B (15.51 GiB) | 8,507,756,960 B (7.92 GiB) |
| Tensors | 1364 BF16 | 432 quantized + 932 preserved |

Reduction 48.9%, conversion time 19.044 s. Sidecar `capybara_v0.1_w4a4_convrot.quant.json` written. Source untouched.

Verification (`tools/verify_w4a4.py --kernel-smoke`):

- Structural: **PASS** (432 ConvRot W4A4 layers).
- Source comparison: **PASS** — all 932 preserved tensors byte-identical to the source.
- Normal ComfyUI backend: `comfy_kitchen.backends.cuda`.
- Real kernel on `double_blocks.0.img_attn_proj` via `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, BF16 output, relative RMSE 0.2326, max abs error 0.5957.

Load through the real diffusion path, traced stage by stage with `tools/stage_probe.py`:

| Stage | Result |
| --- | --- |
| `load_torch_file` | 1796 tensors (1364 − 432 weights + 432 packed + 432 scales), metadata `_quantization_metadata` + `quantization` |
| `convert_old_quants` | 432 `.comfy_quant` keys injected, **0** without a matching weight |
| `model_config_from_unet` | `HunyuanVideo15`, `quant_config {'mixed_ops': True}` |
| `process_unet_state_dict` | 2228 keys; quant keys correctly remapped, e.g. `double_blocks.0.img_attn.proj.comfy_quant` |
| `get_model` + `load_model_weights` | ok |
| Quantized modules after load | **432**, first `double_blocks.0.img_attn.qkv` |
| `model.to(cpu)`, move one block to `cuda:0`, forward | ok, output `(1, 64, 6144)` |

So the metadata naming, the remap interaction and the module targeting are all correct end to end.

### Open: non-deterministic 0xC0000005 during large mmap — host issue, not the file

`tools/diffusion_smoke.py` and one `stage_probe.py` run died with exit `-1073741819` (`0xC0000005`, access violation), no traceback, empty stdout and stderr. The crash is **not reproducible against a specific step**: the identical `comfy.utils.load_torch_file` call on the identical file succeeded twice and then crashed once at stage 3.

This matches the two `torch_cpu.dll` access violations and the `os error 1455` seen earlier during conversion, and it tracks memory pressure rather than file content: the Windows commit charge on this host sits at 92-93 GiB against a 97-103 GiB limit because `vmmemWSL` holds roughly 28 GiB for unrelated work. The pagefile auto-expanded from 97.15 to 103.10 GiB mid-session, which is itself a symptom.

Not a W4A4 defect: `verify_w4a4.py` read both files fully and compared every preserved tensor byte for byte with a PASS, and the full load pipeline completes cleanly whenever the allocation succeeds.

Workaround: rerunning `stage_probe.py --stop-after 15` skips the heaviest allocation and completes. Retrying is currently the only mitigation, since WSL must not be touched and the pagefile must not be resized.

### Native execution on the loaded diffusion model: CONFIRMED

With the instrumented run that survived the mmap, a module taken straight out of the loaded `HunyuanVideo15` graph:

```
[15] forward ok: (1, 64, 6144) torch.float32; native=1 dequant=0
     impls=['comfy_kitchen.backends.cuda.convrot_w4a4_linear']
```

Zero weight dequantizations. This is real ConvRot W4A4 execution through the native CUDA kernel on the normal ComfyUI diffusion path, with no core modification and no helper node — exactly what the text-encoder path could not do.

Layered evidence for native execution:

| Layer | Tool | Result |
| --- | --- | --- |
| Raw kernel | `verify_w4a4.py --kernel-smoke` | `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, BF16 out |
| ComfyUI ops | `convrot_ops_probe.py` | native in every diffusion-path case |
| Loaded model | `stage_probe.py` stage 15 | native=1, dequant=0 |

## Full stack upgrade, 2026-08-16

The user explicitly authorized a full upgrade on this date, overriding the standing
"do not mass-upgrade" rule, on the condition that a backup exists first.

Backup: `_backup_20260816_preupgrade/` — `pip_freeze_BEFORE.txt` (268 packages), `comfyui_HEAD.txt`,
`custom_nodes_HEADS.txt` (61 repos), `comfy_kitchen_base.py.bak`, and a full copy of
`python_embeded` verified at 93,700 files / 8.54 GiB with zero delta. Git tag
`pre-upgrade-20260816` = `42d2aa55`.

| Component | Before | After |
| --- | --- | --- |
| ComfyUI | 0.29.0 `42d2aa55` | **0.33.0** `b963f4ad` (126 commits) |
| torch | 2.12.1+cu130 | **2.13.0+cu130** |
| torchvision | 0.27.1+cu130 | **0.28.0+cu130** |
| torchaudio | 2.11.0+cu130 | unchanged (2.11.0 is the newest published) |
| triton-windows | 3.7.0.post26 | **3.7.1.post27** |
| sageattention | 2.2.0+**cu128**torch2.10.0andhigher.post5 | 2.2.0+**cu130**torch2.10.0andhigher.post6 |
| comfy-kitchen | 0.2.23 | **0.2.31** |
| comfy-aimdo | 0.4.10 | **0.4.13** |
| frontend / templates / docs | 1.47.10 / 0.11.19 / 0.5.9 | 1.49.6 / 0.11.41 / 0.5.10 |
| flash-attn | 2.8.4 | unchanged, deliberately |

Done one component at a time with the full test battery between each, so any break would name its
own cause. Everything passed at every step.

### The cudart64_12.dll workaround is gone

Sage was on a **cu128** wheel while torch was **cu130** — a CUDA *major* version boundary. That
mismatch was the actual reason the `cudart64_12.dll` copy was needed. Installing
`sageattention-2.2.0+cu130...post6` removed the need: with the DLL renamed to `.disabled`,
`_check_accel.py` still reports Sage OK. The file is renamed, not deleted.

### Two predictions of mine that the tests disproved

1. "flash-attn will break on torch 2.13." It did not. The installed wheel is
   `flash_attn-2.8.4+cu130torch2.11.0cxx11abiTRUE-cp313-cp313` — built against torch **2.11**, and it
   runs correctly on **2.13**, two minors ahead, `mean|d|=0.0000` vs SDPA. No cu132 wheel was needed.
2. "cu132 flash wheels would clash with a cu130 Sage." Both are CUDA 13.x; NVIDIA guarantees minor
   version compatibility within a major. The earlier pain was cu128 → cu130, which crosses CUDA 12 → 13.

### Post-upgrade verification

| Check | Result |
| --- | --- |
| `_check_accel.py` | triton 3.7.1, sageattention, flash_attn all OK |
| `verify_w4a4.py --kernel-smoke` | structural PASS (432 layers), source comparison PASS, `comfy_kitchen.backends.cuda`, relative RMSE 0.2286 |
| `stage_probe.py` | 432 quantized modules, `native=1 dequant=0` |
| `compile_w4a4_fix_poc.py` | torch.compile works, `max_abs_diff 0.000000` |
| ComfyUI server boot | 61 custom_nodes loaded, 1 failure (`ComfyUI-AnimateDiff-Evolved`, empty folder, pre-existing) |
| `ConvRotNativeTextEncoder` node | registered |

The PR #52 patch must be re-applied after any comfy-kitchen upgrade — `pip install` overwrites
`tensor/base.py`. It survived the torch 2.13 move unchanged, so `_make_wrapper_subclass`'s
`strides=` signature is stable across 2.12 → 2.13.

Not measured yet: whether torch 2.13 is actually faster. Single-layer eager timing moved from
0.417 ms to 0.228 ms across the upgrade, but the machine was busy with unrelated work throughout,
so that number is not a result.

### Upgrade casualty: ComfyUI-TenserTensor

```
custom_nodes\ComfyUI-TenserTensor\nodes_workflow.py, lines 448 and 511
WorkflowSettings.Output("WORKFLOW_CONFIG")
TypeError: WorkflowSettings.Output.__init__() takes 1 positional argument but 2 were given
```

ComfyUI 0.33 changed that `comfy_api.latest._io` signature. The node **imports** fine and so it
was counted as healthy by the earlier "61 loaded, 1 failure" check — it only fails later, when
`/object_info` asks it to build its schema. Two of its nodes are unusable
(`TT_Sd35GgufWorkflowSettingsAdvancedNode` and one other), the rest of the server is unaffected.

Method note: counting `IMPORT FAILED` lines is not sufficient to certify custom nodes after a
ComfyUI upgrade. Schema construction happens later and can fail on its own. Any future upgrade
check should fetch `/object_info` and count per-node errors as well.

### Newly available in comfy-kitchen 0.2.31: W4A8

The CUDA backend now advertises capabilities that 0.2.23 did not:
`w4a8_int8_linear`, `quantize_w4a8_int8_weight`, `dequantize_w4a8_int8_weight`, and `na3d`.

The earlier 0.2.23-vs-0.2.31 comparison missed these because it only diffed
`torch.library.custom_op("...")` declarations, and these are registry capabilities rather than
custom ops. W4A8 keeps 4-bit weights but 8-bit activations, which on Ampere should map to the far
more mature INT8 tensor-core path and should cost much less accuracy than the current
0.22-0.23 per-layer relative RMSE of W4A4. Worth benchmarking against ConvRot W4A4 on this hardware.

### Benchmark still blocked, on VRAM not on code

`_backup_20260816_preupgrade` and the converted models are all in place, but the A/B needs both
sides resident without offloading, and desktop applications currently hold about 18.8 GiB of the
3090. Free VRAM at the last check: 3090 5.4 GiB, 3080 Ti 9.2 GiB. W4A4 needs 7.92 GiB plus the
text encoder and VAE; the FP16 source needs 15.51 GiB. Measuring one side with offload and the
other without would conflate "quantization is faster" with "fitting in VRAM is faster".

## VERDICT: ConvRot W4A4 does not work for HunyuanVideo 1.5 on this hardware

Measured 2026-08-16 on an idle RTX 3090, ComfyUI 0.33.0, torch 2.13.0+cu130, Sage enabled.
Identical prompt, seed, steps (6), resolution (480x480), sampler (euler), scheduler (simple) and
frame count for every run. Seeds varied between runs so ComfyUI could not serve a cached result.
Timings are ComfyUI's own "Prompt executed" with the model already resident.

| Config | Steady time | VRAM staged | Image |
| --- | --- | --- | --- |
| **FP16 source** | **2.49 s** | 15881 MB | clean, sharp, correct |
| W4A4 ConvRot g256 | 3.21 s | 8113 MB | **unusable** |
| W4A4 ConvRot g64 | 3.86 s | 8113 MB | **unusable** |
| W4A4 ConvRot g16 | 3.42 s | 8113 MB | **unusable** |

The FP16 run produces a clean red apple on a wooden table. Every W4A4 run produces coloured mush in
which the subject is not identifiable. The prompt was "a red apple on a wooden table, soft daylight".

So W4A4 here is **1.3x to 1.55x slower than FP16 and destroys the output**. Its only advantage,
about half the VRAM, is worthless when the result is unusable.

### What was ruled out, and how

- **Not torch.compile.** Compiled output matches eager at `max_abs_diff 0.00000000`, and the
  compiled image shows the same corruption as the uncompiled one.
- **Not the smoke-test settings.** FP16 at byte-identical settings produces a good image.
- **Not load order.** The first measurement had W4A4 at 27.31 s against FP16 at 14.56 s, but that
  charged the cold load of the Qwen 7B text encoder, byt5 and the VAE to whichever model ran first.
  With both resident and seeds varied, the gap narrows and reverses in FP16's favour.
- **Not the Hadamard group size.** 256, 64 and 16 were each converted and run. All three are
  unusable; g64 is visibly the worst and g16 marginally the least bad. The parameter does not rescue it.

### The earlier hardware argument was wrong

Ampere exposes INT4 tensor cores where later architectures deprecated them, so the reasoning went
that native INT4 MMA would make W4A4 pay off here where it does not elsewhere. It does not. A PTT
user measured the same conclusion on an AMD 9070XT through a weight-only dequant path
("INT4 精度實在崩到無法使用" — INT4 precision collapses to the point of being unusable, and INT4
ConvRot generates slower than INT8 ConvRot). Same outcome through a completely different code path
and vendor. The native-kernel advantage did not change the result.

### Where this leaves the converted models

`hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot`, `hv15_w4a4_g64`, `hv15_w4a4_g16` and
`capybara_v0.1_w4a4_convrot` are structurally valid, load correctly and execute the native kernel.
They are simply not usable for generation. Keep them as fixtures for kernel and loader work; do not
use them for output. The Gemma W4A4 text encoder was never visually validated and is now suspect
by association — the same 4-bit activation error applies.

## W4A8 works, and it is the format to use

Converted the same source with `tools/quant_w4a8.py` to comfy-kitchen's `asym_w4a8_int8` format and
re-ran the identical benchmark. Same prompt, seed, steps, resolution, sampler, scheduler.

| Config | Steady time | VRAM staged | On disk | Image |
| --- | --- | --- | --- | --- |
| FP16 source | **2.94 s** | 15881 MB | 15.51 GiB | good |
| ConvRot W4A4 g256 | 4.85 s | 8113 MB | 7.92 GiB | **unusable** |
| **asym_w4a8_int8** | **4.61 s** | **8437 MB** | 8.24 GiB | **good** |

W4A8 is both *faster* than W4A4 and produces a clean, correct image, for 4% more disk. Against
FP16 it is 1.57x slower for 1.88x less VRAM — a real trade, where W4A4 was pure loss.

Why it survives where W4A4 does not, all three at once:

- **Activations stay at 8 bits.** Half the error in W4A4 came from quantizing them to 4. Ampere's
  INT8 tensor-core path is mature; its INT4 path is not.
- **Per-group scales.** `group_size=16` gives a `[N, K/16]` scale grid instead of one scale per row.
  On a `[8192, 2048]` layer that is 128 scales per row rather than 1.
- **A Lloyd-Max codebook.** The 16 int4 levels stop being uniformly spaced and are placed where the
  weights actually are.

ConvRot rotation is still applied underneath, so W4A8 is W4A4 plus three additional defences.

Checkpoint layout, per quantized layer — four tensors, not two:

```
<layer>.weight            int8 container, packed int4        [N, K // 2]
<layer>.weight_s_rel      per-group scale, fp8 stored as u8  [N, K // group_size]
<layer>.weight_s_channel  per-channel scale                  [N]
<layer>.weight_codebook   Lloyd-Max levels                   [16]
```

One trap worth recording: `comfy/ops.py` reads `weight_s_rel`, `weight_s_channel` and
`weight_codebook`, but **never reads the optional asymmetric `correction` tensor**. Quantizing
asymmetrically would have its correction silently dropped and decode wrong, so the converter forces
`symmetric=True` and refuses to continue if a correction tensor comes back.

This also lands where the Chinese-language community already converged empirically (INT8 ConvRot
over INT4), and where the PTT measurements pointed. They reached it by trying; the kernel counter
just made the reason legible.

### Remaining leads for W4A4 specifically, now lower priority

1. **Fewer layers.** The profile quantizes all 432 attention and MLP projections. Attention may not
   survive 4 bits while MLP does.
2. **Not this model.** ConvRot's paper measured 2.26x on FLUX.1-dev, an image model. HunyuanVideo 1.5
   is a video DiT run here far outside its 720p operating point. The technique may simply not transfer.

## Skipped

- `checkpoints/ltx-2.3-22b-dev-fp8.safetensors` — already FP8; not a high-precision requantization source.
- LTX Q5/Q6 GGUF files — no matching BF16/FP16 diffusion source found locally; do not stack GGUF-to-W4A4 quantization.
- Wan FP8/GGUF files — already quantized and no matching high-precision 14B source found locally.
- Gemma 3 12B Q4_K_M GGUF — matching BF16 source exists locally; use the BF16 source instead.
- SEEDVR2 GGUF variants — matching FP16 sources exist locally; use those sources instead.
- VAEs, LoRAs, embeddings, vision encoders, small utilities, and model patches — excluded from the initial pass by policy.

## Failed / Blocked

### SageAttention binary compatibility after cu130 — resolved

Exact error: `ImportError: DLL load failed while importing _fused: The specified module could not be found.`

Confirmed cause: the locally installed `sageattention 2.2.0+cu128torch2.10.0andhigher.post5` binary extension depends on `cudart64_12.dll`. The DLL was no longer present in the embedded Torch library directory after replacing the cu126 wheel with cu130.

Result: **resolved** by restoring the existing CUDA 12.6 runtime DLL side-by-side. No Sage package update, source build, random third-party wheel, core edit, custom-node edit, or launcher edit was needed. Native ConvRot W4A4 and SageAttention both pass on the RTX 3090.

### Windows commit limit blocks large mmap — open

Exact error, twice, from two different callers:

`OSError: The paging file is too small for this operation to complete. (os error 1455)`

1. During conversion, from the converter's own `safe_open` on the 21.93 GiB Gemma source. It also crashed `torch_cpu.dll` twice with `0xc0000005`. **Resolved** by rewriting `tools/quant_w4a4.py` to stream: output header offsets are computed up front, quantized layers are read by byte range and written one at a time, and preserved tensors are copied in 16 MiB chunks. Do not reintroduce `safe_open`/mmap for huge sources on this host.
2. During validation, from **normal ComfyUI** — `comfy.utils.load_torch_file` → `safetensors.safe_open` on `checkpoints/ltx-2.3-22b-dev-fp8.safetensors` (27.14 GiB). This is not a quantization bug; the same call fails for any large file.

Measured at the time of failure:

| Counter | Value |
| --- | --- |
| Commit limit | 98.13 GiB |
| Committed bytes | 89.49 GiB |
| Commit headroom | ~8.6 GiB |
| Free physical RAM | 9.11 GiB (was 21.33 GiB minutes earlier) |
| Pagefile | `C:\pagefile.sys`, 35,826 MB allocated, 20,403 MB peak |

`vmmemWSL` holds the bulk of the committed memory and is running unrelated Qwen/DeepSeek work, so it is deliberately left alone; the pagefile is deliberately not resized. Consequence: **the 21.93 GiB BF16 Gemma source cannot currently be loaded at all**, so the BF16-vs-W4A4 A/B comparison is blocked on commit headroom, not on any converted file.

Workaround used for validation: the split `text_encoders/ltx-2.3_text_projection_bf16.safetensors` (2.15 GiB, 4 BF16 tensors) supplies `text_embedding_projection.{audio,video}_aggregate_embed.{weight,bias}` in place of the 27.14 GiB checkpoint. `comfy.text_encoders.lt.sd_detect` reads the same keys and still selects `dual_linear`, so the `CLIPType.LTXV` → `LTXAVTEModel_` path is exercised unchanged.

---

## 2026-08-18 — Baseline concorrente: SVDQuant W4A4 (Nunchaku) na 3090

Medido para responder uma pergunta que o projeto ConvRot tinha em aberto sem saber: **W4A4 de
difusão já entrega na sm86, hoje, com kernels prontos?** Entrega.

Mesmo modelo dos dois lados — Z-Image-Turbo. `z_image_turbo_bf16.safetensors` pelo loader normal
do ComfyUI contra `svdq-int4_r32-z-image-turbo.safetensors` (nunchaku-ai) pelo
`NunchakuZImageDiTLoader`. Latente, seed, sampler, scheduler, steps e condicionamento idênticos,
pesos residentes, um processo por modelo. Ferramenta: `tools/nunchaku_compare.py`.

| | BF16 | SVDQuant INT4 | razão |
|---|---|---|---|
| disco | 11.46 GiB | 3.36 GiB | 0.29x |
| load | 17.35 s | 5.60 s | 0.32x |
| pico de VRAM | 12.05 GiB | 3.91 GiB | 0.32x |
| passada (8 steps, 1024²) | 7.07 s | 3.12 s | 0.44x |
| s/step | 0.8837 | 0.3895 | **2.27x mais rápido** |

Passadas: BF16 `7.07, 7.12, 7.16`; INT4 `3.12, 3.12, 3.12`. Dispersão intra-run abaixo de 1.5%,
então a razão 2.27x não está competindo com ruído.

**Os kernels W4A4 estão de fato no grafo**, não é dequantização disfarçada. O script conta os
módulos e imprime:

```
quantised modules in graph: ComfyNunchakuZImageAttention×34, ComfyNunchakuZImageFeedForward×34,
                            SVDQW4A4Linear×136
```

`SVDQW4A4Linear` é o caminho que chama `nunchaku._C.ops.gemm_w4a4`. Isso fecha a verificação que
estava pendente: `gemm_w4a4` executa na sm86.

### O que isto **não** mostra

`latent relL2 0.7834`, `cosine 0.7208` contra o BF16. O condicionamento é aleatório, então esses
números medem quanto a trajetória mudou, **não** se a imagem ficou pior. Julgar qualidade exige
prompt real através do text encoder e olhar os pixels. Enquanto isso não for feito, a coluna de
qualidade desta comparação está vazia — e uma diferença de trajetória desse tamanho é grande o
bastante para que "vazia" não seja o mesmo que "sem problema".

### Consequência para o ConvRot

Não invalida o projeto, mas muda a pergunta. O ConvRot converte checkpoints locais arbitrários;
o Nunchaku consome checkpoints pré-quantizados no formato SVDQuant deles, distribuídos por eles,
e quantizar os próprios exige o `deepcompressor`. São escopos diferentes. O que o Nunchaku
estabelece é o **piso de desempenho**: qualquer saída ConvRot W4A4 que renda menos que 2.27x com
mais que 3.91 GiB de pico precisa de justificativa.

### Ambiente adicionado para isto

`nunchaku 1.2.1+cu13.0torch2.11` (wheel de torch 2.11 rodando sob 2.13, kernel verificado),
`ComfyUI-nunchaku 1.2.1`, `tomli`. `insightface` e `facexlib` deliberadamente **não** instalados
— só os nós de PuLID falham por isso, e nenhum caminho de Z-Image passa por eles.

### Repetido com o pipeline completo do ComfyUI (prompt real, CLIP real, VAE real)

A medição acima usava condicionamento sintético, o que deixava a coluna de qualidade vazia.
Refeita com `qwen_3_4b.safetensors` (o `comfy.text_encoders.z_image.te` é selecionado sozinho
para Qwen3-4B) e `ae.safetensors` (Z-Image é `ZImage(Lumina2)`, latente Flux 16ch — a
`qwen_image_vae` é 3D e falha com `IndexError: tuple index out of range` em `memory_used_decode`).

Prompt: `"a red apple on a weathered wooden table, afternoon light, sharp detail"`, 23 tokens,
`|cond| 14767.041` **idêntico nos dois runs** — o comparador recusa a comparação se divergir.

| | BF16 | SVDQuant INT4 | razão |
|---|---|---|---|
| disco | 11.46 GiB | 3.36 GiB | 0.29x |
| load | 7.81 s | 5.69 s | 0.73x |
| pico de VRAM | 12.05 GiB | 3.91 GiB | 0.32x |
| s/step | 0.8920 | 0.3922 | **2.27x mais rápido** |

`latent relL2 0.6344`, `cosine 0.8099`. Passadas BF16 `7.14, 7.17, 7.20`; INT4 `3.14, 3.15, 3.16`.
A razão 2.27x reproduziu exatamente a da corrida sintética.

**Veredito visual:** as duas imagens atendem ao prompt, são fotorrealistas e estão nítidas.
Composição, enquadramento, direção de luz e sombra praticamente iguais. Diferenças reais e
visíveis, nenhuma delas degradação: o BF16 deixa o fundo mais desfocado e cobre a maçã com
gotas d'água finas e numerosas; o INT4 resolve mais grão e rachadura na madeira, dá luz um pouco
mais dura e menos gotas. **Não é a mesma imagem** — `relL2 0.63` já dizia isso — mas não há
perda de qualidade que justifique recusar o INT4. Trajetória diferente, não pior.

Consequência: o piso para o ConvRot fica firme. W4A4 na sm86 entrega 2.27x com 32% da VRAM
**sem** custo de qualidade perceptível neste teste.

Ressalvas honestas: um prompt, um seed, um modelo, 8 steps. Não é avaliação de qualidade em
escala, e não cobre rosto, texto na imagem, nem mãos — que é onde quantização costuma quebrar
primeiro. O decode do BF16 emitiu avisos de OOM do alocador e caiu no caminho tiled do ComfyUI;
saiu imagem correta, mas a VAE disputou memória com os 12 GiB de pesos, coisa que o INT4 não fez.

---

## 2026-08-18 — Sparge + Sage2 + Triton medidos de verdade (e o resultado é humilde)

Até aqui Sparge estava instalado com kernel provado na sm86, mas **nunca medido em modelo**.
Medido agora, plugado na atenção do ComfyUI, mesmo Z-Image-Turbo BF16, mesmo prompt, mesmo seed,
pesos residentes. Backend forçado via `--attention` em `tools/nunchaku_compare.py`.

Antes dos números, três correções de premissa:

- **Sage2++ não existe na 3090.** `sageattention/core.py:162` põe o ramo `# SageAttention2++` em
  `sm89`; `sm80/86/87` cai em `sageattn_qk_int8_pv_fp16_cuda` com acumulador fp32. É Sage2.
- **Sparge não empilha com Sage2, contém Sage2.** As entradas chamam-se `spas_sage2_attn_*` —
  quantização do Sage2 mais esparsidade de blocos. Substitui o `sageattn`, não soma a ele.
- **Triton não é terceira peça.** Roda dentro dos dois, fazendo a quantização INT8 em
  `per_thread_int8_triton` (`core.py:442` tem `qk_quant_gran="per_thread"` como default).

| atenção | s/pass | s/step | speedup | blocos pulados |
|---|---|---|---|---|
| ComfyUI default (SDPA) | 7.03 | 0.8790 | 1.00x | — |
| SageAttention 2 | 6.86 | 0.8572 | 1.02x | — |
| SpargeAttn topk 0.5 | 6.68 | 0.8347 | 1.05x | 47.88% |
| SpargeAttn topk 0.25 | 6.60 | 0.8247 | **1.07x** | **71.82%** |

**Pular 72% dos blocos de atenção comprou 6.5%.** Não é falha do Sparge — é o que sobra quando a
atenção não é o gargalo. No mesmo modelo, na mesma placa, o W4A4 do Nunchaku comprou 127%,
atacando as camadas lineares.

Qualidade: a imagem do `topk 0.25` está íntegra, sem artefato, sem borrão — discutivelmente a
melhor das três. `relL2 0.8264`, `cosine 0.7118` contra o default: trajetória bem diferente,
resultado igualmente bom.

### Dois bugs encontrados no caminho, um deles do SpargeAttn

1. **`UnboundLocalError` com `smooth_k=False`.** `spas_sage_attn/core.py:163` atribui `km` só
   dentro de `if smooth_k:`, e a linha 169 usa `km` incondicionalmente. ComfyUI passa
   `smooth_k=False` em `attention_sage`, então **toda** chamada morre. Bug do upstream, não da
   integração. Contornado forçando `smooth_k=True` no shim.
2. **`assert q.size(-2)>=128`** (`core.py:154`). Toda cross-attention aqui carrega 23 tokens de
   texto e cai nesse assert. Roteada para o Sage em vez do PyTorch — cair no caminho mais lento
   faria a medição parecer um resultado de Sparge sem ser.

Sem esses dois contornos o `attention_sage` do ComfyUI engolia a exceção e caía em
`attention_pytorch` silenciosamente, com a mensagem `Error running sage attention: ..., using
pytorch attention instead` — ou seja, um usuário que ligasse Sparge assim teria ficado **mais
lento** achando que estava mais rápido.

### Ressalva que muda a conclusão para outro caso de uso

Isto é um modelo de **imagem**, 16384 tokens de latente e 23 de texto. Modelos de **vídeo** têm
sequências de ordem de magnitude maiores, e a atenção é quadrática: lá o eixo Sparge/Sage deve
pesar muito mais do que os 7% vistos aqui. Não medido. Não afirmo.

---

## 2026-08-18 — deepcompressor baixado, bloqueado em compilador

`F:\COMFY_PORTABLE\deepcompressor` (clone raso, 11 MB). Instalado no `python_embeded`:
`omniconfig 0.1.10`, `datasets 5.0.1`, `pandas 3.0.5`, `pyarrow 25.0.1`, `multiprocess`,
`docstring-parser`, `tzdata`. Verificado por `pip install --dry-run` **antes** e por
`import torch` **depois**: nada tocou Torch nem numpy, as quatro principais estavam ausentes
(nenhum upgrade). `torch 2.13.0+cu130` intacto.

`import deepcompressor` funciona. `deepcompressor.app.diffusion.ptq` **não**:

```
deepcompressor/data/__init__.py -> data/dtype.py -> data/codebook.py -> csrc/load.py:12
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1
```

`csrc/load.py` compila uma extensão C++ por JIT (`torch.utils.cpp_extension.load`) **no import**,
e ela é alcançada pela raiz do pacote. Precisa de `cl.exe` (Visual Studio Build Tools, workload
C++), que não existe nesta máquina. Não é opcional e não tem flag para pular.

Ainda faltam 12 dependências, quase todas de avaliação: `lm_eval jieba fuzzywuzzy rouge
python-Levenshtein clean-fid dominate bs4 cd-fvd xformers pyav clip image_reward`. Se o caminho
de conversão precisa de todas, não sei — só dá para descobrir depois que o import passar.

Nota para o plano de `.exe`/GUI: **um wrapper não remove o compilador**, porque o JIT roda no
import. A saída melhor é pré-compilar a extensão uma vez e distribuir o `.pyd` junto, como o
woct0rdho faz com Sage e Sparge — mas isso ainda exige MSVC uma vez, aqui.

**Correção, mesmo dia: o bloqueio não existia.** Antes de pedir os 2–7 GB de Build Tools eu
procurei `cl.exe` na máquina. Já havia **quatro** instalações MSVC: VS 2019 BuildTools,
VS 18 BuildTools, VS 2022 BuildTools e VS 18 Community. O problema era PATH, não instalação.
Nada foi instalado a nível de sistema.

### Conversor operacional

`run_deepcompressor.bat` na raiz carrega o `vcvars64.bat` do VS 2022 BuildTools (com fallback
para os outros dois) e chama `deepcompressor.app.diffusion.ptq` com o Python embedded.
`--help` responde.

O `vcvars` é necessário **em toda execução**, não só na primeira: `csrc/load.py` compila por JIT
no import e o `torch.utils.cpp_extension.load` roda `where cl` para validar a toolchain **antes**
de consultar o próprio cache de build. Extensão cacheada não dispensa o compilador.

Três correções locais foram necessárias (todas comentadas no código como `LOCAL PATCH`, nenhuma
enviada para o upstream):

1. `deepcompressor/csrc/load.py` — as listas de flags são de GCC. Com CUDA >= 13 o CCCL recusa o
   pré-processador tradicional do MSVC e o build morre em
   `preprocessor.h(23): fatal error C1189`. Passei `/Zc:preprocessor` direto ao `cl` e via
   `-Xcompiler` ao `nvcc`, e troquei as flags GCC por equivalentes MSVC no ramo Windows.
2. `deepcompressor/app/llm/eval/longbench/eval.py:339` — `open()` sem `encoding` num JSON UTF-8
   com CJK. No Windows o default é cp1252 e o import inteiro morre com
   `UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d`. O `open()` irmão da linha 123 já
   passava `encoding="utf-8"`; a leitura foi esquecida. Um arquivo de benchmark de LLM que nenhum
   workflow de difusão vai ler impedia `app.diffusion.ptq` de importar.
3. `python_embeded/Lib/site-packages/deepcompressor.pth` — o embedded ignora `PYTHONPATH` por
   causa do `python313._pth`, e `pip install -e` falha com
   `BackendUnavailable: Cannot import 'poetry.core.masonry.api'`. Um `.pth` resolve sem instalar
   backend de build nenhum.

### Um dano de stack pego e revertido

Instalar `image_reward` **rebaixou `timm` de 1.0.28 para 0.6.13** (ele pina `timm==0.6.13`).
`timm` é usado por `comfyui-easy-use` e `comfyui-frame-interpolation` — nós reais desta
instalação. Removi `image_reward` e `fairscale` (ambos só de avaliação, fora do caminho de
conversão) e restaurei `timm==1.0.28`. Confirmado: `torch 2.13.0+cu130`,
`torchvision 0.28.0+cu130`, `transformers 5.14.1`, `timm 1.0.28`, tudo intacto.

Lição para o resto do projeto: `pip install --dry-run` mostra o que **seria instalado**, mas eu
li a lista procurando `torch` e `numpy` e não notei o downgrade de `timm`. A checagem certa é
comparar a lista inteira contra o que já está instalado, não procurar nomes suspeitos.

### O que ainda falta para converter um checkpoint local

Configs existentes: `flux.1-dev`, `flux.1-schnell`, `pixart-sigma`, `sana-1.6b`. Presets de
quantização: `int4`, `nvfp4`, `fast`, `gptq`, `__default__`.

**Não há config para Z-Image nem para Qwen-Image**, que são os modelos de imagem em uso aqui. E o
pipeline carrega pelo `diffusers`, não lê `.safetensors` do ComfyUI direto. Então converter um
checkpoint local exige escrever config nova e ter o modelo em formato diffusers — mais o custo de
GPU das três etapas (referência, calibração com 128 prompts do COCO, quantização com avaliação).

---

## 2026-08-18 — SANA validado até a metade, e abortado por custo

O pipeline de três etapas do deepcompressor **roda** nesta máquina. Não terminou porque o preço
não compensa, não porque quebrou.

| etapa | resultado |
|---|---|
| 1. referência | pulada (`--skip-eval true`), não altera o modelo produzido |
| 2. calibração | ✅ 128/128 amostras, **25min42**, 3,43 s/amostra, 7,5 GiB de dataset |
| 3. quantização INT4 | ▶ iniciou e progrediu; **abortada pelo usuário** |

O número que motivou abortar, lido do log em execução:

```
smoothing: 1/20 [36:11<8:48:12, 1668.00s/it]
GPU 9857 MiB, 100%  |  RAM 59.9 GB
```

**36 minutos por camada, 8h48 estimadas, para um modelo de 1.6B.** Cabe em 24 GB de VRAM (usa
9,8 GB), mas consome ~60 GB de RAM — nesta máquina isso disputa commit com o `vmmemWSL`.

Isso confirma empiricamente o que o guia de comunidade [spooknik/deepcompressor-guide] afirma em
teoria (48 GB mínimo, 18–20 h para FLUX 12B): o gargalo do SVDQuant não é capacidade, é tempo e
RAM. Extrapolar para Qwen-Image (20B, 12,5× o SANA) não é conservador.

### Estrutura do formato SVDQuant, mapeada a partir do checkpoint oficial

Feito sem download extra — `svdq-int4_r32-z-image-turbo.safetensors` já estava no disco.
1099 tensores, 136 camadas quantizadas.

Uma camada (`context_refiner.0.attention.to_out.0`):

```
qweight             I8    (3840, 1920)   INT4 empacotado 2-por-byte
wscales             BF16  (60, 3840)     3840/64 = 60 grupos -> group_size 64
smooth_factor       BF16  (3840,)
smooth_factor_orig  BF16  (3840,)
proj_down           BF16  (3840, 32)     branch low-rank, rank 32
proj_up             BF16  (3840, 32)
```

Composição do arquivo: `qweight` 83,3%, BF16 não quantizado 7,7%, `wscales` 5,2%,
low-rank (`proj_up`+`proj_down`) 3,7%, `smooth_factor` 0,0%.

O formato **se declara** no `__metadata__`, então não há engenharia reversa a fazer:

```json
"quantization_config": {"method": "svdquant",
  "weight": {"dtype": "int4", "scale_dtype": null, "group_size": 64},
  "activation": {"dtype": "int4", "scale_dtype": null, "group_size": 64},
  "rank": 32, "skip_refiners": false}
"config": {"_class_name": "ZImageTransformer2DModel", "_diffusers_version": "0.36.0.dev0", ...}
```

`_class_name: ZImageTransformer2DModel` derruba a alegação de que Z-Image dependeria do diffusers
ganhar uma pipeline class — ela existe.

### Por que isto importa para o ConvRot

Comparando com a saída do nosso conversor (`gemma_3_12B_it_heretic_w4a8`, 673 camadas):

```
weight              I8    (3840, 7680)
weight_codebook     F32   (16,)
weight_s_channel    F32   (3840,)
weight_s_rel        U8    (3840, 960)
format: asym_w4a8_int8, group_size 16, convrot_groupsize 256
```

São **duas estratégias diferentes para o mesmo problema** — outliers em 4 bits.

- **SVDQuant absorve**: branch de posto 32 mais suavização de ativação. Encontrar esse branch
  exige SVD sobre ativações reais, logo calibração, logo as 8h48 e os 60 GB de RAM.
- **ConvRot rotaciona**: `convrot_groupsize 256`, escala em dois níveis (F32 por canal + U8
  relativo por grupo de 16) e codebook de 16 entradas. Rotação é transformada direta e
  **não requer calibração**.

Essa é a vantagem prática do ConvRot que não estava registrada: converte em minutos, sem dataset,
sem 128 prompts do COCO, sem noite de GPU. O SVDQuant compra qualidade-por-bit pagando em
calibração. Qual das duas vence em qualidade final não foi medido e não se deduz da estrutura.

### Estado

Processo de quantização morto (PID 5504). Restam órfãos 7,5 GiB de dataset de calibração em
`deepcompressor/examples/diffusion/datasets/` — mantidos por ora, reutilizáveis se o SANA for
retomado. Modelo SANA bf16 (9,07 GiB) em `D:/deepcompressor-models/`.

---

## 2026-08-18 (tarde) — SVDQuant medido em modelo real, com imagens

Tudo abaixo foi executado nesta máquina. Condicionamento real (prompt, text encoder, VAE), pesos
residentes, mesma seed, e o comparador **recusa** a comparação se o `cond_norm` divergir entre os
lados — então "mesmo prompt" aqui significa mesmo tensor, não mesma string.

### FLUX.1-dev — o maior ganho da sessão

| | fp8 e4m3fn | SVDQuant INT4 r32 | razão |
|---|---|---|---|
| disco | 11,08 GiB | 6,30 GiB | 0,57x |
| load | 134,3 s | 75,8 s | 0,56x |
| VRAM (nvidia-smi) | 12.682 MiB | 7.134 MiB | 0,56x |
| s/step @1024², 20 steps | 1,1222 | 0,3899 | **2,88x** |

`latent cosine 0.9344`. Visualmente indistinguível: mesma maçã, mesma madeira, mesma luz e
sombra. Prompt `"a red apple on a weathered wooden table, afternoon light, sharp detail"`,
`cond_norm 156.5873` idêntico nos dois lados.

### Beyond_Reality Z-Image v2 — antes/depois contra o bf16 do próprio autor

O `tonera` publica o transformer bf16 ao lado do quantizado, então este é o único par da sessão
sem proxy: mesmo modelo, mesmo autor, um quantizado e o outro não.

| | bf16 | SVDQuant INT4 r32 |
|---|---|---|
| disco | 11,46 GiB | 3,36 GiB |
| VRAM | 12,05 GiB | 3,91 GiB |
| s/step @1024², 8 steps | 0,8793 | 0,3915 |
| | | **2,25x** |

### Qwen-Image — build de terceiro é legítimo

`QuantFunc/Nunchaku-Qwen-Image-2512` (balance, INT4) carrega no loader **oficial** e roda
idêntico ao `nunchaku-ai/nunchaku-qwen-image` r128: ambos 0,617 s/step, 12,35 GiB de pico,
`SVDQW4A4Linear×480` + `NunchakuQwenImageTransformerBlock×60`, 781 módulos quantizados. Os dois
declaram `rank 128`, `group_size 64` — o rank não os diferencia.

### `tonera/Qwen-Image-Edit-2511-Lightning` NÃO carrega

```
ValueError: Key transformer_blocks.0.img_mod.1.qweight not found in state_dict
```

Causa, verificada tensor a tensor:

```
oficial 2509 — img_mod.1: qweight I32 (4608,1536) + wscales + wzeros   -> AWQ W4A16
tonera  2511 — img_mod.1: weight  F16 (18432,3072)                     -> não quantizado
```

O loader do `ComfyUI-nunchaku 1.2.1` exige as camadas `mod` quantizadas. Isso também explica os
13,72 GiB dele contra 11,79 do oficial. **13,72 GiB de peso morto** até haver suporte ou uma
build com `mod` em AWQ — `QuantFunc/Nunchaku-Qwen-Image-EDIT-2511` é a alternativa óbvia, não
baixada.

Correção de um registro anterior: eu havia escrito "600 camadas quantizadas, 480 com low-rank,
120 sem". O certo é **480 em SVDQuant W4A4 e 120 em AWQ W4A16** — as `mod` usam outro formato,
com zero-point, não é ausência de branch.

### `tools/svdq_to_bf16.py` — reconstrução de peso a partir do SVDQuant

Recupera BF16 de um checkpoint SVDQuant sondando o kernel com a identidade (`W.T = forward(I)`),
em vez de reverter o interleave de tensor core, que o pacote não expõe.

Contra o bf16 original do Beyond_Reality, 20 camadas:

```
mediana 0.0972 | melhor 0.0473 | pior 0.1059
```

**~10% de erro relativo nos pesos** é o que o SVDQuant custou para levar 11,46 → 3,36 GiB.

Três coisas descobertas ao construir, todas por medição:

1. **A camada não é linear.** É W4A4 — a ativação também é int4. Verificar com `x` denso
   aleatório mede erro de ativação, não de reconstrução, e reprova (0,0988) uma recuperação
   correta.
2. **A fusão do feed-forward é `w3+w1`, não `w1+w3`.** Ordem natural dá erro relativo 1,4199
   contra o original; a invertida dá 0,0995. Cortando a matriz recuperada ao meio: metade
   superior casa com `w3` (0,099), inferior com `w1` (0,1001).
3. **O quantizado funde o que o diffusers separa**, e casar só por nome comparava 5 camadas de
   20 — todas `to_out` — reportando isso como se cobrisse a rede. Verificado pelas formas:
   `to_qkv (11520,3840) = to_q+to_k+to_v`, `feed_forward.net.0.proj (20480,3840) = w3+w1`,
   `feed_forward.net.2 = w2`.

### Erro de metodologia meu, registrado

Comparei Qwen-Image **base** contra Qwen-Image-**2512** com 8 steps e cfg 1.0 — ajuste de modelo
distilled. O base saiu cru e eu quase atribuí isso à quantização. Refeito com **20 steps e
cfg 4.0**, mesmo arquivo e mesma seed, a imagem fica nítida e correta (custo: 24,77s contra
4,93s, porque cfg>1 roda cond e uncond).

**Cada modelo tem o seu ajuste; comparar dois num ajuste único mede o ajuste, não o modelo.**
Mesmo erro de categoria do threshold do FBCache, que também não era portável entre arquiteturas.

### Bugs corrigidos nas ferramentas

- `extra_model_paths.yaml` **nunca era lido** por script fora do `main.py`, então todo modelo no
  D: dava "not found" — inclusive dentro dos nós do nunchaku. Agora é carregado explicitamente.
- Ordem de carga invertida: o modelo de difusão entrava **antes** de o prompt ser codificado.
  Qwen-2.5-VL-7B (~15 GiB) + Qwen-Image INT4 (12 GiB) estouravam os 24 GiB. O comentário no
  código já dizia a ordem certa; a implementação fazia o contrário.
- Decode 5D: o latente do Qwen tem eixo temporal mesmo para imagem parada
  (`TypeError: Cannot handle this data type: (1, 1, 3, 1)`).
- Sonda de identidade precisa ser 3D `(1, in, in)`; 2D morre em
  `not enough values to unpack (expected 3, got 2)`.
- Loaders do nunchaku **não compartilham assinatura**: ZImage recebe só o nome, Qwen exige
  `cpu_offload` posicional, FLUX recebe seis argumentos (um deles `cache_threshold`, o
  first-block cache próprio deles — fixado em 0 para não misturar cache com quantização).
- `save_file` do safetensors falha no D: com `os error 50: The request is not supported`. Merge
  dos shards feito com escrita manual (header + streaming).

### Bugs conhecidos e **não** corrigidos

- `svdq_to_bf16` nunca rodou em modelo inteiro, só `--limit 20`.
- Os 6 workflows `.nunchaku.json` nunca foram abertos nem enfileirados (teste 4). Precisa de GPU.

---

## 2026-08-18 - sessao sem GPU

GPU emprestada para trabalho de LLM nao relacionado. Tudo abaixo e CPU, rede ou git. A NVML
confirma **3 processos de terceiros na placa durante toda a sessao**, o que por si so e o motivo
de um dos consertos abaixo ter a forma que tem.

### Os dois bugs de ferramenta acima: corrigidos

**1. `peak VRAM` mentindo no caminho nunchaku.** Causa: `torch.cuda.max_memory_allocated` so
conta o que passou pelo caching allocator do PyTorch, e o nunchaku aloca os pesos dentro da
extensao C++. Reportava `0,21 GiB` para um modelo de ~7 GiB - e o pior nao e errar, e errar ao
lado de uma linha BF16 que estava certa: a coluna lia como um ganho de memoria de 50x.

Corrigido com uma classe `DevicePeak` em `tools/nunchaku_compare.py`: thread daemon amostrando
NVML a cada 50 ms, do instante anterior ao load ate o fim das passadas cronometradas.

Detalhe que **obriga** a forma da solucao: no Windows o driver roda em modo WDDM e a NVML se
recusa a quebrar o total por processo (`nvmlDeviceGetComputeRunningProcesses` devolve entradas
com `usedGpuMemory` indisponivel). Entao o que existe e uso do dispositivo inteiro, e a
atribuicao vem de subtrair uma baseline tomada imediatamente antes de o modelo carregar. Isso so
vale se nada mais na placa crescer no meio - condicao que **nao e assumida**: o numero de outros
processos e medido, gravado no arquivo de resultado e impresso ao lado da figura.

Os dois numeros passam a ser gravados (`peak_gib` da NVML, `torch_peak_gib` do allocator) junto
com `peak_source`. E o `--compare` **recusa** dividir um contra o outro:

```
peak VRAM GiB                       0.21          7.00          --
  !! not comparable: A measured by torch caching allocator (undercounts CUDA extensions),
     B by nvml device-wide minus baseline. Re-run both.
```

Arquivos `.pt` antigos nao tem `peak_source` e sao tratados como allocator por definicao, que e
o que sao. **Consequencia pratica: medicao de VRAM anterior a hoje nao pode ser comparada com as
futuras.** Os numeros de VRAM ja registrados neste log vieram de `nvidia-smi` a mao e continuam
validos; o que nao vale e o que saiu da coluna da ferramenta.

**2. `--verify` do `svdq_to_bf16` testando a propriedade errada.** O antigo exigia
`layer(x) ~= x @ W.T` com folga de 2e-2. Isso e impossivel: W4A4 quantiza **a ativacao** tambem,
entao a camada nao e linear na entrada e nenhuma matriz BF16 a reproduz exatamente. Media 0,0988
e reprovava reconstrucao correta.

O que o formato **tem** e que a sonda de identidade e *exata*, nao aproximada - cada linha de `I`
tem um unico nao-zero, e escala simetrica por grupo representa um nao-zero solitario (e os zeros
ao redor) sem erro nenhum. Medido, nao deduzido: erro maximo `0.0`. Logo `recovered` e o peso
efetivo que o kernel guarda, e o residuo em entrada aleatoria e a quantizacao da *sonda*.

O gate novo e por direcao, com dois limiares, e cada camada e pontuada **ao lado de uma matriz
deliberadamente errada** (as proprias linhas embaralhadas), para que a margem seja impressa em
vez de afirmada. Calibracao medida contra uma camada que quantiza a ativacao em INT4 por grupo
de 64:

| caso | cos | relL2 | gate antigo (rel<=0,02) | gate novo (cos>=0,99 e rel<=0,25) |
|---|---|---|---|---|
| reconstrucao correta | 0,9942 | 0,1072 | **REPROVA** <- o bug | aprova |
| matriz errada (embaralhada) | 0,0010 | 1,4150 | reprova | reprova |
| correta x 1,5 | 0,9943 | 0,5080 | reprova | **reprova** <- cosseno sozinho aprovaria |

A ultima linha e o motivo de manter os **dois** limiares: escala errada mantem cosseno 1.

Adicionado tambem um teste que nao existia e que cobre o caso que o docstring dizia estar
protegido sem nunca ter sido exercitado: com o bias suprimido a camada tem de mapear zero em
zero. Custa um forward num tensor de zeros, pega uma falha que corromperia toda linha da camada,
e por isso roda mesmo com `--verify 0`.

`tools/test_svdq_verify.py` trava essa calibracao: 4 testes, CPU, sem checkpoint e sem GPU. Um
deles falha de proposito se o teto de 0,02 voltar a ser alcancavel.

> **Nota de ambiente:** nao ha `pytest` neste interpretador embedded. Os comandos de teste do
> CLAUDE.md que invocam um nao funcionam como escritos. O arquivo carrega o proprio runner.

### PR #3 upstream: chengzeyi/Comfy-WaveSpeed#149

Porte das correcoes do FBCache para o upstream de verdade. Upstream esta parado desde
**2025-08-02** (`8253745`), e o fork divergiu 2170 linhas em `first_block_cache.py`, entao
cherry-pick nao aplica. Rebase feito a mao, em worktree isolado - o no vivo em `custom_nodes/`
nao foi tocado.

Dos 7 bugs do fork, **so 2 existem no upstream**:

| bug | upstream 8253745 | acao |
|---|---|---|
| 1. clone do input do bloco 0 | **presente**, `fbcache_nodes.py:257` | portado |
| 2. deteccao FLUX larga demais | nao - upstream compara nome exato | descartado |
| 3. Wan sem `blocks` | ausente, mas seria *feature* | descartado |
| 4. igualdade de string vs MRO | presente | portado |
| 5. `modulation_dims` (#120) | ja tem; foi o fork que perdeu | descartado |
| 6. cache unica p/ cond+uncond | design diferente (`sequence_num`) | descartado |
| 7. gate `cos_sim` | adicao do fork (ec3d421) | descartado, como previsto |

O bug 1 foi **provado contra a classe real do ComfyUI**, nao argumentado:

```
block returns the same object it was handed : True
the aliased original was mutated            : True
residual measured against the alias  (max)  : 0.0
residual measured against a copy     (max)  : 0.25027644634246826
```

E a razao de ninguem ter percebido: `are_two_tensors_similar` faz
`(t1-t2).abs().mean() / t1.abs().mean()`, que para dois tensores nulos e `0/0` -> `nan`, e
`nan < threshold` e `False`. Miss permanente, saida bit-identica ao modelo sem patch, nada
levantado e nada logado - e ainda **mais lento**, porque o bloco 0 roda duas vezes.

O mesmo aliasing esta em tres lugares no upstream, os tres corrigidos. A flag
`clone_original_hidden_states` foi **removida** em vez de virar `True`: nao existe configuracao
correta em que pular a copia esteja certo.

Dois testes de CPU acompanham. O segundo **falha no upstream como publicado** e passa com a
mudanca - verificado nos dois sentidos com `git stash`.

Declarado no PR o que ele *nao* cobre: ambiente e ComfyUI 0.29.0 / torch 2.13, nao o de agosto
de 2025; nao ha comparacao de imagem ponta a ponta; e o gate do primeiro bloco na rota FLUX
(`first_hidden_states_residual = img`, estado bruto e nao residuo) foi deixado exatamente como
esta, porque mexer moveria o significado de `residual_diff_threshold` para todo usuario atual.

### Qwen-Image-Edit-2511: substituto baixado, e o diagnostico anterior corrigido

Baixado `QuantFunc/Nunchaku-Qwen-Image-EDIT-2511`, variante **`balance_int4` (rank 128)** ->
`D:/ComfyUI-Models/diffusion_models/svdq-int4_r128-qwen-image-edit-2511.safetensors`,
12.654.443.120 bytes (tamanho conferido contra o manifesto antes de instalar).

Rank 128 escolhido **nao** por ser "balance": e o rank que o slot ja usava
(`svdq-int4_r128-qwen-image-edit-2509`), entao a troca continua sendo de um arquivo so, em vez
de mudar junto o ponto de qualidade/velocidade. FP4 nao foi baixado - sm_86 nao tem FP4.

Comparacao de header das tres (so CPU, so metadados) mostra que o registro anterior sobre o
build da tonera estava **impreciso**:

| checkpoint | camadas `img_mod`/`txt_mod` | formato |
|---|---|---|
| 2511 QuantFunc r128 (novo) | `qweight I32` + `wscales` + `wzeros` | AWQ W4A16 - **igual ao 2509** |
| 2509 (funciona) | `qweight I32` + `wscales` + `wzeros` | AWQ W4A16 |
| 2511-lightning (tonera) | `qweight I8` + `smooth_factor` + `smooth_factor_orig`, mais 4 `weight` F16 | SVDQuant |

O que estava escrito antes - "`img_mod` deixado em F16" - e so a ponta: sao 4 tensores em F16,
mas o problema real e que a tonera quantizou **todas** as 116 camadas de modulacao como
**SVDQuant onde o loader exige AWQ**. Estruturalmente o arquivo novo e identico ao 2509 que
funciona, o que e evidencia forte mas **ainda nao e carga**: nunca foi aberto por um loader,
porque isso precisa de GPU.

O arquivo da tonera (13,71 GiB) continua no disco e continua morto.

> Nota lateral: `D:` neste host e um share de rede (`\\192.168.3.68\estoque`), o que explica
> `save_file` do safetensors falhar la com `os error 50`.

### Pendente ao fim da sessao sem GPU

1. Abrir e enfileirar os 6 workflows `.nunchaku.json` (teste 4).
2. Carregar o 2511 novo por um loader de verdade e comparar contra o 2509.
3. Rodar `svdq_to_bf16` em modelo inteiro.
4. Confirmar as duas colunas de VRAM novas numa medicao real.

---

## 2026-08-18, parte 2 - GPU de volta

Os itens 1, 2 e 4 acima foram executados. O 3 continua aberto. E os **dois consertos da parte 1
foram testados contra hardware e os dois estavam errados** - um na causa, outro na propria ideia.

### O `--verify` estava errado pela SEGUNDA vez

A calibracao da parte 1 (`cos >= 0,99`) veio de um stand-in sem smoothing e sem ramo de baixa
dimensao. Contra o kernel real ela nao sobrevive. Z-Image INT4 r32, 14 camadas amostradas ao
longo da rede:

| matriz | cos | relL2 |
|---|---|---|
| reconstrucao correta | 0,704 .. 0,998 | 0,058 .. 0,894 |
| linhas embaralhadas | 0,004 max | razao 0,56 |
| correta x 1,5 | 0,995 max | razao 0,65 |
| correta + 20% ruido | **0,976 max** | **razao 0,963** |

Uma `feed_forward.net.2` **correta** pontua 0,704. Uma matriz **20% errada** pontua 0,976. A
errada ganha da certa, entao nenhum limiar absoluto separa. Com o gate da parte 1, so 4 de 12
camadas corretas passariam.

Causa medida, nao suposta: o `smooth_factor` por canal. Varrendo o espectro dele num stand-in na
CPU, uma reconstrucao **correta** anda de cos 0,994 ate 0,586 - atravessa exatamente onde uma
matriz ruidosa de camada facil se senta. E o mesmo erro de categoria do threshold do FBCache: o
numero descreve a camada, nao a correcao.

Homogeneidade tambem foi testada (`layer(cI)/c == layer(I)`): exatamente 0 em toda camada, e ~0
tambem em entrada aleatoria. Nao distingue nada - quantizacao simetrica por grupo e homogenea
para qualquer entrada.

O que **sobrou** como checagem de verdade, em `check_recovery`:

- **bias**: com o bias suprimido a camada tem de mapear zero em zero. 0,0 em toda camada de todo
  checkpoint testado.
- **determinismo**: duas sondas de identidade identicas tem de bater bit a bit.
- **magnitude**: matriz toda-zero ou nao-finita e lancamento de kernel que falhou sem levantar.

E o que a sonda de identidade e, agora medido no kernel real e nao argumentado: **exata**.
`layer(8I)/8` bate com `layer(I)` em exatamente 0 em toda camada amostrada.

Os escores em entrada aleatoria viraram **tabela de diagnostico, nao veredito**, com o controle
embaralhado impresso do lado e o aviso de que 20% errado pontua 0,976. Quem so tem o kernel nao
tem oraculo; **so `--reference` responde se a reconstrucao presta**.

`tools/test_svdq_verify.py` reescrito: 5 testes CPU. Um deles reproduz a falha de discriminacao
de proposito - falha se um limiar voltar a parecer defensavel.

### O VRAM estava certo no conserto e errado na causa

Escrevi que "o nunchaku aloca fora do allocator do PyTorch". Medido nas tres rotas, mesmo dia:

| rota | modulos quantizados no grafo | torch allocator | NVML |
|---|---|---|---|
| FLUX.1-dev | **2** | 0,21 GiB | 6,71 GiB (**31x**) |
| Qwen-Edit-2511 | 480 | 12,35 GiB | 12,60 GiB |
| Z-Image | 204 | 3,91 GiB | 4,55 GiB |

Nao e "o caminho nunchaku" - e **so o FLUX**, que carrega o transformer inteiro no engine nativo
e devolve um wrapper (por isso so 2 modulos Python). Qwen e Z-Image constroem `SVDQW4A4Linear` em
Python, cujos buffers **sao** tensores torch, e ali os dois numeros batem dentro de um contexto
CUDA. A linha `quantised modules in graph` que a ferramenta ja imprimia e o sintoma: **2 significa
que a pegada e invisivel ao PyTorch**.

Corrigido tambem um defeito do proprio aviso que escrevi: eu alertava sobre "N outros processos
de compute". Na WDDM a NVML lista 4 processos numa 3090 ociosa a 36 MiB - o processo System, um
servico da AMD, um tray app - todos com `usedGpuMemory` indisponivel. O aviso dispararia sempre.
Agora alerta sobre **memoria residente** na baseline, que e o que de fato ameaca a subtracao.

### ComfyUI 0.33 quebra o loader Z-Image do nunchaku no Windows

Os 6 workflows reescritos: **o swap esta correto**. O no exige exatamente um input (`model_name`),
`swap_to_nunchaku` escreveu exatamente um, com o valor certo, e o diff contra os originais mostra
que **so** o modelo de difusao mudou. Confirmado tambem que `MarkdownNote` sumindo do
`/object_info` era falso positivo do meu proprio checador - no de frontend nao chega ao backend.

Mas ao enfileirar, o loader morre:

```
AttributeError: 'NoneType' object has no attribute 'dtype'
nunchaku/models/linear.py:152  torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
```

O mesmo checkpoint pelo mesmo no, chamado direto no processo com as mesmas flags e os 2297 nos
registrados, carrega sem erro (136 modulos SVDQ). Repro minimo de **9 nos escritos a mao**, sem
nada dos workflows, falha identico - entao nao e workflow, nem swap, nem checkpoint.

Causa-raiz, `ComfyUI/comfy/ops.py:520-538`:

```python
class Linear(torch.nn.Linear, CastWeightBiasOp):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        if (not comfy.memory_management.aimdo_enabled
            or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
            super().__init__(in_features, out_features, bias, device, dtype)
            return
        # "Windows doesn't over-commit memory ... If the commit charge exceeds the ceiling
        #  we can destabilize the system."
        torch.nn.Module.__init__(self)
        self.weight = None
```

ComfyUI 0.33 adicionou lazy-init de `Linear` **especifico para Windows**: o peso so aparece em
`_load_from_state_dict`. O nunchaku le `orig_attn.qkv.weight.dtype` em `patch_model`, antes disso.
`aimdo_enabled` so vira `True` em `main.py:289`, que e por onde o servidor passa e um script
importando comfy como biblioteca nao passa - o que explica os dois resultados de uma vez.

**Workaround, testado com geracao real (nao inferido do codigo):**

```
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190
```

Com essa flag, `TXT2IMG-ZIMG.nunchaku` e `Zimg-TXT2IMG-_multigpu.app.nunchaku` **rodaram e
salvaram imagem** (`z-image-turbo_00118/00119`). Vale PR para ComfyUI-nunchaku - ha repro minimo
e a linha exata.

Dos outros 4 workflows: 2 (`image_qwen_image_edit_2509_relight`, `templates-image_to_real`)
pedem `qwen_2.5_vl_7b_fp8_scaled.safetensors`, que nao esta instalado, e **os originais tambem
pedem** - quebra anterior ao rewrite. Os outros 2 falham em `LoadImage` por PNG de entrada
ausente (`z-image-turbo_00547_.png`, `z-image-turbo_00030_.png`), que e dado do usuario.

### Qwen-Image-Edit-2511: carrega e roda

781 modulos quantizados no grafo (`SVDQW4A4Linear` x480), 4,95s / 8 steps, 12,60 GiB, disco 11,79
GiB. A predicao estrutural da parte 1 (header identico ao 2509) confirmou-se em carga real. O
build da tonera continua morto.

Erro de metodologia meu, repetido: mandei o primeiro par de imagens em 8 steps / cfg 1,0, que e
ajuste de modelo distilled. Qwen-Edit base quer 20 steps / cfg 4,0 - refeito, 24,8s contra 4,9s
(cfg>1 roda cond e uncond). **Ja estava escrito neste log e eu repeti.**

### Benchmark controlado Z-Image: BF16 x INT4

Prompt de contagem do usuario (objetos contaveis: 3 chaves vermelha/amarela/verde, 4 engrenagens,
6 vidracas, 3 itens na prateleira, alca da xicara a direita, lapis amarelo com borracha rosa),
seed 1234 fixa, cfg 1,0, mesmo encoder (`qwen_3_4b`, lumina2) e mesmo VAE (`ae.safetensors`).
`|cond| 30234,8223` identico nas quatro execucoes - a condicionante e a mesma, o unico eixo que
muda e o modelo.

| | BF16 | INT4 SVDQuant | razao |
|---|---|---|---|
| disco | 11,46 GiB | 3,36 GiB | **3,41x mais leve** |
| VRAM | 12,95 GiB | 4,68 GiB | **2,77x menos** |
| 8 steps | 10,29 s | 4,90 s | **2,10x** |
| 16 steps | 21,04 s | 9,85 s | **2,14x** |
| s/step | 1,29 - 1,32 | 0,613 | |
| latent relL2 | - | 0,629 (8s) / 0,638 (16s) | |
| latent cosine | - | 0,816 (8s) / 0,813 (16s) | |

A divergencia de trajetoria **nao piora com mais steps**.

Primeira tentativa foi feita com `--repeats 1` e produziu `7,26 s/step` para o BF16 a 8 steps -
cold start, numero invalido, descartado e refeito com `--repeats 3`. A propria ferramenta avisa
que reporta a passada mais rapida "porque as lentas sao lentas por motivos que nao tem nada a ver
com o modelo", e eu ignorei o proprio aviso.

**INT8 SVDQuant nao existe.** `SVDQW4A4Linear` aceita `int4` e `nvfp4`, mais nada, e nvfp4 e
Blackwell. Nao ha ponto intermediario de 8 bits para comparar; FP8 e4m3fn seria armazenamento
apenas (sem compute FP8 na sm_86) e foi descartado a pedido em vez de ser passado como se fosse
SVDQuant de 8 bits.

### Ambiente: ComfyUI subiu de 0.29.0 para 0.33.0

O CLAUDE.md dizia `0.29.0` / commit `42d2aa55`. O servidor reporta **0.33.0** (`v0.33.0-19-gc1739380`,
lancado 2026-08-18). Corrigido no CLAUDE.md. E essa subida que trouxe o lazy-init acima.

### Ainda pendente

1. ~~Rodar `svdq_to_bf16` em modelo inteiro~~ - **feito**, ver abaixo. Falta o alvo real,
   `svdq-int4-qwen-image-2512-balance`, que nao tem BF16 publicado (e por isso nao tem como
   ser conferido).
2. PR para ComfyUI-nunchaku sobre o lazy-init.
3. ~~`beyond-reality-zimage-v2_bf16.safetensors` pode ter saido do proprio `svdq_to_bf16`~~ -
   **resolvido, nao saiu.** O usuario lembrava de te-lo baixado (a v1 veio quebrada e ele pegou
   esta outra), e o header confirma: **0 chaves fundidas, 102 separadas** (`.to_q.weight`,
   `.w1.weight`, `.w3.weight`). `svdq_to_bf16` recupera as matrizes ja fundidas - e obrigado a
   isso, porque e assim que o kernel as guarda - entao nao teria como produzir q/k/v e w1/w3
   separados. 12,31 GB, 521 tensores, sem metadata. **Serve como referencia limpa.**

   Teste generico util: para saber se um BF16 saiu de uma recuperacao SVDQuant, procurar
   `attention.to_qkv` / `feed_forward.net.0.proj` no header. Presentes = recuperado; ausentes
   com q/k/v separados = publicado.


## 2026-08-18, parte 3 - `svdq_to_bf16` num modelo inteiro, com verdade-terreno

Primeira execucao completa da ferramenta. Fonte `svdq-int4_r32-beyond-reality-zimage-v2`
(3,61 GB), referencia `beyond-reality-zimage-v2_bf16` (12,31 GB, publicado - proveniencia
confirmada pelo teste de chaves separadas).

**136 de 136 camadas recuperadas e casadas contra o original. Zero nao-pareadas.**

| | |
|---|---|
| erro relativo mediano vs BF16 | **0,1013** |
| melhor | 0,0382 |
| pior | 0,1239 (`layers.29.feed_forward.net.2`) |
| taxa | ~2,0 camadas/s |

Isso valida em escala o **mapa de fusao**, inclusive a ordem `w3+w1` descoberta antes em 20
camadas. Ordem errada teria falhado o casamento ou explodido o erro; casou 136/136 em ~0,10.
E o numero responde a pergunta que nenhuma checagem interna responde: **W4A4 custa cerca de 10%
de erro relativo de peso** neste modelo.

`check_recovery` passou nas 8 camadas amostradas (bias, determinismo, finitude). O arquivo saiu
sem nenhuma chave SVDQ residual, 419 tensores, todos BF16.

### Coincidencia que NAO deve ser lida como validacao

O diagnostico de entrada aleatoria deu 0,099 a 0,104 aqui, praticamente igual ao erro real de
0,1013. **E acaso.** No z-image-turbo o mesmo diagnostico espalhou de 0,058 a 0,894 para um erro
real presumivelmente parecido. O diagnostico continua nao servindo de proxy - foi exatamente por
isso que deixou de ser gate.

### Limitacao encontrada so ao rodar inteiro

O arquivo recuperado mantem o layout **fundido** do SVDQuant:

| | tensores | chaves fundidas |
|---|---|---|
| recuperado | 419 | 68 (`to_qkv`, `net.0.proj`) |
| publicado | 521 | 0 |

Os bytes equivalem, os nomes nao. O docstring da ferramenta dizia que o resultado vira "um
safetensors comum que qualquer loader le" - **forte demais**, e ja corrigido no proprio arquivo.
Le quem aceita a forma fundida. Desfundir e possivel (as formas dizem onde cortar) mas nao esta
implementado.

Saida de 11,46 GiB escrita em `D:/_svdq_recover_tmp/` e **apagada depois da conferencia**: e
qualidade INT4 em tamanho BF16, e o BF16 publicado deste modelo ja existe. So faz sentido guardar
para modelo sem BF16 original.


## 2026-08-18, parte 4 - o recuperado separa W4 de A4

Comparacao de tres no Beyond_Reality Z-Image v2, mesmo prompt de contagem, seed 1234, 8 steps,
cfg 1,0, mesmo encoder e VAE:

| | pesos | ativacoes | relL2 vs original | cosine | tempo | VRAM |
|---|---|---|---|---|---|---|
| original BF16 | BF16 | BF16 | - | - | 10,21 s | 12,95 GiB |
| **recuperado** | **INT4** | **BF16** | 0,4019 | 0,9194 | 10,16 s | 12,95 GiB |
| INT4 nativo | INT4 | INT4 | 0,4046 | 0,9154 | 4,83 s | 4,68 GiB |

`recuperado vs int4` da relL2 0,3569 - os dois estao mais perto um do outro do que qualquer um
esta do original. Mesmo modelo em dois trajes.

**O achado.** O usuario notou que uma caneta fina sai perfeita no recuperado e deformada no INT4.
Os pesos sao os MESMOS nos dois (a sonda de identidade e exata, medido), entao a unica variavel
entre essas duas imagens e a **quantizacao da ativacao**. Ou seja: o dano visivel vem do **A4**,
nao do W4. Reduzir peso a 4 bits custou ~10% de erro relativo e nao quebrou geometria fina;
quantizar a ativacao quebrou.

Mecanismo coerente: erro de peso e um desvio fixo que curva a trajetoria suavemente; erro de
ativacao e ruido novo a cada bloco de cada passo, de alta frequencia, e estrutura de poucos pixels
e onde aparece primeiro. Bate com `feed_forward.net.2` - entrada pos-ativacao - ser a pior camada
em todo diagnostico deste projeto.

**Nao exagerar:** no espaco latente as duas estao a MESMA distancia do original (0,4019 e 0,4046).
Muda o *tipo* de erro, nao a magnitude. A parte perceptual e uma imagem, uma seed, olho nu.

**Isso derruba o que este log e o docstring diziam** - que recuperar um modelo com BF16 disponivel
seria "estritamente pior, sem razao para rodar". E o unico jeito de separar W4 de A4, portanto e
instrumento de ablacao. Corrigido no docstring de `svdq_to_bf16.py`.

### O recuperado nao carregava, e o conserto

`KeyError: 'noise_refiner.0.attention.to_k.weight'` - a ferramenta devolve os pesos **fundidos**
como o kernel os guarda, e o ComfyUI quer q/k/v separados. Os cortes:

    attention.to_qkv        -> to_q, to_k, to_v    tres pedacos no dim 0
    feed_forward.net.0.proj -> w3, w1              dois pedacos, w3 PRIMEIRO
    feed_forward.net.2      -> w2                  so renomeia

Validado contra o BF16 publicado antes de gravar: **521 tensores, 0 faltando, 0 sobrando, 0 com
forma errada**, tamanho identico ao byte. A ordem `w3+w1`, provada antes em 20 camadas, segurou
nas 136.

Hoje isso e script avulso no scratchpad. **`svdq_to_bf16` ainda nao faz** - e enquanto nao fizer,
a saida da ferramenta nao carrega, que e exatamente o proposito dela. Deve ser absorvido.

### Pendente

1. Absorver o desfundir em `svdq_to_bf16`.
2. PR para ComfyUI-nunchaku sobre o lazy-init do ComfyUI 0.33.
3. Confirmar o achado W4-vs-A4 com mais seeds antes de trata-lo como conclusao.


## 2026-08-18, parte 5 - o caminho INT4 nao e deterministico entre processos

Medido ao tentar confirmar o achado W4-vs-A4. Mesmo modelo, mesma seed, mesmo prompt, mesma
condicionante (`|cond|` identico), duas execucoes em processos separados:

| modelo | execucoes identicas? | relL2 entre elas |
|---|---|---|
| original BF16 | **sim** | 0,00000 |
| recuperado BF16 | **sim** | 0,00000 |
| **INT4 nunchaku** | **nao** | **0,29794** |

O gap que eu estava lendo como dano do A4 era **+0,038**. O ruido do proprio INT4 entre execucoes
e **0,298**, quase 8x maior. O gap esta afogado.

**Invalida:** todo `relL2` / `cosine` de INT4 reportado neste log (0,629 e 0,638 no z_image_turbo;
0,4046 e 0,4399 no Beyond_Reality) - sao **amostras**, nao medidas. E invalida o teste de
acumulacao por steps que eu propus: o gap encolheu de +0,0380 (8 steps) para +0,0225 (24 steps),
que era o oposto da previsao, mas nenhum dos dois numeros tem significado.

**Sobrevive:** disco, VRAM e velocidade (nao dependem do latente); o erro de peso 0,1013 contra o
BF16 publicado (calculado dos pesos, nao de amostragem); e o mecanismo estrutural de que o
recuperado tem os mesmos pesos com ativacoes BF16 - reforcado, alias, porque o recuperado E
deterministico e o INT4 nao, o que aponta a variacao para o caminho de ativacao/kernel.

**A observacao da caneta continua de pe como observacao, sem medida.** Confirma-la exige N
execucoes da mesma seed para medir o espalhamento, nao mais seeds.

`check_recovery` testa determinismo **dentro** de um processo e passou. Entre processos e outra
coisa - provavelmente autotuning escolhendo GEMM diferente. Nao coberto.

## Desenho decidido para o proximo conversor (nao implementado)

Decisao do usuario, 2026-08-18: o proximo `.py` **nao** deve usar allowlist fixa. Deve **rodar o
modelo algumas vezes, observar as ativacoes durante a geracao** e so entao decidir a precisao por
camada - umas em 4 bits, outras em 8, com inicio e fim em precisao maior.

Isso e calibracao com consciencia de ativacao, e ha evidencia deste projeto a favor:

- o `smooth_factor` do SVDQuant **e** essa estatistica, ja coletada por calibracao;
- varrendo o espectro dele num stand-in, uma reconstrucao **correta** anda de cos 0,994 a 0,586 -
  a sensibilidade e por camada e ja esta escrita no checkpoint;
- `feed_forward.net.2` (entrada pos-ativacao) foi a pior camada em **todo** diagnostico deste
  projeto. O ranking de sensibilidade ja aparece sozinho.

Metade da ideia ja existe: `PROFILE_PATTERNS` em `quant_w4a4.py` ja exclui embeddings, norms,
`lm_head` e vision tower. A mudanca e trocar **lista fixa escrita a mao** por **decisao medida**.

**Verificar ANTES de escrever qualquer linha:** o kernel aceita precisao por camada? Ha
`_w4a8.safetensors` no disco (capybara, hv15, ltx-2.5, minimax), entao o formato existe - mas
nunca foi confirmado se `comfy_kitchen` executa W4A8 **nativo** ou cai em dequant + GEMM BF16.
Regra dura do projeto: se cair para eager, perdeu o sentido. Um conversor que produz arquivo que
nada executa e o pior resultado possivel.


## 2026-08-18, parte 6 - o kernel misto JA existe no comfy_kitchen

Verificado, nao suposto. `registry.get_implementation` resolve **tudo abaixo em
`comfy_kitchen.backends.cuda`** - nenhum cai em eager:

| operacao | resolve para |
|---|---|
| `convrot_w4a4_linear` | cuda |
| `w4a8_int8_linear` | cuda |
| `int8_linear` | cuda |
| `quantize_int8_convrot_weight` | cuda |
| `quantize_w4a8_int8_weight` | cuda |
| `scaled_mm_svdquant_w4a4` | cuda |

O backend CUDA tambem traz `rotate_int8_convrot_weight`, `quantize_int8_convrot_staged`,
`quantize_int4_rowwise_convrot64_to_int8`, `dequantize_w4a8_int8_weight`, alem de nvfp4, mxfp8 e
fp8. Ou seja: **ConvRot em INT8 e W4A8 nativo ja estao implementados.** O conversor com precisao
por camada nao precisa de kernel novo - so precisa escolher.

Correcao de premissa: isso nao e implementacao da Intel. E do proprio `comfy_kitchen`. A Intel tem
INT8 no OpenVINO / Neural Compressor / AutoRound, mas o que viabiliza o plano nesta maquina sao
estes kernels.

**Armadilha a evitar:** o backend `eager` declara **as mesmas capabilities**, e o `triton` esta
`disabled: True`. `w4a8_int8_linear` resolveria sem erro caindo em eager. O conversor tem de
rodar a mesma preflight de `quant_w4a4.py` **para cada op que pretende usar**, nao so para
`convrot_w4a4_*`.

**Consequencia para o achado de hoje:** eu escrevi que "INT8 nao existe como ponto intermediario".
Correto para SVDQuant (`SVDQW4A4Linear` so aceita int4 e nvfp4) e **errado para este projeto** -
existe em ConvRot. Se o dano visivel vier mesmo do A4, `w4a8_int8_linear` mantem a economia de
peso (o disco vem do peso) e devolve a ativacao para 8 bits. E o ponto intermediario que eu
procurei e nao achei.


## 2026-08-18, parte 7 - W4A8 medido: erra ~3x menos que W4A4

`tools/check_w4a8.py` (novo). Mesmo peso, mesma entrada, so muda a precisao da ativacao. As
quatro ops confirmadas resolvendo em `comfy_kitchen.backends.cuda` antes de medir - o script
recusa reportar se qualquer uma cair em eager.

| forma | entrada | W4A8 | W4A4 | W4A8 melhor |
|---|---|---|---|---|
| 3840x3840 | gaussiana | 0,0739 | 0,2230 | 3,02x |
| 3840x3840 | pos-ativacao | 0,0739 | 0,2234 | 3,02x |
| 3840x10240 | gaussiana | 0,0738 | 0,2370 | 3,21x |
| **3840x10240** | **pos-ativacao** | **0,0737** | **0,2464** | **3,34x** |
| 11520x3840 | gaussiana | 0,0738 | 0,2231 | 3,02x |
| 11520x3840 | pos-ativacao | 0,0738 | 0,2233 | 3,03x |

Medicao direta do que a observacao da caneta sugeriu, agora **sem o ruido de amostragem** que
afogou a tentativa anterior: o peso e identico nos dois lados, so a ativacao muda.

**O criterio para o conversor por camada esta neste quadro.** W4A8 fica cravado em 0,0737-0,0739
nas tres formas e nas duas distribuicoes - insensivel. W4A4 varia 0,2230 a 0,2464 e **piora
exatamente na 3840x10240 com entrada pos-ativacao**, que e a forma e a entrada da
`feed_forward.net.2`, a pior camada em todo diagnostico deste projeto desde o inicio. Ativacao
em 4 bits e sensivel a distribuicao; em 8 bits nao e. E isso que uma passada de calibracao
detectaria, e e por isso que promover so algumas camadas faz sentido.

**Limites do que foi medido:** peso sintetico gaussiano, nao peso de modelo real - mede o kernel,
nao o modelo. O piso de 0,0739 do W4A8 e o custo do **peso** em 4 bits com codebook, nao da
ativacao; nao desce sem subir o peso. E a distribuicao `post_activation` e uma imitacao de SwiGLU
com outliers plantados, com faixa dinamica realista mas distribuicao inventada.

Duas armadilhas de assinatura, achadas na marra:

- `quantize_convrot_w4a4_weight(weight, convrot_groupsize=256, quant_group_size=64)` - passar
  `64` primeiro levanta `int4 MMA kernel requires quant_group_size 64`, mensagem que aponta para
  o valor certo no lugar errado.
- `quantize_w4a8_int8_weight` devolve **(qdata, s_rel, s_channel, correction, codebook)**. Ler o
  indice 3 como codebook falha com `correction must have shape (240, 3840), got (16,)` - o 16
  sendo o codebook de 16 entradas, que foi o que denunciou a ordem.


## 2026-08-18, parte 8 - ablacao com 3 seeds, fechando a historia da caneta

18 execucoes (3 modelos x 3 seeds x 8 e 24 steps), Beyond_Reality Z-Image v2.

| steps | modelo | relL2 vs original por seed | media | spread |
|---|---|---|---|---|
| 8 | recuperado | 0,4019 / 0,5611 / 0,4119 | 0,4583 | 0,159 |
| 8 | **int4** | 0,4399 / 0,6690 / 0,5368 | **0,5485** | 0,229 |
| 24 | recuperado | 0,4269 / 0,5957 / 0,4220 | 0,4815 | 0,174 |
| 24 | **int4** | 0,4494 / 0,6962 / 0,5576 | **0,5677** | 0,247 |

**O int4 e pior que o recuperado nas 6 comparacoes pareadas, sem excecao.** Como o recuperado tem
os MESMOS pesos e ativacoes BF16, a diferenca e a quantizacao de ativacao - a mesma conclusao que
a caneta sugeriu.

Dimensionando: gap ~0,09, ruido entre execucoes do mesmo int4 na mesma seed ~0,035 (medido:
0,4046 e 0,4399). Gap ~2,5x o ruido, e 6/6 na mesma direcao daria 1,6% por acaso puro.
**Sugestivo, nao conclusivo.** O spread entre seeds (0,16-0,25) e o que afogou a tentativa
anterior, que comparava seeds diferentes sem perceber.

O que fecha o caso continua sendo `check_w4a8.py`, onde peso e entrada sao identicos e nao ha
amostragem: W4A8 erra 3x menos. Esta parte 8 e **confirmacao independente**, nao prova.


## 2026-08-18, parte 9 - o conversor de precisao mista existe e roda

Tres ferramentas novas, na ordem em que se usam:

```
tools/to_native.py            renomeia diffusers -> nomes nativos do ComfyUI (BF16, sem perda)
tools/calibrate_activations.py roda o modelo de verdade e guarda as ativacoes reais por camada
tools/quant_mixed.py           mede os kernels reais nessas ativacoes e escolhe 4 ou 8 bits por camada
```

### A descoberta que obrigou o passo 1

Um checkpoint Z-Image publicado esta em naming **diffusers** (`attention.to_q/to_k/to_v`), e o
ComfyUI funde os tres num `attention.qkv` no load (`model_detection.py:1498`). A matematica
sobrevive - quantizacao e por linha e concatenar linhas e seguro - mas o **mecanismo nao**:
`sd_map` so mapeia `.weight`. As chaves `weight_scale` / `weight_s_rel` / `comfy_quant` caem no
ramo identidade (`if k not in sd_map: sd_map[k] = k`) e ficam com o nome antigo, longe do modulo
que precisa delas. **O arquivo carrega e a camada fica sem escala.** Falha silenciosa.

`to_native.py` resolve renomeando antes. O plano nao e escrito a mao: vem de
`comfy.utils.z_image_to_diffusers` invertido, e e conferido contra o proprio
`convert_diffusers_mmdit` do ComfyUI rodado em tensores `meta` (custo zero de memoria). 521 chaves
-> 453, batendo exatamente. **Latente bit-identico** ao original no mesmo seed - o remap nao muda
nada, o que separa "bug de remap" de "bug de quantizacao" no resto da investigacao.

Efeito colateral: em naming nativo o Z-Image tem **170** Linears quantizaveis (34 blocos x 5), nao
238. O perfil anterior, escrito em naming diffusers, casava so 102.

### Bug encontrado e corrigido: fp16 no reservatorio

A calibracao guardava as amostras em `float16`. A entrada de `layers.0.feed_forward.w2` chega a
**344064**, que estoura o maximo do fp16 (65504) e vira `inf`. Os tres erros dessa camada viraram
`nan`, `nan > limiar` e falso, e **a camada com a maior ativacao do modelo recebia o formato mais
barato**. Corrigido para `bfloat16` (mesmos 2 bytes, alcance do fp32). Depois da correcao ela
aparece como a pior camada do modelo (W4A4 0,4719) e e promovida.

`quant_mixed.py` agora tambem recusa medicao nao-finita em vez de compara-la.

### O crest factor NAO prediz o erro

Escrevi na propria ferramenta que crest factor era "a estatistica a que o caminho de ativacao do
ConvRot e sensivel". Medido em 170 camadas reais:

| correlacao com err_w4a4 | valor |
|---|---|
| Pearson (crest p99) | **+0,068** |
| Spearman (crest p99) | **+0,096** |
| Spearman (err_w4a8) | **+0,978** |

O mecanismo e real - uma escala por token, um canal outlier define a escala do vetor todo - mas
nao chega na saida, porque o canal que estoura a escala costuma ser tambem o que domina o
resultado. `feed_forward.w2` tem o maior crest do modelo (p50 97,7, contra o maximo teorico
`sqrt(10240) = 101,2`) e `attention.out` tem o menor (p50 15,9) - e `attention.out` contem a
segunda pior camada. Heuristica descartada; a decisao e medida contra o kernel.

### Erro por camada, ativacoes reais (170 camadas, Z-Image v2)

| formato | min | p50 | max |
|---|---|---|---|
| bf16 (piso) | 0,0014 | ~0,0018 | 0,0022 |
| W4A4 | 0,0176 | 0,1254 | 0,4719 |
| W4A8 | 0,0077 | 0,0393 | 0,1676 |

Razao W4A4/W4A8: p50 **3,18x**, faixa 2,29x-4,55x. Consistente com `check_w4a8.py`.

**Ativacao real erra bem menos que gaussiana sintetica**: o `check_w4a8.py` dava W4A4 0,223-0,246
e W4A8 0,0737 fixo. Nas ativacoes reais a mediana do W4A4 e 0,125. O benchmark sintetico
**superestimava o dano em ~2x**.

Por sub-camada (p50 do W4A4): `feed_forward.w3` 0,177 > `feed_forward.w2` 0,133 >
`attention.qkv` 0,124 > `attention.out` 0,107 > `feed_forward.w1` 0,103.

### Resultado, `--promote-error 0.15`

115 camadas em `convrot_w4a4`, 55 em `asym_w4a8_int8`, 0 em BF16. Pior erro W4A4 que sobra no
modelo: 0,1486.

Carregado pelo loader normal do ComfyUI: **115 + 55 modulos com o `quant_format` certo e
`_full_precision_mm` falso em todos** - nenhum caiu em math dequantizada. Os 170 pesos sao
`QuantizedTensor`. Precisao mista num arquivo so e comportamento nativo do formato
(`ops.py:1142` despacha pelo JSON de cada camada), nao truque.

| | BF16 nativo | W4A4 puro | misto |
|---|---|---|---|
| disco | 11,46 GiB | 3,06 GiB (**3,74x mais leve**) | 3,18 GiB (**3,61x mais leve**) |
| VRAM (torch alloc) | 12,21 GiB | 3,74 GiB | 3,86 GiB (**3,16x menos**) |
| s/step | 1,290 | 0,583 (**2,21x menos**) | 0,598 (**2,16x menos**) |
| relL2 vs BF16 | 0 | 0,5651 | **0,4592** |
| cosseno vs BF16 | 1 | 0,8583 | **0,8955** |

Promover 55 de 170 camadas custou **0,12 GiB (3,9%)** e 2,5% de velocidade, porque o peso continua
4-bit nos dois formatos - muda a precisao da **ativacao** e a granularidade da escala.

### O kernel ConvRot do comfy_kitchen E deterministico

Duas execucoes do mesmo arquivo, mesma seed, processos separados: **latente bit-identico**. Isso
contrasta com a parte 5, onde o caminho INT4 do Nunchaku variava 0,29794 entre processos. Logo os
numeros desta parte 9 sao **medidas**, nao amostras - ao contrario dos da parte 5.

### O que estes numeros NAO dizem

`relL2` no latente mede divergencia de trajetoria, nao qualidade. As tres imagens
(`bench/beyond-reality-zimage-v2_native.png`, `bench/mixed.png`, `bench/w4a4.png`) sao coerentes,
sem banding, sem colapso de anatomia, maos integras nas tres. **Nao da para afirmar pela imagem
que o misto e melhor que o W4A4 puro** - precisaria de muitas amostras e julgamento humano. O que
esta provado e a reducao de 11,7% na divergencia e a reducao de 3,18x no erro por camada medido
contra o kernel.

Nenhuma das tres acerta as contagens do prompt (3 chaves de fenda, 4 engrenagens). Isso e o modelo
a 8 passos, nao a quantizacao - o BF16 erra igual.


## 2026-08-18, parte 10 - onde o int4 cruza: M decide tudo

Ferramenta: `tools/m_crossover.py`. Varre M de 1 a 8192 num Linear so, quatro caminhos, mesmo
peso. Motivada por um projeto irmao (Qwen quantizado em vLLM) onde ConvRot W4A4 fez 28,7 tok/s
contra 44,6 do AWQ W4A16/Marlin e foi dado como derrota do formato.

### Nao era derrota do formato. Era o M

Peso 3840x3840, ms, RTX 3090 (duas execucoes, mesmo cruzamento, erros iguais na 4a casa):

> **Substituido pela parte 11 (2026-08-19).** As razoes desta tabela vem de execucoes de disparo
> unico, que depois foram medidas com +/-20% de erro entre execucoes; o cruzamento se sustentou, os
> numeros de duas casas nao. Citar os da parte 11, com o intervalo junto.

| M | bf16 | w4a4 | veredito |
|---|---|---|---|
| 1 | 0,057 | 0,122 | 1,48x mais lento |
| 8 | 0,059 | 0,155 | 2,08x mais lento |
| 64 | 0,066 | 0,122 | 1,84x mais lento |
| 128 | 0,097 | 0,110 | 1,13x mais lento |
| **256** | 0,185 | 0,130 | **1,43x mais rapido** |
| 1024 | 0,560 | 0,175 | 3,19x |
| **5856** | 2,640 | 0,569 | **4,64x** |

**Cruzamento em M ~ 128-256.** Em 10240x3840 o cruzamento cai entre 64 e 128 e o ganho em
M=5856 e 5,03x.

Decode de LLM vive em M=1..8, onde o W4A4 e 1,8-1,9x mais lento - o kernel puro **prediz** o
1,55x que o projeto irmao mediu em producao. Difusao nunca tem fase de decode: o Z-Image entrega
M=5856 a todo Linear, 240 chamadas por geracao de 8 passos. **O mesmo formato, na mesma placa,
inverte de 1,9x mais lento para 4,6x mais rapido so pelo M.**

Contraste dos denominadores, medido nos dois lados:

```
LLM prefill    GEMM 94%   attention  6%
difusao        GEMM 67%   attention 33%   (parte 9)
```

### `linear_dtype="int8"` NAO e o tier W4A8

Confusao real que existia no projeto irmao, resolvida aqui. `convrot_w4a4_linear` aceita
`linear_dtype` em `{"int4", "int8"}` - o `int8` e ativacao 8 bits **no layout de peso do W4A4**.
O tier `w4a8_int8_linear` tem layout proprio (`qdata + s_rel + s_channel` + codebook Lloyd-Max).
Sao coisas diferentes, e a diferenca e grande:

| caminho | erro | vs w4a4 | ms em M=5856 |
|---|---|---|---|
| `w4a4` | 0,2231 | - | 0,569 |
| `w4a4` com `linear_dtype="int8"` | 0,1571 | 1,42x melhor | 1,321 |
| `w4a8_int8_linear` | **0,0737** | **3,03x melhor** | **1,121** |

O tier real e 2,13x mais preciso **e** mais rapido que o knob, em M grande. Em M=1 inverte: o knob
faz 0,085 ms contra 0,121 do tier. Decode quer o knob, prefill quer o tier.

### Ressalva que nao pode ser omitida ao citar isto

O baseline `bf16` aqui e `F.linear` sobre peso bf16: le 28 MiB onde um kernel weight-only-int4
(Marlin) leria 7. Isso o penaliza **exatamente** no regime limitado por banda. Mas acima de M~128
os dois viram compute-bound e o Marlin desempacota para bf16, fazendo os mesmos FLOP - a vantagem
dele so existe abaixo do cruzamento. Estimativa (nao medicao): contra Marlin o cruzamento fica em
M ~ 270 em vez de ~200. **Isto nao e um benchmark de Marlin e nao pode ser citado como um.**

Segunda ressalva: kernel puro nao tem model runner nem captura de CUDA graph. Se o tier W4A8
ganha aqui, isso diz que vale consertar a captura no vLLM - nao que ja funcione la.

### Bug meu no caminho

O guard que recusa rodar com a placa ocupada usava `torch.cuda.mem_get_info()`, que neste host
WDDM reporta **1292 MiB onde o NVML reporta 548** para o mesmo instante - 744 MiB de discordancia,
mais o custo de ~270 MiB de criar o contexto. O guard recusava numa placa ociosa. Passou a ler via
NVML **antes** do torch tocar em CUDA, com teto de 2 GiB.

### Correcao da parte 10: o cruzamento era do relogio, nao do kernel

`tools/w4a4_breakdown.py` perfila os kernels CUDA dentro de uma chamada e separa tempo de GPU de
tempo de host. Duas hipoteses caem, uma minha e uma de fora.

**Cai a hipotese "a administracao custa mais que a multiplicacao" em M=1.** Perfilado:

```
M=1  w4a4   int4_linear_kernel      32,2 us  88,5%   <- o GEMM
            quantize_int4_rowwise    4,2 us  11,5%   <- a quantizacao de ativacao
```

A quantizacao de ativacao e 11,5%, nao a maioria.

**Cai a minha, que era pior.** Relogio contra GPU, peso 3840x3840:

| M | caminho | relogio us | GPU us | host us | host % |
|---|---|---|---|---|---|
| 1 | bf16 | 89,7 | 45,2 | 44,6 | 49,6% |
| 1 | **w4a4** | 196,5 | **35,3** | **161,2** | **82,1%** |
| 1 | w4a8 | 136,7 | 62,2 | 74,5 | 54,5% |
| 128 | bf16 | 93,0 | 69,0 | 24,0 | 25,8% |
| 128 | w4a4 | 122,1 | 27,9 | 94,3 | 77,2% |
| 5856 | bf16 | 2470,2 | 2457,2 | 13,0 | 0,5% |
| 5856 | w4a4 | 529,3 | 472,4 | 56,9 | 10,7% |

**Em tempo de GPU o W4A4 ganha do bf16 em TODO M medido** - 35,3 contra 45,2 em M=1, 27,9 contra
69,0 em M=128, 472,4 contra 2457,2 em M=5856. **Nao existe cruzamento no kernel.** O cruzamento em
M~128-256 registrado acima e do relogio, e o relogio carrega ~160 us de despacho Python por
chamada, fixo, que so deixa de importar quando o trabalho de GPU cresce o bastante para afoga-lo.

Consequencia para quem serve LLM: custo de host e exatamente o que a captura de CUDA graph
elimina. Um decode em M=1 sem CUDA graph mede despacho, nao kernel. A pergunta certa no projeto
irmao nao e "o kernel int4 serve em M=1" e sim "o decode roda com graph ligado".

> **CONTESTADO e depois RESOLVIDO no mesmo dia, 2026-08-29 - e o percurso vale mais que o
> desfecho.** Esta linha afirmava, como fato, que o projeto irmao abandonou o tier
> `w4a8_int8_linear` por quebrar a captura de CUDA graph. Nao era medicao deste lado: veio de
> relato relayado em 2026-08-19. Em 2026-08-29 **o projeto irmao negou** - "nunca testei captura
> nesse tier". Marquei contestado em vez de apagar, porque o `state.md` dele listava o tier dentro
> do plugin de vLLM e as duas versoes nao fechavam.
>
> Ele foi buscar o log e **a evidencia contradiz a negativa dele**. Lido aqui direto da fonte,
> `P:\PROJETOS\W4A4_LLM\.claude\autopilot\state.md:267-290`, secao *"W4A8 / CHECKPOINT MIXED -
> REPROVADO POR MEDICAO"*:
>
> ```
> RuntimeError: info.status != cudaStreamCaptureStatusInvalidated
>   INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:2213
> torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
> ```
>
> **Duas correcoes no enunciado original, e as duas apertam em vez de afrouxar:**
>
> 1. **A captura quebra no caminho do model runner novo** (o que o DSpark exige). *"Sozinho ele
>    captura bem"* - o que bate exatamente com o que foi medido aqui na parte 11. O bloqueio era a
>    integracao, nao o op. Entao nem arqueologia e: esta medido dos dois lados.
> 2. **A decisao nao foi so a captura.** `W4A4 puro 28,73 sem draft e 63,10 com DSpark k=7`; o
>    `mixed (W4A8 em 0/63)` da `27,43` (-4,5%) e **nao sobe** com o draft. A frase do arquivo dele e
>    *"custo sem contrapartida demonstrada"* - nao ha medida de qualidade mostrando que 0/63 em W4A4
>    degradem algo. Captura foi a causa proximal; custo-beneficio foi a decisao.
>
> Licao de metodo, e e a que transfere: **uma sessao nao e autoridade sobre a propria historia
> quando a historia esta em outra sessao.** A negativa dele era verdadeira da sessao em que ele
> estava, e falsa do historico. O arquivo decidiu. Nao fechar como resolvido so porque a outra
> parte afirmou foi o que fez a evidencia aparecer.

Ressalva: os 161 us sao do wrapper Python do `comfy_kitchen` nesta stack; o numero absoluto nao
transfere para outro caminho de chamada. O que transfere e a forma - em M=1 o trabalho de GPU e
minusculo e qualquer overhead de host o domina.

Tambem observado: o W4A4 troca de kernel com M. Em M=1 e 8 usa `int4_linear_kernel`, escrito a mao
em `convrot_w4a4.cu`; em M=128 e acima usa um GEMM CUTLASS (`cutlass::Kernel2<...integer_s...>`).

### Lock de GPU

`tools/gpu_lock.py` e `/c/Users/joaoz/w4a4/gpu_lock.sh` (sessao irma) usam o mesmo arquivo
`F:/GPU_BENCH.lock`, um com `open(path,"x")` e outro com `set -o noclobber` - ambos atomicos.
Interop testado nos dois sentidos: cada um recusa quando o outro segura, e cada um mostra o dono
gravado pelo outro. Sem isso, dois monitores "esperando a GPU liberar" disparam no mesmo segundo.

### Os microssegundos que faltavam: sao `cudaLaunchKernel`

Perfil de CPU (`ProfilerActivity.CPU + CUDA`), M=1, peso 3840x3840, self time por chamada:

| w4a4 | us | contagem |
|---|---|---|
| **cudaLaunchKernel** | **59,5** | **x2** |
| aten::empty | 11,3 | x3 |
| aten::reshape | 8,8 | x7 |
| aten::view | 7,9 | x7 |
| cudaFuncSetAttribute | 1,6 | x1 |
| **total self CPU** | **109,0** | |

| bf16 | us | contagem |
|---|---|---|
| cudaLaunchKernel | 9,4 | x1 |
| aten::mm | 11,0 | x1 |
| **total self CPU** | **64,1** | |

Nao e pybind, nao e validacao de shape, nao e `contiguous`. E launch: **59,5 dos 109 us**. E nao e
so "dois launches em vez de um" - por launch da 29,75 us contra 9,40 do bf16, **3,2x mais caro**.
Mais 3 alocacoes, 14 operacoes de view/reshape e um `cudaFuncSetAttribute` a cada forward, que e
configuracao que normalmente se faz uma vez.

Tudo nessa lista e do tipo que a captura de CUDA graph pode eliminar. **Medido, nao prometido** -
a primeira versao deste paragrafo dizia "o graph elimina tudo", o que era afirmacao e nao medicao:

| caminho | eager us | sob CUDA graph | removido | captura |
|---|---|---|---|---|
| bf16 | 75,4 | 53,7 | 21,7 (29%) | OK, saida confere |
| **w4a4** | 137,3 | **46,6** | **90,7 (83%)** | OK, saida confere |
| w4a8 | 160,0 | 77,5 | 82,6 (52%) | OK, saida confere |

**Em eager o w4a4 e mais lento que o bf16 em M=1; sob graph ele e 1,15x mais rapido** (46,6 contra
53,7). A ordem inverte so por remover o despacho.

Nao remove tudo: sobram ~11 us de host por replay no w4a4 (46,6 de relogio contra 35,3 de GPU).
Entao "o graph apaga o overhead" e falso; "o graph apaga 83% dele nesta stack" e o que foi medido.

**Os dois ops do comfy_kitchen sao capture-safe** - `convrot_w4a4_linear` e `w4a8_int8_linear`
capturam e reproduzem com saida correta. Isso importa para o projeto irmao, que abandonou o tier
W4A8 tendo a quebra de captura como causa proximal - e o log dele, lido em 2026-08-29
(`state.md:284-285`), diz literalmente *"o `w4a8_int8_linear` nao e capture-safe no caminho de CUDA
graph do model runner novo. **Sozinho ele captura bem**"*. As duas medicoes, em duas maquinas e
dois runtimes, concordam: **o que quebrou la nao era o kernel.** Ver o bloco da parte 10 para o
percurso completo dessa afirmacao, que passou por contestada antes de fechar.

> **ERRADO, corrigido em 2026-08-19 (parte 12).** Isto foi medido em M pequeno e escrito como se
> valesse para todo M. `w4a8_int8_linear` **recusa a captura** acima de M x K ~ 21,8 milhoes de
> elementos de ativacao - em K=3840, a partir de M=5680. O w4a4 captura em todos os M testados.
> A frase "o que quebrou la nao pode ser o kernel" nao se sustenta: em batch de prefill o kernel
> quebra aqui tambem. Ver parte 12.

Ressalvas: captura de **um op** com tensores estaticos, nao de um modelo inteiro com batch
dinamico; e Windows/WDDM. Prova que o op e capture-safe, nao que a integracao de outro projeto o
seja.

**Ressalva de magnitude:** isto e Windows/WDDM, onde o launch atravessa o scheduler do SO e custa
caro - 9,4 us ate no caminho bf16 de um unico launch. Em Linux o custo por launch e bem menor, e o
projeto irmao roda em WSL. **Os 30 us por launch nao transferem.** O que transfere e a estrutura
(2 launches + 3 allocs + 14 metadata contra 1 launch) e a aritmetica: com 35 us de trabalho de GPU
em M=1, qualquer overhead de host dessa ordem vira a maioria do tempo.

**Marlin nao existe nesta stack** (`marlin`, `gptqmodel`, `auto_gptq`, `vllm`, `awq` todos
ausentes no interpretador embutido). A linha `Marlin W4A16` da tabela comparativa tem de sair do
lado que serve o LLM; nao da para produzi-la aqui.

**O lock funcionou em producao:** uma execucao minha foi recusada com `dono=diag-w4a8-gemv pid=1530`
enquanto a sessao irma media. Sem ele as duas mediriam contendidas.


## 2026-08-19, parte 11 - a contraprova de ordem passou, e o instrumento estava errado

Item 1 da fila de GPU do handoff: rodar `m_crossover` com M em ordem decrescente, para saber se o
cruzamento medido na tabela ascendente era propriedade dos kernels ou efeito de a placa esquentar
ao longo da varredura. Tres coisas sairam disso, e a segunda e maior que a primeira.

### 0. A ferramenta nunca tinha sido executada

Primeira invocacao morreu antes de tocar a GPU:

```
File "F:\COMFY_PORTABLE\tools\m_crossover.py", line 198, in <module>
    from _bench_guard import BenchGuard
ModuleNotFoundError: No module named '_bench_guard'
```

O interpretador embutido traz um `python313._pth`, que suprime a entrada usual do diretorio do
script, entao um modulo irmao em `tools/` nao importa sem `sys.path.insert`. `attn_dtype_ab.py`,
`check_w4a8.py` e `w4a4_breakdown.py` ja tinham a linha; `m_crossover.py` nao. O guard tinha sido
**escrito na passada de correcoes e nunca executado** - exatamente a distincao que o CLAUDE.md
manda declarar. Corrigido; a partir daqui tudo nesta secao e execucao.

### 1. Efeito de ordem: nao detectavel

Ascendente e descendente, mesma sessao, placa ociosa e travada, 3 repeticoes intercaladas por
ponto. Peso [10240, 3840]:

| M | ascendente | descendente |
|---|---|---|
| 64 | 1,39x mais lento | 1,20x mais lento |
| **128** | **1,10x mais rapido** | **1,17x mais rapido** |
| 512 | 2,66x | 2,65x |
| 1024 | 3,99x | 4,00x |
| 2048 | 4,96x | 5,27x |
| 5856 | 5,65x | 5,75x |
| 8192 | 5,75x | 5,76x |

Cruzamento no mesmo intervalo (64|128) nas duas direcoes; em [3840, 3840], 128|256 nas duas. As
duas curvas caem dentro do proprio min-max uma da outra em quase todo M. **A tabela ascendente nao
era artefato de aquecimento.** Contraprova passou.

### 2. Mas o numero de uma execucao so tem +/-20% de erro

O controle e que entregou isso. Antes de comparar ascendente com descendente rodei ascendente
**duas vezes**, e as duas discordaram mais entre si do que ascendente discorda de descendente:

| M, peso [3840, 3840] | asc #1 | desc | asc #2 |
|---|---|---|---|
| 2048 | 4,47x | 4,55x | **3,26x** |
| 256 | 1,46x | 1,39x | 1,19x |

E em [10240, 3840] o proprio cruzamento andou um degrau entre duas execucoes ascendentes: M=128
saiu 1,34x mais rapido numa e 1,05x **mais lento** na outra.

Com n=2 (uma ascendente, uma descendente) eu teria chamado isso de efeito de ordem e estaria
errado. E ruido entre execucoes. O que separa os dois e o controle, nao o par.

**Consertado na ferramenta, nao no texto:**

- `--repeats` (default 3) cronometra cada caminho varias vezes, **intercalado** - os quatro
  caminhos uma vez, depois de novo. Cronometrar um caminho ate o fim antes de comecar o proximo
  joga toda a deriva do intervalo em cima de quem estava rodando na hora; intercalado, a deriva
  atinge todos igual e cancela na razao.
- O veredito agora imprime o min-max **da razao**, que e a grandeza que sai citada daqui. Com 3
  repeticoes intercaladas o intervalo dentro da execucao fecha muito: `5856  w4a4 5,65x
  [5,54-5,75]`.
- `--reverse` e `--repeats` passam por argparse, para uma flag digitada errado nao produzir uma
  tabela ascendente rotulada como contraprova.

**Ressalva que fica:** o `[min-max]` impresso e a dispersao *dentro* de uma execucao. Entre
execucoes ainda e maior - M=5856 em [10240, 3840] deu 5,11 / 5,90 / 4,87 nas tres execucoes de
disparo unico e 5,65-5,75 nas duas com repeticao. Duas casas decimais continuam sendo mais
precisao do que este instrumento tem.

### 3. Numeros atuais, com a condicao colada

Placa ociosa, lock tomado, 3 repeticoes intercaladas, RTX 3090, torch 2.13.0+cu130:

| | [3840, 3840] | [10240, 3840] |
|---|---|---|
| cruzamento | entre M=128 e M=256 | entre M=64 e M=128 |
| M=5856 | 4,89x [4,72-5,06] | 5,65x [5,54-5,75] |
| M=8192 | 5,10x [4,97-5,15] | 5,75x [5,68-5,76] |
| M=1 | 1,5-2,0x mais lento | 1,0-1,4x mais lento |

**Perto do cruzamento o vencedor nao e confiavel.** Em M=256 [3840, 3840] o w4a8 ganhou nas duas
execucoes com repeticao e o w4a4 tinha ganho nas de disparo unico; a diferenca esta dentro do
intervalo. Abaixo de M~512 vale ler "empate", nao o rotulo.

Os erros relativos sao identicos ate a 4a casa em todas as cinco execucoes (w4a4 0,2231, w4a4/a8
0,1574, w4a8 0,0737 em M=5856) - determinismo confirmado; a variacao e toda de tempo, nenhuma de
numerica.


## 2026-08-19, parte 12 - o W4A8 recusa CUDA graph acima de um tamanho, e o "83%" nao tinha instrumento

Continuacao da parte 11, mesma pergunta aplicada aos outros benches: se um numero de disparo unico
podia estar 40% errado, quais outros numeros deste repo estao?

### 0. O "83% do overhead sai com graph" nao tinha instrumento

Nenhum arquivo em `tools/` media captura de CUDA graph. O numero saiu de um script solto que nao
existe mais - e ja tinha viajado para o projeto irmao, que decide arquitetura em cima disso. Numero
sem instrumento nao pode ser reconferido quando a stack muda por baixo.

Agora tem: `tools/graph_capture_probe.py`. Captura, **compara a saida do replay com a do eager**
(um graph que reproduz lixo marcaria o melhor tempo da tabela), e repete 3 passadas com min-max.

### 1. Quanto o graph tira, medido

RTX 3090, peso [3840, 3840], 3 passadas de 300 iteracoes, placa ociosa e travada:

| M | caminho | eager us | replay us | tirado | % do wall |
|---|---|---|---|---|---|
| 1 | bf16 | 63,8 | 51,1 | 12,7 | 20,0% |
| 1 | **w4a4** | **124,8** | **40,4** | **84,4** | **67,6%** |
| 1 | w4a8 | 132,4 | 70,6 | 61,8 | 46,7% |
| 128 | w4a4 | 110,4 | 35,6 | 74,9 | 67,8% |
| 5856 | w4a4 | 597,5 | 560,5 | 37,0 | 6,2% |
| 5856 | bf16 | 2546,8 | 2522,2 | 24,6 | 1,0% |

O numero que interessa nao e a % do wall, e **o que sobra**. Contra os ~35 us de GPU que o
`w4a4_breakdown` mede em M=1, um replay de 40,4 us deixa **~5-6 us de host por chamada**. O mesmo
piso aparece nos tres caminhos (bf16 51,1 contra 44,3 de GPU; w4a8 70,6 contra 62,8). **O piso do
replay nesta maquina e ~6 us, seja qual for o kernel.**

Isso **nao** reproduz o "83%, ~11 us sobrando" que este log afirmava. Pelo instrumento de hoje sao
~93% do host e ~5,6 us sobrando. Nao da para reconciliar os dois: o instrumento antigo nao existe
mais. Fica o de hoje, que da para rodar de novo.

> **Adendo 2026-08-29 - o projeto irmao mediu a mesma razao em outra maquina, e ela caiu em cima do
> instrumento APOSENTADO.** Numeros dele, RELATO RELAYADO, nao medidos aqui: RTX 3080 Ti, WSL2,
> rodada declarada **suja** (desktop do dono a ~18%, pico 58%), peso de 8 MiB, M=1, os dois bracos
> sob graph -> `w4a8 78,1 / bf16 53,6 = 1,46x`. Ao lado dos dois daqui, mesma razao, mesmo M:
>
> | fonte | maquina | w4a8 replay | bf16 replay | razao |
> |---|---|---|---|---|
> | instrumento aposentado | 3090 / Windows | 77,5 | 53,7 | **1,44** |
> | `graph_capture_probe` (atual) | 3090 / Windows | 70,6 | 51,1 | **1,38** |
> | bench do irmao | 3080 Ti / WSL2 | 78,1 | 53,6 | **1,46** (sujo) |
>
> Ele chamou de "replicacao independente, 1,44 contra 1,46". A leitura honesta e outra: **a
> discordancia entre os dois instrumentos da MESMA maquina (1,44 contra 1,38) e maior que a
> distancia entre as duas maquinas.** O que a tabela sustenta e "a razao vive em 1,38-1,46", nao
> dois decimais. O interessante e que a medida dele bate com o instrumento que foi aposentado, e
> nao com o que ficou - o que reabre *por que* os dois discordam (iteracoes, warmup, ou a checagem
> de saida dentro da regiao cronometrada), pergunta que ninguem respondeu.
>
> **E a condicao que nao pode soltar da razao: tudo isso e M=1.** Em M=5856 a parte 10 mede
> `w4a8_int8_linear` em 1,121 ms contra ~2,55-2,64 ms do bf16 - **~2,3x mais rapido**, ordem
> invertida. "W4A8 fica ~1,45x atras mesmo capturado" so e verdade com "em decode" colado. Sem
> isso vira veredito de formato e contradiz a medicao de prefill deste mesmo repo.

### 2. O achado grande: `w4a8_int8_linear` recusa captura acima de um tamanho

```
M=5856 w4a8  capture failed: AcceleratorError: CUDA error: operation failed due to a
             previous error during capture   (cudaErrorStreamCaptureInvalidated)
```

Nao e contaminacao das capturas anteriores: processo limpo, op sozinho, M=5856 - falha igual.

**Nao e M, e M x K.** Bissecado, um processo por ponto:

| K | ultimo M que captura | primeiro M que falha | M x K no limite |
|---|---|---|---|
| 3840 | 5664 | 5680 | 21.749.760 -> 21.811.200 |
| 2560 | 8400 | 8600 | 21.504.000 -> 22.016.000 |
| 1024 | 20800 | 21600 | 21.299.200 -> 22.118.400 |

Tres K, um deles nao proporcional aos outros dois, e o limiar cai sempre na mesma faixa de
**M x K entre 21,75 e 21,81 milhoes de elementos de ativacao**. N nao entra: foi 3840 em todos.

O `w4a4` captura em todos os M testados, ate 5856 (M x K = 22,5 milhoes, acima do limite do w4a8).
O bf16 tambem. **E especifico do tier W4A8.**

**Nao e troca de algoritmo visivel de fora.** O tempo eager e liso atravessando o limiar - 1182 us
em M=5600, 1192 em 5664, 1154 em 5680 (o que falha), todos dentro do min-max um do outro. E o
dispatch em Python e identico: instrumentei os quatro pontos de entrada `_C.*` e nos dois lados do
limiar so `w4a8_codebook_linear_chunked` e chamado, retornando `True`. **A causa esta dentro do
`.pyd` e nao foi determinada daqui.**

### 3. O que isso significa para os dois projetos

**Aqui:** com M=5856, um layer W4A8 so passa da captura se K < 21,78e6 / 5856 = **3719**. Layers com
K=3840 nao capturam; com K=2560, capturam. Se algum dia se tentar CUDA graph no Z-Image, o
checkpoint misto tem layers dos dois tipos e vai quebrar em alguns e nao em outros.

**No projeto irmao:** eles abandonaram o tier W4A8 por quebrar a captura no model runner do vLLM, e
este log dizia que "o que quebrou la nao pode ser o kernel, porque aqui ele captura". **Isso estava
errado** - estava medido em M pequeno e escrito como se valesse para todo M. Em decode (M=1..8) o
kernel captura; em prefill com batch grande, M x K passa de 21,8 milhoes com folga e o kernel
recusa aqui tambem. A conclusao deles pode estar certa. **Vale mandar o limiar para la.**

### 4. Corrigido junto

- `w4a4_breakdown.py` ainda tinha o proprio teste de ocupacao inline, e ele falhava **aberto**:
  NVML indisponivel imprimia aviso e perfilava assim mesmo. Era o unico que faltava migrar para o
  `_bench_guard`, que falha fechado. Migrado.
- O mesmo arquivo agora repete a passada de wall 3 vezes e imprime o min-max. Os rabos sao longos
  em M pequeno: `[112-252]` us em M=8, `[108-190]` em M=128. Uma passada so podia cair em qualquer
  ponto disso - foi assim que o "host 161 us" aconteceu.
- Host por chamada, remedido com 3 passadas: **w4a4 em M=1 gasta 81,4 us de host contra 34,8 us de
  GPU** (wall 116,2, spread [116-116]). O bf16 gasta 17,1. A penalidade de M pequeno do w4a4 e
  dispatch, nao aritmetica - o kernel dele e *mais rapido* que o do bf16 em M=1.
- Uma captura que falha **envenena o contexto CUDA**: toda chamada seguinte no mesmo processo
  levanta o mesmo erro cascateado. Uma varredura de 5600,5664,5680,5700,5856 morreu dentro da linha
  do 5700. O probe agora para na primeira falha e diz por que, em vez de imprimir cascata como se
  fosse medida.


## 2026-08-19, parte 13 - quem transpoe, e o que a LoRA faz: os dois medidos, os dois negativos

Itens 2 e 4 da fila de GPU. Ferramenta nova: `tools/dispatch_census.py`, que embrulha os handlers
de layout registrados e conta, numa geracao real, qual ramo cada Linear quantizado tomou.

### 1. Quem transpoe: o proprio ComfyUI, 680 vezes, e nao e o ramo perigoso

O achado da auditoria (reportado por dois auditores independentes) e
`comfy_kitchen/tensor/convrot_w4a4.py:237`:

```python
if weight._params.transposed:
    return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
```

Peso com a flag ligada nao chega ao kernel: dequantiza e multiplica em BF16. O que nunca se
estabeleceu por leitura foi **se alguem neste checkout transpoe**. Um auditor deu media por isso,
o outro deu alta olhando so o mecanismo.

Contado numa geracao real do `zimage-v2-mixed`, 4 passos, 512px:

```
w4a4.mm                    460      w4a4.t:flips False -> True   460
w4a4.mm:native kernel      460      w4a8.t:flips False -> True   220
w4a8.mm                    220
w4a8.mm:native kernel      220
kernel nativo: 680    dequantizado para BF16: 0
```

**Alguem transpoe: 680 vezes.** Mas nao cai onde o achado temia. O ComfyUI despacha o Linear como
`aten.t` seguido de `aten.mm`, nunca como `aten.linear` - e no handler de `mm` a flag `transposed`
e o estado **exigido**: `_resolve_convrot_w4a4_rhs` levanta `RuntimeError` se ela estiver falsa, e
com ela ligada roda o kernel. O ramo que dequantiza precisa de `linear` sobre um peso ja
transposto, combinacao que nao ocorre nesse caminho.

Cobertura completa, nao amostra: 115 layers `convrot_w4a4` x 4 passos = 460, 55 layers
`asym_w4a8_int8` x 4 = 220, total 680 = os 170 layers em todos os passos. Nenhum ficou de fora.

**Ressalva:** vale para este modelo, este sampler, esta resolucao. `torch.compile` e qualquer node
que chame `linear()` num peso ja transposto continuam sem medicao.

**A primeira versao da ferramenta errou o rotulo** e marcou os 460 `mm` como "TRANSPOSED
(dequantized to BF16)". So peguei porque o resumo se contradizia com a tabela. `transposed` nao
significa a mesma coisa nos tres handlers, e assumir que significa faz a ferramenta rotular o
caminho correto como o perigoso. Corrigido com a tabela `DEQUANTIZES_WHEN_TRANSPOSED`, derivada
linha a linha do fonte.

### 2. LoRA sobre quantizado: nao dequantiza

Hipotese da auditoria em `comfy/ops.py:1377`: o caminho quantizado exige
`len(self.weight_function) == 0`, e uma LoRA popula `weight_function`, logo toda camada com LoRA
cairia para BF16 em silencio.

Medido com `char_Liria_zimage.safetensors` em forca 1,0, **150 das chaves aplicadas caindo em
layers quantizados**: 680 despachos, todos no kernel nativo, zero dequantizados - identico a
execucao sem LoRA. Camada que tivesse caido para o fallback **sumiria** da contagem, nao apareceria
como dequantizada, entao a contagem e o instrumento certo para esta pergunta.

**O controle importa mais que o resultado.** Contagem igual com LoRA tem duas causas possiveis: ou
a LoRA rodou e o kernel tambem, ou a LoRA foi engolida em silencio. A contagem nao distingue. O
latente distingue:

```
sem LoRA   norm 747,060303   mean -0,436329   first +0,066680
com LoRA   norm 728,985107   mean -0,471541   first +0,047811
```

A LoRA esta em efeito. O kernel continua nativo. **Hipotese refutada neste caminho.**

O WARN do preflight foi reescrito: em vez de repetir a hipotese, agora carrega a medicao e estreita
o aviso para o que continua desconhecido - se aplicar delta de LoRA sobre peso ja de 4 bits custa
**qualidade**. Isso nao foi medido.

### 3. Dois guards que nunca podiam disparar

Achados de tabela, nao de leitura - os dois apareceram porque uma ferramenta se recusou a rodar
quando devia rodar, ou passou quando nao devia.

**`calibrate_activations.py`** recusava checkpoint ja quantizado testando
`named_buffers()` por sufixo `comfy_quant`. Medido no `zimage-v2-mixed`:

```
named_buffers    com comfy_quant:   0
named_parameters com comfy_quant:   0
state_dict       com comfy_quant: 170
modulos cujo .weight e QuantizedTensor: 170
```

O marcador nao e buffer registrado. **O guard nunca protegeu nada** - so nunca tinha recebido um
arquivo quantizado para deixar passar. Trocado para `isinstance(module.weight, QuantizedTensor)`.

**`test_checks.py`** do preflight calculava `ROOT = parents[3]`, aritmetica correta para o stub
dentro de `ComfyUI/custom_nodes/`, mas o pacote mora um nivel acima. Resolvia para `F:/`, e
`MODELS` apontava para `F:/ComfyUI/models/diffusion_models`, inexistente. O teste chamado "contra
os checkpoints reais desta maquina" pulava todos e imprimia **PASS**, com `(0 real checkpoint(s)
exercised)` numa linha que ninguem le. Corrigido para `parents[2]`, e agora **falha** se o
diretorio nao existir ou se nenhum checkpoint for exercitado - teste que nao testa nada tem de
quebrar, nao passar. Passou a exercitar 4.


## 2026-08-19, parte 14 - o widget de dtype num checkpoint quantizado: medido, e o traco errou o sintoma

Item 1 da fila de GPU, o unico candidato a PR upstream. O traco de leitura dizia: `comfy/sd.py`
sabe que o arquivo e quantizado - testa `model_config.quant_config is not None` duas vezes - mas
protege so metade da decisao.

```python
if model_config.quant_config is not None:
    weight_dtype = None                    # 2296: ignora o dtype farejado do arquivo
if dtype is None:
    unet_dtype = model_management.unet_dtype(...)
else:
    unet_dtype = dtype                     # 2303: o widget passa, sem consultar quant_config
if model_config.quant_config is not None:
    manual_cast_dtype = unet_manual_cast(None, ...)   # 2306: protegido
else:
    manual_cast_dtype = unet_manual_cast(unet_dtype, ...)
```

### O que a execucao mostrou

`zimage-v2-mixed`, 4 passos, 512px, mesma seed, so mudando o widget do `UNETLoader`:

| widget | unet_dtype | tensores nao quantizados | norm do latente |
|---|---|---|---|
| default | bfloat16 | bf16 x283 | **747,06** |
| fp8_e4m3fn | float8_e4m3fn | bf16 x76, **fp8_e4m3fn x207** | **713,98** |
| fp8_e5m2 | float8_e5m2 | bf16 x76, **fp8_e5m2 x207** | **828,08** |

`manual_cast` fica `bfloat16` nos tres - a protecao da linha 2306 funciona. E o dispatch fica
**680 de 680 no kernel nativo** nos tres: o widget nao consegue tocar os 170 layers quantizados,
que ja sao de 4 bits.

**O mecanismo do traco estava certo. O sintoma que eu imaginei estava errado.** Nao trava, nao
produz lixo, nao cai para eager. O que ele faz e converter para fp8 os **207 tensores que o perfil
deixou de proposito em alta precisao** - normas, embeddings, modulacao - e a saida muda.

Escala, medida no mesmo latente e na mesma seed: uma LoRA em forca 1,0 move de 747,06 para 728,99.
O `fp8_e5m2` move para 828,08 - **cerca de quatro vezes mais longe que aplicar uma LoRA inteira**.
Sem erro, sem aviso, e sem perda de velocidade que denuncie.

O silencio e o que faz disso ERROR e nao WARN no preflight. Quem marca o widget ve um modelo que
carrega, roda na velocidade cheia, e devolve outra imagem.

### Sobre o PR

O `git log -S` nao decide a intencao: neste checkout o bloco inteiro aparece como adicionado num
commit de atualizacao de templates (`e8e8fee2`), entao nao da para dizer daqui se a assimetria foi
deliberada ou passou batida. **O que da para afirmar** e que existem dois guards adjacentes
neutralizando dtype quando `quant_config` esta setado, e o caminho do dtype explicito escapa dos
dois - e que a consequencia agora esta medida, com repro de uma linha.

O mesmo formato aparece em duas funcoes do arquivo (`unet_dtype = model_options.get("dtype", ...)`
na outra), entao um PR mexeria nos dois pontos.

Nao publiquei nada. Nao ha remote configurado, e abrir issue upstream e acao para fora - fica para
decisao do usuario, com o texto pronto.

### Ferramenta

`tools/dispatch_census.py` ganhou `--weight-dtype`, montado exatamente como `nodes.py:993` monta,
e passou a imprimir o histograma de dtype dos tensores **nao** quantizados. Sem esse histograma o
achado seria "a saida mudou"; com ele e "207 tensores foram convertidos, e sao estes".


## 2026-08-19, parte 15 - o `.T` e a escala do nunchaku: os dois corretos, agora com teste que prova

Itens 5 e 6 da fila. Os dois viviam no mesmo lugar - `recover_weight` em `tools/svdq_to_bf16.py` -
e a suite de testes dizia, em toda execucao, que nao cobria nenhum dos dois:

> "a bug in the shipped recover_weight -- the dropped `.T`, for one -- passes this suite"

Isso porque os testes de CPU exercitam `_recover`, uma reimplementacao no proprio arquivo de teste,
e nao a funcao enviada, que constroi a camada por dentro do nunchaku e precisa de GPU.

### O `.T` esta certo

Camada real do `svdq-int4_r32-beyond-reality-zimage-v2`, comparando a matriz recuperada contra a
propria camada de onde ela saiu:

```
layer(x) vs F.linear(x, recovered)     rel 0,101
layer(x) vs F.linear(x, recovered.T)   rel 1,413
```

**A assercao tem de ser um contraste, nao um limiar.** O `SVDQW4A4Linear` quantiza a ativacao
tambem, entao `layer(x)` com x aleatorio carrega erro de ativacao W4A4 que `F.linear` nao tem - o
0,101 nao pode ser pequeno. Um limiar apertado o bastante para rejeitar a transposta rejeitaria
tambem a resposta certa. A razao entre as duas separa por mais de dez vezes.

### A escala esta certa

Erro relativo sozinho nao distingue ruido de quantizador de bug de escala: os dois sobem. O alpha
de minimos quadrados distingue. Contra o BF16 publicado do mesmo modelo:

| camada | alpha | rel@1 | rel@alpha |
|---|---|---|---|
| context_refiner.0.attention.to_out.0 | 0,992197 | 0,096566 | 0,096248 |
| context_refiner.1.attention.to_out.0 | 0,992261 | 0,096307 | 0,095994 |
| noise_refiner.0.attention.to_out.0 | 0,997170 | 0,056853 | 0,056782 |
| layers.0.attention.to_out.0 | 0,998100 | 0,047259 | 0,047220 |
| layers.17.attention.to_out.0 | 0,992848 | 0,092716 | 0,092438 |

Alpha entre 0,992 e 0,998, e corrigir por alpha melhora o residuo em menos de 0,4% dele mesmo.
**Nao ha fator de escala** - o que sobra e ruido de quantizacao int4, na magnitude esperada.

### E os testes tem dentes

Provado por mutacao, como os de `split_fused`. Derrubando o `.T` da funcao enviada:

```
CAUGHT  gpu_recover_weight_returns_the_layers_own_linear_map: rel 1,4128
CAUGHT  gpu_recovered_weight_has_no_systematic_scale_error: alpha=0,000486
```

Os dois pegam. Um teste que passa nao vale nada ate alguma coisa faze-lo falhar.

### Como os testes de GPU entraram na suite

Nomeados `gpu_*`, nao `test_*`, e o runner so os chama quando ha CUDA, nunchaku e os dois
checkpoints. Faltando qualquer um, imprime **SKIP** e uma linha `GAP` dizendo que o `.T` e a escala
ficaram sem verificacao naquela execucao. Nunca PASS - esse e exatamente o modo de falha que deixou
o teste do preflight exercitar zero arquivos e reportar PASS (parte 13).

E o arquivo tropecou no mesmo `python313._pth` do `m_crossover`: `from svdq_to_bf16 import ...`
morre com `ModuleNotFoundError` porque o interpretador embutido suprime o diretorio do script. A
suite ja carregava o modulo por `importlib`; agora os testes de GPU usam esse mesmo objeto.


## 2026-08-19, parte 16 - o fallback eager do W4A8 e alcancavel, silencioso e 6x mais lento

Item 3 da fila. A auditoria apontou `cuda/__init__.py:2213` e `:2261`: a cadeia do
`w4a8_int8_linear` tenta kernels em sequencia, cada um devolvendo um booleano visivel ao host, e a
cauda e `return eager_w4a8_int8_linear(...)` - matematica dequantizada com o nome de um op
quantizado, sem uma linha de log. O que faltava era **qual shape chega la**.

Ferramenta: `tools/w4a8_fallback_sweep.py`. Nao infere nada - embrulha os quatro pontos de entrada
`_C.*` e le os booleanos, entao uma linha marcada `EAGER` e uma linha em que o C++ disse nao.

### A regra

**`out_features % 8 != 0` cai no eager.** Medido, nao deduzido:

| N | verdito | N | verdito |
|---|---|---|---|
| 1000 | kernel | 1004 | EAGER |
| 1001 | EAGER | 1006 | EAGER |
| 1002 | EAGER | **1008** | kernel |
| 1003 | EAGER | 1016 | kernel |
| 3840 | kernel | 1020 | EAGER |
| 3841 | EAGER | 1024 | kernel |

Nao e paridade - 1002 e par e cai. Nao e multiplo de 128 - 1000 nao e e passa. E divisibilidade
por 8. `K` nao entra: 256, 512, 1024, 2560 e 3840 dao o mesmo resultado para o mesmo N.

### O que custa, e o que nao custa

**Nao custa correcao.** Erro relativo contra referencia float32: 0,0736 no eager, 0,0737 no
kernel. O fallback calcula a mesma coisa.

**Custa velocidade, e muito.** M=5856, K=3840:

```
N=1024  kernel   0,571 ms       N=3840  kernel   1,133 ms
N=1020  EAGER    2,869 ms       N=3841  EAGER    6,762 ms
        5,02x                           5,97x
```

**Cinco a seis vezes**, sem aviso nenhum. Um modelo com um unico layer de `out_features` nao
divisivel por 8 perde o formato naquele layer e nada no log diz isso.

### Relevancia pratica

Os layers do Z-Image sao todos divisiveis por 8 (3840, 11520), entao este projeto nao esbarra
nisso hoje. Quem esbarra: modelo podado, arquitetura com head count esquisito, ou qualquer coisa
com dimensao escolhida por outro criterio que nao alinhamento.

Segundo candidato a reporte upstream, e mais simples que o do widget de dtype: a cauda eager
deveria logar. Nao publiquei - mesma regra do item 1.

### Ressalva

Uma GPU (sm86), um build do comfy-kitchen. A recusa e o teste de capacidade da propria extensao,
entao a regra dos 8 pode ser outra em outra placa. O que transfere e o metodo: ler os booleanos,
nao inferir do tempo.


## 2026-08-19, parte 17 - a recusa de captura do W4A8 esta dentro do kernel chunked

Item 8, aberto na parte 12. A regra dos 8 da parte 16 deu o controle que faltava: o mesmo op, no
mesmo tamanho, tomando caminho interno diferente.

| shape | caminho interno | captura em M=5856 |
|---|---|---|
| N=3840, K=3840 | `w4a8_codebook_linear_chunked` | **falha** |
| N=3841, K=3840 | cauda eager | **captura** |

`N=3841` cai no eager por nao ser divisivel por 8, e captura sem problema exatamente no tamanho em
que `N=3840` recusa. **A falha esta dentro do `w4a8_codebook_linear_chunked`**, nao no Python em
volta, nao nos tensores, nao no tamanho da alocacao - senao o caminho eager, que aloca o mesmo
tanto, falharia junto.

Com isso o reporte fica completo o suficiente para ser util: ponto de entrada exato, limiar exato
(M x K entre 21,75 e 21,81 milhoes), tres K confirmando que a grandeza e M x K, e um controle
mostrando que o caminho alternativo do mesmo op captura. O **mecanismo** continua nao determinado -
esta dentro do `.pyd`, o erro original e engolido pela cascata e `CUDA_LAUNCH_BLOCKING=1` nao o
expoe.

Rascunho em `UPSTREAM_REPORT_w4a8_capture.md`. Nao publicado.

### Estado da fila de GPU

Os seis itens que estavam bloqueados esperando a placa foram medidos: partes 11 (ordem de M), 12
(captura), 13 (quem transpoe, LoRA), 14 (widget de dtype), 15 (`.T` e escala), 16 (fallback eager).
Quatro deram negativo - a hipotese nao se reproduziu - e dois deram positivo com repro. Os dois
positivos viraram rascunho de reporte upstream e nenhum foi publicado.


## 2026-08-19, parte 18 - o threshold varrido: o default nao se sustenta, e a metrica nao mede o que importa

`--promote-error 0.15` estava marcado neste repo como "escolhido, nao derivado". Varrido agora.
Ferramenta: `tools/quality_ladder.py`. Referencia BF16, mesmas seeds nos dois lados, 2 prompts x
6 seeds = 12 runs pareados por checkpoint.

### O que a metrica diz

| checkpoint | promovidos | delta vs w4a4 | vence | s/step | GiB |
|---|---|---|---|---|---|
| BF16 nativo | — | referencia | — | 0,910 | 11,46 |
| w4a4 puro | 0 | baseline | — | 0,348 | 3,06 |
| t0.20 | 10 | -0,0149 | 6/12 | 0,355 | 3,08 |
| **t0.15 (default)** | 55 | **-0,0002** | **3/12** | 0,403 | 3,18 |
| t0.10 | 119 | -0,1033 | **12/12** | 0,467 | 3,32 |
| t0.05 | 156 | -0,1218 | **12/12** | — | 3,40 |

**Pareado, nao media.** A divergencia varia mais entre seeds do que entre checkpoints - na primeira
execucao um checkpoint teve dispersao 0,2301 em tres seeds enquanto o vao inteiro entre o melhor e
o pior era 0,1065. Media de amostras nao pareadas convida a uma ordem que os dados nao sustentam;
os runs sao pareados por construcao (mesma seed, mesmo ruido, mesmo condicionamento), entao a
comparacao que sobrevive e run a run. A ferramenta imprime as duas, com a de media rotulada como a
errada.

Pela metrica, **0,15 nao compra nada**: 55 camadas promovidas, +16% de tempo, empate tecnico com o
W4A4 puro.

### O que as imagens dizem, e por que isso derruba a metrica

Este arquivo dizia, sobre a mistura: *"the images do not visibly separate, and saying otherwise
would be overclaiming"*. **Era falso.** Separam, e muito.

No W4A4 puro o bloco de pistoes do trompete vira um emaranhado de tubos - nas duas seeds olhadas,
nao numa. Com camadas promovidas, o mesmo prompt e a mesma seed dao um instrumento coerente. O
t0.05 sai limpo, com valvulas nitidas.

E aqui esta o problema: **o t0.15, que empata com o W4A4 puro na metrica, e visivelmente melhor que
ele.** A divergencia de latente mede quanto a composicao inteira andou, e ela anda de qualquer
jeito - enquadramento diferente, objetos em outro lugar. Esse movimento afoga o sinal estrutural,
que e o que quebra. **A metrica que este projeto usou para justificar o threshold nao mede o
defeito que o threshold existe para evitar.**

Mesma familia do erro do crest factor (parte 9): um numero que le como medida de qualidade e nao e.

### O que fica decidido e o que nao

**Decidido:** 0,15 nao se defende. Pela metrica perde para 0,10 e 0,05; pela imagem e melhor que o
W4A4 puro, entao o valor certo esta abaixo de 0,15, nao acima.

**Nao decidido:** onde exatamente. A ordem fina entre 0,15, 0,10 e 0,05 nao esta estabelecida
visualmente - olhei 2 seeds para w4a4 e t0.05, e 1 seed para os dois do meio. Um prompt so, um
modelo so. E "melhor" aqui e julgamento meu olhando imagem, nao medida.

**O que falta e uma metrica que veja o defeito.** Divergencia de latente nao serve. Candidatos que
nao foram testados: distancia perceptual contra a imagem BF16 na mesma seed, ou pontuar so a regiao
que quebra. Ate ter isso, a escolha do threshold e feita a olho, e o registro tem de dizer isso.


## 2026-08-19, parte 19 - o que NAO substitui a calibracao: estatistica de peso e ativacao sintetica

Pergunta do usuario: da para decidir a promocao por inferencia, sem a passada de calibracao? Duas
tentativas, as duas negativas, as duas medidas contra as mesmas 170 camadas do Z-Image que ja
tinham `err_w4a4` medido.

### Estatistica do peso: acaso

`tools/predict_promotion.py`. A parte 9 ja tinha matado o crest factor da **ativacao** (+0,10).
Estatistica do **peso** nunca tinha sido testada, e e de graca - nao precisa de amostragem nem de
hook. Nove candidatos, cada um uma forma de perguntar "quanto desta linha e decidido pelos seus
outliers":

| feature | spearman |
|---|---|
| group_waste_mean | +0,225 |
| row_crest_mean | +0,188 |
| row_crest_p99 | +0,149 |
| crest_p99_activation (controle) | +0,095 |
| top0.1pct_mass | +0,036 |
| std | -0,023 |
| kurtosis | +0,002 |

**A correlacao nao e o entregavel.** O conversor escolheria um conjunto de camadas, e o que importa
e se e o mesmo conjunto que a medicao escolhe. Com 119 de 170 promovidas, **um sorteio do mesmo
tamanho ja acerta 83,3 por acaso**. O melhor feature acerta 89. Ou seja: **+5,7 sobre o acaso.**
Nada. A ferramenta imprime a linha do acaso justamente porque `89/119` sozinho le como bom.

### Ativacao sintetica: pior que o acaso

`tools/synthetic_vs_real.py`. Melhor pergunta que a anterior: a medicao em si e barata, o caro e
**obter as ativacoes**. E se rodar os mesmos kernels em ruido gaussiano da largura certa?

| entrada | spearman vs real | overlap@0,10 | acaso |
|---|---|---|---|
| gaussiana | **-0,128** | 78/119 | 83,3 |
| gaussiana x lognormal por canal | +0,004 | 79/119 | 83,3 |

**Abaixo do acaso nas duas.** E o erro medio sai muito inflado: 0,2295 e 0,2442 no sintetico contra
**0,1290** no real. Isso confirma com numero a ressalva que o CLAUDE.md ja carregava sobre o smoke
de `verify_w4a4.py` ("RMSE ~0,25 em entrada aleatoria e sinal de vida, nao metrica de qualidade") -
ruido faz o W4A4 parecer 1,8x pior do que ele e.

**Conclusao:** o que torna uma camada dificil para o W4A4 e propriedade da ativacao real, e nao
esta no peso nem em ruido. A passada de calibracao nao sai por esse caminho.


## 2026-08-19, parte 20 - mas o perfil transfere entre checkpoints da mesma arquitetura

Hipotese do usuario, e melhor que a minha: o padrao nao esta numa formula, esta na **familia**.
Mede uma vez por arquitetura, reusa nos parentes.

Testado na forma mais forte disponivel aqui - tres checkpoints Z-Image, **recalibrados os tres nas
mesmas condicoes** (2 prompts, seeds 1234/5678, 8 passos, 1024px) para nao comparar contra uma
calibracao antiga com outras condicoes:

| par | spearman(err_w4a4) | overlap @0,10 | acaso |
|---|---|---|---|
| turbo vs de-turbo | +0,981 | 119/120 | 84,7 |
| turbo vs beyond-reality-v2 | **+0,997** | **120/120** | 84,7 |
| de-turbo vs beyond-reality-v2 | +0,986 | 119/120 | 84,7 |

**O perfil e da arquitetura, nao dos pesos.** Um finetune muda os pesos e nao muda qual camada
sofre com 4 bits.

### O que mudou no codigo

`quant_mixed.py` recusava reusar uma analise de outro checkpoint - default certo enquanto ninguem
sabia se transferia. Agora e opt-in explicito, `--foreign-analysis`, que:

- exige **a mesma lista de camadas e os mesmos shapes**, senao recusa dizendo que sao arquiteturas
  diferentes (o teste de shape ja existia; o de lista de camadas e novo, porque uma camada que a
  analise nunca mediu cairia no tratamento de `--uncalibrated` em silencio);
- imprime em toda execucao de qual arquivo vieram os erros, porque o risco inteiro da flag e
  alguem esquecer qual medicao produziu o arquivo que esta enviando;
- carrega os numeros acima no proprio `--help`.

Converter um Z-Image novo agora custa **zero calibracao**. Custo medido do reuso: **1 camada de 170
sai diferente** do que o perfil proprio daquele checkpoint escolheria.

### Ressalva que nao pode cair

Isto e uma arquitetura, tres checkpoints, dois deles provavelmente derivados do primeiro. **Nao ha
evidencia nenhuma de transferencia entre arquiteturas ou entre fabricantes** - a ideia de que Qwen
e Z-Image se pareceriam por serem da Alibaba e plausivel e **nao foi testada**. Testar exige
calibrar um modelo de outra familia, o que precisa do encoder e do perfil daquela familia.


## 2026-08-19, parte 21 - segunda arquitetura: o que transfere e o que nao

Teste da hipotese da familia contra uma arquitetura genuinamente diferente. Escolhido
HunyuanVideo 1.5 (`hunyuanvideo1.5_720p_t2v_fp16`, 15,5 GiB) porque o perfil `hunyuan_video_15` ja
existia e cabe na placa. Qwen-Image nao serve: so ha SVDQ int4 e GGUF Q8 no disco, e calibracao
precisa da fonte de alta precisao.

### Antes do resultado: dois bugs, e a armadilha de naming pela quarta vez

**O perfil `hunyuan_video_15` nunca tinha sido executado.** Ele lia
`double_blocks.N.img_attn_qkv` e `img_mlp.fc1`; os modulos carregados chamam-se
`img_attn.qkv` e `img_mlp.0`. Casou **zero** camadas. Custou barato so porque
`calibrate_activations` grita `Profile 'hunyuan_video_15' matched no Linear` em vez de calibrar
conjunto vazio.

**E o motivo do erro e estrutural, nao um typo.** O perfil estava certo *para o arquivo* e errado
*para o modulo*:

```
arquivo   double_blocks.0.img_attn_qkv.weight    double_blocks.0.img_mlp.fc1.weight
modulo    double_blocks.0.img_attn.qkv           double_blocks.0.img_mlp.0
```

`calibrate_activations` percorre **modulos**; `quant_mixed` percorre **chaves do arquivo**. Os dois
liam o mesmo `PROFILE_PATTERNS`. Em Z-Image e LTX as duas nomeacoes coincidem e nada forcou a
distincao a aparecer. Aqui nao coincidem: uma calibracao com chave de modulo descreveria camadas
que o conversor nao acha, e a juncao voltaria vazia - ou pior, meio populada.

Consertado com tres pecas, todas em `calibrate_activations.py` para ficarem juntas do padrao:
`MODULE_TO_FILE` (a traducao), `to_file_name()` (aplicada ao gravar a calibracao, com guarda que
recusa se a traducao colapsar camadas) e `PROFILE_FILE_PATTERNS` (o que o `quant_mixed` casa).
Terceira armadilha de naming deste tipo no projeto, depois do Z-Image diffusers e do
`weight_scale` passando sem renomear.

Tambem: `calibrate_activations` montava latente 4-D sempre. Modelo de video handed um latente 4-D
falha la dentro do transformer com erro de shape que nao fala em latente. Agora le
`latent_dimensions` do proprio `latent_format` e monta 5-D com `--frames`.

### O resultado

432 camadas medidas (54 `double_blocks` x 8).

| | Z-Image | HunyuanVideo 1.5 |
|---|---|---|
| **profundidade vs err_w4a4** | **+0,667** | **+0,220** |
| err_w4a4 mediano | 0,1241 | 0,2136 |
| promovidas em `--promote-error 0.15` | 55 de 170 (32%) | 408 de 432 (94%) |

**A regra de profundidade nao transfere.** Forte numa arquitetura, fraca na outra. O inteiro que
andava metade do caminho no Z-Image nao anda no HunyuanVideo.

**O que aparece nas duas:** a projecao de subida do MLP e a pior das MLP. Z-Image `w3` 0,178 contra
`w2` 0,136; Hunyuan `img_mlp.fc1` 0,429 contra `img_mlp.fc2` 0,192, e `txt_mlp.fc1` 0,222 contra
`fc2` 0,198. E uma comparacao por modelo, entao e hipotese, nao regra.

**Estrutura propria do Hunyuan**, sem paralelo no Z-Image: o fluxo de imagem e mais dificil que o
de texto (`img_attn_qkv` 0,311 contra `txt_attn_qkv` 0,186).

### A consequencia pratica que ninguem tinha visto

**`--promote-error` nao e portavel entre arquiteturas.** O mesmo 0,15 promove 32% do Z-Image e
**94%** do HunyuanVideo - naquele modelo a mistura vira quase-W4A8 e o ganho de tamanho evapora. Um
default fixo so faz sentido dentro de uma familia, exatamente como o perfil.

### Placar da hipotese

| escopo | transfere? | evidencia |
|---|---|---|
| entre checkpoints da mesma arquitetura | **sim** | spearman +0,981 a +0,997; 119-120 de 120 camadas iguais |
| regra de profundidade entre arquiteturas | **nao** | +0,667 contra +0,220 |
| "up-projection e a pior do MLP" | talvez | vale nas duas, uma comparacao por modelo |
| threshold entre arquiteturas | **nao** | 32% contra 94% promovidas no mesmo valor |


## 2026-08-19, parte 22 - tres familias medidas: so a familia transfere

Plano: tres checkpoints por arquitetura, medir, guardar, comparar. O que o disco permitiu:
Z-Image x3, HunyuanVideo 1.5 x2 (so existem dois em alta precisao aqui), WAN 2.1 x1 (arquitetura
nova, perfil escrito nesta sessao). Qwen-Image continua fora: so ha SVDQ int4 e GGUF.

Ferramenta nova: `tools/profile_transfer.py`, que compara N analises par a par.

### Dentro da familia: transfere, e agora em duas familias

| par | camadas | spearman | mesma metade pior | acaso |
|---|---|---|---|---|
| Z-Image turbo vs de-turbo | 170 | +0,981 | 82/85 | 42,5 |
| Z-Image turbo vs beyond-reality-v2 | 170 | **+0,997** | 84/85 | 42,5 |
| Z-Image de-turbo vs beyond-reality-v2 | 170 | +0,986 | 83/85 | 42,5 |
| Hunyuan 720p vs capybara | 432 | **+0,900** | 189/216 | 108,0 |

**A comparacao passou a ser por posto, nao por limiar.** Motivo medido: em `--promote-error 0.10`
os dois Hunyuan promovem 424 de 432, o acaso vira 417, e um 424/424 perfeito pontua **+6,9** - a
estatistica nao diz nada enquanto parece uma vitoria. Metade pior de cada um responde a mesma
pergunta onde ainda da para responde-la.

O resultado do Z-Image nao era quirk: a segunda familia tambem transfere, um pouco mais frouxa
(87,5% da metade pior contra 96-99%).

### Entre familias: nada transfere

| familia | camadas | err_w4a4 mediano | profundidade vs erro | acima de 0,15 |
|---|---|---|---|---|
| Z-Image | 170 | 0,1241 | **+0,667** | 34% |
| HunyuanVideo 1.5 | 432 | 0,2136 | +0,220 | 94% |
| WAN 2.1 | 300 | 0,1511 | **+0,088** | 50% |

**A regra de profundidade morre com a terceira familia**: +0,667, +0,220, +0,088. Ela e propriedade
do Z-Image, nao dos transformers de difusao.

E a hipotese que tinha sobrevivido a duas familias - "a projecao de subida do MLP e a pior" -
**inverte na terceira**:

```
Z-Image   w3 (up)   0,1782   >   w2 (down)  0,1364
Hunyuan   fc1 (up)  0,2565   >   fc2 (down) 0,1956
WAN       ffn.0(up) 0,1334   <   ffn.2(down) 0,2084
```

Era exatamente por isso que valia medir a terceira. Com duas eu teria escrito uma regra.

### Placar final da hipotese da familia

| escopo | transfere? |
|---|---|
| entre checkpoints da mesma arquitetura | **sim**, medido em duas familias |
| regra de profundidade entre arquiteturas | **nao** (+0,667 / +0,220 / +0,088) |
| "up-projection e a pior do MLP" | **nao** (inverte no WAN) |
| valor de `--promote-error` entre arquiteturas | **nao** (34% / 94% / 50% promovido no mesmo 0,15) |
| estatistica do peso, qualquer familia | **nao** (acaso) |
| ativacao sintetica | **nao** (pior que acaso) |

**Sobra uma coisa so, e ela funciona: medir uma vez por arquitetura e reusar via
`--foreign-analysis`.** Todo atalho que tenta pular essa medicao falhou.

### Terceira convencao de nomes, terceira vez

WAN prefixa as chaves com `model.diffusion_model.` e os modulos nao. Hunyuan troca ponto por
underscore. Z-Image (publicado) usa naming diffusers. **Assumir que arquivo e modulo tem nomes
diferentes ate um dump dos dois dizer o contrario** - as tres vezes que isso apareceu aqui, custou
uma investigacao.

### Custo de calibrar, para dimensionar o proximo

| modelo | tamanho | calibracao | medicao |
|---|---|---|---|
| WAN 2.1 VACE 1.3B | 4 GiB | 8,3 s | 300 camadas |
| Z-Image | 11,5 GiB | 40 s | 170 camadas |
| HunyuanVideo 1.5 | 15,5 GiB | 67 s | 432 camadas |

Barato. O que custa e o disco: LTX 2.5 (39 GiB BF16 x2) ficou de fora por nao caber na placa sem
offload, e o Z-Image recuperado de SVDQ por precisar de mais 11,5 GiB num disco a 98%.


## 2026-08-19, parte 23 - o perfil sobrevive a uma volta destrutiva pelo int4

Quarto ponto do Z-Image, e o mais informativo dos quatro: `beyond-reality-recovered-bf16` e o
`beyond-reality-zimage-v2` depois de passar por SVDQuant int4 rank-32 e voltar para BF16 pelo
`tools/svdq_to_bf16.py`. Nao e um finetune - e o mesmo modelo com os pesos danificados.

**Quanto os pesos mudaram**, medido em cinco camadas contra o original:

```
layers.0.attention.qkv        0,0923      layers.29.feed_forward.w3     0,1132
layers.10.feed_forward.w2     0,0963      noise_refiner.0.attention.qkv 0,0708
layers.20.attention.out       0,0935      mediana                       0,0935
```

**Quanto o perfil mudou:** nada.

| par | spearman | metade pior igual |
|---|---|---|
| beyond-reality-v2 vs sua propria versao recuperada | **+0,999** | **85/85** |
| recuperado vs turbo | +0,997 | 84/85 |
| recuperado vs de-turbo | +0,986 | 83/85 |

Os dois ultimos sao **identicos** aos que o original marca contra os mesmos dois checkpoints. O
recuperado ocupa exatamente o lugar do original na familia.

### O que isso fecha

Perturbar **cada peso do modelo em ~9,4%** nao move a decisao de qual camada precisa de 8 bits.
Somado ao que ja estava medido - estatistica do peso nao prediz (acaso), ativacao sintetica nao
prediz (pior que acaso), finetunes diferentes concordam a +0,98 - a conclusao fica dificil de
escapar: **o perfil nao e funcao dos valores dos pesos.** E funcao da arquitetura e das ativacoes
que ela produz, e nenhuma das duas muda quando se requantiza o peso.

Consequencia pratica direta: da para calibrar na variante que estiver a mao. Nao precisa ser a
mais recente, nem a de maior precisao, nem sequer uma intacta.

E uma consequencia para o proprio `svdq_to_bf16.py`: o que ele recupera preserva a estrutura que
importa para esta decisao, mesmo perdendo 9,4% do peso. Isso nao diz que a imagem dele e boa - nao
foi medido aqui - so que o perfil de quantizacao dele e o mesmo.


## 2026-08-19, parte 24 - quanto custa reusar o perfil, em camadas e nao em correlacao

`--foreign-analysis` foi liberado com base em spearman +0,98 a +0,997. Spearman nao e o que o
conversor usa: ele usa um limiar e escreve um formato por camada. A pergunta certa e **quantas
camadas saem atribuidas diferente**, e em que direcao - promover a toa custa tamanho e velocidade,
deixar em 4 bits custa precisao.

`profile_transfer.py` agora reporta isso, com a direcao nomeada, porque os dois lados nao dao o
mesmo numero.

### Z-Image: reuso e exato

```
reusando turbo em beyond-reality-v2 @0.10:   0 de 170 camadas diferem
reusando turbo em recuperado @0.10:          0 de 170
reusando beyond-reality-v2 em recuperado:    0 de 170
```

**Zero.** Nos tres pares, incluindo o checkpoint que passou por int4 e voltou. Nesta familia a
calibracao e literalmente pagavel uma vez.

### HunyuanVideo: reuso custa 11%

```
reusando capybara em hunyuan15 @0.22:  48 de 432 diferem (18 a toa, 30 expostas, pior 0,3358)
reusando hunyuan15 em capybara @0.22:  48 de 432 diferem (30 a toa, 18 expostas, pior 0,2339)
```

11,1% das camadas. E a assimetria importa: na direcao que interessa aqui - usar o perfil do
`hunyuanvideo1.5_720p` para converter o `capybara` - sao 18 camadas expostas, e a pior delas mede
0,2339 contra um corte de 0,22. **Todas as discordancias sao rentes ao limiar**, que e onde a
decisao menos importa: uma camada em 0,2339 e uma em 0,22 sao o mesmo caso duvidoso.

### A regra que sai disso

**A seguranca do reuso e ela propria propriedade da familia, e tem de ser medida uma vez por
familia.** Z-Image: zero. HunyuanVideo: 11%, concentrado no limiar. Nao da para transportar "reuso
e seguro" de uma para a outra, exatamente como nao deu para transportar o limiar nem a regra de
profundidade.

O procedimento que isso sugere, para uma familia nova: calibrar **dois** checkpoints dela uma vez,
rodar `profile_transfer.py`, e so entao decidir se os proximos podem usar `--foreign-analysis`. O
custo e uma calibracao extra - 8 a 67 segundos nesta maquina - contra converter todos os proximos
as cegas.

### Nota sobre o limiar usado

0,22 para HunyuanVideo, nao 0,10. Em 0,10 aquela familia promove 424 de 432 e a mistura deixa de
ser mistura. O valor foi escolhido perto da mediana daquela familia (0,2136), que e o que torna a
comparacao informativa - e e mais uma instancia de que o limiar e por familia.


## 2026-08-19, parte 25 - por que nenhum atalho pelo peso funciona: o peso nao varia

Pergunta do usuario: com um checkpoint ja quantizado ao lado do BF16, da para estimar
matematicamente qual camada precisa do formato caro, sem calibrar?

### Primeiro, o que o checkpoint quantizado da LTX contem

`ltx-2.5-22b-dev-transformer-comfy-int8-convrot` nao traz `_quantization_metadata` no header - usa
os marcadores `comfy_quant` inline, 1440 deles, um por camada quantizada. Decodificados, os 1440
sao **a mesma string**:

```
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

Formato uniforme. **Nao ha perfil por camada para ler ali** - a Lightricks nao escolheu camada a
camada, aplicou um formato a todas.

### Segundo, e o achado: o peso nao carrega a informacao

Testado o que faltava testar. Nao uma *estatistica* do peso (ja medido: acaso), mas **o erro de
reconstrucao do proprio peso**: quantizar, desquantizar pela identidade - o mesmo truque do
`recover_weight`, exato porque uma linha one-hot nao sofre com o quantizador de ativacao - e
comparar com o original. Zero calibracao, zero amostragem, zero encoder.

Z-Image, 170 camadas:

| grandeza | min | max | CV | max/min |
|---|---|---|---|---|
| erro do **peso** w4a4 | 0,1566 | 0,2128 | 6,1% | 1,36x |
| erro do **peso** w4a8 | 0,0730 | 0,0734 | **0,1%** | **1,01x** |
| erro da **ativacao** w4a4 | 0,0195 | 0,4758 | 45,5% | **24,4x** |
| erro da **ativacao** w4a8 | 0,0079 | 0,1666 | 45,3% | 21,1x |

HunyuanVideo 1.5, 432 camadas:

| grandeza | min | max | CV | max/min |
|---|---|---|---|---|
| erro do peso w4a4 | 0,1496 | 0,1823 | 4,5% | 1,22x |
| erro da ativacao w4a4 | 0,0475 | 0,5259 | 35,4% | **11,1x** |

**Todo peso quantiza praticamente igual.** No W4A8 o erro de peso e literalmente constante -
1,01x entre a melhor e a pior camada de 170. A variacao que a decisao de promocao depende de -
11x a 24x - esta inteiramente na ativacao.

Spearman entre os dois: **-0,079** no Z-Image e **-0,396** no HunyuanVideo. Nao e so ausencia de
sinal: no Hunyuan o pouco que existe aponta para o lado errado, e usar erro de peso como preditor
la seria pior que sortear.

### O que isso fecha

Isto explica de uma vez todos os negativos das partes 19 e 22, que ate agora eram uma lista de
tentativas fracassadas sem causa comum:

- estatistica de peso nao prediz -> **porque o lado do peso nao varia**
- erro de reconstrucao do peso nao prediz -> mesma razao, medida diretamente
- ativacao sintetica prediz pior que o acaso -> porque troca a unica fonte de variacao por outra
- perfil transfere entre checkpoints da mesma arquitetura -> **porque requantizar o peso nao muda
  a ativacao**, e a parte 23 mostrou isso ao vivo: 9,4% de dano em cada peso, perfil identico

**Nao da para estimar pelo peso. Nao por falta de uma formula melhor, mas porque a grandeza que
distingue as camadas nao esta la.** A calibracao nao e um atalho que ainda nao foi encontrado; ela
mede a unica coisa que varia.

O que continua valendo, e agora com mecanismo por tras: medir uma vez por arquitetura e reusar.


## 2026-08-19, parte 26 - W8A8 existe, e o convrot vale a pena nele

Pedido: fazer um W8A8 do LTX 2.5. Duas respostas, e a primeira e que ele ja existe.

### `int8_tensorwise` e W8A8 de verdade, medido

O layout `TensorWiseINT8Layout` chama `torch.ops.comfy_kitchen.int8_linear` com a ativacao em
bf16, e o comentario do fonte diz "TensorWise needs dynamic row-wise quant" - o que sugere que a
ativacao e quantizada dentro do kernel, mas sugerir nao e medir.

Discriminador: recuperar o peso efetivo pela identidade (exato, uma linha one-hot nao sofre com
quantizador simetrico de ativacao), depois rodar o kernel com entrada comum e comparar contra
`x @ W_hat.T`. Se baterem, a ativacao ficou em alta precisao; se nao, foi quantizada.

| formato | erro do peso | kernel vs `x @ W_hat.T` | veredito |
|---|---|---|---|
| convrot_w4a4 | 0,1566 | 0,15699 | ativacao em int4 |
| asym_w4a8_int8 | 0,0731 | 0,00966 | ativacao em int8 |
| **int8_tensorwise** | **0,0094** | **0,00938** | **ativacao em int8** |

A contribuicao da ativacao do `int8_tensorwise` (0,00938) e a mesma do W4A8 (0,00966) - os dois
quantizam a ativacao para int8. A diferenca entre eles esta toda no peso: 0,0094 contra 0,0731,
**8x**.

Erro total aproximado por formato: W4A4 ~0,22, W4A8 ~0,074, **W8A8 ~0,013**.

### E ja existe em disco, tres vezes

| arquivo | GiB | origem |
|---|---|---|
| `ltx-2.5-22b-dev-transformer-comfy-int8-convrot` | 20,0 | Lightricks, marcadores `comfy_quant` inline |
| `ltx-2.5-22b-distilled-transformer-bf16_int8` | 20,0 | nosso, sem convrot |
| `ltx-2.5-22b-distilled-transformer-bf16_int8_convrot` | 20,0 | nosso, com convrot |

### O convrot vale a pena no int8

Ninguem tinha checado. No W4A4 a rotacao e essencial; no int8 poderia ser custo sem retorno.
Vinte camadas do LTX distilled, desquantizadas pelos ops reais
(`dequantize_int8_convrot_weight_dtype` e `dequantize_int8_simple_dtype`, nao por multiplicacao a
mao, que ignoraria a rotacao) e comparadas contra o BF16 fonte:

| | mediana | min | max | max/min |
|---|---|---|---|---|
| int8 sem convrot | 0,01144 | 0,00970 | 0,01926 | **2,0x** |
| int8 com convrot | 0,00942 | 0,00902 | 0,01065 | **1,2x** |

**Corta 17,1% do erro na mediana e ate 51,0% na pior camada.** O que importa nao e a mediana: a
rotacao **comprime a dispersao**, de 2,0x para 1,2x entre a melhor e a pior camada. Ela conserta as
camadas ruins, nao a media - que e exatamente o que uma rotacao contra outliers deveria fazer.

Entao, para W8A8, usar a variante com convrot. Ja esta pronta.

### O que continua sem ter sido feito

Nenhum desses tres arquivos foi verificado pelo padrao deste projeto: carregar pelo loader normal,
confirmar que despacha nativo, e medir contra o BF16 com parametros casados. `dispatch_census.py`
faria a parte do despacho, mas hoje so instrumenta os layouts W4A4 e W4A8 - o
`TensorWiseINT8Layout` nao esta na lista, entao rodar como esta reportaria zero e pareceria uma
resposta.


## 2026-08-19, parte 27 - aceitacao do LTX 2.5 W8A8: passa no que da para testar, e o resto esta bloqueado

`ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors`, 20,0 GiB.

### Antes: o instrumento nao enxergava o formato

`tools/diffusion_smoke.py` embrulhava `convrot_w4a4_linear` e `w4a8_int8_linear`. O layout int8
nao chama funcao de registry - chama `torch.ops.comfy_kitchen.int8_linear` direto - entao um
checkpoint int8 reportava **zero linears e zero dequantizacoes**, que le como "nada rodou" e nao
como "esta ferramenta nao olha esse layout".

Terceira vez que essa mesma classe de buraco aparece: o proprio docstring do `diffusion_smoke`
registra que ele ja tinha sido consertado uma vez pelo mesmo motivo, e continuava incompleto. O
`dispatch_census.py` tinha o mesmo (corrigido junto, mais o latente de video 5-D).

Consertado embrulhando o handler registrado no `_LAYOUT_DISPATCH_TABLE` em vez da funcao.

### O que passou

```
loader                     comfy.sd.load_diffusion_model, sem no custom
VRAM apos carga            20,02 GiB
modulos quantizados        1440
formatos                   int8_tensorwise x1440
layouts                    TensorWiseINT8Layout x1440
convrot_groupsize          256 x1440
camadas sondadas           4, todas nativas
  native_linear_calls      2 por camada
  weight_dequant_calls     0
all_native                 true
```

O despacho vai por `comfy_kitchen.tensor.int8.t` seguido de `.addmm` - o mesmo padrao `t` + `mm`
que a parte 13 contou no Z-Image, e nao por `linear`, entao o ramo que dequantiza continua fora do
caminho tambem neste formato.

### O aviso que parecia grave e nao e

A carga emite `WARNING: unet unexpected: [...]` com **1440 chaves**, todas `.comfy_quant`. Parece
que os marcadores nao foram consumidos - o que seria a mesma armadilha do Z-Image em naming
diffusers, onde a escala passa batida e a camada carrega sem erro.

Nao e. Os numeros casam exatamente: **1440 chaves reportadas como sobra, 1440 modulos carregados
quantizados**, todos com o formato e o groupsize certos. O loader consome o marcador para montar a
config e depois lista a mesma chave como nao-consumida no state dict. Se tivesse ignorado,
`quantized_module_count` seria zero.

Vale registrar porque o aviso e alarmante e a verificacao e barata: comparar as duas contagens.

### O que continua bloqueado

**O benchmark casado contra o BF16 nao foi feito.** Nao por falta de GPU: amostrar LTX 2.5 nao e
uma chamada de `KSampler`. Os workflows deste projeto usam `LTXAVTextEncoderLoader`,
`LTXVAudioVAELoader` e `CheckpointLoaderSimple` de um pacote de custom nodes, com o gemma de 12B
carregando a projecao a partir do proprio checkpoint - e os arquivos 2.5 aqui sao transformer
sozinho, em `diffusion_models`. Montar isso e integracao, nao comando.

Entao a aceitacao deste arquivo esta em: **carrega e despacha nativo, sim, medido. Qualidade e
velocidade contra o BF16, nao medido.** Nao chamar de aceito sem essa segunda metade.

## 2026-08-19, parte 28 - o widget `type` do CLIPLoader derrubou tres runs, e o MultiGPU nao roda quantizado no Windows

Tentativa de rodar o workflow de aceitacao do LTX 2.5 int8 pela UI. Nenhuma imagem saiu. Tres
tracebacks distintos, um deles com cara de bug de kernel. Nenhum era.

### Tres erros, um knob - **tracado no codigo, nao executado**

| erro | classe de CLIP que carregou |
|---|---|
| `NotImplementedError: Cannot copy out of meta tensor` | `SD1ClipModel` |
| `ValueError: invalid tokenizer` | tokenizer gemma3_4b via `lumina2.py` |
| `RuntimeError: Tensors must have same number of dimensions: got 4 and 3` | `Gemma3_12BModel_` |

`nodes.py:1024` e a origem dos tres:

```python
clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
```

Qualquer `type` que nao seja `ltxv` **nao levanta erro** - cai no fallback STABLE_DIFFUSION, que
fareja o state dict, ve GEMMA_3_12B e constroi o encoder errado (`sd.py:1813`).

Cadeia do erro de dimensao:

1. `lt.py:96` - `Gemma3_12BModel(layer="all")` devolve todas as camadas escondidas empilhadas. A
   condicao sai **4-D** `[B, L+1, T, C]`.
2. Esse caminho nao tem `text_embedding_projection` - ela vive no `LTXAVTEModel`, que so o
   `type=ltxv` monta. Nada achata o eixo de camadas.
3. `av_model.py:587` testa `context.shape[-1]`; nao bate com
   `cross_attention_dim + audio_cross_attention_dim`, entao nao faz curto-circuito e segue com o
   tensor 4-D.
4. `embeddings_connector.py:290` faz `.unsqueeze(0).repeat(shape[0],1,1)`, que e 3-D. `torch.cat`
   de 4-D com 3-D.

O sinal no log, e ele e barato: `clip missing: ['vision_model.embeddings.patch_embedding.weight',
...]`. Um Gemma3-12B puro espera torre de visao; o text encoder do LTX nao tem. Esse warning
aparece em **todo** run que morre na dimensao e **nao** aparece no que carregou `LTXAVTEModel_`.

Correto para os checkpoints deste projeto: **`type: ltxv`**. Os dois workflows em
`user/default/workflows/` ja estao assim em disco; a alteracao foi feita no canvas.

### ComfyUI-MultiGPU quebra qualquer modelo quantizado fora do device de execucao - **medido**

`custom_nodes/ComfyUI-MultiGPU/p2p_registry.py:20`:

```python
_libcudart = ctypes.CDLL("libcudart.so")
```

Nome de biblioteca Linux, sem ramo Windows. `can_access_peer` (`p2p_registry.py:62`) **nao captura**
a excecao, e `__init__.py:557` chama a funcao sempre que `tensor_device.index != exec_device.index`.
Como o pacote monkeypatcha o `_wrap_for_dlpack` do comfy_kitchen com
`wrap_for_dlpack_with_device_guard`, o patch e **global**: basta qualquer tensor quantizado estar
num device diferente do de execucao.

`FileNotFoundError: Could not find module 'libcudart.so'` reproduzido em 5 runs, por tres caminhos
diferentes: `dequantize_int8_convrot_weight_dtype`, `dequantize_per_tensor_fp8` e
`dequantize_w4a8_int8_weight`.

Consequencia pratica: **a 3080 Ti e inutilizavel para modelo quantizado nesta maquina** enquanto o
pacote estiver instalado. Nao adianta evitar os nos MultiGPU - o patch entra no import. A defesa e
nao cruzar device nenhum: tudo em `cuda:0`, offload pelo DynamicVRAM.

Terceiro candidato a report upstream, e o de repro mais curto dos tres
(ver `AUDITORIA_2026-08-18.md` para os outros dois).

### O metadata do checkpoint esta limpo - **medido, so o header**

Levantada a hipotese de que os `.comfy_quant` reportados como `unet unexpected` indicassem despacho
perdido. Nao indicam - a parte 27 ja tinha medido isso contando modulos. O header corrobora por
outro angulo, e custa um `read` de 8 bytes mais o JSON:

```
ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors
  layers no _quantization_metadata:      1440
  layer + ".weight" casa com tensor:     1440/1440
  conjuntos de campos distintos:         1  -> {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
  weight I8 [2048,2048]   scale F32 [2048,1]
```

Nomes corretos, nenhuma camada orfa, nenhum campo faltando, `convrot_groupsize` presente nos 1440 -
que e o parametro que `comfy_kitchen/tensor/int8.py:172` passa para
`dequantize_int8_convrot_weight_dtype`. O nome do arquivo nao mente: neste esquema *int8-convrot* e
`format: int8_tensorwise` mais a flag, nao um format separado.

Controle que **nao** serve de espelho: o `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
oficial nao tem `_quantization_metadata` **nenhum** e tem 7229 tensores contra 5789 do nosso, com
tamanho em disco praticamente igual. Esquema diferente, provavelmente resolvido por
`convert_old_quants`.

Nota de modo de falha, caso a duvida volte: o I8 tem a **mesma largura** do slot bf16
(`[2048,2048]`, nao empacotado). Se o marcador fosse perdido de verdade, o `copy_` passaria calado e
os codigos -127..127 virariam pesos sem escala. Isso e ruido visivel, nao degradacao sutil.

### comfy-cli / comfy-mcp: validacao de workflow offline

Instalado a pedido, em **venv isolada** - `venvs/comfymcp`, criada com o Python 3.12.10 do sistema.
`python_embeded` intocado, verificado antes e depois. `comfy-cli 1.16.0`, `comfy-mcp 0.10.0`.

O que vale, e nao e a parte de gerar imagem:

```bash
comfy validate --workflow <wf.json> --input <object_info.json>   # offline, sem servidor, sem GPU
comfy stop --port 8190                                            # mata servidor que o CLI nao subiu
comfy free --unload-models --free-memory                          # devolve VRAM sem matar o servidor
```

`validate` converte UI->API sozinho e confere class_types, shapes, enums e fiacao. Medido nos dois
workflows, com a GPU ocupada por outra sessao:

```
LTX25-int8-acceptance-v2   valid: true  errors: 0  warnings: 0  converted_from_ui: 18 nos
LTX25-int8-acceptance      valid: true  errors: 0  warnings: 0  converted_from_ui: 18 nos
```

**E nao teria pego o bug desta parte.** Os dois passam limpos e os dois morreram em runtime.
`type: lumina2` e um enum *valido*; o validate nao sabe que so `ltxv` produz o embedding certo. Ele
pega fiacao e digitacao - que foi o que queimou tres tentativas de montar API-format na mao - e nao
pega escolha errada.

Telemetria, checada antes de registrar: `comfy_cli/tracking.py` manda para Mixpanel e PostHog em
`https://t.comfy.org`. Sanitiza `prompt`, `changelog`, `set_overrides` e credenciais. **As duas
travas nao sao equivalentes**: a config `enable_tracking` cobre so a telemetria passiva de comando,
enquanto `submit_feedback` (`tracking.py:648`) e explicitamente *nao* gated no consentimento e so
para com as env `DO_NOT_TRACK` / `COMFY_NO_TELEMETRY`. O `comfy_mcp` em si nao tem telemetria
nenhuma. Registrado em escopo user com **as duas** env vars ligadas, mais
`COMFY_LOCAL_URL=http://127.0.0.1:8190` - sem isso ele falaria com a 8188, que aqui nao existe.

Nenhuma ferramenta do MCP foi exercitada ainda. O `comfy validate` acima rodou pela linha de
comando, nao pelo MCP.

### O launcher

Nao existe registro de que algum `.bat` estivesse errado. `run_nvidia_gpu_8190_loopback.bat` e o
certo e foi o usado - os `Arguments` do report de erro batem com ele exatamente. A unica flag
condicional documentada e `--disable-dynamic-vram`, e **so** para loader Nunchaku SVDQuant.

**Nao adicionar `--disable-dynamic-vram` para LTX 2.5.** Medido e registrado em
`tools/_dynamic_vram.py`: sem DynamicVRAM esse modelo leva o processo a 44.5 GiB de working set
numa maquina de 63.1, antes de qualquer bloco chegar na GPU.

### O que continua bloqueado

O mesmo da parte 27, e nada mudou: **benchmark casado contra o BF16 nao foi feito.** Alem disso o
workflow v2 ainda nao rodou de ponta a ponta - parou no `CLIPTextEncode` por causa do `type`, e a
correcao nao foi testada porque a GPU passou para a outra sessao.

## 2026-08-31, parte 29 - o checkpoint emite 4 bits; o ComfyUI nunca deixa, em text encoder nenhum

Tudo abaixo foi **executado** na 3090 (cc 8.6) com o lock `w4a4:winnougan_int4_dispatch`, contra
`ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-int4_convrot.safetensors` (13,20 GiB, 351
camadas, `converted_by: Star Ultimate Model Converter`).

### O que estava em aberto

Os dois ConvRot publicos grandes (`Abiray/...FL2VA`, `...Ref2VA`) gravam `linear_dtype: "int8"` em
cada camada, que e a primeira condicao do desvio em `convrot_w4a4_linear`, antes de qualquer teste
de hardware. Este arquivo e a excecao: `linear_dtype` **ausente** nas 351, e ausente cai no default
da assinatura, `"int4"`. Faltava executar.

### O kernel: 15/15, o ramo nativo roda com os bytes deste arquivo

`tools/probe_winnougan_int4.py`. Uma camada por forma convrot distinta (5 formas no modelo), pesos
lidos por faixa de bytes do proprio arquivo -- nada e requantizado. Um eixo varia:
`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, lida no import, por isso um subprocesso por braco.

```
15/15 pares diferem entre nativo e fallback   -> sao dois kernels distintos
native_mma_supported (executado)              -> True
M=1     fallback ganha em 4 de 5 formas (1,04x a 2,16x)
M=64    nativo   ganha em 4 de 5 formas (1,66x a 3,13x)
M=1024  nativo   ganha em 5 de 5 formas (1,37x a 2,46x)
```

A curva do `m_crossover` de novo, na mesma direcao ja registrada nas partes anteriores.

### O loader: carrega pelo caminho de estoque, e a minha primeira leitura estava errada

`tools/probe_winnougan_load.py`. Hipotese inicial, por leitura: `load_text_encoder_state_dicts`
chama `detect_layer_quantization` so no ramo do MiniMax **Music3**, nunca no `QWEN3VL_32B`, entao
seria um PR de uma linha. **Derrubado pela execucao.** O ramo QWEN3VL_32B carrega e codifica sem
tocar em nada: 1604 tensores, 351 `.comfy_quant`, `cond [1, 8, 5120]`, sem nan, pico de 14093 MiB.

Erro meu no meio do caminho, registrado porque a mensagem engana: chamar
`load_text_encoder_state_dicts` com um `load_torch_file` cru **nao** e o caminho de estoque. Ele
pula `convert_old_quants` (`comfy/sd.py:1536`), nenhum `.comfy_quant` existe, e o erro que sai e
`size mismatch 2560 contra 5120` -- que parece "modelo errado" e e "metadata nao aplicada". O
caminho de estoque e `comfy.sd.load_clip`, que e o que `CLIPLoader` chama (`nodes.py:1031`).

### O achado: o encode nao emite 4 bits, e nao e por causa deste arquivo

Terceiro braco do mesmo probe, com `FORCE_INT4_INT8_FALLBACK=1` (`visto pelo modulo=True`): o
condicionamento saiu **identico bit a bit**. Se os dois bracos dao o mesmo numero, nenhum dos dois
passou pelo kernel. `tools/probe_te_fullprecision_mm.py` conta, em vez de ler:

```
Linear inspecionadas        100/100  MixedPrecisionOps.Linear, quant_format convrot_w4a4,
                                     peso QuantizedTensor, layout TensorCoreConvRotW4A4Layout
_convrot_w4a4_forward            0
QuantizedTensor.dequantize     350
```

Zero. O peso fica 4 bits na VRAM e a **matematica e dequantizada**. Economia de memoria, nao de
tempo.

### Qual trava, medida termo a termo -- porque a primeira resposta obvia estava errada

A leitura apontava `comfy/sd1_clip.py:114`, que fixa `full_precision_mm=True` para todo text
encoder. Virei so esse termo: **nada mudou**, ainda 0 chamadas. Instrumentando os termos de
`_use_quantized` (`comfy/ops.py:1372-1377`) na primeira passagem de uma Linear real:

```
input_ndim 3   input_is_qt False   layout_type TensorCoreConvRotW4A4Layout
full_precision_mm True   comfy_force_cast_weights True   <- este
```

`comfy_force_cast_weights` vem de `model_patcher.py:1016`, que le `force_cast_weights`, ligado em
`model_patcher.py:743` por `set_model_compute_dtype(dtype)` -- chamado em **`comfy/sd.py:269`,
`self.patcher.set_model_compute_dtype(torch.float32)`, para todo objeto CLIP**, com o comentario
"Match torch.float32 hardcode upcast in TE implemention".

São **duas travas independentes**, e a que morde e a segunda. Soltando as duas nas 351 Linear:

```
                        fase 1 (estoque)      fase 2 (as duas soltas)
_convrot_w4a4_forward              0                    350
dequantize                       350                      0
norma do cond               15724.97               14215.17
tempo do encode (mediana de 5)  312,1 ms              114,3 ms   -> 4 bits 2,73x mais rapido
  [SUPERSEDIDO pela parte 32: estes numeros vieram de um destravamento que so pegava por
   acaso. Refeitos pelo metodo confiavel dao 311,6 -> 117,7 ms, 2,65x, e 9,73e-2.]
                        [313,1 311,7 312,0 313,2 312,1]  [128,6 112,7 114,7 114,3 111,3]
rel-RMSE fase2 contra fase1                        9,70e-2
```

### O que isso muda para este projeto

A regra do `CLAUDE.md` -- "W4A4 significa execucao nativa ConvRot" -- e **inalcancavel por
construcao para qualquer text encoder no ComfyUI de estoque**, independente do checkpoint. Isso
inclui o perfil `gemma`, que e o carro-chefe do conversor e e carregado como text encoder. **Nao
medido ainda:** se a conversao Gemma deste projeto sofre o mesmo (esperado que sim, pelo mesmo
`sd.py:269`), e quanto custa em s/it no LTX. E o proximo teste.

O modelo de difusao e outro caminho (`comfy/ops.py:1667`, que passa `disabled=` e nao
`full_precision_mm`) e nao foi tocado aqui.

### Nao coberto

Sem SASS: "o ramo nativo roda" significa "produz numero diferente do fallback", nao "a instrucao
`m16n8k64.s4` foi observada emitindo". Ativacao sintetica gaussiana no probe de kernel, sem os
outliers que a rotacao existe para suprimir. **Sem referencia BF16 deste modelo nesta bancada**,
entao o `9,70e-2` e contra o proprio braco dequantizado, e nao ha nenhuma afirmacao de fidelidade
aqui. Um prompt, uma placa sm86. As travas foram soltas por monkeypatch pos-load, nao por um
caminho que o ComfyUI ofereca -- `custom_operations` em `model_options` e a unica saida real, e
nenhum no a expoe.

## 2026-08-31, parte 30 - a divisao e limpa: difusao roda quantizado, text encoder nao

`tools/probe_quant_dispatch.py`, executado na 3090 sob o lock `w4a4:gemma_te_dispatch`. Mesmo
instrumento da parte 29, generalizado: carrega pelo caminho de estoque, instrumenta **depois** do
load, e conta o kwarg `weight_only_quant` que o proprio `comfy/ops.py:1414-1419` calcula a partir
de `_use_quantized` -- em vez de re-derivar a expressao, que seria mais uma leitura.

```
zimage-v2-w4a4              difusao  170 convrot_w4a4    340 quant, 0 sem, 0 dequantize
gemma_3_12B_it_heretic_w4a8  TE      336 asym_w4a8_int8    0 quant, 336 sem, 336 dequantize
```

### O que isso salva

**As medicoes de Z-Image desta bancada estavam certas.** Erro por camada, epsilon por passo,
nativo-contra-INT8 -- tudo rodou em matematica de fato quantizada, `_full_precision_mm` False e
`comfy_force_cast_weights` False nas 170 camadas. Se tivesse dado o contrario, aquelas comparacoes
teriam sido entre duas dequantizacoes e nao valeriam nada.

### O que isso custa

**O Gemma deste projeto e economia de memoria e nada mais.** As duas travas `True` nas 336 camadas,
336 dequantize, zero forward quantizado. E `asym_w4a8_int8` nao e ConvRot W4A4 -- **o formato nao
importa**, o que importa e ser text encoder. `comfy/sd.py:269` nao olha para o formato.

### O erro de contagem, registrado porque quase virou conclusao

A primeira versao do probe embrulhou `forward_comfy_cast_weights` na **classe**
`MixedPrecisionOps.Linear` e contou toda chamada. Resultado: `MISTO: 340 quantizados contra 76
nao`, para um modelo em que 170 de 170 camadas quantizadas estavam quantizadas. As 76 eram Linear
que **nunca tiveram peso quantizado** -- a classe serve todas as Linear do modelo, quantizada ou
nao. Contagem correta filtra por `layout_type is not None`. O numero errado nao era menor nem
maior: era de outra populacao.

### Nao coberto

Sem SASS. Sem referencia BF16 casada, entao nada aqui afirma fidelidade. Uma placa (3090, sm86).
Difusao rodou 2 passos em 512, o bastante para o dispatch acontecer e nao para julgar imagem.
**Nao medido:** quanto custa em s/it num render LTX real, e se soltar as travas num text encoder e
seguro para a qualidade da saida.

## 2026-08-31, parte 31 - o criterio ponderado por sigma foi testado e nao paga

A ideia era de 30/08: `tools/probe_epsilon_per_step.py` mostrou que o dano da quantizacao se
concentra em sigma alto e decai monotonico (6,17e-1 em 1,000 -> 5,31e-2 em 0,300), entao um
criterio que pesasse as camadas pela contribuicao delas la seria coisa diferente de um que trata
todo passo igual. Executado hoje na 3090 sob o lock `w4a4:sigma_criterion`.

### O que foi preciso construir

O reservoir mistura os passos por construcao (Algoritmo R sobre o fluxo inteiro) e depois nao da
para separar -- **uma calibracao antiga nao pode ser reponderada**. Entao:

- `calibrate_activations.py` grava agora `sample_sigma`, um sigma por linha guardada, escrito nos
  mesmos slots do reservoir. O sigma vem de um `model_function_wrapper` em `apply_model`, porque o
  forward-pre-hook da Linear nao ve o timestep. Meta nova: `sigma_tagged_layers`, `sigma_min`,
  `sigma_max`, `sigma_distinct` -- se o wrapper nunca disparasse, a coluna sairia toda NaN e o
  arquivo pareceria normal.
- `quant_mixed.py` ganhou `--sigma-weight none|sigma|sigma2|high`, **default `none`**: mudar o
  default reinterpretaria em silencio toda analise ja gravada. A analise e o sidecar registram o
  modo, e reusar uma analise medida com outro modo e recusado -- as colunas `err_*` tem o mesmo
  nome nos dois casos e significam coisas diferentes.
- `tools/test_quant_mixed_sigma.py`, 15 checagens, sem GPU. A que carrega o resto: **peso uniforme
  tem de dar exatamente o mesmo numero que nenhum peso**, senao `--sigma-weight` nao e uma
  ponderacao da metrica existente, e uma metrica nova com o mesmo nome.
- `tools/probe_epsilon_per_step.py` teve o corpo de script guardado em `main()`. Estava tudo em
  escopo de modulo, entao importar os templates ARM executava o `parse_args()` do outro programa.

Calibracao: 170/170 camadas rotuladas, 8 sigmas distintos (1,000 a 0,300), reservoir espalhado
pelos 8 passos como o Algoritmo R promete.

### O criterio ponderado e quase o mesmo criterio

Spearman de `err_w4a4` contra o plano, 170 camadas: **+0,9935** (`sigma2`), **+0,9629** (`high`).
Cruzam o limiar 0,15: 6 e 9 camadas. Contagens: 117/53, 115/55, 114/56.

### A primeira comparacao disse 8/8 e estava confundida

Plano contra `high`, trajetoria do BF16 imposta aos dois: `sigma_alto` ganhava **8/8 passos** e
1,047x na media. Mas ele promovera **56 camadas a 8 bits contra 53** -- modelo maior contra modelo
menor. O eixo que eu achava segurado era o orcamento, e ele estava variando.

### Com o orcamento igualado, o efeito some e inverte

Braco plano reconstruido com `--promote-error 0,1484`, que promove exatamente 56. Tres sementes:

```
                  todos os passos       sigma ALTO          sigma BAIXO      passos
seed 1234   plano56  1,0045x        sigma_alto 1,0079x   plano56 1,0443x      6x2
seed 5678   plano56  1,0102x        sigma_alto 1,0005x   plano56 1,0511x      7x1
seed 4242   plano56  1,0188x        plano56    1,0116x   plano56 1,0438x      5x3
```

O plano ganha no geral nas tres. **Em sigma alto, que e onde a ponderacao foi desenhada para
ganhar, a direcao inverte entre sementes na casa de 1% -- ou seja, nada.** Em sigma baixo perde
uns 4,5%, estavel. Todo o ganho aparente eram as tres camadas de 8 bits a mais.

`ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` de novo, com o eixo segurado sendo o orcamento de
promocao. Regra que fica: **igualar o orcamento antes de comparar dois criterios de selecao.**

### Nao coberto

Um prompt, um modelo, tres sementes, sem metrica perceptual. `sigma` e `sigma2` foram medidos como
criterio mas so `high` virou checkpoint e foi amostrado -- `sigma2` e mais suave (81% da massa na
metade alta contra 100% do `high`) e nao foi para a ponta a ponta. E o BF16 continua sendo o alvo,
nao a verdade.

## 2026-08-31, parte 32 - destravar um text encoder: quanto custa, quanto rende, e a medicao que dependia de sorte

Continuacao direta da parte 29. Executado na 3090 sob o lock `w4a4:te_lock_cost`.

### Primeiro, item fechado: nada nesta instalacao liga `_FORCE_INT4_INT8_FALLBACK`

Varredura da arvore inteira: as unicas ocorrencias sao a leitura em
`comfy_kitchen/backends/cuda/__init__.py:212` (mais a copia em `_backup_20260816_preupgrade/`),
documentacao e os probes deste projeto. Conferidos tambem os pontos cegos de um grep: variaveis de
ambiente de usuario e de maquina (nenhuma), arquivos `.env` (nao existem) e atribuicoes dinamicas a
`os.environ[...]` (a unica que casa com "fallback" e `PYTORCH_ENABLE_MPS_FALLBACK`, de outro
pacote). **Na sm86 o ramo nativo sempre roda.**

### O erro de metodo, que vale mais que os numeros

A parte 29 destravava escrevendo `comfy_force_cast_weights = False` nos modulos. Isso e apagado:
`model_patcher.py:1016` faz `m.comfy_force_cast_weights = self.force_cast_weights` em **cada
modulo, cada vez que o modelo sobe para a GPU**. A escrita so sobrevive se o modelo ja estava
residente -- o que depende do estado da VRAM naquele instante.

**A mesma linha de comando deu 350 chamadas ao kernel numa execucao e 0 na seguinte.** O que pegou
foi o contador de dispatch: ele imprimiu `convrot 0` e `dequantize 252` no braco que deveria estar
destravado. Um probe que so comparasse saidas teria concluido "destravar nao muda nada" -- com os
dois bracos identicos bit a bit, o que e exatamente a aparencia de um resultado limpo.

Forma correta: `clip.patcher.force_cast_weights = False` na FONTE, mais remover
`manual_cast_dtype` dos object patches; `patch_model` propaga. `_full_precision_mm` continua sendo
escrito no modulo, porque esse `patch_model` nao toca. E o probe agora le o estado **depois** do
primeiro load, nao antes -- antes ele reportava um estado que ainda ia ser sobrescrito.

### Os tres encoders, medidos pelo metodo confiavel

`tools/probe_te_lock_cost.py`. Tres bracos onde ha referencia BF16 (A bf16, B travado, C
destravado) e dois onde nao ha. Mediana de 5 encodes, apos dois aquecimentos.

```
encoder                  formato  quant     travado  destravado   tempo               C-vs-B    cos
qwen_3_4b (4B)             W4A4   2,4 GiB    70,2 ms    82,9 ms   1,18x MAIS LENTO   5,99e-1  0,949
gemma_3_12B_it_heretic     W4A8   8,1 GiB  1772,3 ms   478,9 ms   3,70x mais rapido  2,11e-1  0,982
qwen3vl_32b_minimax_h3     W4A4  13,2 GiB   311,6 ms   117,7 ms   2,65x mais rapido  9,73e-2  0,99989
```

Dispatch conferido em cada braco: travado sempre `dequantize = n_camadas`, destravado sempre
`convrot`/`w4a8 = n_camadas`, zero do outro lado.

O `qwen_3_4b` foi quantizado aqui (`tools/quant_w4a4.py --profile qwen`, 8,04 GiB -> 2,42 GiB, 252
camadas) justamente porque tem gemeo BF16 no disco. So nele da para separar as duas metades do
custo: **o peso de 4 bits sozinho ja custa 1,44e-1** contra o BF16 (cos 0,9896), e destravar leva a
6,09e-1 (cos 0,9492) -- **4,23x**.

### Duas coisas que um modelo so teria escondido

**Destravar pagou nos dois encoders maiores e perdeu no menor** -- e o POR QUE nao esta
estabelecido. A primeira versao deste paragrafo afirmava um mecanismo que nao foi medido ("o
caminho travado materializa o peso inteiro a cada encode, entao o custo escala com o tamanho do
peso enquanto a contagem de tokens fica minuscula"). Os proprios numeros recusam essa historia: os
tempos travados sao 70,2 ms com 2,4 GiB, **1772,3 ms com 8,1 GiB** e 311,6 ms com 13,2 GiB -- nao e
monotonico no tamanho do peso. E os tres casos diferem tambem em comprimento de sequencia por
ordens de grandeza (o condicionamento do Gemma e `[1, 49, 1024, 3840]`, o do MiniMax e
`[1, 8, 5120]`), entao tamanho e M andam juntos aqui e estes dados nao separam os dois. Fica como
observacao, nao como explicacao, ate alguem variar um eixo sozinho -- o teste obvio e repetir o
Qwen com um prompt longo.

**O custo em precisao nao e propriedade do formato.** Dois arquivos W4A4 diferem por 6x no erro que
destravar adiciona (5,99e-1 contra 9,73e-2). "W4A4 custa X" nao pode ser citado sem dizer qual
checkpoint.

### E o que os publicos da Abiray fazem, lido camada a camada

Nao pelo campo de resumo: pelos 200 tensores `.comfy_quant` que estao no proprio header.

```
117 convrot_w4a4  com linear_dtype "int8"      83 int8_tensorwise  (sem linear_dtype)
```

E o resumo do arquivo se contradiz: `convrot_w4a4_mixed` traz `"linear_dtype": "int4"` no topo e
`"w4a4_int4mm_layers": 0` tres chaves abaixo. Quem ler o campo de topo conclui int4; **zero camadas
usam MM de int4**. Como sao modelos de DIFUSAO, eles nao pegam a trava do text encoder: aquelas 117
camadas executam matematica quantizada, pelo ramo INT8 -- que a medicao desta bancada em 30/08 diz
ser 1,49x mais fiel que o nativo. E uma escolha, e os dados daqui a sustentam.

### Nao coberto

Um prompt por encoder (tres no Qwen), uma placa sm86, sem metrica perceptual, sem gerar imagem --
mede o CONDICIONAMENTO, nao o resultado. Nao ha referencia BF16 para o Gemma (o fonte de 23,5 GiB
foi apagado) nem para o MiniMax, entao nesses dois so existe o delta C-contra-B. E destravar
continua sendo monkeypatch: `custom_operations` em `model_options` e a unica saida real, e nenhum
no a expoe.
## 2026-08-31, parte 33 - os publicos de difusao despacham quantizado, e o instrumento que mediu isso estava errado tres vezes

Executado na 3090 sob o lock `w4a4:te_lock_cost`.

### Os dois arquivos mais baixados da Abiray nao abrem sem dynamic-VRAM

Ambos carregam bytes DEPOIS do ultimo tensor -- 83 no FL2VA, 64 no Ref2VA -- e
`safetensors.safe_open` recusa:

```
SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

Nao e download quebrado, e conferir custou dois `curl`: o `Content-Length` do servidor bate com o
nosso byte a byte (15903012791 e 15093774276) e um `Range: bytes=-83` devolve exatamente a mesma
cauda que temos. A do FL2VA e texto legivel --
`\nL2P_bypass_MiniMax_H3_FL2VA_..._convrot.safetensors_1785789862\n` -- e a do Ref2VA sao 64 bytes
binarios. **Estao no arquivo publicado.**

O caminho do dynamic-VRAM usa outro leitor (`comfy/utils.py:85`, `comfy_aimdo.model_mmap`) e
aceita: com o aimdo inicializado os dois abrem, 932 e 1132 tensores. Ou seja, funcionam para quase
todo mundo -- dynamic VRAM e o default -- e falham exatamente na configuracao que esta bancada
precisa para Nunchaku e LTX 2.5. O `minimax_h3_fl2va_pruned-w4a8_convrot_pruned` do Winnougan nao
tem cauda e abre dos dois jeitos.

### O instrumento errou tres vezes, e as tres valem mais que o resultado

**1. Imprimiu veredito sobre um forward que nao rodou.** O MiniMax H3 e audio-video e quer uma
LISTA de latentes; o probe passou um tensor so e morreu em `audio_src = x[1]`. Ele imprimiu
`ZERO forwards quantizados` mesmo assim. Contador zerado por nao ter executado e contador zerado
por ter executado dequantizado sao o mesmo numero com significados opostos. Agora ele recusa dar
veredito quando `rodou` e False.

**2. Chamou os modulos na CPU.** O modo `--forward-only` chama Linear do modelo carregado com
entrada sintetica -- e nada tinha subido o modelo para a GPU. Os pesos ficaram no device de
offload, o registry do comfy-kitchen resolveu para o backend **eager**, e o A/B de
`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` virou no-op, porque essa flag so existe no backend CUDA.
Os quatro "8/8 quantizado" da primeira rodada mediram o backend errado. O que denunciou foi um
`ValueError: Expected a cuda device, but got: cpu` ao instrumentar a chamada -- ate ali o probe
estava alegre e coerente. Agora ele chama `load_models_gpu` antes e reporta o device dos pesos.

**3. A impressao digital era grossa demais.** Quatro elementos em bf16 batiam entre os dois ramos
ate no Z-Image, que sabidamente muda de kernel com a flag. Trocada por soma, norma e absmax sobre
o tensor inteiro.

E a correcao de fundo, que e a mesma da AUDITORIA item 18: **contar "quantizado" nao diz qual
kernel rodou.** O probe agora grava tambem `impl:<op>=<modulo>` -- a implementacao que o registry
escolheu -- e o `linear_dtype` que chegou no despachante.

### Validacao do instrumento, no Z-Image

```
flag=0  pesos cuda:0  impl comfy_kitchen.backends.cuda  linear_dtype=int4
        FP [-14911.6582, 16050.6924, 160.0] ...
flag=1  pesos cuda:0  impl comfy_kitchen.backends.cuda  linear_dtype=int4
        FP [-17727.7383, 15852.9805, 178.0] ...
```

Fingerprints diferem, entao os dois ramos sao kernels distintos e o Z-Image roda o nativo.

### Qual ramo cada checkpoint toma, medido e nao lido

Mesmo probe, `--forward-only`, pesos na GPU, um eixo variando (`FORCE_INT4_INT8_FALLBACK`):

```
checkpoint                                linear_dtype  impl                          flag muda?  ramo
zimage-v2-w4a4                (nosso)         int4      comfy_kitchen.backends.cuda      SIM      nativo int4
LTX25-distilled-DiT-comfy-w4a4 (riftcast)     int4      comfy_kitchen.backends.cuda      SIM      nativo int4
MiniMax_H3_FL2VA        (Abiray, 791k dl)     int8      comfy_kitchen.backends.cuda      NAO      INT8, por instrucao
minimax_h3_fl2va-w4a8      (Winnougan)         --       comfy_kitchen.backends.cuda      n/a      w4a8, outro kernel
```

A linha da Abiray e a que fecha tres semanas de duvida: o arquivo **executa matematica quantizada
no backend CUDA**, e nao muda nada quando se desliga o MMA de 4 bits, porque aquelas camadas nunca
tomam esse ramo. `linear_dtype: "int8"` gravado camada a camada -- e o resumo do proprio arquivo se
contradiz, com `"linear_dtype": "int4"` no topo e `"w4a4_int4mm_layers": 0` tres chaves abaixo.
**Nao e um atalho: e a escolha que a medicao desta bancada em 30/08 diz ser 1,49x mais fiel.**

E aparece um segundo W4A4 publico que roda 4 bits de verdade: `LTX25-distilled-DiT-comfy-w4a4`,
`quantized_by: riftcast/ltx25-quant-lab`, 1440 camadas `convrot_w4a4` com `linear_dtype` ausente.
**Setimo escritor conhecido de checkpoint** para o ticket 08.

### Nao coberto

Sem SASS: "ramo nativo" continua significando "produz numero diferente do fallback". Tres camadas
por checkpoint no `--forward-only`, entrada sintetica gaussiana, M=256, uma placa sm86. O
`--forward-only` prova que ESTES modulos, como o loader os deixou, despacham assim -- nao que uma
geracao inteira o faca; para o Z-Image as duas coisas foram medidas e concordam (340/340 no
sampler), para os outros nao. E nao ha nenhuma afirmacao de fidelidade aqui: nenhum destes tem
gemeo BF16 nesta bancada.

## 2026-08-31, parte 34 - o que decide se destravar um text encoder paga e o COMPRIMENTO DO PROMPT

A parte 32 mediu tres encoders e viu destravar perder no menor e ganhar nos dois maiores. O
paragrafo que escrevi para explicar isso afirmava um mecanismo que **nao tinha sido medido** -- que
o custo do caminho travado escala com o tamanho do peso enquanto a contagem de tokens fica
minuscula. Os proprios numeros ja recusavam: 70,2 ms com 2,4 GiB, **1772,3 ms com 8,1 GiB** e 311,6
ms com 13,2 GiB nao e monotonico em tamanho. E os tres encoders diferiam em tamanho **e** em
comprimento de sequencia ao mesmo tempo (o condicionamento do Gemma e `[1, 49, 1024, 3840]`, o do
MiniMax e `[1, 8, 5120]`), entao aquela tabela nao separava os dois eixos.

Separado agora: mesmo arquivo, mesma placa (3080 Ti), so o prompt muda.

```
qwen_3_4b W4A4, mediana de 3
tokens   travado  destravado
    22     80,7      120,1    1,49x MAIS LENTO
    75    100,1      105,8    1,06x MAIS LENTO
   199    151,9       95,2    1,60x mais rapido    <- cruza entre 75 e 199
   424    245,2      102,0    2,40x
   850    456,1      124,3    3,67x
  1496    824,5      249,0    3,31x
```

**O cruzamento fica entre 75 e 199 tokens.** O caminho destravado quase nao se move de 22 a 424
tokens (120 -> 102 ms) enquanto o travado sobe com a sequencia: o kernel de 4 bits carrega um custo
fixo por camada que prompt curto nao amortiza, e o caminho dequantizado paga um GEMM BF16 que
cresce com os tokens. A curva do `m_crossover` de novo, agora do lado do text encoder.

Consequencia pratica: prompt de verdade costuma estar acima do cruzamento. Em 850 tokens o encoder
quantizado e destravado bate ate o **BF16 original** (372,1 ms perto de 700 tokens contra 124,3 ms).

Tamanho do peso pode continuar importando por cima disso -- o MiniMax de 13,2 GiB ganhou com 8
tokens -- mas esse eixo **nao** foi isolado.

### O erro de metodo, de novo e por pouco

A primeira tentativa de isolar o eixo rodou o prompt curto na 3090 e o longo na 3080 Ti. Dois eixos
mexendo. Refeito na mesma placa antes de qualquer conclusao: 62,9 / 82,1 / 96,2 ms no curto contra
372,1 / 375,3 / 95,6 no longo, e ai sim a inversao de sinal e do prompt.

### Nao coberto

Um modelo (qwen_3_4b W4A4), uma placa (3080 Ti), mediana de 3 por ponto, um unico texto cortado em
comprimentos crescentes -- os prompts nao sao independentes, sao prefixos do mesmo. Nao mede
qualidade: o erro do condicionamento cresce com o comprimento (o peso quantizado sozinho custa
1,44e-1 no curto e 2,55e-1 no longo) e isso nao foi investigado. E nao diz onde fica o cruzamento
em outro modelo ou outra placa.

## 2026-08-31, parte 35 - oito sementes dizem que os tres criterios sao indistinguiveis, e isso corrige a parte 31

A parte 31 comparou dois bracos (plano contra ponderado por sigma alto) em tres sementes, o plano
ganhou nas tres, e o texto tratou isso como achado. Com o terceiro braco (`sigma2`) e mais cinco
sementes, o ranking muda de semente para semente.

Desenho pareado -- dentro de uma semente os tres bracos veem a MESMA trajetoria imposta pelo BF16,
entao a comparacao que vale e a diferenca por semente, nao a media de cada braco solta. Todos os
tres com o **mesmo orcamento**, 56 camadas promovidas.

```
epsilon medio, 8 sementes     media       min       max    espalhamento entre sementes
plano56                     1,3326e-1  1,1220e-1 1,5577e-1        1,388x
sigma2                      1,3684e-1  1,1731e-1 1,5216e-1        1,297x
sigma_alto                  1,3685e-1  1,2276e-1 1,4695e-1        1,197x

diferenca pareada contra o plano:  sigma2     +3,37%  (erro-padrao 3,95%,  vence 4/8)
                                   sigma_alto +3,46%  (erro-padrao 3,93%,  vence 2/8)
```

**O espalhamento entre sementes dentro de um unico braco chega a 1,388x; a diferenca entre criterios
e 3,4%.** O efeito esta dentro do proprio ruido. Vencedor por semente: plano56 em 1234, 31337 e 555;
sigma2 em 5678, 4242 e 8080; sigma_alto em 777 e 90210. Tres, tres e dois.

O sinal e consistente -- os dois ponderados saem um pouco PIORES, nunca melhores -- mas com oito
sementes isso e uma pista, nao um resultado.

**A licao e sobre tamanho de amostra, e o "3/3" da parte 31 e o exemplo.** Duas coisas conspiraram:
so dois bracos (nao da para ver o ranking trocar com dois) e tres sementes contra um ruido de 39%.
Um criterio de parada escrito antes de olhar teria pedido mais sementes; nao havia.

### E o encoder destravado toma mesmo o ramo nativo

Fechando a simetria com o lado da difusao: `qwen_3_4b_w4a4_convrot` destravado, mesma placa, um eixo
(`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, agora passado pelo probe e nao herdado do ambiente):

```
destravado nativo x destravado fallback-int8   rel-RMSE 1,384   cosseno 0,961   identicos? NAO
controle: travado (dequantizado) x destravado nativo   rel-RMSE 0,599
```

Sao kernels distintos. O caminho do text encoder, uma vez destravado, chega ao MMA de 4 bits.

### Nao coberto

Um modelo, um prompt, um scheduler, oito sementes. Nao e teste formal de hipotese: e a distancia
entre o efeito e o proprio espalhamento dele, que e o minimo para nao chamar de resultado uma
diferenca que troca de sinal entre sementes. E `sigma` (linear) foi implementado e testado como
aritmetica, mas nunca virou checkpoint.

## 2026-08-31, parte 36 - o aviso publico de agosto reproduziu, e o que ele mede e o MODELO

O repositorio publico deste projeto -- renomeado hoje de `ComfyUI-ConvRot-Quant` para
`comfy-quant-bench`, ver abaixo -- carrega desde 16 de agosto uma manchete em negrito:
**"Use `tools/quant_w4a8.py`. Do not use W4A4."**, apoiada numa foto de tres macas onde a do meio,
ConvRot W4A4, esta destruida. Modelo: HunyuanVideo 1.5, 480x480, um quadro, seed 12345, 6 passos,
cfg 6, euler/simple.

Desde entao esta bancada mediu o oposto no Z-Image: W4A4 1,83x-1,93x mais rapido por passo, 3,6x
mais leve, imagem boa (parte 29 e o portao de aceitacao). Duas afirmacoes publicas incompativeis
sobre o mesmo formato.

### O que se moveu entre as duas, e o que nao

A primeira hipotese foi o `to_native.py`: o ComfyUI funde `attention.to_{q,k,v}` em `attention.qkv`
na carga e so `.weight` esta no mapa de renome, entao `weight_scale` passa sem renomear e a camada
carrega **sem escala e sem erro** -- exatamente a cara daquele dano.

**Essa hipotese morreu antes de custar uma janela de GPU**, e morreu lendo o comentario que o
proprio perfil carrega (`tools/quant_w4a4.py`, perfil `hunyuan_video_15`):

```
# HunyuanVideo.process_unet_state_dict rewrites them ... and its substring replacements
# ("_attn_qkv." -> "_attn.qkv.", ...) carry the injected .comfy_quant and .weight_scale
```

O HunyuanVideo tem o proprio `process_unet_state_dict` e ele **carrega as escalas junto**. O buraco
do `to_native.py` e especifico do Z-Image (nomes diffusers). Nao explica nada aqui.

Sobrou a pilha. Diferenca real entre 16 e 31 de agosto:

```
comfy-kitchen   0.2.23  ->  0.2.31
ComfyUI         0.29    ->  0.33 (c1739380)
torch           2.12.1  ->  2.13.0+cu130
conversor       praticamente o mesmo codigo (flag nova, sonda extraida para _native_probe.py)
```

### EXECUTADO: requantizado e re-renderizado hoje, 3090

Requantizacao com `tools/quant_w4a4.py --profile hunyuan_video_15`. O sidecar registra
`backend: comfy_kitchen.backends.cuda`, `convrot_groupsize: 256`, 432 tensores quantizados, 929
preservados, **saida de 7,92 GiB -- identica a de agosto.** Mesmo perfil, mesmas camadas, mesmo
group size.

Render com `tools/quality_ladder.py`, `CUDA_VISIBLE_DEVICES=0`, exatamente os parametros do
`SMOKE_HunyuanVideo15_W4A4.json` de agosto:

```
                    s/passo    GiB   divergencia   imagem
FP16 (controle)       1,858   15,51            -   boa
ConvRot W4A4          1,961    7,92       0,8255   DESTRUIDA
```

1,055x **mais lento** que o FP16, e destruida. As duas metades do aviso de agosto reproduziram numa
pilha tres versoes mais nova. **`comfy-kitchen` 0.2.31 nao consertou nada.**

### O que isso realmente mede

Nao e bug de pilha e nao e bug de conversor. E o modelo:

| modelo | W4A4, por passo | imagem |
|---|---|---|
| Z-Image | 1,83x-1,93x **mais rapido** | boa |
| HunyuanVideo 1.5 | 1,055x **mais lento** | destruida |

Mesmo formato, mesmo kernel, mesmo conversor, mesmo group size. O que separa e qual modelo entra.

Consequencia para o README publico: **nao e retratacao, e escopo.** A frase
*"ConvRot W4A4 destroys HunyuanVideo 1.5"* fica, e fica mais forte -- foi reproduzida quinze dias
depois. A frase *"Do not use W4A4"* e generalizacao de um modelo so, e o Z-Image e o contraexemplo
medido.

**Vale registrar por que isso nao foi publicado antes de rodar.** A tentacao era virar a manchete
com a foto boa do Z-Image na mao. Se tivesse feito isso, o repositorio publico estaria hoje
afirmando que o W4A4 foi consertado -- falso, e falso num repo cujo log de commits e feito de
retratacoes cuidadosas (`Retract the smoothing recommendation`, `correct two claims it refutes`).
E `ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` outra vez: dois bracos, quatro eixos movidos.

### Divergencia de latente nao tem limiar

```
HunyuanVideo W4A4   divergencia 0,8255   ->  destruida
Z-Image      W4A4   divergencia 0,7173   ->  boa
```

Dois numeros proximos, desfechos opostos. Terceira vez que esta bancada bate nisso: **distancia nao
e qualidade**, e nenhum corte nesse eixo separa "funciona" de "nao funciona". So a imagem decidiu.

### Nao coberto

Uma semente, um prompt, 480x480, um quadro. Sem SASS -- "kernel nativo" aqui significa que o
registry resolveu `comfy_kitchen.backends.cuda` e o sidecar gravou, nao que a instrucao foi
observada emitindo. O s/passo inclui carga sob dynamic VRAM nos dois bracos, nao isolada. Nenhuma
metrica perceptual. Nao foi testado nenhum group size alem de 256 nesta rodada (agosto testou 256,
64 e 16 e reportou os tres inutilizaveis).

## 2026-08-31, parte 37 - o gemeo BF16 chegou, e o ramo INT8 ganha pela terceira vez seguida

`tools/probe_winnougan_int4.py` fechava toda execucao com esta ressalva, escrita por ele proprio:
*"NAO mede fidelidade contra o BF16 original: o BF16 deste modelo nao esta aqui. Diz qual ramo
executa e quanto custa, nao qual erra menos."* Baixado hoje:
`Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, **51.506.295.256
bytes (47,97 GiB)**, que bate ao byte com o publicado. **351 de 351** nomes de camada casam com o
arquivo quantizado do Winnougan. A ressalva caiu, e o item do handoff marcado `bloqueado` deixou de
estar.

### EXECUTADO: `tools/probe_winnougan_fidelidade.py`, 3090

Um eixo (`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, um subprocesso por braco). Referencia
`F.linear(x, W_bf16)` em float32, peso do gemeo lido por faixa de bytes. 20 camadas, 5 formas
distintas, 4 profundidades, M em {1, 64, 1024}:

```
media rel-RMSE contra BF16   nativo 2,3415e-1   int8 1,6737e-1
int8 e 1,40x mais fiel
vitorias por camada-M: nativo 0, int8 60, empate 0
```

**Sessenta de sessenta.** E agora sao tres medicoes independentes na mesma direcao:

| medicao | modelo | quem quantizou | ativacao | resultado |
|---|---|---|---|---|
| erro por camada (parte 27) | Z-Image, difusao | nos | real | int8 1,49x, 24/24 |
| epsilon por passo (parte 29) | Z-Image, difusao | nos | real | int8 1,33x, 8/8 |
| erro por camada (aqui) | Qwen3-VL-32B, text encoder | **Winnougan** | sintetica | int8 1,40x, 60/60 |

Familia de modelo diferente, quantizador diferente, ativacao diferente. O ramo INT4 do ConvRot e
consistentemente menos fiel que o ramo INT8, e a razao fica entre 1,33x e 1,49x nas tres.

### O erro nao cresce com a profundidade

```
mesma forma, bloco 0 -> bloco 49
  nativo  2,2776e-1 -> 2,2762e-1
  int8    1,6010e-1 -> 1,6002e-1
```

Quatro casas iguais entre o primeiro e o quinquagesimo bloco. Nenhum acumulo ao longo do modelo,
pelo menos medindo camada isolada -- o que nao diz nada sobre acumulo na ATIVACAO, que este probe
nao propaga.

### A sonda estava errada, e o comentario dela dizia como

A primeira execucao mediu **cinco camadas, todas de `model.layers.0`**, e deu 1,39x, 15/15. O
`pick_layers` pegava "uma camada por forma distinta", com um comentario explicando que isso era
melhor que pegar as N primeiras *"porque as N primeiras seriam todas do bloco 0"*. Num transformer
as formas se repetem bloco a bloco, entao parar na primeira ocorrencia de cada forma **e** parar no
bloco 0: a funcao fazia exatamente o que o proprio comentario dizia evitar. A forma variava e a
profundidade ficava presa -- `teste-varia-o-eixo-errado`, dentro do instrumento.

O primeiro conserto tambem estava errado: passo fixo a partir do inicio dava 0/10/20/30 num modelo
de 50 blocos, deixando os ultimos dezenove blocos inteiros fora da amostra. Agora e linspace
**fechado nos dois extremos**, entao o primeiro e o ultimo bloco entram sempre.

O numero quase nao mudou (1,39x -> 1,40x), mas isso e sorte deste modelo, nao defesa do metodo: o
resultado antigo era sobre um bloco e estava escrito como se fosse sobre o modelo.

### Nao coberto

Ativacao gaussiana sintetica, sem os outliers que a rotacao existe para suprimir -- e este eixo ja
mudou de resposta entre sintetico e real nesta bancada (1,4x -> 1,49x na parte 27). Sem SASS. Erro
por CAMADA, que preve o erro de predicao do modelo mas nao preve a imagem final. Uma placa, sm86.
Nao carrega nada no ComfyUI: prova o kernel, nao o loader. E o BF16 do publicador e o alvo, nao a
verdade -- nunca foi validado contra float32 aqui.

## 2026-08-31, parte 38 - por que o Z-Image aguenta o W4A4 e o HunyuanVideo nao: e o PESO

A parte 36 deixou uma pergunta aberta e escrita como aberta: o mesmo formato, o mesmo kernel e o
mesmo conversor produzem imagem boa no Z-Image e lixo no HunyuanVideo 1.5. Tres hipoteses foram
testadas hoje. **Duas morreram, e as duas morreram na direcao contraria a esperada.**

### Hipotese 1: "o Z-Image aguenta porque quantiza menos". MORTA, ao contrario

Contagem de parametro pelo cabecalho, sem GPU, reconstruindo o numel real das camadas empacotadas:

```
                params totais   quantizados        %   camadas quantizadas
Z-Image W4A4          6,155 B       6,016 B    97,7%                  170
Hunyuan W4A4          8,327 B       5,436 B    65,3%                  432
```

O Z-Image e o mais agressivamente quantizado dos dois **por muito** -- 97,7% dos parametros em 4
bits -- e e o que sobrevive. O Hunyuan deixa um terco do modelo em FP16 e quebra assim mesmo.

### Hipotese 2: "as ativacoes do Hunyuan tem outliers piores". MORTA, ao contrario

E era a hipotese que o proprio README acusa: o caminho de ativacao do ConvRot e um absmax por
token cobrindo todos os canais, 15 niveis, entao um canal outlier fixa a escala do vetor inteiro.

`tools/probe_por_que_zimage_aguenta.py`, sobre as ativacoes REAIS ja capturadas em 2026-08-19
(nada recapturado, nenhuma GPU), mediana sobre as camadas:

```
                                   Z-Image   Hunyuan
crest por token, SEM rotacao       19,1121    5,8893   Z-Image 3,25x PIOR
crest por token, COM rotacao        4,1642    3,9638   praticamente igual
pior canal / canal mediano         57,4915    4,0708   Z-Image 14,12x PIOR
fracao de ativacoes no codigo 0     0,2534    0,2324   Z-Image pior
erro int4 uma escala por token      0,1760    0,1682   Z-Image pior
```

O Z-Image tem ativacao **dramaticamente mais feia** -- canal outlier quatorze vezes pior -- e e ele
que sobrevive.

### Hipotese 3: e o PESO. Sobreviveu, e e a unica que separa

Das analises por camada de 2026-08-19, medidas com os kernels reais nas ativacoes reais:

```
                          Z-Image   Hunyuan
mediana err_w4a8 (base)    0,0394    0,0695   Hunyuan 1,76x pior
mediana err_w4a4           0,1241    0,2136   Hunyuan 1,72x pior
mediana da razao a4/a8     3,1670    3,0466   IGUAL, 4% de diferenca
```

**O custo de descer a ativacao para 4 bits e o mesmo nos dois modelos.** O que difere e de onde
eles partem:

```
Z-Image   0,0394 x 3,17 = 0,125   fica abaixo da linha de uso
Hunyuan   0,0695 x 3,05 = 0,212   passa
```

`err_w4a8` e peso de 4 bits com ativacao de 8, e o proprio README ja mediu que nessa configuracao o
peso domina por ~8x (0,0731 contra 0,0092). Entao esse numero e um medidor de qualidade do PESO, e
o peso do Hunyuan e 1,76x pior.

A distribuicao inteira acompanha, e nao e cauda -- e o corpo:

```
camadas com err_w4a4 > 0,15   Z-Image  57/170 (33,5%)   Hunyuan 408/432 (94,4%)
camadas com err_w4a4 > 0,25   Z-Image   3/170 ( 1,8%)   Hunyuan 140/432 (32,4%)
```

`0,15` nao e um numero escolhido depois: e o `--promote-error` **padrao do `quant_mixed.py`**. O
criterio desta propria bancada diz que 94,4% do HunyuanVideo deveria ser 8 bits, e nos rodamos ele
100% em 4.

Os piores sao estruturais, nao aleatorios: os oito piores do Hunyuan sao **todos**
`double_blocks.NN.img_mlp.fc1`, blocos 31 a 43, uma faixa contigua. Por tipo, `img_attn_qkv` tem
mediana 0,3105 sobre 54 camadas; o pior tipo do Z-Image e `w3` a 0,1782.

### A resposta ja estava no repositorio publico, escrita em agosto

`comfy-quant-bench`, secao "The matrix: which axis actually decides", 2026-08-16:

> So the weight quantizer, not the activation precision, decides whether the model works.

Foi medido no **Gemma**, com uma bateria de perguntas, e nunca foi aplicado ao lado da difusao.
Aplicado hoje, da a mesma coisa.

E esta na primeira FOTO do repositorio: as tres macas de agosto sao FP16 boa, W4A4 destruida,
**W4A8 boa**. W4A8 e o mesmo peso de 4 bits com ativacao de 8 -- funciona no Hunyuan exatamente por
isto, a base de 0,0695 sem o multiplicador de 3x fica abaixo da linha.

### Nao coberto

Nao explica **por que** o peso do Hunyuan e 1,76x pior; mede que e. As duas calibracoes tem passos,
resolucao e execucoes diferentes (8/1024/4 contra 6/512/2) -- a razao `a4/a8` e interna a cada
modelo e sobrevive a isso, as medianas absolutas comparadas entre modelos sobrevivem menos. A
rotacao usada no probe de ativacao e uma Hadamard normalizada do proprio arquivo, nao a do kernel:
a escala dos numeros pode diferir, a comparacao entre os dois modelos nao, porque ambos passam pela
mesma. E mede a entrada de cada camada isolada, nunca o acumulo pelo residual.

## 2026-08-31, parte 39 - a linha de uso existe, e esta entre 0,1837 e 0,2147

A parte 38 concluiu que o que separa o Z-Image do HunyuanVideo e o PESO: mesma penalidade por
descer a ativacao para 4 bits (razao a4/a8 de 3,17 contra 3,05), bases diferentes (`err_w4a8`
mediano 0,0394 contra 0,0695). A previsao que isso faz e testavel: se o que decide e o **nivel
absoluto** de erro por camada, entao puxar o erro do Hunyuan para baixo deve trazer a maca de volta,
e existe um nivel onde isso acontece.

O criterio foi escrito antes de rodar, em `bench/criterio_hunyuan_misto.md`, e incluia o desfecho
que me contradiria (a maca voltar ja em 0,40, o que significaria que manda a cauda e nao o corpo da
distribuicao) e o que mataria a teoria (nao voltar em nenhum).

### Primeiro: os guardas da bancada barraram o atalho, duas vezes

`quant_mixed.py` recusou a analise de 2026-08-19 e depois a calibracao, as duas por
`carries no 'source_identity_sha256'`, com a mensagem *"'nothing to check' is not 'checked'"*. Sao
anteriores as chaves de proveniencia. Custou uma recaptura e estava certo: eu ia construir tres
checkpoints a partir de numeros que ninguem consegue amarrar ao arquivo de entrada.

**A recaptura virou uma replicacao independente da parte 38**, e replicou:

```
                med_w4a4   med_w4a8   razao a4/a8   acima de 0,15 / 0,25 / 0,40
agosto            0,2136     0,0695        3,0466        408 / 140 /  31
hoje              0,2230     0,0708        3,0207        408 / 150 /  30
```

Prompt, resolucao, passos e semente diferentes, doze dias e tres versoes de pilha de distancia, e a
razao `a4/a8` -- o numero em que a conclusao se apoia -- moveu **0,9%**.

### EXECUTADO: tres builds mistos, mesma maca, 3090

```
build              4 bits / 8 bits   mediana do erro EFETIVO   imagem
W4A4 puro              432 /   0                     0,2230    DESTRUIDA (papa)
misto  --promote 0,40  402 /  30                     0,2147    destruida, ja com forma de maca
misto  --promote 0,25  282 / 150                     0,1837    CORRETA, granulada
misto  --promote 0,15   24 / 408                     0,0739    CORRETA, limpa
Z-Image W4A4 (ref)     170 /   0                     0,1241    CORRETA
```

"Erro efetivo" e, para cada camada, o erro do formato que ela **realmente recebeu** -- `err_w4a4`
nas que ficaram em 4 bits, `err_w4a8` nas promovidas.

**A linha de uso fica entre 0,1837 e 0,2147.** O desfecho foi o segundo do criterio, que era o
melhor dos tres vivos: a maca volta com **65% do modelo ainda em 4 bits**.

E dentro do "correta" a qualidade acompanha o erro monotonicamente: 0,1837 granulada, 0,1241 boa,
0,0739 limpa. Nao e binario.

### A regra de bolso que isto entrega, e e a parte util

O erro por camada sai da **calibracao**, antes de converter e sem renderizar nada. Entao:

```
mediana err_w4a4 > 0,21   ->  W4A4 puro vai quebrar
mediana err_w4a4 < 0,15   ->  W4A4 puro funciona
entre                     ->  funciona, com granulacao visivel
```

Isso transforma "converte e reza" em uma checagem previa. **Nao coberto:** dois modelos, uma
familia de arquitetura cada, uma semente. A linha e o intervalo entre duas medidas adjacentes, nao
um valor com incerteza estimada.

### Divergencia de latente errou a ordem outra vez, e agora com quatro pontos

```
                divergencia   imagem
W4A4 puro            0,8255   destruida
misto 0,40           0,7978   destruida
misto 0,25           0,3741   correta, granulada
misto 0,15           0,4246   CORRETA, LIMPA     <- divergencia MAIOR, imagem MELHOR
```

O build de melhor imagem tem divergencia pior que o anterior. Quarta vez que esta bancada mede que
distancia no latente nao ordena qualidade, e a primeira com uma serie de quatro pontos em vez de um
par.

### Nao coberto

Uma semente, um prompt, 480x480, um quadro, sem metrica perceptual -- o julgamento e olho humano
sobre "quebrou ou nao", que e o unico uso que esta bancada ja mediu que a imagem final tem. O ganho
pratico do misto 0,25 sobre o W4A8 puro e pequeno em disco (8,03 contra 8,24 GiB) e a imagem e
pior; o valor aqui e o numero da linha, nao o checkpoint. E a analise do Z-Image continua sendo a
de agosto, sem chaves de proveniencia: o lado do Hunyuan replicou, o do Z-Image ainda nao foi
refeito.

## 2026-08-31, parte 40 - os dois lados refeitos com proveniencia, e a conclusao nao se moveu

A parte 39 fechou com uma ressalva propria: o lado do Hunyuan tinha sido recapturado, o do Z-Image
continuava sendo o de 19 de agosto, sem chaves de proveniencia. Recapturado hoje, nas condicoes do
render dele (1024 px, 8 passos, cfg 1.0, semente 1234, o prompt da maca), mesmo tratamento dado ao
Hunyuan.

```
                 med w4a8   med w4a4   razao a4/a8   camadas > 0,15
Z-Image agosto     0,0394     0,1241        3,1670            33,5%
Z-Image hoje       0,0390     0,1216        3,2021            31,8%
Hunyuan agosto     0,0695     0,2136        3,0466            94,4%
Hunyuan hoje       0,0708     0,2230        3,0207            94,4%
```

Deriva maxima em qualquer campo: **4,4%**. A razao `a4/a8`, que e onde a conclusao se apoia, moveu
**+1,1%** no Z-Image e **-0,9%** no Hunyuan.

A conclusao recomputada usando **somente** numeros de hoje, mesma ferramenta, mesmo dia, os dois
lados com `source_identity_sha256`:

```
base (peso)     Z 0,0390   H 0,0708    Hunyuan 1,82x pior
razao a4/a8     Z 3,2021   H 3,0207    5,7% de diferenca
camadas > 0,15  Z  31,8%   H  94,4%
```

**A razao difere por 5,7% e a base por 82%.** E o argumento inteiro: os dois modelos pagam
praticamente o mesmo por descer a ativacao para 4 bits, e partem de lugares muito diferentes.

Os tres desfechos escritos antes de rodar como "me obriga a corrigir a parte 38 e o README publico"
-- a base do Z-Image subir perto de 0,0695, a razao dele se afastar de 3,05, ou os numeros nao
replicarem -- nenhum aconteceu.

### O crest continua sem explicar nada, pela terceira vez

A calibracao nova do Z-Image reporta crest p99 batendo **100** nas camadas `feed_forward.w2`,
contra ~26-31 nas piores do Hunyuan. Somado ao que a parte 38 ja tinha medido:

```
crest por token, mediana        Z 19,1   H  5,9    Z pior, Z funciona
pior canal / canal mediano      Z 57,5   H  4,1    Z pior, Z funciona
crest p99 do pior tipo          Z  100   H   31    Z pior, Z funciona
```

O modelo com a ativacao mais feia em todos os eixos e o que sobrevive. Isto e consistente com o que
esta bancada ja tinha medido e escrito no CLAUDE.md num contexto diferente: crest contra erro W4A4
da Spearman **+0,10** sobre 170 camadas. Aquele numero foi publicado como "nao escolha precisao por
crest"; aparece agora como "nao explique por crest qual modelo quebra". Mesma estatistica, mesma
inutilidade preditiva, dois usos distintos.

### Nao coberto

Continua sendo dois modelos, um por familia de arquitetura. As duas recapturas usaram cada uma as
condicoes de render do seu proprio modelo (1024/8/cfg 1,0 contra 480/6/cfg 6), o que e o certo para
representar cada um, mas significa que as medianas absolutas cruzadas entre modelos nunca foram
medidas sob condicoes identicas -- e nao podem ser, porque os dois nao rodam nas mesmas condicoes.
A razao `a4/a8` e interna a cada modelo e nao tem esse problema, que e por isso que a conclusao se
apoia nela.

## 2026-08-31, parte 41 - o "1,055x mais lento" era de uma execucao, e esta refutado

A parte 36 e o README publico afirmam que o ConvRot W4A4 e **1,055x mais lento** que o FP16 no
HunyuanVideo. Aquilo veio de **uma execucao de cada lado**, e a propria ferramenta imprimiu
`1 runs is a small sample for a quantity this noisy` na tela enquanto eu publicava.

Refeito com tres sementes, mesma placa, mesmo prompt:

```
                    s/passo (3 sementes)   antes (1 execucao)
FP16                              0,990                1,858
ConvRot W4A4                      0,536                1,961
                        1,85x MAIS RAPIDO     "1,055x mais lento"
```

**O sinal inverteu.** E o mesmo aconteceu com um numero que eu tinha chamado de "claramente
contaminado" sem medir: o `hunyuan15-misto-t021` deu `11,812 s/passo` numa execucao e `0,635` em
tres -- **18x** de diferenca.

### Quanto esta bancada realmente consegue medir de tempo

O MESMO par (Z-Image BF16 contra `zimage-v2-w4a4`) foi medido tres vezes, em condicoes diferentes:

```
1 semente, prompt da maca      1,50x
2 sementes, outro prompt       1,90x
3 sementes, prompt da maca     2,63x
```

A razao anda de **1,50x a 2,63x** para a mesma comparacao. Nao e ruido pequeno em volta de um
valor: e o valor nao estando determinado pelo que foi controlado. **Esta bancada nao sustenta duas
casas decimais em tempo**, e qualquer razao de velocidade daqui deve sair como faixa, com o numero
de execucoes ao lado.

O que NAO muda: as conclusoes cientificas nao dependiam de tempo. "W4A4 destroi o HunyuanVideo"
reproduziu em toda execucao (divergencia 0,8426, espalhamento 0,0664 em 3 sementes). A linha de uso
0,1837-0,2147 vem do erro por camada. "E o peso, nao a ativacao" vem da razao a4/a8, medida duas
vezes em cada modelo.

O que muda, e melhora: o W4A4 no Hunyuan **entrega a velocidade que promete** e destroi a saida.
Isso e mais limpo que "lento e quebrado".

### A licao, e ela e sobre confirmacao e nao sobre leitura

O aviso estava na tela. Nao passou despercebido por descuido: o numero **casava com o que agosto
dizia**, entao confirmacao de expectativa foi tratada como confirmacao de medida. Um numero que
contrariasse teria sido repetido antes de publicar.

## 2026-08-31, parte 42 - metrica de imagem existe agora, e ela recusa ordenar

`tools/metricas_imagem.py`: PSNR, SSIM, MS-SSIM, grao (desvio-padrao do laplaciano) e LPIPS
opcional. Ate hoje todo julgamento de imagem aqui era olho humano.

**Ela confirma o corte com folga.** Nos seis bracos do Hunyuan, os quebrados ficam num mundo a
parte -- MS-SSIM 0,26-0,29 e LPIPS 0,70-0,76 contra 0,62-0,78 e 0,43-0,56 dos bons. Nao e
gradiente, e abismo.

**E ela recusa ordenar os bons, discordando de mim e de si mesma:**

```
meu olho     t015 > t021 > t022
MS-SSIM      t021 > t022 > t015
LPIPS        t015 > t021 > t022
```

MS-SSIM e LPIPS invertem primeiro e terceiro.

**O caso que fecha isso** veio do portao do Z-Image. `zimage-v2-mixed-t0.20` tirou o **pior**
MS-SSIM (0,6537) e o **pior** PSNR (13,35) dos cinco, e a imagem dele e a melhor maca que esta
bancada ja gerou. Nao e sutil e nao precisa de interpretacao: a metrica poe em ultimo, o olho poe
em primeiro. E trajetoria livre indo para outro lugar, e o outro lugar tambem sendo otimo.

Quinta confirmacao independente de que **imagem de trajetoria livre responde "quebrou ou nao" e nao
responde "qual e melhor"** -- agora tambem contra metrica, e nao so contra olho.

### Portao nos cinco que nunca tinham sido testados

Todos passam: amostram ponta a ponta em 3 sementes e produzem imagem correta.

```
                  s/passo    GiB   divergencia (espalh.)   MS-SSIM (s1234)
BF16                0,889  11,46             -                    -
zimage-v2-w4a4      0,338   3,06   0,5131 (0,0726)            0,6680
zimage-v2-mixed     0,390   3,18   0,3529 (0,1594)            0,8121
mixed-t0.05         0,487   3,40   0,3372 (0,0956)            0,8489
mixed-t0.10         0,454   3,32   0,2996 (0,1519)            0,8497
mixed-t0.20         0,347   3,08   0,5202 (0,1573)            0,6537
```

O espalhamento da divergencia chega a 0,20 -- da ordem das diferencas entre os bracos. Nada aqui
separa um do outro.

### Nao coberto

Uma semente por linha de metrica (s1234), um prompt, uma placa. Nenhuma das metricas foi validada
contra julgamento humano nesta bancada. O `capybara_v0.1_w4a8` continua sem teste.

## 2026-09-01, parte 43 - terceira familia derruba a linha universal, e ela vira uma linha por modelo

Wan 2.1 VACE 1.3B, o unico ponto barato que arriscava a regra em vez de confirma-la dentro de
casa: perfil `wan_2_1` ja existia, checkpoint ja no disco, terceira arquitetura. Criterio escrito
antes em `bench/criterio_wan21.md`, com a previsao "mediana acima de 0,21, W4A4 puro quebra".

**A previsao errou.** Mediana `err_w4a4` = **0,1602** sobre 300 camadas, ativacao real
(`calib/wan21_2026-09-01.calib.pt`, recalibrado porque o de 2026-08-19 e anterior as chaves de
proveniencia e o `quant_mixed` o recusa, de proposito). Caiu na faixa do meio, 0,15-0,21, que
nenhum modelo tinha ocupado e onde a regra prevê "correta, granulada".

### A imagem contradiz a regra, e nao por pouco

Tres sementes, 25 passos, 33 quadros, 480x480, `vace_strength 0`, mesmo prompt e mesma semente nos
dois bracos:

| build | erro efetivo mediano | 4 bits / 8 bits | GiB | divergencia | resultado |
|---|---|---|---|---|---|
| referencia FP16 | - | - | 4,01 | - | oficina nitida, pessoa na bancada |
| `--promote-error 0,05` | 0,0546 | 2 / 298 | 2,15 | 0,2612 | **correta** |
| `--promote-error 0,15` | 0,0793 | 134 / 166 | 2,11 | 0,3022 | estrutura volta, tudo borrado, inutilizavel |
| W4A4 puro | 0,1602 | 300 / 0 | 2,07 | 0,3949 | destruida, sem sujeito |

**A linha do Wan fica entre 0,0546 e 0,0793.** O misto em 0,0793 ja e pior que o Z-Image em
0,1241, que sai bom.

### A regra da parte 39 nao e do formato, e do modelo

| modelo | parametros | tolerado | nao tolerado |
|---|---|---|---|
| Wan 2.1 VACE | 1,3 B | 0,0546 | 0,0793 |
| Z-Image v2 | ~6 B | 0,1241 | **nao medido** |
| HunyuanVideo 1.5 (familia) | ~13 B | 0,1837 | 0,2147, e 0,2163 no capybara |

**Correcao, 2026-09-01, no mesmo dia.** A tabela acima dizia `Z-Image v2 | 0,1241 | 0,2163 (capybara)`.
O `capybara_v0.1` NAO e um checkpoint Z-Image. Lido do arquivo: 1364 tensores e 54 `double_blocks`,
arquitetura do HunyuanVideo 1.5, contra 453 tensores e zero do Z-Image; e 16 653 435 264 bytes contra
16 653 368 128 do `hunyuanvideo1.5_720p_t2v_fp16`, 67 KiB de diferenca. A evidencia era real e estava
na fila errada. Ela passa para ~13 B, onde e a **segunda** quebra independente medida nessa
arquitetura e concorda com o 0,2147. E deixa o teto do Z-Image vazio: 0,1241 funciona, e nada alem
foi tentado. A monotonia da coluna do tolerado nao depende dessa celula e sobrevive.

Monotona no tamanho, 2,4x a 3,4x entre as pontas. A leitura por capacidade entrou no criterio como
argumento nao medido e sobreviveu **na forma oposta a que eu previ**: nao "modelo pequeno tem erro
por camada maior" -- o Wan tem o menor dos tres -- e sim "modelo pequeno aguenta menos erro por
camada". Com tres pontos isso e hipotese, nao lei; o proximo modelo pode derruba-la como este
derrubou a anterior.

Consequencia pratica imediata: **`--promote-error 0,15` nao e um default seguro.** Ele foi
escolhido no Z-Image e transportado; no Wan produz um arquivo que carrega, despacha, passa em todo
verificador estrutural e gera borrao.

### Quatro renderizacoes gastas antes de perceber que a referencia estava quebrada

A referencia **FP16, sem quantizacao nenhuma**, saiu como uma trama tecida em 6 passos/1 quadro,
em 25/33, em cfg 6 e cfg 1, com e sem `ModelSamplingSD3 shift 8` (que aplica -- `shift` vira 8.0 --
mas nao muda sigma nenhum no scheduler `simple`, medido). A primeira rodada reportou
`divergence 1,2365`, o pior numero ja visto nesta bancada, e ele nao media qualidade nenhuma.

Causa, um eixo variado (`tools/probe_vace_strength.py`):

    vace_strength 1.0 (default do ComfyUI)   |latente| 607,6    trama tecida
    vace_strength 0.0                        |latente| 1543,2   oficina, pessoa, cena

`WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) preenche `vace_frames` com **zeros**
quando nao ha no VACE, passa cada bloco por `process_latent_in` -- que subtrai a media do formato
latente, entao **zero vira valor nao nulo** --, concatena mascara toda de **UNS**, e aplica com
forca **1,0**. Nao e "sem controle": e controle constante em forca total. Qualquer checkpoint VACE
num workflow T2V comum sai destruido, sem erro e sem aviso. `quality_ladder.py` ganhou
`--vace-strength`.

Licao que ja esta em `CLAUDE.md` e foi cara de novo: **quando o braco nao quantizado tambem quebra,
o numero nao e sobre quantizacao.** Olhar a referencia primeiro custa uma imagem; nao olhar custou
quatro renderizacoes e quase uma conclusao publicada.

### Confirmado de passagem

- **Despacho real:** `probe_quant_dispatch.py --forward-only` conta 300 modulos quantizados, 12/12
  forwards com matematica quantizada, 0 `dequantize`, `linear_dtype int4`,
  `comfy_kitchen.backends.cuda`. O `WARNING: unet unexpected [...comfy_quant]` no load e
  **cosmetico** -- os tensores sao consumidos antes e reclamados depois. Confundi-lo com falha de
  carga custaria a rodada inteira.
- **Razao a4/a8 = 2,932**, contra 2,996 / 3,021 / 3,202. Terceira familia independente:
  espalhamento 9% enquanto o erro absoluto varia 78%. "E o peso, nao a ativacao" ganha um ponto
  fora de casa.
- **Velocidade troca de sinal com o lote, de novo.** 1 quadro: 0,204 contra 0,363 s/passo, W4A4
  **1,78x mais lento**. 33 quadros: 0,870 contra 0,653, W4A4 **1,33x mais rapido**. Curva
  `m_crossover`, agora num modelo de video.

### Nao coberto

Um prompt, tres sementes (duas no braco 0,05), 480x480, 33 quadros, um scheduler, uma placa. O
checkpoint e **fp16** e e uma variante **VACE rodada como T2V comum**, que nao e o uso para o qual
foi treinada -- a tolerancia medida pode ser do modo, nao do modelo. Os `vace_blocks` ficam fora do
perfil por construcao e permaneceram fp16; com forca 0 nao contribuem, entao nao houve carona.
Nenhuma metrica perceptual: "correta", "borrada" e "destruida" sao julgamento de quem olhou.

## 2026-09-01, parte 44 - o avaliador em lote, e o que ele achou na primeira passada

`tools/avaliar.py`, camada 1. Le cabecalho, sidecar e `.analysis.json`. **Sem GPU, sem torch, sem
carregar modelo: 163 checkpoints em 0,68 s**, entao "deixar rodando em tudo" nem chega a ser um
trabalho em lote.

A mediana do erro efetivo sai inteira do disco, porque as duas metades ja estao la: o sidecar diz
que formato cada camada levou e a analise diz o erro medido daquele formato naquela camada. Ela
reproduz todos os numeros que esta bancada publicou:

```
Wan 2.1 VACE      0,0546   0,0793   0,1602
HunyuanVideo 1.5  0,0739   0,1837   0,2147   0,2230
capybara_v0.1                       0,2163
```

`APROVADO` nao existe no conjunto de vereditos. Sao `REPROVADO`, `OLHAR` e `SEM VEREDITO`, porque
nenhum corte medido aqui separa usavel de inutilizavel nos dois eixos que temos -- 0,1837 correta
contra 0,2147 destruida, 0,7173 boa contra 0,8255 destruida. O numero reprova sozinho, aponta e
preve; nao aprova.

### Achou um numero publicado na linha errada, no primeiro dia

O card do Z-Image no HuggingFace creditava **0,1241 ao build misto** e deixava o W4A4 puro em
branco. E o contrario, e a inversao aparece em **nove calibragens independentes**:

```
build                     camadas          faixa nas 9 calibragens
zimage-v2-w4a4         170 convrot            0,1213 - 0,1285
zimage-v2-mixed        115 int4 / 55 int8     0,0771 - 0,0839
```

A identificacao dos arquivos no card estava certa (as contagens de camada batem exatas); so o
numero estava na outra linha. O README do GitHub sempre esteve certo -- `170 / 0 | 0,1241 |
correct` -- e foi o card que o contradizia. **Nada mais se mexe**: 0,1241 continua sendo um build
que funciona, e agora sabe-se que e o *mais agressivo* medido, entao o `tolerado` do Z-Image nao
muda de valor, so ganha dono. Corrigido no card no mesmo dia.

Mesma classe do erro do capybara, achado do mesmo jeito: por algo que **recalcula em vez de citar**.

### A semente da calibragem move a mediana 2% a 6%

Nunca tinha sido medido. Duas calibragens de `wan2.1_vace_1.3B_fp16` diferindo so na semente (1234
contra 12345) dao 0,051807 e 0,054631 no mesmo checkpoint. No Z-Image, nove calibragens espalham
5,7% no W4A4 puro e 8,8% no misto.

O espalhamento e menor que a propria banda (a faixa do Wan vai de 0,0546 a 0,0793, 45%), entao a
linha por modelo sobrevive com folga. Mas um numero de quatro casas saido de uma calibragem so
reivindica precisao que esta bancada nao tem, e agora ele viaja com o espalhamento ao lado. A
ferramenta so levanta achado quando as calibragens cairiam em **lados diferentes** da linha.

### Quatro defeitos da propria ferramenta, achados rodando

O que fez a construcao valer foi rodar em arquivos de terceiros, nao so nos nossos.

**Ela era cega em silencio para o segundo dialeto.** `LTX25-distilled-DiT-comfy-w4a4` -- 1440
camadas de 4 bits de verdade -- saiu `SEM VEREDITO` sem um unico achado, que le como "nada a ver
aqui". Ele nao traz `_quantization_metadata` nenhum: traz um tensor `<camada>.comfy_quant` com o
JSON, que e o outro dialeto que o ComfyUI aceita. **Qualquer ferramenta desta bancada que so leia
o metadata do arquivo enxerga zero camada quantizada num arquivo inteiramente quantizado.** Ler o
segundo dialeto custa um seek e ~50 bytes por camada.

**As regras de int4 aplicadas a todo formato reprovaram um FP8 publico e valido** (`flux-2-klein`,
container F8_E4M3 nao empacotado, escala escalar). Uma checagem que reprova arquivo bom ensina a
desligar checagem.

**Comparar mediana crua com tabela arredondada** marcou como suspeito exatamente o build do Wan que
*definiu* o valor tolerado: 0,054631 > 0,0546.

**A mediana sumia de quem passava**, porque so era reportada dentro de um achado. Ela e fato, nao
achado.

E a regra de escala do `int8_tensorwise` foi escrita errada **tres vezes seguidas**, cada vez a
partir do primeiro arquivo que eu tinha lido, cada vez reprovando arquivo bom: `[linhas]` do
ConvRot acusou o Dasiwa inteiro; `[linhas, 1]` do Dasiwa acusou o `embed_tokens` do encoder MiniMax
que esta bancada ja mediu rodando kernel nativo 15/15; `[linhas,1] se per_row senao []` acusou as
561 camadas do Gemma 4 E2B, que tem `[linhas, 1]` sem declarar `per_row`. Os dois formatos existem
no mundo real e nada no JSON da camada diz qual e o certo, entao a ferramenta **parou de afirmar**:
aceita os dois e declara no campo `cego` que nao pega escala torta nesse formato. Um quarto palpite
seria o mesmo erro pela quarta vez.

### Nao coberto

Nada aqui executa. Ninguem contou forward quantizado nem chamada a `dequantize`, entao um arquivo
pode passar tudo acima e rodar dequantizado -- e as camadas `--dispatch` e `--erro` do plano nao
existem. Nenhuma renderizacao, entao nenhum veredito de qualidade. O braco **nao** quantizado nunca
e exercitado, que e justamente a guarda que teria salvado quatro renderizacoes no Wan; ela precisa
de GPU e ficou para a camada 3. As bandas sao tres pontos, e a monotonia no tamanho do modelo segue
hipotese.

## 2026-09-01, parte 45 - a guarda do braco de referencia: "este modelo esta ouvindo?"

Camada 3 do avaliador, `tools/avaliar_referencia.py`. **Executada**, tres familias, quatro bracos.

Ela nao pergunta se a imagem esta boa -- essa pergunta esta proibida aqui, porque nenhum corte
medido separa usavel de inutilizavel. Ela pergunta se o modelo **responde ao proprio
condicionamento**, que tem resposta.

Dois prompts sem nada em comum, duas sementes, no modelo NAO quantizado:

```
d_prompt   quanto o PROMPT move o latente
d_semente  quanto a SEMENTE move            <- controle negativo, e o que torna a medida legivel
resposta   d_prompt / d_semente
```

Sem o denominador a medida nao vale: um `d_prompt` pequeno sozinho pode so significar que o modelo
e estavel. Pequeno *em relacao ao que a semente move* significa que o condicionamento nao chega.

### Criterio escrito antes, e ele se sustentou

`bench/criterio_guarda_referencia.md`, com as tres condicoes que refutariam a guarda escritas antes
de qualquer medicao.

| braco | resposta | d_prompt | d_semente | veredito | previsao | acertou |
|---|---|---|---|---|---|---|
| Wan VACE 1.0 — **destruido**, verdade conhecida | **0,2477** | 0,075 / 0,055 | 0,250 / 0,285 | REPROVADO | `< 0,5` | sim |
| Wan VACE 0.0 — **bom**, verdade conhecida | **0,7911** | 0,905 / 0,755 | 1,098 / 0,996 | OLHAR | `> 0,8` | **nao**, por 1,1% |
| Z-Image v2 BF16 — bom | **1,3459** | 1,064 / 1,150 | 0,862 / 0,789 | SEM VEREDITO | `> 0,8` | sim |
| HunyuanVideo 1.5 FP16 — bom | **2,8603** | 1,584 / 1,481 | 0,846 / 0,385 | SEM VEREDITO | `> 0,8` | sim |

**As tres condicoes de refutacao ficaram todas em silencio.** 3,19x separam o quebrado do sadio
mais proximo; o limiar de reprova nao rejeitou nenhum dos tres sadios; o destruido nao passou.
**Ela teria abortado a rodada do Wan na primeira imagem em vez da quarta.**

O mecanismo aparece cru, nao so na razao: no braco destruido o prompt move o latente **0,075** e a
semente move **0,250**. O modelo gera a partir do ruido e ignora o que se pede. Nos tres sadios o
prompt move de 0,76 a 1,58. E `|latente| 607,6` reproduz exato a medicao da parte 43.

### A previsao do limite SAO errou e o limiar nao foi mexido

O Wan bom deu 0,7911 contra os `> 0,8` previstos, e caiu em `OLHAR`. **O limiar continua 0,8.**
Mudar um limiar depois de ver o numero que ele deveria classificar nao e calibrar, e descrever. Um
limite superior defensavel precisa de bracos sadios novos, medidos depois -- nao dos mesmos que
testaram este.

O custo pratico do erro e pequeno: um braco sao ouve "olhe a imagem", que aqui e sempre verdade. O
erro caro seria reprovar arquivo bom, e esse nao aconteceu.

### Dois defeitos meus no caminho

**A ferramenta saiu com codigo 1 e duas linhas de saida**, sem dizer por que. A `BenchGuard`
recusou -- legitimamente, o 3080 Ti tinha 2,8 GiB de outro trabalho acima do teto de 2,0 -- e eu
nao imprimia `guarda.refused`, como o `quality_ladder` faz. Mesmo defeito de cegueira silenciosa
que a camada 1 tinha com o segundo dialeto, no mesmo dia. A saida documentada e
`CUDA_VISIBLE_DEVICES`, que tira a placa da run e da guarda ao mesmo tempo.

E o `d_semente` varia de **0,38 a 1,10** entre familias e ajustes, o que confirma que a estatistica
tinha que ser a razao e nao o `d_prompt` sozinho -- um limiar absoluto sobre `d_prompt` teria
classificado errado assim que mudasse de familia.

### Nao coberto

O lado da **reprova** se apoia em **um** braco quebrado de verdade, um caso com uma causa num
modelo; a guarda pode nao reconhecer uma quebra de condicionamento com outro mecanismo. A razao e
barulhenta em valor absoluto -- no Hunyuan as duas sementes deram 1,87 e 3,85, espalhamento 2,06x
-- entao vale a distancia ao limiar, nao a segunda casa. Nada decodifica imagem: a medida e no
latente. Uma placa, um scheduler, um tamanho por familia, dois prompts, duas sementes. E um modelo
que legitimamente responde pouco ao prompt (refinador, upscaler, modelo de controle) reprovaria
sem estar quebrado.
