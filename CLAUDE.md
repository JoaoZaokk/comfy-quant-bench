# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`F:\COMFY_PORTABLE` is a **live ComfyUI Portable installation**, not a clean source project. It is currently the working bench for the **ConvRot W4A4 quantization project** (converting local high-precision checkpoints to native 4-bit weight / 4-bit activation format).

The root is **not** a Git repository. `ComfyUI/` is the Git checkout (version **0.33.0**, `v0.33.0-19-gc1739380`, 2026-08-18 -- it was 0.29.0/`42d2aa55` earlier in this project's life). Commits belong to `ComfyUI/` or to an individual repo under `ComfyUI/custom_nodes/` (60 installed).

Also read [AGENTS.md](AGENTS.md) (root policy) and [ComfyUI/AGENTS.md](ComfyUI/AGENTS.md) (upstream engineering style — mandatory before editing anything under `ComfyUI/`).

## Hard rules

- **Only** `.\python_embeded\python.exe`. Never global Python, pip, or Conda. Always pass `-s` when running scripts directly so user site-packages stay out.
- **Never** delete, move, overwrite, or requantize an original model. Outputs go beside their source with a `_w4a4_convrot` suffix; the converter refuses to overwrite anything.
- **Do not mass-upgrade** Torch / CUDA / ComfyUI / comfy-kitchen. Inspect installed versions and local APIs before proposing any package change. `_pip_freeze_before_w4a4_cu130_20260816.txt` is the current known-good freeze baseline.
- WSL on this host runs unrelated Qwen/DeepSeek jobs. Do not stop, restart, or reconfigure it — but note `vmmemWSL` eats ~28 GB RAM and can starve conversions (see *Memory gotchas*).
- **W4A4 means native ConvRot CUDA execution**, not weight-only INT4 followed by BF16 GEMM. Any change that lets the work fall back to eager/dequantized math defeats the entire project.

## Say which one it was: traced, or executed

Reading code and running code produce the same confident prose. That is the specific failure mode
this project keeps hitting — not carelessness, and not reluctance to be wrong, but that a careful
trace through a call chain *feels* like evidence and *reads* like a measurement, and nothing in
the writing separates them. It has cost real work here more than once:

| written as fact | what it actually was | what measurement said |
|---|---|---|
| "crest factor is the statistic ConvRot's activation path is sensitive to" | a mechanism argument, in a tool's own docstring | Spearman **+0.096** over 170 layers. No relationship. |
| "CUDA graph removes the 109 us" | inference from what graphs do | removes **83%**; ~11 us of host survives per replay |
| "host overhead is 161 us" | one median, from a contaminated process | **77 us** clean. Passed to a sibling project before it was checked. |
| "`verify_w4a4.py` runs the same check" | a claim in this very file | it resolves one op; the converter resolves two |
| "the probe is exact, with no error at all" | true of the weight, argued for the activation | BF16 scale rounding, ~1.4e-3 mean relative |

The fix is mechanical, because intent does not survive the next session. **When a conclusion comes
from reading, the artifact that carries it must say so, in the artifact.** Not in the chat, which
is gone by then.

- Tools print it. `tools/verify_w4a4.py`, `tools/test_svdq_verify.py` and
  `custom_nodes/comfy-quant-preflight/` each end their output with what they did *not* cover, on
  every run, pass or fail — a column of PASS lines otherwise reads as "verified".
- Docstrings carry the provenance and the caveat inline, next to the number, not in a paragraph
  below it. Numbers get pasted out of this repo into other sessions; a ratio with its condition
  attached ("4.89x [4.72-5.06] at M=5856 on weight [3840, 3840]; 1.5-2.0x *slower* at M=1")
  cannot be misquoted the way a bare "4.6x" can.
- **A single run is not a measurement.** Three consecutive `m_crossover` runs on an idle, locked
  3090 disagreed by up to 1.4x at the same M and shape, and the crossover itself moved a step
  between two runs in the *same* direction (2026-08-19, `W4A4_PROGRESS.md` part 11). The tool now
  repeats interleaved and prints the ratio's own min-max, because a two-decimal number from one
  burst claims a precision this bench does not have. Anything quoted from here needs a repeat
  behind it and the spread beside it.
- An unverified finding stays labelled unverified all the way into the file that acts on it. The
  preflight package's two audit-derived checks are WARN and say "not confirmed by execution" in
  the message itself, because a check that blocks on a hypothesis teaches people to disable checks.
- **No upstream PR from a trace.** Only from something run here. `AUDITORIA_2026-08-18.md` lists
  three PR candidates; only one has been proved, and the other two wait for the GPU.

Corollary that has paid off repeatedly: when a measurement contradicts a claim in this repo, the
claim is what changes, including claims in this file — and the tool that produced the wrong number
gets fixed too, not just the sentence.

## Environment

| Item | Value |
|---|---|
| Python | 3.13.12 (embedded) |
| Torch | 2.13.0+cu130 (torchvision 0.28.0, torchaudio 2.11.0, all `+cu130`) — subiu de 2.12.1 em algum ponto; `_check_accel.py` confirma Triton/Sage/FlashAttention ainda executando kernel sob 2.13 |
| comfy-kitchen | 0.2.23 |
| spas_sage_attn (SpargeAttn) | 0.1.0+cu130torch2.9.0andhigher.post4 — wheel abi3 do woct0rdho, traz `_qattn_sm80.pyd`, roda kernel na sm86 |
| nunchaku (SVDQuant) | 1.2.1+cu13.0torch2.11 — build de torch 2.11 rodando sob 2.13; `ops.attention_fp16` executa na sm86, `ops.gemm_w4a4` ainda não exercitado |
| GPUs | RTX 3090 24 GB (`cuda:0`, cc 8.6), RTX 3080 Ti 12 GB (`cuda:1`) |

Non-obvious environment facts:

- The cu130 Torch wheel ships only `cudart64_13.dll`, but the installed SageAttention 2.2.0 binary extensions (`_fused.pyd`, `_qattn_sm80.pyd`) link `cudart64_12.dll`. A CUDA 12.6 runtime DLL was copied side-by-side into `python_embeded\Lib\site-packages\torch\lib\cudart64_12.dll` (556,544 bytes, SHA-256 `D954CA54...F9DAD`, sourced from `venvs\ultravox311`). **Do not remove it** — Sage, FlashVSR, and the Sage launchers depend on it.
- Extra models are mounted from a second drive via `ComfyUI/extra_model_paths.yaml` (`D:/ComfyUI-Models/`). A missing model may live there, not under `ComfyUI/models/`.
- `venvs/ultravox311` is an unrelated side venv (Ultravox/TTS experiments — `teste_*.py` at root). Not part of ComfyUI.

## Commands

Launch:

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build
```

```bash
.\run_nvidia_gpu_8190_loopback.bat
```

**Any workflow using a Nunchaku SVDQuant loader needs `--disable-dynamic-vram`**, verified 2026-08-18 by a real generation. ComfyUI 0.33 added a Windows-only lazy `Linear` (`comfy/ops.py:520-538`, `self.weight = None` until `_load_from_state_dict`), enabled by `main.py:289`. ComfyUI-nunchaku reads `orig_attn.qkv.weight.dtype` in `patch_model` before that happens and dies with `AttributeError: 'NoneType' object has no attribute 'dtype'`. Nothing about this points at the loader or the checkpoint, and the same node called directly in-process works fine — so the error is easy to misattribute.

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190
```

Launcher notes: `run_nvidia_gpu_8190_loopback.bat` (Sage, `127.0.0.1:8190`) is the one to use. `run_nvidia_gpu_8190.bat` binds `0.0.0.0` and on this host the asyncio accept loop dies with `OSError(22, 'The specified network name is no longer available', 64)` whenever NordLynx/NordVPN reconnects — process stays alive, port stops accepting. Other launchers: `run_nvidia_gpu.bat`, `run_nvidia_gpu_fast_fp16_accumulation.bat`, `run_nvidia_gpu_8190_flash.bat` (FlashAttention), `run_cpu.bat`.

Tests (run from the root; `pytest.ini` lives in `ComfyUI/` with `pythonpath = .`).

**Neither `pytest` nor `ruff` is installed in the embedded interpreter** — verified 2026-08-18,
both fail with `No module named ...`. The commands below are the correct ones *once they are*,
and installing either is a package change, so it is the user's call. Test files written for this
project (`tools/test_svdq_verify.py`, the suites under `Comfy-WaveSpeed-Fixed/tests/`) therefore
carry their own runner and are executed directly:

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests-unit
```

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests -m "not inference"
```

Single test / single case:

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests-unit\path\to\test_file.py::test_name
```

Markers defined: `inference`, `execution` (deselect with `-m "not inference"`).

Lint:

```bash
.\python_embeded\python.exe -m ruff check ComfyUI
```

Verify the attention accelerators actually run kernels (import success is not proof — this does forward + numeric comparison against SDPA for Triton, Sage, FlashAttention, xformers):

```bash
.\python_embeded\python.exe -s .\_check_accel.py
```

Updates go through `update\update_comfyui.bat`; `update\update_comfyui_and_python_dependencies.bat` is destructive to the pinned stack — do not run it casually.

## Quantization pipeline

Three tools in `tools/`, all run with the embedded interpreter from the root:

```bash
.\python_embeded\python.exe .\tools\quant_audit.py
```

Recursively walks `ComfyUI/models` (metadata only — parses Safetensors headers, uses the installed GGUF reader for GGUF, safe/meta loading for `.pt`) and rewrites `quantization_inventory.json` + `quantization_inventory.md`. Rerun after producing any new output so the inventory stays current.

```bash
.\python_embeded\python.exe .\tools\quant_w4a4.py --input <src.safetensors> --profile gemma --dry-run
```

Drop `--dry-run` to write. Output name defaults to `<stem>_w4a4_convrot.safetensors` plus a `.quant.json` sidecar manifest.

```bash
.\python_embeded\python.exe .\tools\verify_w4a4.py <out.safetensors> --source <src.safetensors> --kernel-smoke
```

### Mixed precision, calibrated on real activations

A second pipeline that picks 4-bit or 8-bit **per layer** by measuring the real kernels on the
activations the model actually produced. Three tools, in order — see `W4A4_PROGRESS.md` part 9 for
the measurements and `W4A4_HANDOFF.md` for the invocation.

`tools/to_native.py` rewrites a diffusers-named checkpoint under ComfyUI's own module names.
**Not optional for Z-Image.** ComfyUI fuses `attention.to_{q,k,v}` into `attention.qkv` at load
(`comfy/model_detection.py:1498`), and only `.weight` is in that map — `weight_scale`,
`weight_s_rel` and `comfy_quant` pass through unrenamed, so the layer loads with no scale and no
error. The remap is derived from `comfy.utils.z_image_to_diffusers` and checked against ComfyUI's
own `convert_diffusers_mmdit` on meta tensors; its output is bit-identical in the latent.

`tools/calibrate_activations.py` runs real sampling with forward-pre-hooks on every candidate
Linear and reservoir-samples the real input rows (Algorithm R, so the sample is not just step 0).
Samples are stored **bfloat16, never fp16**: real Z-Image activations reach 344064 and fp16 caps
at 65504, and an `inf` there makes every error `nan`, which silently routes the worst layer in the
model to the cheapest format.

`tools/quant_mixed.py` measures `err_bf16` / `err_w4a4` / `err_w4a8` per layer against a float32
reference on those activations, then assigns `convrot_w4a4`, `asym_w4a8_int8`, or no quantization.
Mixed formats in one file are native behaviour: `comfy/utils.py:1454` turns each
`_quantization_metadata["layers"]` entry into a per-layer `<layer>.comfy_quant` tensor and
`comfy/ops.py:1142` dispatches on that layer's own JSON.

Do not choose precision by crest factor. Measured over 170 Z-Image layers its rank correlation
with the W4A4 error is **+0.10**. What correlates is the W4A8 error (+0.978).

### How the converter is built (and why)

- **Native-backend preflight.** Before touching a single tensor, `normal_comfy_backend()` spawns a subprocess that imports `comfy.quant_ops` and asks `comfy_kitchen.registry` which implementation would be selected. If either `quantize_convrot_w4a4_weight` or `convrot_w4a4_linear` does not resolve to `comfy_kitchen.backends.cuda.*`, the conversion **hard-refuses**. Keep this — it is the guard against silently producing a checkpoint that only ever runs dequantized.

  This paragraph used to claim "`verify_w4a4.py` runs the same check". It does not, verified 2026-08-18: `verify_w4a4.py:64` resolves only `convrot_w4a4_linear`, while `quant_w4a4.py:93-94` resolves both. Two more converters run **no** preflight at all — `quant_w4a4_smooth.py` (which writes the same `convrot_w4a4` format) and `quant_int8.py`. Do not treat "the tool ran" as proof the CUDA backend was used; check the `backend` field in the sidecar, and note that `quant_w4a4_smooth.py` does not write one. See [AUDITORIA_2026-08-18.md](AUDITORIA_2026-08-18.md) items 8 and 17.
- **Streaming writes, never mmap.** Output header offsets are computed up front, then tensors are streamed: quantized layers are read by byte range → CUDA → `ck.quantize_convrot_w4a4_weight` → written; everything else is `copy_range`'d verbatim in 16 MiB chunks. **Do not reintroduce `safe_open` / mmap for large sources.** On this Windows host mapping the 21.93 GiB Gemma source failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
- **Atomic output.** Writes go to `<output>.partial`, then `os.replace`. Refuses stale partials, refuses existing outputs/sidecars, refuses a source that already has `_quantization_metadata`.
- **Profiles are strict allowlists**, not heuristics. `PROFILE_PATTERNS` matches only `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `model.layers.N.mlp.{gate,up,down}_proj.weight`; embeddings, norms, `lm_head`, and vision towers are excluded. Only `gemma` and `qwen` exist today. **Do not extend a profile to a new architecture without confirming that architecture's loader and layer config** — Flux, Hunyuan, SeedVR2, and Z-Image each need their own recipe.
- **Group sizes.** `CONVROT_GROUP_SIZE = 256` (rotation), `QUANT_GROUP_SIZE = 64`. Layer selection requires `shape[1] % 256 == 0`. The backend-probe subprocesses in both tools use dummy `64/64` values only to resolve the implementation, which is why they differ from the real conversion values — not a bug, but don't copy those numbers into real calls.

### Output format

Standard Safetensors. Per quantized layer: `<layer>.weight` as `I8` of shape `[rows, cols/2]` (INT8 container holding signed INT4), plus `<layer>.weight_scale` as `F32` of shape `[rows]`. Everything else preserved byte-for-byte. `__metadata__` carries `_quantization_metadata` (`format_version` 1.0 + per-layer `{format: convrot_w4a4, convrot_groupsize: 256}`) and `quantization: "ConvRot W4A4"`. See [ComfyUI/QUANTIZATION.md](ComfyUI/QUANTIZATION.md) for the upstream `QuantizedTensor` / `Layout` / `MixedPrecisionOps` model this format plugs into.

`verify_w4a4.py` checks, in order: metadata structure and per-layer dtype/shape → packed shape vs source shape → **byte-identical comparison of every preserved tensor against the source** → native backend resolution → optional real-kernel smoke against `F.linear` on the BF16 source. The smoke's relative RMSE on random inputs (~0.25 for Gemma) is a liveness signal, **not** a quality metric.

`inspect_quant.py <file>` at root is a quick header dump (tensor count, dtype histogram, metadata, scale-like keys).

## Testing a converted model

Structural verification is not acceptance. Every output must additionally pass, in order: normal ComfyUI loader compatibility (real node, not a hand-rolled load), a prompt-encoding smoke test through a long-lived ComfyUI process, and a matched-parameter benchmark against the BF16 source (identical prompt, seed, steps, resolution, sampler, scheduler, frames) recording disk, VRAM, load time, s/it, GPU utilization, power, warnings, and visual quality. Workflows for this live in `ComfyUI/user/default/workflows/` (e.g. `Video-LTX2_MultiGPU.app.json` for the Gemma text encoder).

Record negative and unsupported results rather than hiding them.

## Project tracking

- [W4A4_PROGRESS.md](W4A4_PROGRESS.md) — the running log: candidate ranking, completed steps, skipped models with reasons, failed/blocked with exact error strings. **Update it as part of the work**, not afterwards.
- [W4A4_HANDOFF.md](W4A4_HANDOFF.md) — current state snapshot and the ordered next-steps list.
- `quantization_inventory.{json,md}` — generated; do not hand-edit.

## Memory gotchas

Conversions check free disk (`estimate + 1 GiB`) and free RAM (`3 × largest selected tensor + 2 GiB`) and exit rather than thrash. If it refuses, the fix is to close memory-heavy WSL/worker processes **manually** — never change the pagefile or kill processes automatically.

Known benign noise: `ModelPatcher.__del__` prints an `ON_DETACH` AttributeError on short-lived interpreter shutdown; loading itself still exits 0.
