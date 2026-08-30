# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`F:\COMFY_PORTABLE` is a **live ComfyUI Portable installation**, not a clean source project. It is currently the working bench for the **ConvRot W4A4 quantization project** (converting local high-precision checkpoints to native 4-bit weight / 4-bit activation format).

**Three separate Git repositories overlap here, and confusing them is easy.** The root *is* a repo — on `master`; **run `git ls-files | wc -l` for the count, do not quote one from here.** (This file used to say the root was not a repo, which stopped being true once `tools/` was committed.) The number is deliberately absent: it was written as 88, corrected to 89, and was 93 one commit later — three staleness events in a single session, the last one caused by the commit that fixed the previous one. It is an **allowlist**: `.gitignore` ignores `/*` and re-includes tracked paths one by one, because models alone are ~600 GiB. `ComfyUI/` is a *separate* checkout (version **0.33.0**, `v0.33.0-19-gc1739380`, 2026-08-18 -- it was 0.29.0/`42d2aa55` earlier in this project's life) and is **not** tracked by the root repo. `ComfyUI/custom_nodes/` holds dozens of packages — **count them, do not quote a number from here**, with `ls -A ComfyUI/custom_nodes | wc -l` and `find ComfyUI/custom_nodes -mindepth 1 -maxdepth 1 -type d | wc -l`. This paragraph said "66 entries, 64 directories, 63 packages" at 06:00 on 2026-08-21 and `ls -A` gave **67 entries, 65 directories** the same afternoon — the third staleness event on this one number, which is why it now goes the way the tracked-file count went. It aged a fourth time: measured 2026-08-30, `ls -A` gives **93 entries, 91 directories** — the package count grew by a third in nine days, so any number written here is wrong before it is read. Two of the entries are loose files (`example_node.py.example`, `websocket_image_save.py`), one directory is `__pycache__`, one directory is literally named `.disabled`, and **four** entries end in `.disabled` and are hidden from plain `ls`: subtract those before calling anything a package count. **Roughly half have a `.git` of their own** — **42** measured 2026-08-30 (this line said 31), with `find ComfyUI/custom_nodes -maxdepth 2 -name .git | wc -l`. An earlier version of this very paragraph claimed each entry was its own repo; that is false for half of them, which matters the moment someone tries to `git -C <package>` anything. Only **four** files under `custom_nodes/` are ours and tracked by the root, in **two** packages: three in `custom_nodes/comfy-quant-preflight/` and one in `custom_nodes/comfy-void-stage-tools/` (tracked 2026-08-29; this line said "three files, all in comfy-quant-preflight" until then). Count it, do not quote it: `git ls-files custom_nodes`.

Count these before quoting them. Both the file count and the node count were wrong in this file until they were recounted — see the memory `escrevo-mais-rapido-do-que-confiro`.

Also read [AGENTS.md](AGENTS.md) (root policy) and [ComfyUI/AGENTS.md](ComfyUI/AGENTS.md) (upstream engineering style — mandatory before editing anything under `ComfyUI/`).

## Hard rules

- **Only** `.\python_embeded\python.exe`. Never global Python, pip, or Conda. Always pass `-s` when running scripts directly so user site-packages stay out.
- **Never** delete, move, overwrite, or requantize an original model. Outputs go beside their source with a `_w4a4_convrot` suffix; the converter refuses to overwrite anything.
- **Do not mass-upgrade** Torch / CUDA / ComfyUI / comfy-kitchen. Inspect installed versions and local APIs before proposing any package change. `_pip_freeze_before_w4a4_cu130_20260816.txt` is the current known-good freeze baseline.
- **WSL on this host runs the owner's containers. Do not stop, restart, or reconfigure it.** The running distro is `docker-desktop`; `Ubuntu` was *Stopped* and holds no project, so this rule was never about the Qwen/DeepSeek jobs it used to name. `wsl --shutdown` takes every container down with it. `vmmemWSL` eats ~28 GB RAM and can starve conversions (see *Memory gotchas*).

  **The ERP left this machine, on purpose, and the migration is done.** This rule used to say "eleven live containers — the `baselinker-erp-offline` stack (`zhao-*` / `baselinker-*`), a postgres, and a WhatsApp gateway", measured 2026-08-21. Measured 2026-08-30 in **both** docker contexts (`default` and `desktop-linux` — same engine, so one `docker ps` was not enough to claim absence): **five** containers, and **no `zhao-*` or `baselinker-*` at all**, not even stopped. Confirmed by the owner the same day: **the ERP is on Proxmox now, and MacroLog migrated too.** What still runs here is the **rollback copy**, which he does not expect to need. `F:\zhao-hub-data` (6.7 GB, 16258 files) is the old data, kept.

  So the rule's original reason is gone, but the rule stands until the owner says otherwise, because `wsl --shutdown` is still destructive to what *is* running:

  ```
  glm-w4              qwen38-pp-dspark:0.27.1   <- the owner's own GPU work; holds the 3090
  macrozao            tributario-macrozao       (healthy)  } project `tributario`
  gestao-db           postgres:16-alpine        (healthy)  } the MacroLog rollback copy
  macrozao-api        app_api-api
  nostalgic_torvalds  cloudflare/cloudflared
  ```

  **Consequence worth knowing: the `.vhdx` compaction is no longer blocked by the ERP.** ~140 GB sits in `docker_data.vhdx` (229 GB, ~90 GB of build cache with zero active entries) and `ext4.vhdx` (63 GB with 22 GB used). `Optimize-VHD` needs `wsl --shutdown` because all distros share one VM — that was unthinkable with the ERP here and is now a scheduling question, not a hazard. Still the owner's call, and still never `--allow-unsafe`.

- **A GPU held from inside WSL is invisible to the Windows-side CUDA free-memory query.** Measured 2026-08-30 while `glm-w4` held the 3090: `torch.cuda.mem_get_info` in a fresh Windows process reported **23332 MiB free** where `nvidia-smi` reported **6664 MiB free** — a 16.7 GiB disagreement on the same card in the same second. Same direction on the 3080 Ti: ComfyUI 9771-10024 MiB against 220-239 MiB from the driver, **41x**, stable over six interleaved rounds. The query is not broken — allocating 512 MiB *in this process* moves `vram_free` by exactly 512 MiB — it simply does not see other processes' memory. This matters because `get_free_memory()` **is** ComfyUI's own fit test (`comfy/model_management.py`, in `:978`, `:1096-1098` and `:1210-1213`); on a shared card it will happily decide a model fits when it does not. Anything reading `/system_stats` from outside to size a load must cross-check NVML.
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
- An unverified finding stays labelled unverified all the way into the file that acts on it, and
  **the label changes when the measurement arrives** — in both directions. The preflight package
  started with two audit-derived WARNs. `check_full_precision_matrix_mult` still says "not
  confirmed by execution" in the message itself, because a check that blocks on a hypothesis
  teaches people to disable checks. `check_lora_over_quantized` no longer says it: the hypothesis
  was **measured on 2026-08-19 and did not reproduce** — 680 of 680 dispatches stayed on the
  native kernel with the LoRA applied and in effect (latent norm moved 747.06 → 728.99, so it was
  genuinely applied). It is still a WARN, narrowed to what is actually still unknown: whether the
  delta costs *accuracy* once the weight is already 4-bit, which nobody asked.
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
| comfy-kitchen | **0.2.31** (era 0.2.23 neste arquivo; subiu em algum ponto e a nota não acompanhou. É o registry que decide se `convrot_w4a4_linear` resolve para `comfy_kitchen.backends.cuda.*` — reconfira o preflight antes de confiar em conversão antiga) |
| spas_sage_attn (SpargeAttn) | 0.1.0+cu130torch2.9.0andhigher.post4 — wheel abi3 do woct0rdho, traz `_qattn_sm80.pyd`, roda kernel na sm86 |
| nunchaku (SVDQuant) | 1.2.1+cu13.0torch2.11 — build de torch 2.11 rodando sob 2.13; `ops.attention_fp16` executa na sm86, `ops.gemm_w4a4` ainda não exercitado |
| GPUs | RTX 3090 24 GB (`cuda:0`, cc 8.6), RTX 3080 Ti 12 GB (`cuda:1`) |

Non-obvious environment facts:

- ~~The cu130 Torch wheel ships only `cudart64_13.dll`, but the installed SageAttention 2.2.0 binary extensions link `cudart64_12.dll`, so a CUDA 12.6 runtime DLL was copied side-by-side into `torch\lib\cudart64_12.dll`. **Do not remove it.**~~ **OBSOLETE — but the file is NOT gone, and the sentence that said it was is a lesson.** From 2026-08-21 to 2026-08-22 this paragraph read *"the file is already gone"* and offered as proof: `find python_embeded -iname "cudart64*.dll"` returns only `cudart64_13.dll`. That command **cannot match a name ending in `.disabled`**, so it was blind by construction and returned a clean-looking result. Drop the `.dll` from the pattern and:

```
python_embeded/Lib/site-packages/torch/lib/cudart64_12.dll.disabled   556,544 bytes
  sha256 d954ca542b3b6bcf03cc2b798a7d00051501cf734ca751050e986af505cf9dad
```

It was **renamed, not deleted** — which `W4A4_PROGRESS.md:339-340` already recorded, executed, and this file contradicted for a day without either of them noticing. Windows will not load a `.dll.disabled`, so the runtime conclusion below stands unchanged; the *record* was wrong. **Keep the sha256**: the file is still sitting there unlabelled, and the hash is the only thing that identifies it. Leave it disabled.

This is the memory `arquivo-plausivel-nao-e-o-caminho` and this file's own rule — *absence in a grep is never absence in the system* — broken by the file that states it. The mechanical fix is the one already applied elsewhere here: **do not quote a command's output as proof without checking the command can see what it claims to rule out.**

What remains true, and is the part that matters: `sageattention._qattn_sm80`, `sageattention._fused`, `spas_sage_attn._qattn_sm80` and `flash_attn` **all import successfully** with the 12.6 DLL disabled — that is the Windows loader resolving the whole DLL chain, not a grep. The accel stack was reinstalled on 2026-08-16 with cu130 builds (`sageattention 2.2.0+cu130torch2.10.0andhigher.post6`, installed 15:01; `cudart64_13.dll` timestamped 15:34) which link `torch_cuda.dll` rather than cudart directly. **Do not "restore" the 12.6 DLL.** The caveat that used to travel with this — *an import proves DLL resolution, not kernel execution* — is **closed**. `_check_accel.py` was executed on the 3090 on 2026-08-22 with `CUDA_VISIBLE_DEVICES=0`: `triton v3.7.1 kernel compiled+ran`, `sageattention mean|d|=0.0006 vs SDPA`, `flash_attn mean|d|=0.0000`, xformers not installed. **ALL GOOD.** The accel stack runs kernels without the 12.6 DLL, and that is now a measurement rather than an inference. This rule survived five days after the file it protected stopped existing, in a file every session reads.
- Extra models are mounted via `ComfyUI/extra_model_paths.yaml`, which declares them at `D:/ComfyUI-Models/`. A missing model may live there, not under `ComfyUI/models/`. **`D:` is not a disk.** Measured 2026-08-22: `net use` reports `D: -> \\192.168.3.68\estoque`, a mapped SMB share, and `tools/quant_audit.py` resolves the declared path to that UNC and records both. So **408 GiB of the installation's ~1.01 TiB of model files arrive over the network** — which changes what a full walk costs, and means "the D: mount is offline" is a normal state rather than a broken one. That is a different NAS from the one holding the ERP code (`192.168.3.40`). A tool that walks both roots must treat a missing declared root as a warning, not a crash; one that is *told* to walk a path that is not there must still refuse (`quant_audit.py`, and `.scratch/varredura-2026-08-22/issues/15`).
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

**The LTX 2.5 workflow on this bench needs `--disable-dynamic-vram` too**, verified 2026-08-19 by a real render (executed, not traced; the run is `F:\cortiq\e2e_comfy_512.log`, timestamped 2026-08-19 21:28 -- an earlier pass of this note said 2026-08-21, which was the date it was written down, not the date it was measured). This is a second, independent reason to pass the flag — do not read it as Nunchaku-only. The `cortiq` ledger asserted the opposite for this workflow; that assertion was a trace, not a run, and the run below is what overrides it. Without the flag the render dies with:

```
Model LTXAVTEModel_ prepared for dynamic VRAM loading. 14612MB Staged.
Model LTXAV        prepared for dynamic VRAM loading. 20484MB Staged.
aimdo: src/hostbuf.c:283:ERROR:hostbuf_read_file_slice: device copy failed
RuntimeError: HostBuffer.read_file_slice failed
torch.AcceleratorError: CUDA error: out of memory
```

14612 + 20484 = 35 GB staged on a 24 GB card. PyTorch's own summary, printed alongside, says `Allocated memory 141211 KiB` and `CUDA OOMs: 0` — the allocator that overflowed was dynamic-vram's, not torch's. **The message never names the subsystem at fault** and points the reader straight at "the model is too large", on a card that had 18 GB free at the moment of failure.

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190
```

Launcher notes. There are **13** `.bat` launchers in the root, not seven — count them, do not quote this. The claim that **none** of them passes `--disable-dynamic-vram` was true on 2026-08-21 and is **false since 2026-08-26**: measured 2026-08-30, **four** pass it — `run_nvidia_gpu_8190_dual_component.bat`, `run_nvidia_gpu_8190_ultra_image.bat`, `run_nvidia_gpu_8190_ultra_video.bat`, `run_nvidia_gpu_8190_void_cache_none.bat`. Those four also set `COMFYUI_MGPU_DISABLED=1`. `run_nvidia_gpu_8190_loopback.bat` (Sage, `127.0.0.1:8190`) still does **not** pass the flag, so for a Nunchaku SVDQuant loader or the LTX 2.5 workflow above, use one of the four, call `main.py` by hand, or use `F:\cortiq-cmf\run_e2e_comfy.ps1`. Following "use the loopback launcher" alone will reproduce the 35 GB staging OOM. Other launchers: `run_nvidia_gpu.bat`, `run_nvidia_gpu_fast_fp16_accumulation.bat`, `run_nvidia_gpu_8190_flash.bat` (FlashAttention), `run_8190_limpo.bat`, `run_cpu.bat`.

**The `WinError 64` accept-loop death is NOT specific to the `0.0.0.0` launcher.** This file said `run_nvidia_gpu_8190.bat` binds `0.0.0.0` and the asyncio accept loop dies with `OSError(22, 'The specified network name is no longer available', 64)` whenever NordLynx/NordVPN reconnects — which reads as "loopback is immune". It is not. Observed 2026-08-30 at 04:29:19 on a server launched by `run_nvidia_gpu_8190_ultra_video.bat`, from its own log:

```
socket: <asyncio.TransportSocket fd=3000, family=2, type=1, proto=6, laddr=('127.0.0.1', 8190)>
```

`127.0.0.1`. Binding loopback does not protect against it. What *did* hold is the other half of the note: **the process stays alive with a dead port** — accept died at 04:29:19 and the process only ended at 04:35:18, six minutes of a listening-looking server that accepts nothing. `curl` returns `http=000` and `netstat` shows nothing `LISTENING`; the PID being alive proves nothing. Both `NordLynx` and `CloudflareWARP` tunnels were `Up`, plus a `cloudflared` container — three things that churn routes on this host.

**`main.py --windows-standalone-build` re-executes itself as a child process**, verified 2026-08-19 (executed, not traced). Killing only the PID that `Start-Process` returns leaves that child running as an orphan. On this bench the orphan held **20,578 MiB of the 3090** and **16.67 GB of RAM**, invisibly: `Get-Process -Id` reported the launched PID as dead, while `nvidia-smi` still showed the VRAM occupied. Kill the process tree, not the PID.

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

  **This paragraph has now been wrong twice, in opposite directions, and the second time is the instructive one.** It first claimed `verify_w4a4.py` ran the same check; it did not. It was corrected on 2026-08-18 to say that `quant_w4a4_smooth.py` and `quant_int8.py` run **no** preflight and that smooth writes no `backend` sidecar field. All three of those became false in `d14ae48` and the paragraph did not follow — so for four days this file told readers that two converters were **less safe than they are**, which is the direction that gets acted on. Re-read against the tree on 2026-08-22: `quant_w4a4_smooth.py:182-191` imports `normal_comfy_backend` and hard-refuses, `:328-330` writes `backend` and `backend_linear`, and `quant_int8.py:187-198` preflights when `--device cuda --convrot` and prints why the `--no-convrot` path is exempt instead of pretending to.

  What is still true, and is the part worth keeping: **the converters do not all resolve the same thing.** `verify_w4a4.py` resolves only `convrot_w4a4_linear` while `quant_w4a4.py` resolves both ops — and there are **six** definitions of `normal_comfy_backend` in the tree with four different answers to "is the backend ready". Line numbers are deliberately absent here: they were `:64` and `:93-94` in this sentence and are `:70` and `:98-99` today. Grep for `def normal_comfy_backend` and count. Do not treat "the tool ran" as proof the CUDA backend was used; check the `backend` field in the sidecar. See [AUDITORIA_2026-08-18.md](AUDITORIA_2026-08-18.md) items 8 and 17, and `.scratch/varredura-2026-08-22/issues/05`.

  **Measured 2026-08-22, on the 3090, so nobody re-derives it:** the resolved implementation is *invariant* to `convrot_groupsize` and to dummy-vs-real probe tensors. All four combinations — cg 64 and 256, `torch.empty` and real quantized tensors — resolve to `comfy_kitchen.backends.cuda`, and both real calls succeed. So `quant_w4a4.py`'s hardcoded 64/64 preflight against a 256 conversion is untidy, **not** wrong; and `_native_probe.py`'s own docstring claim that dummy kwargs let the check pass where a real call would not **did not reproduce** for these two ops on this build. That is "did not reproduce under the only conditions anyone has tried", not "is false" — the mechanism at `registry.py:246` may still bite elsewhere. `tools/probe_backend_resolution.py` re-runs it.
- **Streaming writes, never mmap.** Output header offsets are computed up front, then tensors are streamed: quantized layers are read by byte range → CUDA → `ck.quantize_convrot_w4a4_weight` → written; everything else is `copy_range`'d verbatim in 16 MiB chunks. **Do not reintroduce `safe_open` / mmap for large sources.** On this Windows host mapping the 21.93 GiB Gemma source failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
- **Atomic output.** Writes go to `<output>.partial`, then `os.replace`. Refuses stale partials, refuses existing outputs/sidecars, refuses a source that already has `_quantization_metadata`.
- **Profiles are strict allowlists**, not heuristics. `PROFILE_PATTERNS` matches only `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `model.layers.N.mlp.{gate,up,down}_proj.weight`; embeddings, norms, `lm_head`, and vision towers are excluded. Only `gemma` and `qwen` exist today. **Do not extend a profile to a new architecture without confirming that architecture's loader and layer config** — Flux, Hunyuan, SeedVR2, and Z-Image each need their own recipe.
- **Group sizes.** `CONVROT_GROUP_SIZE = 256` (rotation), `QUANT_GROUP_SIZE = 64`. Layer selection requires `shape[1] % 256 == 0`. The backend-probe subprocesses in both tools use dummy `64/64` values only to resolve the implementation, which is why they differ from the real conversion values — not a bug, but don't copy those numbers into real calls.

### Output format

Standard Safetensors. Per quantized layer: `<layer>.weight` as `I8` of shape `[rows, cols/2]` (INT8 container holding signed INT4), plus `<layer>.weight_scale` as `F32` of shape `[rows]`. Everything else preserved byte-for-byte. `__metadata__` carries `_quantization_metadata` (`format_version` 1.0 + per-layer `{format: convrot_w4a4, convrot_groupsize: 256}`) and `quantization: "ConvRot W4A4"`. See [ComfyUI/QUANTIZATION.md](ComfyUI/QUANTIZATION.md) for the upstream `QuantizedTensor` / `Layout` / `MixedPrecisionOps` model this format plugs into.

**Does `.backends.cuda` in `__module__` prove the native path ran? Measured 2026-08-22: yes, here.** The whole preflight rests on that string match, and the test that settles it is cheap: W4A4 quantizes the **activation** to 4 bits too, so a dequantized-weight fallback (`F.linear(x, W_deq)`, which is what `comfy_kitchen/tensor/convrot_w4a4.py:237` does under one condition) must agree with a real call. It does not — `native` vs `W4-only` is **1.43e-1**, not 1e-6, on a `[1024, 1024]` bf16 weight at cg=256. The A4 half is real. Re-run with `tools/probe_backend_resolution.py`.

`verify_w4a4.py` checks, in order: metadata structure and per-layer dtype/shape → packed shape vs source shape → **byte-identical comparison of every preserved tensor against the source** → native backend resolution → optional real-kernel smoke against `F.linear` on the BF16 source. The smoke's relative RMSE on random inputs (~0.25 for Gemma) is a liveness signal, **not** a quality metric.

`inspect_quant.py <file>` at root is a quick header dump (tensor count, dtype histogram, metadata, scale-like keys).

## Testing a converted model

Structural verification is not acceptance. Every output must additionally pass, in order: normal ComfyUI loader compatibility (real node, not a hand-rolled load), a prompt-encoding smoke test through a long-lived ComfyUI process, and a matched-parameter benchmark against the BF16 source (identical prompt, seed, steps, resolution, sampler, scheduler, frames) recording disk, VRAM, load time, s/it, GPU utilization, power, warnings, and visual quality. Workflows for this live in `ComfyUI/user/default/workflows/` (e.g. `Video-LTX2_MultiGPU.app.json` for the Gemma text encoder).

Record negative and unsupported results rather than hiding them.

## Project tracking

- [W4A4_PROGRESS.md](W4A4_PROGRESS.md) — the running log: candidate ranking, completed steps, skipped models with reasons, failed/blocked with exact error strings. **Update it as part of the work**, not afterwards.
- [W4A4_HANDOFF.md](W4A4_HANDOFF.md) — current state snapshot and the ordered next-steps list.
- `quantization_inventory.{json,md}` — generated; do not hand-edit.

A second, separate effort also runs on this stand:

- [CORTIQ_LTX25_HANDOFF.md](CORTIQ_LTX25_HANDOFF.md) — the `cortiq` / LTX-2.5 investigation: what was **measured** (the 52× end-to-end gap, where the time goes, the PV-NT accuracy result) and the five claims that had to be withdrawn. Read this before touching `F:\cortiq-cmf`.
- [.scratch/estado-entregavel/map.md](.scratch/estado-entregavel/map.md) — the **plan**: 17 tickets across both repos, with the blocking graph. Every debt ticket carries its closing criterion, written before anyone looked at the result. `.scratch/` is the local issue tracker; nothing in it touches GitHub.

### The GPU window

Three elements, agreed 2026-08-21, and deliberately only three: **block size, how to ask, how to
let go.** A longer protocol is one nobody follows.

**The block is 30 minutes and it has a name.** Not "I need the card" — `bench:ticket25_fbcache_visual`.
The owner string is what the other side reads when it is refused, and `cortiq:bench` tells it
nothing it can plan around. Work that cannot state its block in advance does not get one: it takes
the card for a measurement, not for a session.

**Since 2026-08-22 the benchmark tools take the lock themselves, so do NOT take it for them.**
`tools/_timing.py`'s `compare()` acquires `BenchGuard`, and every timing tool now routes through it
— `m_crossover`, `attn_bench`, `attn_dtype_ab`, both `compile_*` probes, `nunchaku_compare`,
`fbcache_probe`, `fbcache_visual`. Taking `Assert-GpuLock` first and *then* running one of them
makes the tool refuse its own run:

```
F:\GPU_BENCH.lock is held: owner=bench:m_crossover_regressao_primitiva pid=74572 alive=True
Not reclaiming it. If that run is genuinely dead, delete the file by hand
```

Measured 2026-08-22, by doing exactly that. The in-process `active_guard()` lets `compare()` join a
guard the **same process** already entered; a lock held by a separate PowerShell cannot be joined,
and correctly is not. So: **take the lock by hand only for work that does not go through
`_timing.compare()`** — a converter, `verify_w4a4 --kernel-smoke`, `quant_audit`, an ad-hoc probe.
For a benchmark, just run it; it announces the acquire, the occupancy of every device, and the
release.

**Asking is `Assert-GpuLock -Owner '<repo>:<what>'`.** It throws; never `Take-GpuLock | Out-Null`,
which returns `$false`, prints "not taking", and then runs the benchmark anyway — that happened on
2026-08-19 against a live sibling with a 2-second-old heartbeat. If refused: **measure the card
itself with NVML for about a minute — the lock file is not the card.** Card idle *and* lock held is
a problem, not a wait: say so in chat ("waited X, lock held by `<owner>`") and move to work that
needs no GPU. **Do not wait, and never take a live lock silently.**

**Letting go is `Release-GpuLock`, after the work stops, never before.** Holding the lock while
idle — even announced, even for "I want the card for a comparison later" — is a false claim on a
shared card; a free lock over a busy GPU is the same lie pointing the other way. `Release-GpuLock`
removes the file **only if it is still ours**, so a lock somebody legitimately took after ours went
stale is never clobbered.

Why any of this matters: contention moves numbers. Same code, same day, `im2col 74.4 s` on a quiet
machine against `87.0 s` on a loaded one — **17%**, against a 10 s effect being measured. And on
2026-08-19 a hand-written lock named a pid that was dead the instant it was written, so the card sat
idle behind it for roughly forty minutes.

**Holding the lock is not the same as using the locked card.** Verified 2026-08-21, executed:
`F:\cortiq-cmf` selects its wgpu adapter with `request_adapter(HighPerformance)` when
`CMF_GPU_ADAPTER` is unset, and on this host that resolves to the **RTX 3080 Ti**, not the 3090.
A whole measurement session ran on `cuda:1` — which was 9-26% busy in every `nvidia-smi` sample —
while the lock sat on an idle 3090. The probe cache is what exposed it, because it stamps the
adapter name into every line:

```
0.5.95	NVIDIA GeForce RTX 3080 Ti/Vulkan	gemm-nt	gpu
```

So for any cortiq measurement: **pin the card** with `CMF_GPU_ADAPTER=3090` (index into
`cortiq gpu`, or a case-insensitive substring of the adapter name), and confirm it afterwards by
pointing `CMF_PROBE_CACHE` at a file and reading which adapter it names. A lock on the wrong card
protects nothing and reads as protection.

**The lock itself was broken until 2026-08-22, in one direction only.** `gpu_lock.py` wrote
JSON; `gpu_lock.ps1` parses `key=value`. Every regex missed, so `Get-GpuLockState` returned an
empty owner and `StaleFor = [int64]::MaxValue`, and `Take-GpuLock` **reclaimed a live lock** while
printing `reclaiming from ` with a blank owner — which is indistinguishable from a leftover file
and is the one situation where reclaiming is correct. The reverse direction always worked
(`open(path, "x")` refuses correctly), and **that asymmetry is why it survived**: one direction was
perfect, so nobody had a reason to test the other.

Fixed on the **Python** side deliberately. Teaching the `.ps1` to read JSON would have reproduced
the same theft pointing at a sibling still running the old script — a format change is not
symmetric when the other party may be running last week's code. `key=value` already had two
readers, so it won, and no sibling has to update anything. `tools/test_gpu_lock.py` drives each
implementation against a lock the *other* one wrote (23 checks, all passing), because reading
either half alone never reveals a format disagreement — which is the general lesson, not a detail
about this lock.

**Provenance of this section:** agreed with the bench owner on 2026-08-21, when only one session
held the card ("gpu liberada, so tem voce agora"). It is therefore **one side's protocol written
down, in force until contested** — not a negotiated settlement between two live sessions. The
mechanism it describes (`tools/gpu_lock.ps1`, the detached heartbeat, the 55 s staleness limit) was
executed; the *agreement* is a decision, not a measurement.

## Agent skills

Written by `/setup-matt-pocock-skills` on 2026-08-22. These three files are what the engineering
skills read as input; edit them directly rather than re-running the setup.

### Issue tracker

Local markdown under `.scratch/`, one file per ticket — this repo has **no git remote**, so there
are no GitHub issues and `gh` has nowhere to write. Every debt ticket carries its closing criterion,
written before anyone looked at the result. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).

### Triage labels

The five canonical roles, unchanged, recorded as a `Status:` line in each ticket file.
`ready-for-human` is load-bearing here: package installs, GPU windows and anything touching WSL are
the owner's call and cannot be delegated. See [docs/agents/triage-labels.md](docs/agents/triage-labels.md).

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the root, both created lazily. The provenance
rule extends to them: a glossary entry derived from reading rather than running says so in the
entry. See [docs/agents/domain.md](docs/agents/domain.md).

## Memory gotchas

Conversions check free disk (`estimate + 1 GiB`) and free RAM (`3 × largest selected tensor + 2 GiB`) and exit rather than thrash. If it refuses, the fix is to close memory-heavy WSL/worker processes **manually** — never change the pagefile or kill processes automatically.

Known benign noise: `ModelPatcher.__del__` prints an `ON_DETACH` AttributeError on short-lived interpreter shutdown; loading itself still exits 0.
