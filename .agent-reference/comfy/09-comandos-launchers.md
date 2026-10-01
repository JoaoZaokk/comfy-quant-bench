# Comandos, launchers, testes e dynamic VRAM

> Referência preservada do CLAUDE.md original, linhas 686–836, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Commands

Launch:

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build
```

```bash
.\iniciar_comfy.bat fila
```

**2026-09-29: os 12 `run_*.bat` da raiz viraram 3 modos de `iniciar_comfy.bat`**: `imagem` (padrão; MGPU desligado, `--disable-dynamic-vram`), `video` (idem + `--cache-none --preview-method none`) e `fila` (orquestrador, dynamic VRAM ligado). Todos em `--listen 0.0.0.0 --port 8190`, **sem** `--enable-manager`: o dono tirou a flag em 2026-09-30 porque o Manager embutido bloqueia o `custom_nodes\ComfyUI-Manager`, que é o que ele usa. Argumentos extras vão para o main.py; `COMFY_DRYRUN=1` mostra a linha sem subir. Os originais estão em `_launchers_antigos/`, fora do Git. `imagem-ultra` caiu porque `--cache-ram` já é o padrão do 0.37 (só tirava o preview); `lan` caiu porque o WinError 64 também atinge 127.0.0.1 (nota abaixo). Não migrados: `ultra_image - sem dynamic` (dynamic VRAM ligado + `--disable-pinned-memory`). As notas abaixo citam os nomes antigos e continuam valendo como histórico.

**2026-09-30: LTX deu OOM no modo `video` porque o arquivo estava sem `--disable-dynamic-vram`** (lido no `comfyui_8190.log`, 08:00). O LTXAV (Eros 10 W4A8, do NAS) passou por `prepared for dynamic VRAM loading. 14270MB Staged` e morreu no primeiro passo em `comfy/model_prefetch.py` → `cast_to_gathered` com `CUDA error: out of memory`, seguido de `aimdo memory compile error` no cleanup. O workflow também usa `ModelMemoryUsageFactorOverride` (0,077 → 0,046), que reduz a reserva de ativação estimada; hipótese não medida: com dynamic VRAM isso deixa pesos demais residentes. Flag devolvida ao `video` em 2026-10-01 (dry-run conferido).

**2026-09-29: raiz extra lenta não trava mais a subida.** `ComfyUI/utils/extra_config.py` testa as pastas de cada seção do `extra_model_paths.yaml` numa thread e pula a seção que não responde em 10 s (`SECTION_TIMEOUT`), com aviso `Skipping extra model paths '<seção>'`. Motivo: com o NAS (`P:`) ocupado por cópia, cada `getmtime` levava dezenas de segundos, o servidor não atendia e o navegador caía em WinError 10054. A seção volta sozinha na próxima subida em que responder. Só age na subida. Patch: `patches/comfyui_extra_paths_timeout.patch`.

**Atualização 2026-10-01: os loaders Nunchaku Z-Image e Qwen-Image rodam com dynamic VRAM** depois do patch local `patches/nunchaku_eager_linear_dynamic_vram.patch`. O patch desliga o lazy Linear só durante a construção do modelo. Ele também aceita o `fast_disk` que o `clone()` do 0.37.4 passa, falha que independe do dynamic. Medido em 20/20 renders. Dyn × nodyn diverge tanto quanto o mesmo braço repetido, porque o Nunchaku não é determinístico: NZ ~14 dB, NQ ~32 dB. O Z-Image comum sai idêntico pixel a pixel nos dois braços. Detalhes em `.scratch/dynamic_2026-10-01/resultado.md`. Continuam sem teste com dynamic: LoRA stack, Flux, TE, IP-Adapter e PuLID Nunchaku. O parágrafo abaixo vale para o nó sem o patch, e para o LTX 2.5, que segue precisando da flag.

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

**And dynamic VRAM has a second failure mode that has nothing to do with memory: its lazy load
does not convert dtype.** Measured 2026-09-12. With aimdo on, `comfy/ops.py:552` assigns a
Linear's weight **straight from the file** and never casts it to the module's dtype. That is
invisible while a checkpoint carries one floating dtype, and every checkpoint this bench had run
until then did. `krea2_turbo_bf16` does not: its 256 `blocks` and `txtfusion` weights are BF16
and its **ends** -- `first`, `last.linear`, `tmlp`, `tproj`, `txtmlp` -- are **F32**, which is
the model's own precision policy. The result, after reading all 24.5 GiB:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
  comfy/ldm/krea2/model.py:331, img = self.first(img)
```

The message names the first layer of the model and says nothing about the mechanism. Without
dynamic VRAM the same file loads with `first.weight` in bfloat16 and samples normally -- verified
by varying that one axis, after three probes that held it fixed and reproduced nothing. Note the
trap inside the trap: `load_clip(..., disable_dynamic=True)` and
`load_diffusion_model(..., disable_dynamic=True)` do **not** protect you, because
`tools/_dynamic_vram.enable()` turns aimdo on globally before either call.

`tools/_dynamic_vram.perigoso_para_lazy()` answers it from the header alone: the conflicting
floating dtypes among **2-D** tensors, or `None`. 2-D because that is what becomes an `F.linear`
weight -- a 1-D F32 scale beside BF16 weights is normal. `beyond-reality-zimage-v2_native`
returns `None`; all four Krea2 arms return `{BF16, F32}`.

**And a uniform-dtype file is not safe either — the second mode, measured 2026-09-14.**
`capybara_v0.1` is BF16 in all 568 of its 2-D tensors, so `perigoso_para_lazy()` returns `None`,
and the ladder died in the sampler anyway: `mat1 and mat2 must have the same dtype, but got Half
and BFloat16`, at `time_in`, the first lazy Linear after the input conv. `HunyuanVideo15` lists
`supported_inference_dtypes = [float16, bfloat16, float32]` and `unet_dtype()` walks that list in
order, so the model computes in **fp16** whatever the file holds; the normal loader casts every
BF16 weight to fp16 on the way in, the lazy loader leaves the Linears in BF16, and the conv, which
is not lazy, comes out fp16. The guard was asking the wrong question — "does the file disagree
with itself?" — when the comparison that fails is file against **compute dtype**.
`quality_ladder.py` now casts every arm to `model.model.get_dtype()` after loading, skipping
`QuantizedTensor` weights (int8 container with the scale inside the layout; `.to(dtype)` on it is
not a weight cast), and `casta_pesos_divergentes` takes that target explicitly — its
majority-dtype default would have cast the conv to BF16 and moved the crash one layer earlier.
[JULGAMENTO] the 2026-09-01 capybara ladder never hit this because the ladder's lazy load arrived
with the Krea 2 work around 2026-09-12 (the tool's own comments date it); not re-run to confirm.

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190
```

**But `--disable-dynamic-vram` also makes two of the most-downloaded ConvRot checkpoints unloadable, and the error blames the file.** Measured 2026-08-31. Both `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` files carry bytes **past the last tensor** — 83 in FL2VA, 64 in Ref2VA — so `safetensors.safe_open` refuses them:

```
SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

That is not a broken download, and checking was cheap: the server's `Content-Length` matches ours byte for byte (15903012791 and 15093774276), and a `Range: bytes=-83` request returns the same trailing bytes we have. FL2VA's are readable text — `\nL2P_bypass_MiniMax_H3_FL2VA_..._convrot.safetensors_1785789862\n` — and Ref2VA's are 64 bytes of binary. **They are in the published file.**

The dynamic-VRAM path uses a different reader (`comfy/utils.py:85`, `comfy_aimdo.model_mmap`) which accepts them: with aimdo initialised both files open, 932 and 1132 tensors. So these checkpoints work for almost everyone — dynamic VRAM is the default — and fail for exactly the configuration this bench needs for Nunchaku and LTX 2.5. `minimax_h3_fl2va_pruned-w4a8_convrot_pruned` (Winnougan) has no trailing bytes and opens either way.

Launcher notes. There are **13** `.bat` launchers in the root, not seven — count them, do not quote this. The claim that **none** of them passes `--disable-dynamic-vram` was true on 2026-08-21 and is **false since 2026-08-26**: measured 2026-08-30, **four** pass it — `run_nvidia_gpu_8190_dual_component.bat`, `run_nvidia_gpu_8190_ultra_image.bat`, `run_nvidia_gpu_8190_ultra_video.bat`, `run_nvidia_gpu_8190_void_cache_none.bat`. Those four also set `COMFYUI_MGPU_DISABLED=1`. `run_nvidia_gpu_8190_loopback.bat` (Sage, `127.0.0.1:8190`) still does **not** pass the flag, so for a Nunchaku SVDQuant loader or the LTX 2.5 workflow above, use one of the four, call `main.py` by hand, or use `F:\cortiq-cmf\run_e2e_comfy.ps1`. Following "use the loopback launcher" alone will reproduce the 35 GB staging OOM. Other launchers: `run_nvidia_gpu.bat`, `run_nvidia_gpu_fast_fp16_accumulation.bat`, `run_nvidia_gpu_8190_flash.bat` (FlashAttention), `run_8190_limpo.bat`, `run_cpu.bat`.

**The `WinError 64` accept-loop death is NOT specific to the `0.0.0.0` launcher.** This file said `run_nvidia_gpu_8190.bat` binds `0.0.0.0` and the asyncio accept loop dies with `OSError(22, 'The specified network name is no longer available', 64)` whenever NordLynx/NordVPN reconnects — which reads as "loopback is immune". It is not. Observed 2026-08-30 at 04:29:19 on a server launched by `run_nvidia_gpu_8190_ultra_video.bat`, from its own log:

```
socket: <asyncio.TransportSocket fd=3000, family=2, type=1, proto=6, laddr=('127.0.0.1', 8190)>
```

`127.0.0.1`. Binding loopback does not protect against it. What *did* hold is the other half of the note: **the process stays alive with a dead port** — accept died at 04:29:19 and the process only ended at 04:35:18, six minutes of a listening-looking server that accepts nothing. `curl` returns `http=000` and `netstat` shows nothing `LISTENING`; the PID being alive proves nothing. Both `NordLynx` and `CloudflareWARP` tunnels were `Up`, plus a `cloudflared` container — three things that churn routes on this host.

**`main.py --windows-standalone-build` re-executes itself as a child process**, verified 2026-08-19 (executed, not traced). Killing only the PID that `Start-Process` returns leaves that child running as an orphan. On this bench the orphan held **20,578 MiB of the 3090** and **16.67 GB of RAM**, invisibly: `Get-Process -Id` reported the launched PID as dead, while `nvidia-smi` still showed the VRAM occupied. Kill the process tree, not the PID.

Tests (run from the root; `pytest.ini` lives in `ComfyUI/` with `pythonpath = .`).

**`pytest` 9.1.1 and `ruff` 0.16.5 ARE installed since 2026-09-01** — the owner asked for them. This
section said for two weeks that neither existed, which was true from 2026-08-18 until then.

**Installed with the whole stack pinned, and that is the part worth copying.** A `-c constraints.txt`
built from `pip freeze` (287 pins) means a resolver cannot move anything that already exists while
satisfying the new packages. Diff of the freeze afterwards: **exactly four additions** — `pytest`,
`ruff`, `pluggy`, `iniconfig` — nothing removed, nothing changed, and torch 2.13.0+cu130 /
numpy 2.4.6 / safetensors 0.8.0 / transformers 4.57.6 all re-imported with `cuda.device_count() == 2`.
Do the same for any future install here; a bare `pip install` on this stack is what the
*do-not-mass-upgrade* rule exists to prevent.

**`ruff` earned itself back in ten minutes.** First run over `tools/`: 538 findings, and one of them
was a `NameError` **committed the same hour** — `avaliar_despacho.py` referenced a variable renamed
during a rule rewrite, so every freshly-computed `NAO_DESPACHA` would have raised instead of
producing a verdict. It did not show up in the table because the only row with that verdict came
from cache. Fixed and verified by forcing a non-cached run. The rest of the 538 is mostly noise
worth reading rather than fixing wholesale: 182 unused-`noqa`, 81 unsorted imports, but also
**64 blind-except** — the exact `except Exception: pass` that this repo records as having swallowed
evidence for three debugging rounds — and 30 `function-uses-loop-variable`, concentrated in the
benchmark files. Do **not** mass-apply `--fix`: 322 are auto-fixable and that would be a large diff
across a toolchain whose numbers are published.

Test files written for this project (`tools/test_svdq_verify.py`, the suites under
`Comfy-WaveSpeed-Fixed/tests/`) still carry their own runner and are executed directly, because
they were written when pytest was absent:

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

