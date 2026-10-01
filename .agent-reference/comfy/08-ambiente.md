# Ambiente e dependências

> Referência preservada do CLAUDE.md original, linhas 658–685, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

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

