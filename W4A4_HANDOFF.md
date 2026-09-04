# ConvRot W4A4 Handoff

## Scope and safety rules

This is the live ComfyUI Portable installation at `F:\COMFY_PORTABLE`. Always use `F:\COMFY_PORTABLE\python_embeded\python.exe`; never use global Python. Do not delete, overwrite, move, or requantize original models. Do not mass-upgrade dependencies. W4A4 must execute through the native ConvRot CUDA backend, not eager/dequantized BF16 GEMM. **Do not stop or reconfigure WSL** — but not for the reason this line used to give: it said "WSL currently hosts unrelated Qwen/DeepSeek work", and that stopped being true. Measured 2026-08-30 in both docker contexts and confirmed by the owner: the ERP moved to Proxmox, and what runs there now is `glm-w4` (his own GPU training, which holds the 3090), a MacroLog rollback copy, and a cloudflared tunnel. The rule stands because `wsl --shutdown` is still destructive to *those*. See CLAUDE.md for the current container list.

**And the W4A4 rule above has a boundary, measured 2026-08-31:** it is unreachable for any **text encoder** in stock ComfyUI, whatever the checkpoint says, because `comfy/sd.py:269` calls `set_model_compute_dtype(torch.float32)` on every CLIP object. Counted on a real encode: 0 calls to the 4-bit path, 336 dequantizes. Diffusion models are unaffected — Z-Image W4A4 measured 340 quantized forwards and 0 dequantizes. See `W4A4_PROGRESS.md` parts 29-32.

## Current environment

- Python 3.13.12
- Torch 2.13.0+cu130, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130
- CUDA reported by Torch: 13.0
- nunchaku 1.2.1 (built for torch 2.11, running under 2.13), spas_sage_attn 0.1.0 (SpargeAttn)
- `pytest` 9.1.1 e `ruff` 0.16.5 **ESTAO instalados desde 2026-09-01** (esta linha dizia que
  nenhum dos dois existia, o que foi verdade de 2026-08-18 ate la). Instalados com
  `-c constraints.txt` feito do `pip freeze`: exatamente quatro adicoes, nada movido. Os testes sob
  `tools/` ainda carregam o proprio runner porque sao anteriores a instalacao.
- comfy-kitchen **0.2.31**, read from `python_embeded/Lib/site-packages/comfy_kitchen-*.dist-info/METADATA`
  on 2026-08-22. This line said `0.2.23` for eight releases; it is the registry that decides whether
  `convrot_w4a4_linear` resolves to a CUDA backend, so recheck it rather than quoting this line.
- ComfyUI version `0.33.0` (`v0.33.0-19-gc1739380`). It was `0.29.0`/`42d2aa55` earlier in this
  project's life; anything in this file that assumes 0.29 behaviour is suspect.
- GPUs: RTX 3090 24 GB (`cuda:0`) and RTX 3080 Ti 12 GB (`cuda:1`)
- Only the Torch/vision/audio trio was replaced, using embedded pip, same versions, `--no-deps`. Pre-change freeze: `_pip_freeze_before_w4a4_cu130_20260816.txt`.
- `pip check` passes.

~~The existing SageAttention wheel initially failed because it needs `cudart64_12.dll`, so a
pre-existing local CUDA 12.6 runtime DLL was copied into
`python_embeded\Lib\site-packages\torch\lib\cudart64_12.dll`.~~

**OBSOLETE — and the file is NOT gone: it was renamed to `cudart64_12.dll.disabled` and is still
sitting there. Leave it disabled; do not restore it.** (This headline said "the file is gone" until
2026-09-01, which is false and is the exact claim the correction three paragraphs below refutes. It
is fixed here because a skimmer reads the bold line and stops.) This paragraph told a reader to
restore a DLL that the stack no longer needs, which is the most expensive kind of stale doc —
following it would reintroduce a CUDA 12.6 runtime beside a cu130 Torch. The accel wheels were
reinstalled as cu130 builds on 2026-08-16 and link `torch_cuda.dll`, not `cudart64_12.dll`.

This is settled by **execution, not by import success** — an import only proves DLL resolution; a
forward pass proves the kernel runs. `_check_accel.py` was run on the **RTX 3090 on 2026-08-22**
and returned ALL GOOD: triton v3.7.1 compiled and ran; `sageattention` mean|d| = 0.0006 against
SDPA; `flash_attn` mean|d| = 0.0000.

**Correction, same day, and it is worth more than the fact it corrects.** The line above first
read "the file is gone", citing `find python_embeded -iname "cudart64*.dll"` → only
`cudart64_13.dll`. **That pattern cannot match a name ending in `.disabled`**, so the evidence was
blind by construction. Dropping the `.dll`:

```
python_embeded/Lib/site-packages/torch/lib/cudart64_12.dll.disabled   556,544 bytes
  sha256 d954ca542b3b6bcf03cc2b798a7d00051501cf734ca751050e986af505cf9dad
```

The file was **renamed, not deleted** — which `W4A4_PROGRESS.md:339-340` already recorded, from a
run. Windows will not load a `.dll.disabled`, so "do not put it back" stands; **"it is gone" does
not.** Keep the sha256: the file is still in `torch/lib/` unlabelled and the hash is the only
thing that identifies it. It is byte-identical to the copy at
`venvs/ultravox311/Lib/site-packages/torchvision/cudart64_12.dll`, which belongs to the unrelated
Ultravox venv — leave that one alone too.

Absence in a glob is not absence on disk. This repo's own rule, broken twice in one day by two
different writers, in the two documents that state it.

If anyone doubts it again, re-settle it the same way instead of copying the DLL back:

```powershell
.\python_embeded\python.exe -s .\_check_accel.py
```

That needs the card — read the GPU-window rules in `CLAUDE.md` before taking it.

## Audit and tools

- Inventory: `quantization_inventory.json` and `quantization_inventory.md`. **Do not quote a file
  count or a total size from here.** This line said "157 files, 608.56 GiB"; the generated JSON
  (stamped 2026-08-19) says 159 entries and 609.61 GiB, and it moves every time a model lands.
  Read it from the artifact instead — `python_embeded\python.exe -s -c "import json;d=json.load(open('quantization_inventory.json'));print(len(d['models']), d['total_size_bytes']/1024**3)"`
  — and rerun `tools\quant_audit.py` if the `generated_at` in that file is older than the last output.
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

### Estado em 2026-09-01 (leia isto antes da lista de 2026-08-19 abaixo)

O avaliador em lote existe, em duas camadas, e a lista mais antiga desta secao nao o conhece.

| camada | ferramenta | GPU? | o que decide |
|---|---|---|---|
| 1 | `tools/avaliar.py` | **nao** | cabecalho, sidecar, analise. 163 checkpoints em 0,68 s. Mediana do erro efetivo offline. |
| 2 | ainda nao existe | sim | contar forward quantizado de verdade contra chamadas a `dequantize` |
| 3 | `tools/avaliar_referencia.py` | sim | o arquivo NAO quantizado responde ao proprio condicionamento? |

Os vereditos sao `REPROVADO` / `OLHAR` / `SEM VEREDITO`. **`APROVADO` nao existe** e a ausencia e
deliberada: nenhum corte medido nesta bancada separa usavel de inutilizavel, nos dois eixos que
temos. A camada 3 foi validada contra o par de verdade conhecida do Wan e separou destruido de bom
por **3,19x**, com criterio pre-registrado em `bench/criterio_guarda_referencia.md`.

Aberto e sem GPU: a camada 2 precisa de placa; a migracao dos conversores para `tools/_conversion.py`
(ticket 08) nao precisa. Ver `W4A4_PROGRESS.md` partes 44 e 45.

### A lista de 2026-08-19

Two separate tracks. Everything in the first needs the GPU; the second does not.

### Needs the GPU (blocked while it is lent out)

1. ~~**`--promote-error 0.15` was picked, not derived.**~~ **Varrido 2026-08-19** (parte 18).
   Duas correcoes ao que esta escrito abaixo:

   - **"the images do not visibly separate" era falso.** Separam muito. O W4A4 puro transforma o
     bloco de pistoes de um trompete num emaranhado, nas duas seeds olhadas; com camadas promovidas
     o mesmo prompt sai coerente. Ver `bench/quality_ladder/*.png`.
   - **Divergencia de latente nao acompanha o defeito visivel.** O default 0,15 fica
     estatisticamente empatado com o W4A4 puro na metrica (delta -0,0002, vence 3 de 12 runs
     pareados) e mesmo assim a imagem dele e claramente melhor. A metrica mede quanto a composicao
     inteira andou, e a composicao anda de qualquer jeito.

   **0,15 nao se sustenta**: promove 55 camadas, custa +16% de tempo e nao ganha na metrica.
   0,10 (119 camadas) vence 12 de 12 com -0,1033. Texto original abaixo, mantido porque a parte da
   metodologia continua valendo: sweep contra imagens reais em varias seeds; `--analysis`
   re-decide em segundos, entao a varredura custa so o tempo de geracao.
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
`nunchaku/models/linear.py:152` but `ComfyUI-nunchaku/models/zimage.py:62` carries the same pattern on
another path. Each PR points at the other. Until they land, this installation needs
`--disable-dynamic-vram` for any SVDQuant workflow.

Local modification not covered by any PR: `python_embeded/Lib/site-packages/spas_sage_attn/core.py`
carries the same `km = None` fix as SpargeAttn#121, applied to the 3 occurrences the woct0rdho
wheel has (upstream has 5). A reinstall of that wheel silently reverts it.

Do not start by installing packages. Do not touch the WSL jobs. No original model has been
modified.

## Estado em 2026-08-19

**O repositorio existe.** A raiz virou repo git com `.gitignore` em allowlist. **Nao decore o
conjunto rastreado — liste.** Esta frase dizia "`tools/`, `custom_nodes/`, os `.md` da raiz e dois
scripts", e desde entao entraram `docs/`, `.scratch/` (o issue tracker local) e `calib/`. Em
2026-08-22 o conjunto e:

```bash
git ls-files | wc -l                                    # quantos
git ls-files | awk -F/ 'NF==1{print "raiz"} NF>1{print $1"/"}' | sort | uniq -c   # onde
```

**`git add -An --dry-run` antes de qualquer `git add`** — um denylist que erra uma entrada tenta
commitar um safetensors de 42 GiB.
`ComfyUI/` e checkout aninhado com remote proprio e git nao desce nele; nada la dentro pode ser
rastreado daqui.

### Ferramentas novas

```
tools/to_native.py             diffusers -> naming nativo do ComfyUI (obrigatorio antes de quantizar Z-Image)
tools/calibrate_activations.py ativacoes reais capturadas durante amostragem
tools/quant_mixed.py           4 ou 8 bits por camada, medido contra o kernel
tools/m_crossover.py           onde o int4 passa a ganhar do 16-bit (M ~ 128-256; --repeats/--reverse)
tools/w4a4_breakdown.py        kernel a kernel, e a divisao host/GPU de uma chamada
tools/graph_capture_probe.py   CUDA graph: captura? replay bate com eager? quanto de host sai?
tools/dispatch_census.py       conta o ramo que cada Linear quantizado tomou numa geracao real
tools/w4a8_fallback_sweep.py   quais shapes fazem o W4A8 desistir do kernel (le os booleanos)
tools/quality_ladder.py        divergencia de latente + imagens por checkpoint, pareado por seed
tools/predict_promotion.py     estatistica de peso preve err_w4a4? (nao: acaso)
tools/synthetic_vs_real.py     ativacao sintetica substitui a calibracao? (nao: pior que acaso)
tools/profile_transfer.py      compara N analises par a par; perfil de um checkpoint serve noutro?
tools/attn_dtype_ab.py         fp16 vs bf16 nos backends de attention
tools/gpu_lock.py              exclusao mutua com a sessao irma
tools/_bench_guard.py          lock + ocupacao NVML, falhando fechado
tools/_ram_guard.py            acumulacao real, nao a estimativa de streaming
custom_nodes/comfy-quant-preflight/   recusa workflow cuja config de quantizacao nao pode ser verdade
```

### O que foi corrigido, e o que nao foi

`AUDITORIA_2026-08-18.md` tem os 127 achados (marcados como **hipoteses nao verificadas**) e a
secao 6 lista o que foi consertado sem GPU. Nao consertado de proposito:

- `test_svdq_verify` cobre `split_fused` agora (provado por mutacao) mas **nao** `recover_weight`
  — essa monta a camada por dentro do nunchaku e precisa de GPU. O runner imprime esse buraco em
  toda execucao.
- `quant_audit` conta INT4 empacotado certo, mas `dtype_bytes` continua sendo bytes de container.
  Isso e proposital e agora esta documentado no topo do `.md` gerado.

### Fila que precisa de GPU

1. ~~**O mestiço fp8/4-bit.** `comfy/sd.py:2303` da o `dtype` do widget ao `unet_dtype` mesmo com
   `quant_config` setado, enquanto `:2306` protege o `manual_cast_dtype`.~~ **Medido 2026-08-19**
   (parte 14). Nao trava e nao cai para eager: 680/680 nativo nos tres widgets. Converte para fp8
   os **207 tensores nao quantizados** (normas, embeddings, modulacao) e move o latente mais que
   uma LoRA inteira — 747,06 → 828,08 com `fp8_e5m2`, contra 728,99 da LoRA. Silencioso.
   **Texto de PR/issue pronto, nao publicado** — falta decisao do usuario.
2. ~~`tensor/convrot_w4a4.py:237` — quem transpõe? Monkeypatch contador num forward real.~~
   **Feito 2026-08-19** (parte 13). Transpõe o próprio ComfyUI, 680x numa geração de 4 passos —
   mas via `aten.t` + `aten.mm`, onde `transposed=True` é o estado *exigido* e o kernel roda.
   O ramo que dequantiza precisa de `aten.linear`, que nunca é chamado nesse caminho.
   680/680 nativo. Falta `torch.compile`.
3. ~~`cuda/__init__.py:2213` e `:2261` — achar shape que o CUTLASS recusa, provar o fallback eager.~~
   **Feito 2026-08-19** (parte 16). Regra: `out_features %% 8 != 0` cai no eager. Silencioso,
   numericamente correto (0,0736 vs 0,0737) e **5-6x mais lento**. Repro em
   `tools/w4a8_fallback_sweep.py`. Nota anterior: instrumentei os quatro `_C.*` do caminho W4A8 e em M=5600 e 5700 so
   `w4a8_codebook_linear_chunked` e chamado, retornando `True` — o fallback eager nao foi
   alcancado por esse lado. Falta achar shape que faca `used` voltar `False`.
8. **Parcial 2026-08-19** (parte 17): esta dentro do `w4a8_codebook_linear_chunked` — o mesmo
   op com N=3841 cai no eager e **captura** no mesmo M em que N=3840 recusa. Falta o mecanismo,
   que esta no `.pyd`. Rascunho de reporte em `UPSTREAM_REPORT_w4a8_capture.md`.
   Enunciado original: achar a causa da recusa de captura do W4A8 acima de
   M x K ~ 21,8e6 (parte 12). Esta dentro do `.pyd`; daqui so deu para caracterizar. Se o
   comfy-kitchen tiver fonte disponivel, e um bug reportavel com repro exato em tres linhas.
4. ~~LoRA sobre modelo quantizado (`ops.py:1377`) — hipotese, hoje so aviso no preflight.~~
   **Feito 2026-08-19** (parte 13). Nao dequantiza: 680/680 nativo com a LoRA aplicada e em
   efeito (latente move de 747,06 para 728,99). Hipotese refutada. Continua em aberto o outro
   lado: se o delta de LoRA sobre peso de 4 bits custa **qualidade**. Isso ninguem mediu.
5. ~~A mutacao do `.T` em `recover_weight`.~~ **Feito 2026-08-19** (parte 15). O `.T` esta
   correto: orientacao certa da rel 0,101 contra a propria camada, a transposta da 1,413.
   Coberto agora por `gpu_recover_weight_returns_the_layers_own_linear_map`, provado por mutacao.
6. ~~Escala BF16 do nunchaku contra o quantizador real.~~ **Feito 2026-08-19** (parte 15). Sem
   fator sistematico: alpha de minimos quadrados entre 0,992 e 0,998 em cinco camadas, e corrigir
   por alpha melhora o residuo em menos de 0,4%. O resto e ruido int4 esperado.
7. ~~`m_crossover` em ordem invertida de M (contraprova de efeito de ordem).~~ **Feito
   2026-08-19**, parte 11 do `W4A4_PROGRESS.md`. Sem efeito de ordem: ascendente e descendente
   cruzam no mesmo intervalo. Mas o controle (duas execucoes ascendentes) mostrou +/-20% de erro
   entre execucoes de disparo unico, e a ferramenta ganhou `--repeats` intercalado. E ela nunca
   tinha sido executada — morria em `ModuleNotFoundError: No module named '_bench_guard'`.

### PR pendente, com permissao ja dada e nao usado

Nenhum remote esta configurado, entao nada foi publicado. O unico candidato que **eu mesmo provei**
e o das chaves de quantizacao em `convert_diffusers_mmdit` — provado por leitura do mapa e por
`to_native.py` produzir latente bit-identico. Os outros dois candidatos (`_full_precision_mm`
inerte, `weight_correction` nunca lido) sao achados de auditoria nao verificados e **nao devem
virar PR antes de medicao**.

## Retomada: LTX 2.5 int8 pela UI (parte 28 do PROGRESS)

**Estado:** o workflow de aceitacao nao rodou de ponta a ponta. Parou no `CLIPTextEncode`. A causa
esta identificada e a correcao **nao foi testada** — a GPU passou para a sessao irma no meio.

**Proximo passo, nesta ordem:**

1. Subir pelo `run_nvidia_gpu_8190_loopback.bat`. Se subir por ferramenta, usar `Start-Process`
   para o usuario ter janela e poder fechar — subir em background pelo harness deixa o servidor
   sem como matar pela UI. Alternativa que funciona nos dois casos: `comfy stop --port 8190`.
2. Carregar `user/default/workflows/LTX25-int8-acceptance-v2.json` **do disco**, sem reaproveitar
   canvas editado.
3. Rodar. Se falhar, ler o log antes de mexer em widget.

**Duas armadilhas que custaram a sessao inteira. Nao redescobrir:**

- **`type` do CLIPLoader tem de ser `ltxv`.** Qualquer outro valor nao da erro: cai no fallback
  STABLE_DIFFUSION (`nodes.py:1024`), fareja o state dict e monta um Gemma3-12B puro, cuja saida e
  4-D. O sintoma final e `RuntimeError: Tensors must have same number of dimensions: got 4 and 3`
  no `comfy/ldm/lightricks/embeddings_connector.py:290`, a tres camadas de distancia da causa.
  O caminho completo importa: ha **tres** `embeddings_connector.py` nesta arvore (mais dois em
  `custom_nodes/ComfyUI-LTXVideo/` e `custom_nodes/ComfyUI_LTX2_SM/`), e so o do `comfy/ldm/` tem
  o `torch.cat` que estoura. O sinal barato no log e
  `clip missing: ['vision_model...']`.
- **Nenhum no MultiGPU, e nada fora de `cuda:0`.** `ComfyUI-MultiGPU/p2p_registry.py:20` faz
  `ctypes.CDLL("libcudart.so")` sem ramo Windows, e o chamador nao captura. Como o pacote
  monkeypatcha o `_wrap_for_dlpack` do comfy_kitchen no import, **qualquer** tensor quantizado num
  device diferente do de execucao mata a run. Nesta maquina a 3080 Ti esta fora para modelo
  quantizado enquanto o pacote existir.

**Ferramenta nova disponivel** (venv isolada `venvs/comfymcp`, `python_embeded` intocado):

```bash
venvs/comfymcp/Scripts/comfy.exe validate --workflow <wf.json> --input <object_info.json>
```

Valida grafo offline, sem servidor e sem GPU — converte UI->API e confere class_types, shapes,
enums e fiacao. **Nao pega semantica**: os dois workflows que quebraram passam limpos nele.
Tambem ha `comfy stop --port` e `comfy free --unload-models --free-memory` (devolve VRAM sem
derrubar o servidor). Registrado como MCP em escopo user, com `DO_NOT_TRACK` e
`COMFY_NO_TELEMETRY` ligados; nenhuma ferramenta do MCP foi exercitada ainda.

**Encerrado, nao reabrir:** o `WARNING: unet unexpected: [... .comfy_quant]` **nao** indica perda
de despacho. Medido duas vezes por caminhos independentes — contagem de modulos na parte 27, e
auditoria do header na parte 28.


---

## Estado em 2026-09-01, fim do dia: esparsidade medida e a foto que derrubou a conclusao

Sessao longa. Duas frentes, e as duas terminaram com a imagem contradizendo o numero.

### 1. Esparsidade 2:4 -- o caminho executa, e nao serve para o Z-Image

**O que estava bloqueado e nao esta mais.** O kernel 2:4 do xformers recusava esta placa com
`Got CUTLASS error: Error Internal`. Nao era o cuSPARSELt (o CUTLASS nao o usa) nem a placa: o tile
que eles compilam pede **139.264 bytes** de shared e a 3090 aceita **101.376** -- dimensionado para
a A100. `tools/sparse24_sm86/` compila o SparseGemm com um tile que cabe.

**MEDIDO**, contra `torch.mm` denso bf16, nos shapes reais do Z-Image, controle bit-exato:

    2:4 bf16    9,0 bits/peso    1,7x - 1,95x
    2:4 int8    5,0 bits/peso    3,2x - 4,1x
    2:4 int4    2,5 bits/peso    5,0x - 7,6x

A escada e consistente: cada metade de largura vale ~2x, que e o que o tensor core faz. E
`torch._int_mm` **denso** da 1,0x -- o ganho e do tensor core esparso, nao de ser inteiro.

**Descoberto por experimento, nao lido:** o tensor core INT4 mascara em **PARES**, nao em valores.
`kElementsPerElementE` e 32 no int4 contra 16 no int8, o que forca 4 bits de metadata por 8 valores.
Testadas as duas granularidades contra referencia inteira exata: por elemento falha 6/6, por par
bate 6/6, codificacao `nibble = idx0 | (idx1 << 2)`. Da **2,5 bits/peso**, nao 3,0.

**E a foto diz nao.** Render real do Z-Image, 3 sementes:

    BF16                                        maca
    W4A4 ConvRot (4,0 bits, erro 0,0956)        maca
    2:4 par Wanda + int4 (2,5 bits, 0,1391)     RUIDO
    + ConvRot antes da poda                     RUIDO

**Isolado, um eixo por vez, e o culpado e a granularidade:**

    so poda 2:4 por ELEMENTO, bf16, sem quantizar   maca arruinada, mas EXISTE
    so poda 2:4 por PAR,      bf16, sem quantizar   RUIDO

Nao e o int4 (a escala por linha que usei da 0,0685, **melhor** que os 0,0923 do W4A4 que funciona).
Nao e a rotacao. **E o par** -- a restricao que o hardware INT4 impoe. Logo:

- **INT4 esparso esta morto para o Z-Image**, por restricao de hardware e nao por bits.
- **INT8 esparso sobrevive** (aceita por elemento): 4,0x, 5,0 bits/peso, sujeito preservado.
- Mesmo por elemento esta ruim demais para enviar sem **treino de recuperacao** -- a proposta
  original do dono, ainda **nao testada**, e agora o unico caminho aberto nesta frente.

### 2. O card do Wan estava com todas as imagens fora do ponto de operacao

Republicado. As imagens antigas foram feitas a `cfg 1.0, sem shift, sem negative` -- o regime do
Z-Image **Turbo**, que e destilado. O Wan 2.1 nao e. Re-renderizado com
`uni_pc + shift 8 + cfg 6 + negative`, 6 sementes, os tres vereditos **sobrevivem** e agora tem
imagem que mostra:

    FP16                     nitido 6/6
    misto005  0,0546         nitido 6/6, no nivel do FP16
    misto015  0,0793         sujeito volta, tudo empastado
    W4A4 puro 0,1602         destruido 6/6

**A banda 0,0546 / 0,0793 fica confirmada.** Validado mecanicamente: README publicado
byte-identico, 12/12 imagens referenciadas existem, sha256 confere.

### 3. Tres instrumentos numericos apontaram para o lado errado no mesmo dia

Vale mais que qualquer numero acima:

    erro por camada    0,1391 (1,46x o W4A4)  ->  imagem DESTRUIDA
    divergencia        SUBIU 0,2936 -> 0,4750 ->  imagem MUITO MELHOR
    RMSE no pixel      ordena ruido contra ruido, nao diz nada

Nenhum corte nesses eixos separa usavel de inutilizavel. **So o render decide**, e nenhuma
verificacao automatica desta bancada pegou nenhum dos dois defeitos -- foi o dono olhando a foto.

### 4. Ferramentas novas

    tools/sparse24_sm86/{sp24_gemm.cu,sp24_int.cu,roda.py,roda_int.py}  GEMM 2:4, bf16/int8/int4
    tools/probe_esparso_granularidade.py     custo do par vs elemento, um eixo por vez
    tools/probe_esparso_visual.py            8 bracos de render do Z-Image
    tools/decode_esparso_visual.py           decode + folha rotulada
    tools/decode_wan_ladder.py               decode avulso do ladder (o do ladder morre no aimdo)
    tools/probe_encoder_visual.py            a foto que falta no card do Qwen -- ESCRITA, NAO RODADA

As de CUDA precisam do CUTLASS (~43 MB, nao fica no repo) e do `vcvars64`; ver o docstring de
`tools/sparse24_sm86/roda.py`.

### 5. Armadilhas que custaram tempo hoje

- **`sys.argv = ["main.py"]` no topo do modulo apaga os argumentos da propria ferramenta** antes do
  argparse. Um `--dir` foi ignorado em silencio e tres regimes decodificaram o mesmo diretorio.
  Corrigido nos dois decodificadores; **procure esse padrao antes de escrever o proximo**.
- **`TaskStop` mata o wrapper, nao o `bash` filho.** Um script encadeado sobreviveu, disparou no
  horario e brigou por VRAM -- deixando um lock apontando para pid morto.
- **O `BenchGuard` mede residencia da placa para detectar OUTRO inquilino.** Entrar nele DEPOIS de
  alocar os pesos faz a ferramenta recusar a si mesma. Um guard por run, entrado antes de alocar.
- **O contador de controle do 2:4 estava mal especificado**: `== 2 pares vivos` conta como falha um
  par que a quantizacao zerou. O invariante e `<= 2`.
- **A sonda que descobriu a codificacao do INT4 usou padrao uniforme**, que e invariante a
  reordenamento -- ela nao podia detectar o layout `ColumnMajorInterleaved<2>`, e sem o scatter o
  kernel roda e erra 65407 de 65536 sem avisar.
- **`ls | head -10` cortou a lista antes do `w`** e eu declarei que o VAE do Wan nao existia. Ele
  existe. Terceira vez no dia lendo saida de instrumento cego como ausencia.
- **Heredoc do bash com apostrofo na mensagem de commit** mata o comando inteiro no parser, sem
  executar nem a parte de cima. Escrever arquivo por Python quando o conteudo tem aspas.

### 6. Proximos passos, em ordem

1. **Rodar `tools/probe_encoder_visual.py`.** O card do Qwen3-4B faz alegacao de fidelidade
   (cosseno 0,98957) e **nao tem uma unica imagem**. A ferramenta esta escrita e tem controle;
   falta a placa. Depois republicar aquele card.
2. **Treino de recuperacao sobre 2:4 por elemento.** Unico caminho aberto na frente de esparsidade;
   sem ele, INT8 esparso da 4x e uma imagem que ninguem envia.
3. **Cachear o conditioning em `probe_esparso_visual.py`.** Cada braco recarrega 15,5 GB para
   amostrar 15 s porque o encoder e refeito toda vez, sendo identico entre bracos.
4. **Decidir sobre as 6 imagens orfas** do regime errado que continuam no repo do Wan. Apagar
   arquivo publicado e destrutivo e e decisao do dono.
5. **Rever a parte 43 do PROGRESS** -- a conclusao "nao existe limiar do formato, existe um por
   modelo" usou o Wan como o ponto que quebrou a regra, e o numero do Wan veio do regime errado. A
   banda foi reconfirmada, entao a conclusao provavelmente sobrevive, mas nao foi reverificada.


---

## Estado em 2026-09-03: janela de GPU do dono, das 13h30 as 20h (com ~2h45 perdidas num apagao)

Tres frentes planejadas com os ramos escritos ANTES de rodar. Duas fecharam, a terceira mediu e
estava renderizando quando a janela acabou. O plano combinado tinha uma quarta (smooth x convrot)
que nao chegou a rodar; o criterio dela ja esta escrito.

### 1. FECHADO -- a receita de destrava do text encoder que este repo publica NAO funciona

O `CLAUDE.md` e o card do Qwen mandavam soltar as travas **na fonte**
(`clip.patcher.force_cast_weights = False`) e avisavam que escrever nos modulos "sobrevive so por
acidente do estado da VRAM". Medido no proprio arquivo, um eixo por vez, contando chamadas de kernel:

    o que se escreve                     4 bits   dequantize   force_cast
    nada                                      0          504   True
    so a fonte (a receita publicada)          0          504   True
    so os modulos                           252            0   False
    fonte + unload_all_models() forcado     252            0   False
    os dois                                 252            0   False

`ModelPatcher.load` foi instrumentado e **nao roda durante o encode**: o arquivo de 2,4 GiB sobe
inteiro dentro do `load_clip`, entao a linha 1016 rodou uma vez, antes da escrita, com True. A linha
do unload forcado e o controle que nomeia o mecanismo. A regra real e **se `load` roda entre a
escrita e o forward**. As duas ferramentas agora escrevem nos dois lugares.

**A foto que o card nao tinha:** seis sementes, prompt longo pedindo um pescador remendando uma
rede. As 18 imagens sao boas e ha pessoa em todas; o que muda e a rede -- BF16 6/6, travado 5/6,
solto **0/6**. Falha de aderencia ao prompt, nao de qualidade, e nenhum cosseno mostraria.

### 2. FECHADO -- a celula vazia do Z-Image, com o mecanismo invertido no caminho

O bloqueio era o caminho **W4A8** (so aceita `convrot_groupsize` 256), nao o ConvRot.
`quant_mixed --somente-w4a4` destrava, e em cg 256 produz arquivo byte a byte identico ao
`zimage-v2-w4a4` publicado.

O criterio previa que grupo MAIOR daria mais erro. Quatro pontos monotonicos na direcao oposta
(cg 16 `0,1926`, cg 64 `0,1516`, cg 256 `0,1312`, cg 1024 menor): uma rotacao de Hadamard de tamanho
N espalha cada outlier por N canais, entao N maior mistura MAIS. O controle escrito antes disparou.

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1421        0,1848
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147

Monotonica nas duas colunas, faixas sem sobreposicao, e 0,1848 destroi um ~6 B enquanto 0,1837 e
tolerado num ~13 B -- 0,6% de distancia.

**E o `avaliar.py` era cego a esse eixo inteiro:** casava a analise pelo sha da FONTE e lia
`err_w4a4` sem olhar o groupsize, entao os tres builds recebiam a mesma mediana 0,1216 -- o que
desenha bem e o que desenha lixo, mesmo numero e mesmo veredito. Corrigido.

### 3. MEDIDO, RENDER NAO FECHOU -- recuperacao por camada sobre poda 2:4

A proposta do dono, e a unica frente aberta na esparsidade. Reconstrucao por camada (gradiente
conjugado mascarado sobre `H = X^T X`), nao fine-tuning. Sobre as MESMAS linhas de teste:

    W4A4 ConvRot (o de hoje)      4,0 bits/peso    0,0907
    2:4 elemento, so podado       9,0              0,0736
    2:4 elemento RECUPERADO       9,0              0,0052     17,4x mais fiel que o W4A4
    2:4 elem RECUPERADO + int8    5,0              0,0052
    2:4 par RECUPERADO            5,0              0,0062
    denso recuperado (controle)                    piso do int8, exato

**Dois erros meus de desenho, os dois pegos por controle:**

- A primeira versao media o residuo de TREINO. Com X de `[128, 3840]` -- 128 equacoes para 1920
  incognitas por linha -- o CG zera o residuo por construcao e "poda 2:4 sai de graca, 110x".
  Nao sai. Com separacao treino/teste o mesmo run mostrou **32,1x de distancia** entre os dois.
  O conserto foi **mais linhas de ativacao**, nunca mais iteracoes: uma calibragem de 8192 linhas
  (`calib/zimage_v2_rows8192.calib.pt`, 12,9 GiB) leva teste/treino a **1,1x**.
- O controle denso reprovava o caso CORRETO sob `--quantiza`: com int8 ligado o braco denso tambem
  e quantizado, entao ele mede o piso do int8 e o limiar fixo de 1e-3 o reprovava. O piso agora e
  medido, nao suposto.

**NAO FECHOU: tres tentativas, tres hipoteses erradas sobre por que o patch de peso nao sobrevive a amostragem. O caminho que sobra e gravar os pesos recuperados de um processo separado e carregar so eles no render. 17,4x nao e resultado ate a foto existir.**

### O que ficou de fora, e por que

- **Bloco 4, smooth x convrot no Gemma**, nao rodou. O criterio esta escrito em
  `bench/criterio_smooth_vs_convrot.md`, com um desenho que tem controle embutido: SmoothQuant move
  outlier da ATIVACAO para o peso, e no caminho travado do text encoder a ativacao nunca e
  quantizada -- entao ele tem de ser inutil travado e util solto. Se ganhasse nos dois por igual, o
  ganho nao seria de SmoothQuant. Os dois arquivos estao em disco desde 2026-09-01.
- **Publicar.** O token do HuggingFace ativo e `Tesla P4_VM`, role **read**. Os quatro cards estao
  reescritos e validados em disco e nenhum subiu. Nao dá para eu criar token.
- **As 6 imagens orfas** do regime errado no repo do Wan continuam la (o dono autorizou apagar; a
  autorizacao vale, faltou o token de escrita).

### Fila para a proxima sessao, em ordem

1. **FEITO em 2026-09-03.** Os quatro cards estao no ar e conferidos por sha256 contra o local
   (Qwen3-4B com as duas imagens novas, e Wan/Z-Image/Hunyuan com a linha do Z-Image). As 6 orfas
   do regime errado foram apagadas do repo do Wan (18 -> 12 imagens). As copias locais estao em
   `bench/hf/wan21-vace-w4a4/images_regime_errado/`, FORA de `images/`, porque de dentro de
   `images/` o proximo `--imagens` as subiria de volta calado.

   O token de escrita vive em `F:\COMFY_PORTABLE\.hf\token`, com escopo de DIRETORIO: exporte
   `HF_TOKEN_PATH` apontando para ele antes de publicar. Sem a variavel o processo le o token de
   leitura global e a publicacao falha. **Cuidado:** a variavel de ambiente `HF_TOKEN` vence o
   arquivo (`huggingface_hub/utils/_auth.py:49`), entao um `setx HF_TOKEN` global mataria o escopo
   de diretorio em silencio -- escolha um mecanismo, nao os dois.

   Armadilha encontrada ao apagar, e ela quase custou evidencia: **detector de orfa por substring
   do NOME DO ARQUIVO da falso positivo.** O card do Hunyuan cita as imagens como `t015`, nao como
   `misto_t015.png`, e a linha 75 dele diz que `images/` carrega a escada INTEIRA de proposito.
   Tres imagens foram classificadas como orfas e nao eram. Confira as mencoes em PROSA, nao so os
   links markdown.

2. **FECHADO em 2026-09-03: a frente 2:4 nao paga, e a foto existe.** Ver `W4A4_PROGRESS.md`
   parte 47. Nao reabrir sem ler; o resumo e:

   ```
   so poda (sem quantizar nada)      destroi o modelo. 3 prompts, 2-3 sementes cada
   recuperado + int8      5,0 bits   funciona -- e custa MAIS que os 4,0 do W4A4
   recuperado + int4      2,5 bits   destruido
   W4A4 ConvRot           4,0 bits   funciona, e e o mais barato que funciona
   ```

   A recuperacao **funciona de verdade**: `so poda ELEM` e `recuperado` usam a MESMA mascara e a
   diferenca e so o gradiente conjugado nos sobreviventes -- fantasma vira imagem. O que mata a
   frente nao e ela falhar, e o unico braco que funciona custar 25% mais bits que o W4A4 pronto.

   Artefatos em `bench/esparso_visual` (maca), `bench/esparso_rosto`, `bench/esparso_tijolo` e
   `bench/esparso_tijolo_int4`, cada um com sua grade rotulada. Os pesos assados
   (`bench/pesos_recup_elem_int{4,8}.safetensors`, 11,2 GiB cada) e o calib de 8192 linhas foram
   MANTIDOS por decisao do dono -- ha um teste que ainda os usaria, no item 5.

   **Refutado no mesmo dia, com o eixo isolado:** a hipotese de que faltava a rotacao ao braco
   esparso. `esp_peso` contra `convrot_peso`, 6 de 6 medicoes, a rotacao PIORA.

3. **Rodar o bloco 4** (`probe_te_lock_cost.py` duas vezes, uma por arquivo do Gemma, com `--bf16`
   apontando para o original de 23,5 GiB). Criterio e previsoes ja escritos. **E o proximo item
   sem bloqueio nenhum.**

4. **Recuperado + ConvRot, a 2,5 bits -- a unica combinacao que sobrou e a unica que poderia ganhar
   do W4A4 em bits.** Nunca construida. Cuidado ao estimar: a medicao de que a rotacao piora foi
   feita em pesos PODADOS SEM recuperacao, e aplicar rotacao antes de recuperar muda a base em que
   o CG resolve -- e outro problema, nao o mesmo com um filtro a mais. Extrapolar de um para o
   outro seria a mesma deducao que a parte 47 registra como refutada. Custa um bake (~18 min) mais
   um render, com o calib que foi mantido para isso.
3. **Rodar o bloco 4** (`probe_te_lock_cost.py` duas vezes, uma por arquivo do Gemma, com `--bf16`
   apontando para o original de 23,5 GiB). Criterio e previsoes ja escritos.
5. **Limpeza de disco, autorizada pelo dono e NAO executada.** Em 2026-09-03 ele autorizou apagar
   e depois mandou SEGURAR, entao nada foi apagado. F: com 29 GiB livres. Quando liberar:
   `calib/zimage_v2_rows8192.calib.pt` (12,6 GiB, so vai depois do item 4 -- e o insumo dele),
   `bench/pesos_recup_elem_int{4,8}.safetensors` (11,2 GiB cada, reproduziveis em 18 min), e
   **8 calibracoes que ninguem cita** (2,0 GiB somados, medido casando nome de arquivo contra
   sidecars, analises, `tools/*.py` e os tres markdowns). Ressalva do proprio metodo: "ninguem
   cita" e ausencia num grep. `xfer_recovered` e `zimage_v2_native` tem cara de caminho montado por
   convencao -- conferir o gerador antes desses dois.
5. **Reconstrucao SEQUENCIAL** (cada camada ve a entrada ja degradada pelas anteriores, que e o que
   o SparseGPT faz) se a independente nao bastar na foto.

### Armadilhas do dia, para nao repetir

- **`sys.argv = ["main.py"]` no topo do modulo apaga os argumentos da propria ferramenta.** Ja estava
  no handoff de ontem e eu reproduzi num arquivo novo: quatro modos rodaram como se fossem um so.
  Guardar `ARGV = sys.argv[1:]` ANTES.
- **Heredoc de bash morre com aspas/apostrofos no conteudo.** Duas vezes hoje. Escrever o script em
  arquivo pelo Write e executar.
- **`\n` dentro de heredoc vira quebra de linha literal** no arquivo gerado, quebrando a f-string.
- **Matar o wrapper NAO mata o filho, e isso custou 6,1 GiB de RAM do dono.** Esta armadilha ja
  estava escrita no handoff de ontem e eu a repeti **quatro vezes hoje**: `taskkill` filtrado por
  `MEMUSAGE` mata o `probe_*.py`, o filho `python -c "..."` que segura o modelo continua vivo, e
  nada na saida diz isso. As 19h o dono perguntou quem estava com 99% da RAM -- eram quatro orfaos
  meus somando 6,1 GiB, mais o meu proprio trabalho legitimo. Livre foi de 1,9 GiB para 44,3 GiB
  depois da limpeza.

  Escrever a armadilha nao impediu de repetir. O que impede e o procedimento:

  ```powershell
  # ANTES de dar por encerrado qualquer run: listar por linha de comando, nao por nome.
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select ProcessId, @{N='GB';E={[math]::Round($_.WorkingSetSize/1GB,2)}},
           @{N='Pai';E={$_.ParentProcessId}}, CommandLine | Sort GB -Desc | Format-List
  # e matar a ARVORE, pai e filho, nunca so o que aparece no filtro
  ```

  O filtro por memoria e o pior criterio possivel: o filho recem-nascido ainda nao alocou nada e
  escapa; minutos depois ele esta com 2 GiB e sem pai.

- **Tomar `Assert-GpuLock` a mao numa chamada PowerShell deixa lock com heartbeat morto** quando a
  chamada acaba, e as ferramentas que tomam o proprio lock recusam a si mesmas. Deixe as ferramentas
  tomarem.
- **`quality_ladder.py` recusa quando QUALQUER placa visivel esta ocupada.** Com o vizinho na
  cuda:1, rode com `CUDA_VISIBLE_DEVICES=0`.
- **Calibragem grande residente faz o ComfyUI despejar o modelo no meio da amostragem** e apagar
  patches de peso em silencio. O controle do probe pegou; a calibragem agora e liberada antes de
  amostrar.
