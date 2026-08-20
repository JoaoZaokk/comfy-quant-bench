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
`nunchaku/linear.py:152` but `ComfyUI-nunchaku/models/zimage.py:62` carries the same pattern on
another path. Each PR points at the other. Until they land, this installation needs
`--disable-dynamic-vram` for any SVDQuant workflow.

Local modification not covered by any PR: `python_embeded/Lib/site-packages/spas_sage_attn/core.py`
carries the same `km = None` fix as SpargeAttn#121, applied to the 3 occurrences the woct0rdho
wheel has (upstream has 5). A reinstall of that wheel silently reverts it.

Do not start by installing packages. Do not touch the WSL jobs. No original model has been
modified.

## Estado em 2026-08-19

**O repositorio existe.** A raiz virou repo git com `.gitignore` em allowlist. Rastreia `tools/`,
`custom_nodes/`, os `.md` da raiz e dois scripts. **`git add -An --dry-run` antes de qualquer
`git add`** — um denylist que erra uma entrada tenta commitar um safetensors de 42 GiB.
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
  no `embeddings_connector.py:290`, a tres camadas de distancia da causa. O sinal barato no log e
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
