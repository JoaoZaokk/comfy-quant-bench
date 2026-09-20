# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## RECADO TEMPORARIO — do projeto de TTS/STT (2026-09-18). Apagar esta secao depois de agir.

Escrito pela sessao que roda em `P:\PROJETOS\TTS_TRAINING_LORA` a pedido do dono. Sao tres coisas
que sairam de la' e tocam este repo. **Verificar, agir, e remover este bloco inteiro.**

### 1. Risco latente: `.hf/` nao esta no `.gitignore`

MEDIDO aqui em 2026-09-18. **Nada vazou** — o risco e' prospectivo, nao retroativo. Conferido nos
**260 commits** do historico, por quatro caminhos independentes:

    objetos cujo path contem `.hf/`, em todo o historico      0
    `git log --full-history -- .hf .hf/*`                     vazio
    token literal `hf_` + 30 ou mais caracteres, 260 commits  nenhum
    `sk-`, `ghp_`, `github_pat_`, `AKIA`, chave PEM           nenhum

(Um `git log -S "hf_"` devolve ~10 commits, mas e' `hf_parallel_get.py` e `hf_download` — nome de
arquivo e de funcao, nao token. Por isso a busca com comprimento minimo, que separa os dois.)

**Nao ha' nada a revogar nem historico a reescrever.** O que existe e' que `.hf` **nao aparece no
`.gitignore`**, enquanto o `W4A4_HANDOFF.md:611` documenta por escrito que o token de ESCRITA do
HuggingFace vive em `.hf/token`. Hoje o arquivo esta' fora do repo porque ninguem o adicionou, nao
porque alguma regra impeca — um `git add -A` distraido bastaria.

**FEITO em 2026-09-20.** O dono concordou e `.hf/` esta no `.gitignore:119`, conferido por prova
positiva (`git check-ignore -v .hf/token` responde `.gitignore:119:.hf/`). A porta fechou antes de
ser usada, e a conclusao do levantamento acima -- nada vazou nos 260 commits -- nao mudou.

**E no mesmo dia essa mesma pasta cobrou o preco por outro lado:** `HfApi()` sem token explicito
pega o que o ambiente tiver, e `F:/hf-cache/token` e de **leitura**. O upload comeca e morre em
`403 Forbidden: you must use a write token` **depois** de a transferencia ja estar andando. Os dois
arquivos tem **37 bytes**, entao tamanho nao distingue qual e qual. Quem sobe qualquer coisa daqui
fixa `HF_HOME = <repo>/.hf` primeiro -- `.scratch/sobe_fecha.py` agora faz isso a partir de
`__file__`, e a permissao se confere por **prova positiva** (subir 1 byte e apagar), nunca por
`whoami`, que responde igual para token de leitura.

### 2. Ferramenta: varredor de vazamento antes do push

`github.com/JoaoZaokk/speech-quant-lab` -> `bench/varrer_antes_de_publicar.py`. Procura caminho de
disco, compartilhamento de rede, home de usuario, e-mail, chave/token e caminho de dado de saida.
Sai com codigo 1, entao cabe num hook de pre-commit.

Rodando aqui ele acusa **1708 ocorrencias**, e a maioria **nao e' defeito** — este repo publica
handoffs que citam `F:\COMFY_PORTABLE` de proposito, porque sao instrucao para o agente. Se for
usar, alimente `.publicacao-isencoes` com esses casos e o relatorio fica util; sem isso vira ruido,
que e' pior que nao ter relatorio porque da' a sensacao de ter conferido.

### 3. Achado tecnico que toca o `compile_support.py` deste repo

Este repo registra `convrot_w4a4_linear` como `torch.library.custom_op` porque o Dynamo tracava
para dentro do kernel. Ha' uma segunda armadilha na mesma familia, e ela NAO aparece como erro —
aparece como ganho que some:

**Escrever o backward como `torch.autograd.Function` em Python anula o ganho sob `torch.compile`.**
O Dynamo nao consegue tracar o `backward()` dela (e' Python arbitrario), embrulha em
`autograd_function_apply` e marca o ponto como opaco; o AOTAutograd entao nao gera o backward dentro
do grafo e **o backward inteiro volta a ser eager**. MEDIDO no S2 Pro: com DOIS lineares trocados o
estrago ja' era de 75 ms num passo de 183 ms — nao e' custo por chamada, e' o grafo desmontando.

O conserto e' `torch.library.register_autograd` no op, e ai' o backward vira um op comum que o
AOTAutograd poe no grafo. So' importa se algo aqui fizer BACKWARD (treino/LoRA); para inferencia
pura nao muda nada.

De quebra, dois numeros que podem servir de referencia, medidos numa 3090 sm86 com
`comfy_kitchen::int8_linear` (cuBLASLt, peso ja' em int8, ativacao quantizada por linha dentro do
kernel), M = 145 linhas:

    GEMM              bf16    int8    _int_mm + sanduiche
    155776 x 2560    2,302   0,892    3,844     2,58x  contra  0,60x
    9728 x 2560      0,220   0,119    0,339     1,85x
    2560 x 9728      0,201   0,102    0,500     1,98x

`torch._int_mm` fica **pior que o bf16** porque obriga a quantizar fora, multiplicar, e reescalar
fora — tres kernels, dois trafegando bf16. Se algum lugar deste repo usa `_int_mm` como referencia
de "int8 nao acelera", a referencia esta' furada; o problema e' a API, nao o formato.

A nota completa: `github.com/JoaoZaokk/speech-quant-lab` -> `notes/acelerar-o-treino.md`.

---

## REGRA ZERO: nunca assumir. Testar, ou pesquisar, antes de afirmar.

Posta pelo dono em 2026-09-01, depois de ele me corrigir **quatro vezes num dia**. Fica no topo
porque é a regra que todas as outras abaixo pressupõem, e porque eu a quebrei justamente enquanto
escrevia sobre rigor.

**Os dois modos de falha são diferentes, e o segundo é mais traiçoeiro:**

1. **Afirmar sem medir.** "Ninguém põe modulação em 4 bits, logo não se deve." Isso é consenso
   usado como evidência — três fontes concordando pode significar que as três copiaram a mesma
   suposição. Medido: `adaLN_modulation` é a camada com o **menor** erro em W4A4 de todo o bloco
   (0,1263 contra 0,1569 das demais). O argumento não existia.

2. **Declarar um limite e PARAR, em vez de procurar a volta.** Escrevi "2:4 não executa nesta
   máquina" e encerrei. O dono respondeu: *"executa, aceita A40, que é anterior mas mesma série da
   3090. O que falta é o kernel, e eu não vi você mandando ninguém procurar."* Ele estava certo: A40
   é **GA102, o mesmo silício** da 3090. Eu tinha um agente disponível e não usei. **Um limite que
   você não tentou contornar é uma hipótese, não um fato.**

   **Fechado por medição em 2026-09-01, e a hipótese era falsa em toda linha.** 2:4 executa nesta
   3090 a **1,7x–1,95x sobre o denso bf16**, medido em duas passadas independentes nos shapes reais
   do Z-Image (`bench/sparse24_na_sm86_2026-09-01.md`). Não faltava kernel: o kernel estava dentro
   do wheel Windows do xformers o tempo todo. Faltava um `tile` que coubesse — a config que eles
   compilam pede **139.264 bytes** de memória compartilhada e esta placa aceita **101.376**, porque
   foi dimensionada para a A100. Mesmo tile com dois estágios pede 69.632 e roda. E o `cuSPARSELt`,
   que três ferramentas deste repo culpavam por escrito, **não é usado pelo CUTLASS** e nunca foi o
   obstáculo. Três afirmações, todas erradas, todas na forma "não dá" — que é exatamente a forma que
   esta regra proíbe.

Mais dois do mesmo dia, para mostrar que não é caso isolado: (a) medi que Hadamard destrói o padrão
2:4 e chamei de achado — era **tautologia**, rotação densa preenche zero de qualquer matriz, e
faltava o controle; (b) disse "esparsidade custa 3x a quantização" comparando **erro de peso** de um
contra **erro de saída** do outro, e usando o método de poda que ninguém sério usa. Com critério
guiado por ativação a ordem **inverte**.

**Como aplicar, mecanicamente:**

- Antes de escrever "não dá", "não existe", "não suporta": rodar o teste, ou mandar um agente
  procurar. Custa minutos; a afirmação errada custa a confiança em tudo que veio junto.
- Antes de escrever "todo mundo faz assim, logo": perguntar se todo mundo **mediu**, ou se todo
  mundo **copiou**.
- Toda comparação carrega a métrica no nome. Erro de peso e erro de saída não se comparam, e foi
  assim que uma conclusão inteira nasceu errada.
- Todo conjunto de braços precisa do braço que pode falhar — o controle. Sem ele, um resultado bom
  não se distingue de sorte.

Ver as memórias [[prove-before-asserting]] e [[escrevo-mais-rapido-do-que-confiro]].

---

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
  recensa             s40911120/recensa         (healthy, up 24h)
  gestao-db           postgres:16-alpine        (healthy, up 37h)  <- the MacroLog rollback copy
  macrozao-api        app_api-api               (up 37h)
  nostalgic_torvalds  cloudflare/cloudflared    (up 37h)
```

**That list was measured 2026-09-20 and it is NOT the one this file carried until then.** The old
one named `glm-w4` (`qwen38-pp-dspark`) as the container holding the 3090 and `macrozao` as
healthy. Neither is true: **`glm-w4` does not exist** -- today's GPU container is `glm46v`
(`vllm/vllm-openai`), **Exited 3 days ago**, and the 3090 measured **49 MiB used at 0%**, i.e.
free; `macrozao` is `Exited (137) 2 weeks ago`. Of **20** containers in `docker ps -a`, **4** run.
Every paragraph below that says "while `glm-w4` held the 3090" is still a valid *measurement of
2026-08-30*; it is not a statement about today's machine.

```
  ```

  **Consequence worth knowing: the `.vhdx` compaction is no longer blocked by the ERP.** `Optimize-VHD` needs `wsl --shutdown` because all distros share one VM — that was unthinkable with the ERP here and is now a scheduling question, not a hazard. Still the owner's call, and still never `--allow-unsafe`.

  **But the numbers this paragraph carried were wrong, and so was the mechanism.** It said "~140 GB sits in `docker_data.vhdx` (229 GB, ~90 GB of build cache with zero active entries) and `ext4.vhdx` (63 GB with 22 GB used)". Measured 2026-09-20:

```
  docker_data.vhdx   318.86 GiB   (was 229 -- it grew ~90 GiB)
  ext4.vhdx (Ubuntu)  39.61 GiB   (was 63)
  docker system df   TOTAL  ACTIVE   SIZE      RECLAIMABLE
    Images              54      15   191.1 GB   101.6 GB (53%)
    Local Volumes       23       9    56.77      44.62   (78%)
    Build Cache        161       0    63.3        8.798
    Containers          20       4     2.36       2.359  (99%)
                                                 157.4 GB
```

  191.1 + 2.36 + 56.77 + 63.3 = **313.5 GB of real content inside a 318.86 GiB file**. **The vhdx is not inflated — it is genuinely full**, so compacting *before* pruning returns almost nothing. The order is **prune first, compact second**, and the big item is now **images** (101.6 GB reclaimable), not the build cache the old text blamed. Two cautions: `docker volume prune` targets 44.62 GB across 14 inactive volumes and one of them may hold the MacroLog rollback or the rakazo postgres data, so that one goes volume by volume, never in bulk; and `wsl --shutdown` drops the four running containers including `gestao-db`.

  **Also measured the same day, and it makes every commit-charge number in this file worse:** the pagefile is **35.07 GiB**, not 61, so the commit limit is **98.72 GiB**, not 124.8 — and free commit was **24.27 GiB**. A 38 GiB safetensors needs 76 GiB of commit through ComfyUI's normal reader. Re-read the pagefile before trusting any commit arithmetic here; it is system-managed and moves.

  **And two distro facts this file had wrong:** `Ubuntu` is **Running** (this file says it was *Stopped* and "holds no project"), with its `ext4.vhdx` written the same day; and `NVIDIA-Workbench` **exists, Stopped** — the memory `topologia-da-maquina-e-o-erp` recorded it as `wsl --unregister`ed and that is false.

- **On a shared card, `torch.cuda.mem_get_info` reports memory that is real but belongs to somebody else.** Measured 2026-08-30 while the `glm-w4` container (inside WSL2) held the 3090: torch reported **23332 MiB free** where `nvidia-smi` reported **8362 MiB** — a 15 GiB disagreement on the same card in the same second. Same direction on the 3080 Ti, 41x, stable over six interleaved rounds.

  **The obvious reading of that gap is wrong, and this file carried the wrong one for a few hours.** It said the number is phantom and ComfyUI "will decide a model fits when it does not". It fits. Probed by allocating 1 GiB at a time on `cuda:0` with the tenant present (`scratchpad/probe_wddm_evict.py`, under `Assert-GpuLock`): **22 GiB allocated, zero failures, 14166 MiB past what the driver called free.** WDDM evicts the neighbour's pages to system RAM on demand, so torch's number is honest about what this process can get.

  What it is *not* honest about is the price, and the probe shows the eviction happening:

  ```
   held MiB  alloc  torch free  smi free  smi used
      10240     OK         977      1002     23574
      11264     OK       12068      1002     23574   <- 11 GiB reappears: the neighbour was evicted
      18432     OK           0      1012     23564
  ```

  The neighbour's resident VRAM fell from 16214 to 13054 MiB and it was still paging back minutes later. It survived — the training kept running at 100% — but a 4-hour job silently trading VRAM for system RAM is a real cost that nothing in either tool reports.

  So the rule is not "distrust the number". It is: **`get_free_memory()` answers "can I get this?", never "is the card free?"** — and it is ComfyUI's own fit test (`comfy/model_management.py`, in `:978`, `:1096-1098`, `:1210-1213`). On a card with a tenant, ComfyUI will load successfully *by taking memory from whatever else is running*. Cross-check NVML before starting work on a card someone else is using, and read `/system_stats` as a statement about the reader, not about the hardware.

  Not covered: the neighbour's throughput was never measured, only its resident memory; Linux, TCC-mode cards, and a native-Windows neighbour are all untested.

- **`--reserve-vram` does bind, and the way this bench proved otherwise for twenty minutes is the lesson.** The flag's help says it reserves vram *"for use by your OS/other software"*. Reading the code, the reserve never enters `get_free_memory()` — it is added to the **demand** side — so recomputing the comparison by hand (`tools/probe_reserve_vram.py`) said `loads? YES` at every value up to 20 GiB and the conclusion written here was "the guard cannot bind". **That conclusion was wrong.** Measured end to end on 2026-08-30 with a real 15881 MiB checkpoint, a real `comfy.sd.load_diffusion_model`, and the real `load_models_gpu`, while `glm-w4` held the 3090:

  ```
   --reserve-vram   model    really free   landed on the card   neighbour
        (default)  15881 MiB   10924 MiB          +6672 MiB     unchanged
            8 GiB  15881 MiB   10924 MiB             +0 MiB     unchanged
           20 GiB  15881 MiB   10924 MiB             +0 MiB     unchanged
  ```

  The flag changes the outcome from a partial 6.7 GiB residency to nothing at all, and at the default it stayed **under** what the card actually had — no eviction, the neighbour ended exactly where it started.

  **What made the hand-recomputation wrong was an axis held, not an axis varied.** The first end-to-end attempt passed `force_full_load=True`, which is not what a sampler uses; under it every arm loaded and the neighbour *was* evicted, which looked like confirmation. The real path streams weights under a budget that the reserve genuinely shrinks. Two separate self-inflicted wounds in one probe: modelling a guard instead of calling it, and then calling it with the one argument that disables it. See the memory `teste-varia-o-eixo-errado`.

  **`--gpu-only` and `--highvram` bypass the reserve.** Same model, same card, same tenant, one arm per flag:

  ```
                           arm   really free    landed   placed
                       default     10924 MiB     +6672   cuda:0
                    --gpu-only     10924 MiB    +10169   cuda:0
                    --highvram     10999 MiB    +10252   cuda:0
   --gpu-only --reserve-vram 8     10999 MiB    +10207   cuda:0
  ```

  The last row is the finding: with `--gpu-only`, asking for 8 GiB of reserve lands **+10207** against **+10169** without it — the flag is simply not consulted. On the default path the same reserve took residency to zero. So `--reserve-vram` protects only while nothing else has overridden the placement policy, and the two flags people reach for to "make it use the GPU" are exactly the two that switch it off.

  **It does not back off, and "ComfyUI evicted nobody" was a measurement error.** That claim came from watching the card's `memory.used` rise by only ~10.2 GiB for a 15.9 GiB model and concluding ComfyUI had stopped early. It had not. Reproduced 2026-08-30 against a **synthetic tenant** — a separate native-Windows process holding 13 GiB with `torch.empty` and then sleeping, which unlike a live training does not touch its pages and page them back:

  ```
  --gpu-only, tenant holding 13790 MiB
    smi before                13790 MiB
    smi after                 23428 MiB      delta only +9638
    ComfyUI's own accounting  15881 MiB resident, no exception
    tenant afterwards          7402 MiB      <- it lost 6.2 GiB
  ```

  `7318 + 15881 = 23199`, against the 23428 measured. The full model landed. **The delta looked small because the other occupant was shrinking at the same time** — I attributed to ComfyUI a number that was the sum of two moving quantities, which is `ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` with the arms being two processes instead of two code paths.

  So WDDM evicts, and ComfyUI does ask it to. Against the live `glm-w4` this was invisible precisely because that training was at 100% and kept faulting its pages back in — its *resident* number barely moved while it was actually thrashing against the loader. **A neighbour that looks unharmed by resident VRAM may be the worst case, not the best.**

  The default path is different and is fully explained by arithmetic, not by any backoff: `MIN_WEIGHT_MEMORY_RATIO` is **0.0 on NVIDIA** (`comfy/model_management.py:454-457`), so the budget collapses to `lowvram_model_memory = max(0, free − (model_size + reserve))`. Measured `6750 MiB` against `23332 − (15881 + 700) = 6751`. Note the shape of that formula: it subtracts the **whole model** from free and loads the remainder, so a bigger model puts *less* on the GPU, and any reserve at or above `free − model_size` puts nothing there at all. That is what the `+0 MiB` rows above are.

  **Why `--gpu-only` differs is also not a policy decision:** under it, `comfy.sd.load_diffusion_model` writes weights straight to the GPU while reading the file (`smi_after_read` was already 23428 MiB, before `load_models_gpu` was ever called), so `load_models_gpu` and its reserve arrive after the memory is spent.

  One thing this closed by accident: the synthetic tenant is a **native Windows process**, not WSL, and `get_free_memory` over-reported identically (23332 MiB with 13567 MiB held). The inflation is not a WSL artifact.

  **Not covered:** no sampling was run, so this is placement, not a completed generation — inference-time allocation is a separate budget and untested here. The neighbour's throughput was never measured, only its resident VRAM. Re-run both with `tools/probe_reserve_vram.py` (arithmetic) and `tools/probe_reserve_e2e.py` (real load).
- **W4A4 means native ConvRot CUDA execution**, not weight-only INT4 followed by BF16 GEMM. Any change that lets the work fall back to eager/dequantized math defeats the entire project.

  **But this rule names two options where the hardware offers three, and the third one wins on accuracy.** Measured 2026-08-30 on the 3090, varying one axis — `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` (`comfy_kitchen/backends/cuda/__init__.py:212`), which forces the same `convrot_w4a4_linear` down the other branch. `_cuda_device_supports_native_int4_mma` is `major == 8` (`:293`), so Ampere and Ada reach the `m16n8k64 s4` MMA and **Hopper and Blackwell are routed to the INT8 branch deliberately**. The native branch is not merely selected here — it executes: outputs differ from the fallback in 6/6 synthetic cases (`tools/probe_int4_mma_dispatch.py`).

  The third option is **INT4 weight × INT8 activation on tensor cores**, which is neither "native INT4 MMA" nor "dequantized BF16 math". On **real** Z-Image activations and real weights — 24 layers spanning crest 4.4 to 43.0, from `calib/xfer_z_image_turbo_bf16.calib.pt` (`tools/probe_int4_vs_int8_real_acts.py`):

  ```
  media rel-RMSE   nativo 1,28e-1   int8 8,59e-2
  nativo ganha em 0/24 camadas      -> int8 e 1,49x mais fiel
  Spearman(crest, nativo-menos-int8) = +0,247
  ```

  Zero of twenty-four. And this was the run meant to *rescue* the native path: the earlier synthetic measurement used gaussian input, which has no outliers, and outliers are what the rotation exists to suppress — so real activations were expected to narrow the gap. They widened it slightly, 1.4x to 1.49x. Crest does not explain it either (+0.247, weak, and this repo already measured crest against W4A4 error at +0.10).

  Speed is the other half and it points the other way: native is **1.41x and 1.67x faster at M=1024**, and **1.3x to 1.74x slower at M=1** — the `m_crossover` curve shape again.

  So "anything that is not native INT4 defeats the project" is contradicted by measurement, and the two largest public ConvRot distributors — `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` (791k downloads, `w4a4_int4mm_layers: 0`) and `joeygambino/...surgical_int8_convrot` (`target_dtype: int8_tensorwise`) — ship exactly the third option. That reads as a deliberate trade, not a shortcut.

  **Executed 2026-08-31, with the models on the GPU** — `tools/probe_quant_dispatch.py --forward-only` records the device of the weights, the implementation the registry resolved, and the `linear_dtype` that reached the dispatcher, then varies `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`:

  ```
  checkpoint                                 linear_dtype  impl     flag muda?  ramo
  zimage-v2-w4a4                 (nosso)         int4      cuda        SIM      nativo int4
  LTX25-distilled-DiT-comfy-w4a4 (riftcast)      int4      cuda        SIM      nativo int4
  MiniMax_H3_FL2VA         (Abiray, 791k dl)     int8      cuda        NAO      INT8, por instrucao
  ```

  The Abiray file **does run quantized math on the CUDA backend**; it is simply insensitive to the int4 fallback flag because those layers never take that branch. Its own summary contradicts itself, too — `convrot_w4a4_mixed` carries `"linear_dtype": "int4"` at the top and `"w4a4_int4mm_layers": 0` three keys below, while all 117 per-layer `comfy_quant` tensors say `"int8"`. Read the layers, not the summary. And `LTX25-distilled-DiT-comfy-w4a4` (`quantized_by: riftcast/ltx25-quant-lab`, 1440 layers, `linear_dtype` absent) is a **second public W4A4 that really executes 4 bits**, and a seventh known checkpoint writer for ticket 08.

  **The rule stands until the owner decides otherwise**, because this is a trade and the choice is his: accuracy and small-batch latency favour the INT8 branch, large-batch throughput favours native.

  **And whether native W4A4 is worth reaching at all depends on the MODEL, which this rule does not mention.** Measured 2026-08-31 on the 3090, requantizing `hunyuanvideo1.5_720p_t2v_fp16` with today's pipeline and re-rendering the exact prompt, seed and sampler of the public repo's August test:

  ```
  modelo             W4A4 por passo   imagem       divergencia
  Z-Image                1.83x-1.93x MAIS RAPIDO   boa      0.7173
  HunyuanVideo 1.5       1.055x MAIS LENTO         DESTRUIDA 0.8255
  ```

  Same format, same kernel, same converter, same `convrot_groupsize` 256, output byte-size identical to August's 7.92 GiB. The public repo's warning — *"ConvRot W4A4 is slower than FP16 and destroys the output"* — **reproduced fifteen days later** across `comfy-kitchen` 0.2.23→0.2.31, ComfyUI 0.29→0.33 and torch 2.12.1→2.13.0. **Nothing in the stack fixed it**, because nothing in the stack was broken: the model is the axis.

  The leading hypothesis before the run was the `to_native.py` fused-qkv gap (scales passing through unrenamed, layer loading with no scale and no error). It died on reading, not on the GPU: `HunyuanVideo.process_unet_state_dict` carries `.comfy_quant` and `.weight_scale` through its own substring replacements — the gap is Z-Image-specific.

  Two consequences. First, **a checkpoint that converts cleanly, resolves the CUDA backend and writes a valid sidecar can still produce garbage** — the preflight proves dispatch, never quality; only a render does. Second, **latent divergence has no threshold**: 0.8255 destroyed against 0.7173 fine is a 15% gap separating "unusable" from "ship it", so no cut on that axis decides anything.

  Not covered: one seed, one prompt, 480x480, one frame, no perceptual metric, no SASS, and only `convrot_groupsize` 256 in this round. Re-run with `tools/quant_w4a4.py --profile hunyuan_video_15` then `tools/quality_ladder.py`; see `W4A4_PROGRESS.md` part 36.

  **A per-layer error threshold was derived from that work, and a third architecture family broke it on 2026-09-01.** The rule read: median `err_w4a4` above 0.21 breaks, below 0.15 works, between the two is correct-but-grainy. Wan 2.1 VACE 1.3B measures **0.1602** — the untested middle — and the render is destroyed, no subject, no bench. Bracketing it with mixed builds puts Wan's line **between 0.0546 and 0.0793**:

  ```
  modelo               parametros   tolerado   NAO tolerado
  Wan 2.1 VACE             1,3 B     0,0546        0,0793
  Z-Image v2                ~6 B     0,1421        0,1848
  Krea2 Turbo             12,82 B    0,1377      NAO ALCANCADO (ver abaixo)
  HunyuanVideo 1.5         ~13 B     0,1837        0,2147  (e 0,2163 no capybara)
  Qwen-Image-Edit 2511    20,43 B    0,0358*       0,1080   (*outro FORMATO, ver abaixo)
  ```

  **A linha do Qwen tem um asterisco porque as duas colunas dela nao sao o mesmo experimento, e
  isso e o achado.** Em toda linha acima, as duas colunas sao W4A4 com groupsize ou limiar de
  promocao diferentes. Na do Qwen, `0,1080` e o W4A4 e `0,0358` e um build **todo em
  `asym_w4a8_int8`** -- mesmos pesos de 4 bits, ativacao de 8 em vez de 4. Medido 2026-09-12, 12
  renderizacoes por braco, 6 prompts x 2 sementes:

  ```
  build                      peso    ativacao   GiB   divergencia  s/passo  imagem
  int8_convrot (Comfy-Org)   8 bits   8 bits   19,09      0,1942    1,410   boa
  w4a8 (nosso)               4 bits   8 bits   10,79      0,4997    1,575   boa
  w4a4 (nosso)               4 bits   4 bits    9,60      1,7440    1,053   ESTATICA
  misto, 607/840 em A4       4 bits   4 bits    9,88      1,8846    1,236   ESTATICA
  ```

  Publicado, com as duas grades como prova -- a do build bom e a dos dois que falharam:
  **https://huggingface.co/JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot**. Os pesos `w4a4` e
  `misto` NAO subiram: negativo medido se publica como prova, nao como checkpoint que alguem
  baixa e usa.

  **O Wan 2.2 confirmou o eixo numa SEGUNDA familia, com outro modo de falha.** Medido
  2026-09-13, `wan2.2_ti2v_5B`, 6 corridas de 33 quadros a 480px, **os tres bracos RESIDENTES**
  na 3090 -- nenhum descarregado, nenhum espalhado -- entao o s/passo abaixo e comparacao real:

  ```
  braco            GiB   s/passo   divergencia   min-max          imagem
  FP16 original   9,31     1,579        --          --            nitida
  W4A8            2,75     0,994      0,2115   0,1185-0,3275      indistinguivel a olho
  W4A4            2,46     0,753      0,3847   0,2431-0,5748      BORRADA
  ```

  W4A4 e W4A8 tem os MESMOS pesos de 4 bits; so a ativacao difere. No Qwen Edit o W4A4 deu
  estatica, aqui da borrao. **Modo de falha diferente, mesmo eixo, segunda familia.** Publicado:
  **https://huggingface.co/JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot**.

  **E aqui esta o contraexemplo que muda como esta bancada le divergencia de latente.** `0,3847`
  esta DENTRO da faixa que sempre funcionou -- abaixo do Z-Image v2 (0,7854) e do Krea2 (0,5843),
  os dois bons -- e a imagem esta degradada. Ate agora so existia a falha oposta aqui:
  divergencia ALTA com imagem boa, porque uma perturbacao minima manda o sampler para outro lugar
  que tambem presta. Este e o espelho, e o mecanismo e obvio depois de visto: **borrao e uma
  mudanca PEQUENA de latente.** Suavizar afasta menos da referencia do que ir para algo nitido e
  diferente.

  Entao a regra "divergencia de latente nao tem limiar", que este arquivo ja registrava, e fraca
  demais. O certo e: **a divergencia e ENVIESADA para falha macia.** Um build que borra sempre se
  elogia nessa metrica, e o tipo de dano que ela esconde e justamente o mais facil de nao notar
  numa olhada rapida. Nao usar divergencia sozinha para aceitar um build -- nunca foi suficiente,
  e agora se sabe para que lado ela erra.

  Uma armadilha pega antes de custar, no mesmo dia: o latente do `ti2v_5B` tem **48 canais** e o
  `wan_2.1_vae` que mora ao lado tem **16**. Decodificar com o VAE errado da imagem plausivel e
  silenciosamente errada -- o aviso que `decode_latents.py` imprime em toda execucao. Contados os
  canais ANTES, baixado o `wan2.2_vae` (1.409.400.960 B), e so entao decodificado.

  **Nao e o peso, e a ativacao.** E o build misto fecha o mecanismo: 233 camadas ja promovidas a
  ativacao de 8 bits, erro efetivo **0,0744** -- metade do erro de um Z-Image que presta -- e
  ainda assim estatica. **Basta sobrar camada no caminho A4.** E penhasco, nao ladeira.

  **Consequencia para esta tabela inteira: `0,1080` e o MENOR erro por camada que ja produziu
  lixo nesta bancada**, abaixo do Krea2 (0,1199) e do Z-Image (0,1254), ambos funcionando em
  W4A4. O criterio por camada **ordena formatos corretamente** (ele disse que W4A8 era 3,02x
  melhor, e era) e **nao localiza penhascos**. Isso deixou de ser suspeita e virou demonstracao.

  Descartada por medicao a leitura facil de que o arquivo estava quebrado: `probe_quant_dispatch
  --forward-only` da 840 modulos `convrot_w4a4`, **8/8 forwards quantizados, 0 dequantize**,
  `convrot_linear_dtype=int4`, `backends.cuda`, 840 pesos em `cuda:0`; `verify_w4a4` passa em
  estrutura, na comparacao byte a byte com a fonte, e da `relative_rmse` 0,2265 no kernel real.

  **E uma hipotese nova morreu aqui no mesmo dia: erro mediano x numero de camadas.** Ela separa
  NOVE builds perfeitamente (funciona ate 30,9; destruido a partir de 31,4) e morre no decimo,
  que ja estava no disco: `hunyuan15-misto-t025` marca **79,0 e renderiza correto**, contra
  `zimage-v2-teto-cg16` que marca **31,4 e renderiza lixo**. Terceira hipotese de transferencia
  a morrer nesta bancada, depois da monotonia por tamanho e da razao entre groupsizes.

  **A monotonia por tamanho MORREU em 2026-09-12, e ela era o unico argumento para extrapolar
  desta tabela.** O Krea2 Turbo tem **12.820.073.036 parametros** (somados do header, nao
  estimados do tamanho do arquivo) e a previsao escrita antes de medir -- `bench/criterio_quant_krea2.md`
  -- era mediana de `err_w4a4` entre 0,15 e 0,22, porque e ai que o Hunyuan de ~13 B esta. Medido
  sobre 224 camadas, todas calibradas, cg 256: **0,1199**. Um modelo de 12,8 B mede MENOS erro por
  camada que o Z-Image de ~6 B. A linha entra com a coluna `NAO tolerado` vazia de proposito: nada
  foi medido acima de 0,1199 neste modelo, e `tolerado` registra o maior erro que ja se viu
  funcionar, nunca um teto.

  **O teto foi PROCURADO no mesmo dia e o unico eixo suportado nao alcanca.** `bench/criterio_teto_krea2.md`,
  criterio com seis previsoes escrito antes de converter: 4 confirmadas, 2 refutadas. Reconvertendo
  `--somente-w4a4 --uncalibrated fail` nos dois groupsizes menores, medido na **intersecao das
  mesmas 224 camadas** (P4 confirmada, nenhuma populacao diferente comparada):

  ```
  cg     mediana      p25      p75      max   razao vs 256   render (5 prompts x 2 sementes)
  256     0,1199   0,0822   0,1475   0,2942        1,000x     10/10 boas
   64     0,1238   0,0886   0,1553   0,3397        1,033x     10/10 boas
   16     0,1377   0,1079   0,1961   0,4331        1,149x     10/10 boas
  ```

  Monotonico na mediana e em **215 das 224** camadas. **No menor groupsize legal o modelo nao
  quebra**, entao `tolerado` sobe para 0,1377 e a coluna do teto fica `NAO ALCANCADO` -- com motivo,
  nao por falta de tentativa. Descartada a leitura facil e errada de que a imagem sobreviveu porque
  o kernel nao rodou: `probe_quant_dispatch --forward-only` nos dois builds novos da 224 modulos,
  **8/8 forwards quantizados, 0 dequantize**, `convrot_linear_dtype=int4`, `backends.cuda`.

  **A segunda hipotese de transferencia morreu aqui, no mesmo checkpoint e no mesmo dia.** As
  razoes entre groupsizes medidas no Z-Image (1,155x e 1,468x) foram aplicadas ao Krea2 como
  previsao P2; ele mede **1,033x e 1,149x**, tres vezes menos sensivel, e erra para o mesmo lado nas
  duas pontas. A razao entre groupsizes e do MODELO, como ja era a tolerancia -- nao do formato.

  **E `quant_group_size` NAO e um eixo utilizavel, apesar de o parametro existir e o sidecar
  carregar a chave.** `comfy/ops.py:1201` escreve `"quant_group_size": 64` como constante literal,
  enquanto as duas linhas ao redor leem `convrot_groupsize` (`:1197`) e `linear_dtype` (`:1202`) do
  JSON da camada. Quantizar com outro valor produz arquivo que o loader le como 64: qualquer quebra
  seria desacordo loader-vs-arquivo, nao tolerancia do formato. Lido no codigo, nao executado.

  Sobra um eixo nao tentado, a **cobertura**: o perfil `krea2` seleciona 224 Linears e exclui 41
  tensores 2-D -- as 32 do `txtfusion`, `tproj [36864, 6144]`, `tmlp`, `txtmlp`, `last.linear`,
  `last.modulation.lin` -- dos quais 39 passariam o filtro de divisibilidade. Isso nao move a
  mediana (muda QUAIS camadas degradam), entao responde outra pergunta, e exige recalibrar.

  E funciona bem: 5 prompts x 2 sementes x 4 bracos, **40 renderizacoes, nenhuma quebrada** --
  maca (controle), rosto com pele e ruga, placa "OPEN" legivel em 8 de 8 celulas, mercado noturno
  coerente, cristal de gelo com estrutura fina. A divergencia de latente do W4A4 contra o BF16 e
  **0,5843** e a imagem presta: o que esse numero mede e o sampler indo para outro lugar que
  tambem e bom.

  **O int8 do proprio Comfy-Org ganha do nosso W4A4 em 10 de 10 corridas pareadas** (0,2435 contra
  0,5843, 2,40x mais fiel), que e a quarta medicao independente nesta bancada na mesma direcao --
  agora numa quarta familia. E o W4A4 e **1,47x mais rapido por passo** (0,839 contra 1,233) e
  1,68x menor (7,50 contra 12,57 GiB). A escolha e uma troca, nao um erro; o que nao se pode e
  chamar o W4A4 de mais fiel.

  **A celula do Z-Image foi preenchida em 2026-09-03, e a hipotese mecanica que ia preenche-la
  estava invertida.** O eixo e o `convrot_groupsize`: `bench/criterio_teto_zimage.md` previa que
  grupo MAIOR daria mais erro ("rotacao mais grossa"). Medido sobre a intersecao de camadas que
  todos os valores aceitam, quatro pontos monotonicos na direcao **oposta** -- cg 16 `0,1926`,
  cg 64 `0,1516`, cg 256 `0,1312`, cg 1024 menor ainda. Uma rotacao de Hadamard de tamanho N
  espalha cada outlier por N canais, entao N maior mistura MAIS. "Mais grosso" era a intuicao de um
  quantizador por grupo, onde grupo maior significa uma escala para mais valores; a rotacao nao e
  isso. O controle escrito antes (`grupo menor tem de reduzir o erro`) disparou e impediu a leitura
  errada.

  Renderizado com quatro bracos e tres sementes: BF16 bom (controle), cg 256 (0,1216) bom, cg 64
  (0,1421) bom, **cg 16 (0,1848) destruido 3/3**. As tres previsoes escritas antes bateram. Repare
  que a tabela agora e monotonica nas duas colunas e que **0,1848 destroi um modelo de ~6 B
  enquanto 0,1837 e tolerado num de ~13 B** -- 0,6% separando as duas faixas, que e o que se
  esperaria se o tamanho fosse o eixo. Tres pontos continuam sendo tres pontos.

  **E o avaliador offline era cego a este eixo inteiro.** `tools/avaliar.py` casava o
  `.analysis.json` pelo sha da FONTE e lia `err_w4a4` sem olhar o `convrot_groupsize`, entao os
  tres builds -- mesma fonte -- recebiam a **mesma mediana 0,1216** e o mesmo veredito: o que
  desenha bem e o que desenha lixo. Corrigido: camadas com groupsize diferente do da analise sao
  descartadas, e um arquivo sem analise no proprio groupsize ganha o achado
  `analise_de_outro_groupsize` em vez de sair calado. O bloqueio que impedia tudo isso era o
  caminho **W4A8** (so aceita cg 256), nao o ConvRot; `quant_mixed --somente-w4a4` nao mede nem
  escreve W4A8, e em cg 256 produz arquivo **byte a byte identico** ao `zimage-v2-w4a4` publicado.

  Nao coberto: um prompt, tres sementes, um tamanho, uma placa. Nada foi medido entre 0,1421 e
  0,1848, entao o ponto exato da virada nao e um fato -- os fatos sao as duas pontas.

  **A linha do capybara estava na fila errada, e este arquivo a publicou assim.** `capybara_v0.1` foi tratado como checkpoint da familia Z-Image e seu 0,2163 virou o teto do Z-Image. Lido do arquivo em 2026-09-01: **1364 tensores e 54 `double_blocks`** — arquitetura do HunyuanVideo 1.5 — contra os **453 tensores e zero** do Z-Image. Confere tambem por tamanho: 16 653 435 264 bytes contra os 16 653 368 128 do `hunyuanvideo1.5_720p_t2v_fp16`, 67 KiB de diferenca. A quebra e real e passa para a linha de ~13 B, onde concorda com o 0,2147 medido no proprio Hunyuan — e **o teto do Z-Image nunca foi medido**: sabe-se que 0,1241 funciona, e nada alem disso foi tentado. A monotonia na coluna do tolerado sobrevive; uma celula mudou de linha e outra ficou honestamente vazia. Corrigido nos tres cards do HuggingFace e no README publico no mesmo dia.

  Monotone in the tolerated column. So **there is no threshold of the format — there is one per model**, and the practical consequence is that `--promote-error 0.15`, chosen on Z-Image and carried everywhere since, is **not a safe default**: on Wan it writes a file that loads, dispatches natively, passes every structural check, and renders a smear. Three points make the size reading a hypothesis, not a law.

  **And the reference arm broke first, which cost four renders.** The FP16 Wan — no quantization at all — came out as woven fabric at 6 steps/1 frame, at 25/33, at cfg 6 and cfg 1, with and without `ModelSamplingSD3 shift 8` (which applies, and changes no sigma under the `simple` scheduler). The first run's `divergence 1.2365` measured nothing. Cause, one axis varied (`tools/probe_vace_strength.py`): `vace_strength` **1.0** gives `|latent| 607.6` and fabric, **0.0** gives `|latent| 1543.2` and a real workshop. `WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) fills `vace_frames` with zeros when no VACE node is present, runs each block through `process_latent_in` — which subtracts the latent format's mean, so **zero becomes non-zero** — concatenates an all-ones mask, and applies it at full strength. **Any VACE checkpoint in a plain T2V workflow is destroyed, with no error and no warning.** `quality_ladder.py` now takes `--vace-strength`.

  One counting trap that nearly ended the run early: loading a quantized Wan prints `WARNING: unet unexpected [...comfy_quant]` for all 300 layers. It is **cosmetic** — the tensors are consumed before that check and complained about after. `probe_quant_dispatch.py --forward-only` counts 300 quantized modules, 12/12 quantized forwards, 0 dequantize, native int4 on the CUDA backend.

  Not covered: one prompt, three seeds, one scheduler, one card, no perceptual metric; the checkpoint is **fp16** and is a **VACE variant run as plain T2V**, so the tolerance measured may belong to the mode rather than to the model. See `bench/criterio_wan21.md` (criterion written before the result) and `W4A4_PROGRESS.md` part 43.

  **The "is 1.49x visible?" question is now answered, and answering it corrected how this bench measures.** Three measurements of the same pair, 2026-08-30/31:

  ```
  erro por camada, ativacao real     int8 1,49x mais fiel   24/24 camadas
  epsilon por passo, entrada casada  int8 1,33x mais fiel    8/8 passos
  imagem final, trajetoria livre     2x1 em 3 sementes, ambos a ~0,3-0,5 do BF16
  ```

  The final-image comparison is the one that carries no signal, and it is the one that looked most like an answer. At 8 steps a tiny perturbation reroutes the sampler, and the destination is still a good image — so the free-running image measures chaos, not fidelity. **Do not use a generated image to compare two quantizations of the same model.** `tools/probe_epsilon_per_step.py` is the instrument that does work: the BF16 arm records every `(x, timestep)` it was called with via `model_options["model_function_wrapper"]`, and the quantized arms replay exactly those inputs, so every step is a matched comparison and trajectory divergence cannot exist by construction.

  This also **rehabilitates the per-layer criterion** that `tools/quant_mixed.py` uses. It was written off here on 2026-08-30 as non-predictive; it is not. It predicts the model's prediction error (1.49 against 1.33, same direction, same winner). What it does not predict is the free-running image, and nothing does.

  New information only the per-step view gives: the error is **concentrated at high sigma**. Native goes 6.17e-1 at sigma 1.000 down to 5.31e-2 at sigma 0.300, cosine 0.787 → 0.9986. Quantization damage lands hardest where structure is decided, and decays monotonically into texture. A criterion that weights layers by their contribution at high sigma is therefore a different thing from one that weights all steps equally. **Tested on 2026-08-31, and it does not pay.**

  `calibrate_activations.py` now records the sigma of every sampled row (`sample_sigma`), and `quant_mixed.py` takes `--sigma-weight none|sigma|sigma2|high` (default `none`, unchanged). The weighted criterion is nearly the same criterion: Spearman **+0.9935** (`sigma2`) and **+0.9629** (`high`) against flat on `err_w4a4` over 170 Z-Image layers, moving 6 and 9 layers across the 0.15 threshold.

  **The first comparison said it won 8/8 steps at 1.047x, and that was a confound, not a result.** The weighted build promoted 56 layers to 8-bit against the flat build's 53 — a bigger model, measured against a smaller one. **`ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` again, with the held axis being the promotion budget.** Always match the budget before comparing two selection criteria.

  **Rebuilt at exactly 56 promotions in all three arms and run over eight seeds, the three criteria are indistinguishable.** Paired design — within a seed the arms share the imposed BF16 trajectory:

  ```
  epsilon medio, 8 sementes    media       min       max    espalhamento entre sementes
  plano56                    1.3326e-1  1.1220e-1 1.5577e-1        1.388x
  sigma2                     1.3684e-1  1.1731e-1 1.5216e-1        1.297x
  sigma_alto                 1.3685e-1  1.2276e-1 1.4695e-1        1.197x

  diferenca pareada contra o plano:  sigma2 +3.37% (erro-padrao 3.95%, vence 4/8)
                                 sigma_alto +3.46% (erro-padrao 3.93%, vence 2/8)
  ```

  The between-seed spread inside a single arm reaches **1.39x**; the between-criteria difference is **3.4%**. The effect sits inside its own noise, and the win counts are coin flips. The sign is consistent — both weighted criteria come out slightly *worse*, never better — but at eight seeds that is a hint, not a result.

  **This corrects the three-seed version of this paragraph**, which reported "flat wins 3/3" and read as a finding. It was a small-sample artifact: adding a third arm made the ranking change between seeds, and the winner across eight seeds is 3 / 3 / 2 split between the three.

  Re-run with `tools/probe_epsilon_ckpt_ab.py`, which imposes the BF16 trajectory on N checkpoints at once. Not covered: one prompt, one model, one scheduler, eight seeds, no perceptual metric, and no formal hypothesis test — only the distance between the effect and its own spread.

  **Not covered:** one prompt, one seed, no perceptual metric, only Z-Image, only sm86, no SASS. And BF16 is the target, not the truth — it was never itself validated against float32.

  **And for a text encoder the rule is unreachable by construction, whatever the checkpoint says.** Measured 2026-08-31 on the 3090 against a real 13.20 GiB public W4A4 encoder (`qwen3vl_32b_minimax_h3-int4_convrot`, 350 `convrot_w4a4` layers, `linear_dtype` absent so the signature default `"int4"` applies). Called by hand, the kernel is genuinely native — 15/15 outputs differ from the forced INT8 fallback. Loaded through the stock `CLIPLoader` path and *counted* during a real encode:

  ```
  100/100 Linear   MixedPrecisionOps.Linear, quant_format convrot_w4a4, weight QuantizedTensor
  _convrot_w4a4_forward        0
  QuantizedTensor.dequantize 350
  ```

  Zero. The weight stays 4-bit in VRAM and the **math is dequantized** — memory saved, no time saved, kernel never reached.

  **The obvious culprit is the wrong one, and flipping it changes nothing.** `comfy/sd1_clip.py:114` hardcodes `full_precision_mm=True` for every text encoder; setting it False left the count at 0. Instrumenting each term of `_use_quantized` (`comfy/ops.py:1372-1377`) on a real forward names the one that actually bites: **`comfy_force_cast_weights=True`**, which comes from `model_patcher.py:743` via `set_model_compute_dtype`, called at **`comfy/sd.py:269` for every CLIP object** — `set_model_compute_dtype(torch.float32)`, commented "Match torch.float32 hardcode upcast in TE implemention". Two independent locks, and only releasing both fires the kernel.

  **Release it at the source, never on the modules — and this cost a run that looked like a result.** `model_patcher.py:1016` rewrites `m.comfy_force_cast_weights = self.force_cast_weights` on every module *every time the model is loaded to GPU*. Writing the attribute on the modules survives only if the model happened to be resident already, which depends on VRAM state at that instant. **The same command line gave 350 kernel calls on one run and 0 on the next.** The fix is `clip.patcher.force_cast_weights = False` (plus popping `manual_cast_dtype`), which `patch_model` then propagates. What caught it was the dispatch counter reporting 0 — a probe that only compared outputs would have reported "releasing the locks changes nothing" and been believed.

  With that fixed, the same measurement on three real encoders, each `--sem-bf16` except Qwen, which has a BF16 twin on disk:

  ```
  encoder                 formato  quant   travado  destravado    tempo            erro C-vs-B   cos
  qwen_3_4b (4B)            W4A4  2.4 GiB   70.2ms      82.9ms   1.18x MAIS LENTO    5.99e-1   0.949
  gemma_3_12B_heretic       W4A8  8.1 GiB  1772.3ms    478.9ms   3.70x mais rapido   2.11e-1   0.982
  qwen3vl_32b_minimax       W4A4 13.2 GiB   311.6ms    117.7ms   2.65x mais rapido   9.73e-2   0.99989
  ```

  **The sign is set by the sequence length, and that took varying one axis alone.** The first version of this paragraph asserted a mechanism it had not measured ("the cost scales with weight size while the token count stays tiny"), and the table above refuses it: the locked times are 70.2 ms at 2.4 GiB, **1772.3 ms at 8.1 GiB**, 311.6 ms at 13.2 GiB — not monotonic in weight size. The three encoders differ in weight size *and* in sequence length at once (Gemma's conditioning is `[1, 49, 1024, 3840]`, MiniMax's `[1, 8, 5120]`), so that table cannot separate them. Same file, same card, only the prompt changed:

  ```
  qwen_3_4b W4A4, RTX 3080 Ti, mediana de 3
  tokens   travado  destravado
      22     80.7      120.1    1.49x MAIS LENTO
      75    100.1      105.8    1.06x MAIS LENTO
     199    151.9       95.2    1.60x mais rapido    <- cruza entre 75 e 199
     424    245.2      102.0    2.40x
     850    456.1      124.3    3.67x
    1496    824.5      249.0    3.31x
  ```

  **The crossover is between 75 and 199 tokens** on this model and card. The released path is nearly flat from 22 to 424 tokens (120 → 102 ms) while the locked path climbs with the sequence, so the 4-bit kernel carries a per-layer fixed cost that a short prompt cannot amortise and the dequantized path pays a BF16 GEMM that grows with the tokens. Real prompts usually sit above that crossover: at 850 tokens the quantized encoder also beats the **BF16 original** (372.1 ms at ~700 tokens against 124.3 ms). Weight size may still matter on top of this — MiniMax won at 8 tokens — but it has not been isolated. And **the accuracy cost is not a property of the format**: two W4A4 files differ by 6x in the error releasing adds (5.99e-1 against 9.73e-2), so "W4A4 costs X" cannot be quoted without naming the checkpoint; on Qwen the long prompt is also worse than the short one (weight-only 2.55e-1 against 1.44e-1). For Qwen the full picture is available: the 4-bit *weight* alone already costs 1.44e-1 against its BF16 twin, and releasing takes it to 6.09e-1 — 4.23x.

  **Measured the same day on this project's own outputs, and the split is clean.** `tools/probe_quant_dispatch.py` counts the same way against a real load and a real forward:

  ```
  zimage-v2-w4a4        difusao  170 convrot_w4a4   340 quantizados, 0 sem, 0 dequantize
  gemma_3_12B_heretic   TE       336 asym_w4a8_int8   0 quantizados, 336 sem, 336 dequantize
  ```

  So: **the diffusion path really does run quantized** — every Z-Image number this bench has published (per-layer error, epsilon per step, the INT4-vs-INT8 comparison) was measured on genuinely quantized math, not on two dequantizations. And **our own Gemma is memory-only**, exactly as predicted, with both locks `True` on all 336 layers. The format does not matter: `asym_w4a8_int8` is blocked by the same `comfy_force_cast_weights`, so this is about being a text encoder, not about being W4A4.

  One counting trap worth keeping: `MixedPrecisionOps.Linear` is the class of **every** Linear in the model, quantized or not. Patching the class and counting all its calls first reported Z-Image as "MISTO, 340 against 76" — the 76 were layers that never had a quantized weight at all. Scope the count to `layout_type is not None`.

  **Still not measured:** what it costs in s/it on a real LTX render, and whether releasing the locks on a text encoder is safe for output quality.

  **The MiniMax half of that caveat is closed since 2026-08-31: its BF16 twin is now on disk.** `Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, 47.97 GiB, byte-exact against the published size, **351 of 351 layer names matching** the Winnougan quantization. Measured with `tools/probe_winnougan_fidelidade.py` over 20 layers, 5 shapes, 4 depths (0/16/33/49), M in {1, 64, 1024}: **int8 1.40x more faithful, 60 of 60**. That makes three independent measurements in the same direction — 1.49x on Z-Image with real activations, 1.33x on epsilon-per-step, 1.40x here — across a different model family, a **different quantizer** (theirs, not ours), and a different activation kind. Per-layer error is also flat in depth: same shape, block 0 to block 49, 2.2776e-1 → 2.2762e-1.

  Not covered: no SASS; there is no BF16 reference for the Gemma on this bench, so its error column is against its own dequantized arm and is **not** a fidelity claim; one prompt per encoder, one card; the locks were released by post-load monkeypatch, not by anything ComfyUI offers — `custom_operations` in `model_options` is the only real escape and no node exposes it. Re-run with `tools/probe_winnougan_int4.py`, `tools/probe_winnougan_load.py`, `tools/probe_te_fullprecision_mm.py` and `tools/probe_te_lock_cost.py`.

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

  What is still true, and is the part worth keeping: **the converters do not all resolve the same thing.** `verify_w4a4.py` resolves only `convrot_w4a4_linear` while `quant_w4a4.py` resolves both ops. Do not treat "the tool ran" as proof the CUDA backend was used; check the `backend` field in the sidecar. See [AUDITORIA_2026-08-18.md](AUDITORIA_2026-08-18.md) items 8 and 17, and `.scratch/varredura-2026-08-22/issues/05`.

  **The "six definitions" half of this paragraph is fixed and this text was stale about it.** It used to say there were **six** definitions of `normal_comfy_backend` with four different answers, and told the reader to grep and count. Counted 2026-09-01: **zero** — the name is gone entirely, and there is exactly **one** definition of the probe, `native_backend_ready` in `tools/_native_probe.py:315`. `tools/test_native_probe.py` enforces it mechanically (`test_one_definition_of_the_probe_remains`, assembling the needle from two string pieces so its own source is not counted); executed here, **20/20 pass**. A doc that sends someone hunting for six functions that no longer exist costs them the same hour whether the claim was too harsh or too kind.

  **That adoption is DONE and verified on the card — six of seven writers by re-running them, and the numbers are byte-identity, not a code read.** `tools/_conversion.py` holds the write contract once (atomic `.partial` + fsync + `os.replace`, the disk/RAM guards, the refusals) and all seven writers go through it. Executed 2026-09-01 on the 3090, reconverting to a fresh path and comparing the whole file:

  ```
  quant_w4a4    hunyuanvideo1.5 fp16      8 507 690 240 B   sha256 IDENTICO
  quant_w4a8    hunyuanvideo1.5 fp16      8 847 567 376 B   sha256 IDENTICO
  quant_mixed   wan2.1 vace 1.3B          2 310 437 144 B   sha256 IDENTICO
  quant_mixed   beyond-reality-zimage-v2  3 403 133 032 B   sha256 IDENTICO
  to_native     beyond-reality-zimage-v2       11,46 GiB    sha256 IDENTICO
  ```

  `quant_int8` and `svdq_to_bf16` have **no earlier output on disk**, so their acceptance is weaker and says so: a real conversion plus a load through ComfyUI's normal loader. int8 counts 432 `int8_tensorwise` modules with **8/8 quantized forwards, 0 dequantize** on `comfy_kitchen.backends.cuda`; the recovered BF16 loads as `Lumina2`, 6 154 908 736 params, 34 fused qkv keys — for *this* architecture the docstring's fused-layout warning does not bite, because ComfyUI's own `Lumina2` wants `attention.qkv`.

  **`quant_w4a4_smooth` ran end to end on 2026-09-01, and verifying it found two defects in the VERIFIER.** The owner said to stop listing the blocker and download the model. `DreamFast/gemma-3-12b-it-heretic`, matched by **exact size** rather than by name — 23 545 681 250 bytes, the number the w4a8 sidecar already recorded. One conversion then unblocked both missing inputs: `quant_w4a4 --profile gemma` over the BF16 produced the `--calibrate-with`. Smooth wrote 6.91 GiB in 23.3 s, 96/96 norms observed, and the channel outlier ratio went **82.52 → 9.60**.

  Its acceptance then **failed the file twice, and both times the verifier was wrong** — which is the part worth keeping, because both defects had existed for as long as `smooth` has and could only surface by running it to completion:

  - **`Source comparison` reported 96 corrupted tensors** — exactly 48 `input_layernorm` + 48 `pre_feedforward_layernorm`, and nothing else. Rewriting the norm *is* the SmoothQuant mechanism (`norm ← (norm+1)/λ − 1`, `W ← W·λ`), so "preserved == byte-identical" is true for `quant_w4a4` and false by construction here. The fix is not an exemption: for those norms the check **inverts** — they must have changed. A byte-identical norm in a SmoothQuant file means λ=1 there, i.e. the smoothing did nothing, and the file still loads and dispatches.
  - **`--kernel-smoke` failed at rel-RMSE 16.21 against a 0.90 ceiling.** The smoke compares against `F.linear(x, W_source)` while the file stores `W·λ` — two different functions, not two implementations of one. **Tested before fixing**, recovering λ from the converter's own formula inverted (`λ = (norm_src+1)/(norm_out+1)`, nothing stored):

  ```
  λ recovered                              min 4.90  max 103.47  mean 13.60  (3840 channels)
  reference F.linear(x, W_source)          rel-RMSE 15.3650   <- failed
  reference F.linear(x, W_source · λ)      rel-RMSE  0.2127   <- Gemma's normal band
  ```

  72x. Fixed, and λ is now **printed in the report** (min/max/mean) so a reader can check the correction instead of trusting it. Final: structure PASS, source comparison PASS, backend `comfy_kitchen.backends.cuda`, `relative_rmse 0.18702`. `o_proj` and `down_proj` stay outside the correction on purpose — they are quantized but not smoothed, since no norm feeds them directly, and the function returns `None` for them so the strict comparison still applies.

  **A verifier that fails a correct file teaches people to switch verifiers off** — the same argument this repo makes about a WARN resting on a hypothesis. The migration created neither defect; it created the occasion to find them.

  Still open, and now cheap: nothing compared the `_w4a4_convrot` and `_w4a4_smooth` files against each other, so the question smooth exists to answer — *is channel smoothing the largest term of the SVDQuant recipe?* — is unanswered with both files sitting on disk.

  **The negative control in that converter's own refusal test is what found two real defects**, which is the reusable part. The control exists so a converter that died on every invocation could not pass all the refusal cases; it failed, and the failure was the finding. `smooth` had **no zero-layer guard** — pointed at a Wan it printed `Layers: 0 quantized: 0` and exited **0**, so `--dry-run`, the thing you run *before* spending hours, answered SUCCESS for a conversion with nothing to convert (the other four already refused: `quant_w4a4.py:386`, `quant_w4a8.py:248`, `quant_int8.py:139`, `quant_mixed.py:581`). And it had **no dtype guard** — it validated names only, then crashed ~3 minutes later inside another file's `read_tensor`; it now refuses in **1.8 s**, and `tools/test_smooth_guards.py` (7/7) asserts that *time*, because the regression to catch is moving the check back after calibration.

  One thing only the run teaches: **`convrot_groupsize` must be a power of 4, not of 2.** 128 raises `Regular Hadamard size must be a power of 4` at `comfy_kitchen/tensor/int8_utils.py:22`, which is why 64 and 256 are the only values in this tree. The backend preflight caught it before any work.

  Not covered: the three Z-Image builds from before 2026-08-22 are **not reconvertible** — their analyses carry no `source_identity_sha256` and `quant_mixed` refuses rather than skipping the check. That is the guard working; "did not reconvert" is not "reconverted and differed".

  **Measured 2026-08-22, on the 3090, so nobody re-derives it:** the resolved implementation is *invariant* to `convrot_groupsize` and to dummy-vs-real probe tensors. All four combinations — cg 64 and 256, `torch.empty` and real quantized tensors — resolve to `comfy_kitchen.backends.cuda`, and both real calls succeed.

  **That last clause is true of W4A4 and false of W4A8, and the paragraph did not separate them.** Measured 2026-09-01 with `tools/probe_convrot_groupsize.py`, K=2048 so no failure below is about divisibility:

  ```
  cg     quantize_convrot_w4a4_weight   convrot_w4a4_linear   quantize_w4a8_int8_weight
  16     ok                             ok                    RuntimeError
  64     ok                             ok                    RuntimeError
  128    ValueError (power of 4)        --                    RuntimeError
  256    ok                             ok                    ok
  512    ValueError (power of 4)        --                    RuntimeError
  1024   ok                             ok                    RuntimeError
  ```

  So **W4A4 accepts 16 / 64 / 256 / 1024 end to end — quantize *and* execute — while W4A8 accepts only 256.** The error the W4A8 path raises is `convrot rotate kernel only supports group_size 256`, which names no format and reads as a property of the ConvRot kernel; it is not. I was one paragraph away from recording "256 is the only usable value" until the probe refuted it.

  The practical consequence: `quant_mixed` measures **both** formats per layer to choose between them, so it touches the W4A8 path even when the result will be 170/170 in W4A4 — which pins it to cg 256. `quant_w4a4`, which would accept 1024, has no `zimage` profile. That is why the Z-Image ceiling could not be probed on this axis; see `bench/criterio_teto_zimage.md`. So `quant_w4a4.py`'s hardcoded 64/64 preflight against a 256 conversion is untidy, **not** wrong; and `_native_probe.py`'s own docstring claim that dummy kwargs let the check pass where a real call would not **did not reproduce** for these two ops on this build. That is "did not reproduce under the only conditions anyone has tried", not "is false" — the mechanism at `registry.py:246` may still bite elsewhere. `tools/probe_backend_resolution.py` re-runs it.
- **Streaming writes, never mmap.** Output header offsets are computed up front, then tensors are streamed: quantized layers are read by byte range → CUDA → `ck.quantize_convrot_w4a4_weight` → written; everything else is `copy_range`'d verbatim in 16 MiB chunks. **Do not reintroduce `safe_open` / mmap for large sources.** On this Windows host mapping the 21.93 GiB Gemma source failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
- **Atomic output.** Writes go to `<output>.partial`, then `os.replace`. Refuses stale partials, refuses existing outputs/sidecars, refuses a source that already has `_quantization_metadata`.
- **Profiles are strict allowlists**, not heuristics. `PROFILE_PATTERNS` matches only `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `model.layers.N.mlp.{gate,up,down}_proj.weight`; embeddings, norms, `lm_head`, and vision towers are excluded. Only `gemma` and `qwen` exist today. **Do not extend a profile to a new architecture without confirming that architecture's loader and layer config** — Flux, Hunyuan, SeedVR2, and Z-Image each need their own recipe.
- **Group sizes.** `CONVROT_GROUP_SIZE = 256` (rotation), `QUANT_GROUP_SIZE = 64`. Layer selection requires `shape[1] % 256 == 0`. The backend-probe subprocesses in both tools use dummy `64/64` values only to resolve the implementation, which is why they differ from the real conversion values — not a bug, but don't copy those numbers into real calls.

### Output format

Standard Safetensors. Per quantized layer: `<layer>.weight` as `I8` of shape `[rows, cols/2]` (INT8 container holding signed INT4), plus `<layer>.weight_scale` as `F32` of shape `[rows]`. Everything else preserved byte-for-byte. `__metadata__` carries `_quantization_metadata` (`format_version` 1.0 + per-layer `{format: convrot_w4a4, convrot_groupsize: 256}`) and `quantization: "ConvRot W4A4"`. See [ComfyUI/QUANTIZATION.md](ComfyUI/QUANTIZATION.md) for the upstream `QuantizedTensor` / `Layout` / `MixedPrecisionOps` model this format plugs into.

**Does `.backends.cuda` in `__module__` prove the native path ran? Measured 2026-08-22: yes, here.** The whole preflight rests on that string match, and the test that settles it is cheap: W4A4 quantizes the **activation** to 4 bits too, so a dequantized-weight fallback (`F.linear(x, W_deq)`, which is what `comfy_kitchen/tensor/convrot_w4a4.py:237` does under one condition) must agree with a real call. It does not — `native` vs `W4-only` is **1.43e-1**, not 1e-6, on a `[1024, 1024]` bf16 weight at cg=256. The A4 half is real. Re-run with `tools/probe_backend_resolution.py`.

`verify_w4a4.py` checks, in order: metadata structure and per-layer dtype/shape → packed shape vs source shape → **byte-identical comparison of every preserved tensor against the source** → native backend resolution → optional real-kernel smoke against `F.linear` on the BF16 source. The smoke's relative RMSE on random inputs (~0.25 for Gemma) is a liveness signal, **not** a quality metric.

`inspect_quant.py <file>` at root is a quick header dump (tensor count, dtype histogram, metadata, scale-like keys).

**There are TWO dialects, and a tool that reads only the first is blind in silence.** The section above describes `__metadata__._quantization_metadata`. A checkpoint may instead ship the per-layer JSON *as tensors* — `<layer>.comfy_quant`, UTF-8 bytes, same content — which is what `comfy/utils.py` produces at load and `comfy/ops.py` dispatches on. Both are valid and a file can carry only the second. Measured 2026-09-01: `LTX25-distilled-DiT-comfy-w4a4` (riftcast, **1440** genuinely 4-bit layers) and both `MiniMax_H3_*_pruned_mixed_int4_int8_convrot` files (117 int4 + 83 `int8_tensorwise`) carry **no `_quantization_metadata` at all**. A reader that checks only the metadata reports **zero quantized layers on a fully quantized file** — and reports it as a clean result, which is the dangerous part. Reading the second dialect costs one seek and ~50 bytes per layer, no torch: see `ler_dialeto_por_tensor` in `tools/avaliar.py`. `quant_audit.py`'s `read_quant_dialects` reads three *metadata* dialects and still does not read this one.

**Batch evaluation, no GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar.py ComfyUI\models --saida .scratch\avaliacao
```

Header + sidecar + `.analysis.json` only — **163 checkpoints in 0.68 s**, no torch, no model load. It computes the median effective error entirely offline (the sidecar says which format each layer got; the analysis says that format's measured error on that layer) and reproduces every number this bench has published. Verdicts are `REPROVADO` / `OLHAR` / `SEM VEREDITO`; **`APROVADO` is deliberately absent**, because no cut on either axis separates usable from unusable here — 0.1837 correct against 0.2147 destroyed, 0.7173 fine against 0.8255 destroyed. It rejects, points and predicts. It does not approve.

**Layer 2, does the quantized math actually run TODAY — needs the GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar_despacho.py --controles
```

The sidecar's `backend` field records the **conversion**, not today's load, and comfy-kitchen, ComfyUI and torch have all moved since. `tools/avaliar_despacho.py` loads through ComfyUI's normal path and counts, reusing `probe_quant_dispatch.py` rather than reimplementing the count. Criterion, predictions and three refutation conditions in `bench/criterio_camada2_despacho.md`, written before running; **all three stayed silent**. Executed 2026-09-01 over the 37 checkpoints layer 1 marked as quantized:

```
26  DESPACHA              every diffusion checkpoint on this bench, 0 dequantize
 6  TRAVADO_PELO_COMFY    all six text encoders
 2  SEM VEREDITO          the two Abiray MiniMax -- die in 7-9 s, not a forward
 2  NAO_PROBAVEL          live in `checkpoints/`, which the probe cannot resolve
 1  NAO_DESPACHA          flux-2-klein-base-4b-fp8
```

So the `backend` field still describes today's execution — including **thirteen builds never loaded here before** and three third-party files (riftcast's `LTX25-distilled-DiT-comfy-*`, 1440 layers each; `DasiwaWAN22I2V14BLightspeed`; Winnougan's `minimax_h3_..._w4a8_convrot`).

**`DESPACHA` is not approval.** It answers one binary question — was the kernel called? — and nothing else. On this bench HunyuanVideo 1.5 W4A4 dispatches natively and the render is destroyed. `APROVADO` stays absent from every layer of the evaluator.

**fp8 runs dequantized, and the `impl` counter says so outright.** `flux-2-klein-base-4b-fp8`, a *diffusion* model with `comfy_force_cast_weights=False` — so not the CLIP lock — makes 0 quantized forwards, 8 `dequantize`, and the op literally called is `dequantize_per_tensor_fp8=comfy_kitchen.backends.cuda`. Memory saved, time not. Open, and not worth guessing at: where `full_precision_mm=True` comes from on a diffusion model, given `comfy/ops.py:1667` passes `disabled=` and not `full_precision_mm` on that path.

**There are TWO text-encoder locks, and the rule that knew only one misclassified the very file that established the finding.** This file documents both — `comfy_force_cast_weights` (from `comfy/sd.py:269`) and `full_precision_mm` (hardcoded at `comfy/sd1_clip.py:114` for every text encoder) — and layer 2's first rule asked only for the first:

```
5 encoders                            force_cast {'True': N}                 -> TRAVADO
qwen3vl_32b_minimax_h3-int4_convrot   force_cast {'False': 351}
                                      fpmm       {'True': 350}               -> NAO_DESPACHA
```

That sixth file is the one this bench measured on 2026-08-31 to establish the lock in the first place. `NAO_DESPACHA` reads as a defect in the checkpoint and would send someone to reconvert a public file that is fine. Either lock alone drops the math, so the rule now asks for either and **names which**. And the defect that let the other one hide: the report did not record the field its own verdict rested on — a verdict whose evidence is not in the report is an opinion.

**Layer 3, the reference-arm guard — needs the GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar_referencia.py --modelo <fp16>.safetensors --clip <te>.safetensors --clip-type wan --steps 25 --size 480 --frames 33 --vace-strength 0.0
```

It does not ask whether the image is good. It asks **whether the model responds to its own conditioning** — two unrelated prompts by two seeds on the *unquantized* file, and the statistic is `d_prompt / d_semente`, where the seed distance is the negative control that makes the ratio readable. Criterion and the three refutation conditions were written in `bench/criterio_guarda_referencia.md` **before** measuring. **Executed 2026-09-01, three families:**

```
braco                                     resposta   veredito       previsao   acertou
Wan VACE 1.0  destruido, verdade          0,2477     REPROVADO      < 0,5      sim
Wan VACE 0.0  bom, verdade                0,7911     OLHAR          > 0,8      NAO, por 1,1%
Z-Image v2 BF16       bom                 1,3459     SEM VEREDITO   > 0,8      sim
HunyuanVideo 1.5 FP16 bom                 2,8603     SEM VEREDITO   > 0,8      sim
```

All three refutation conditions stayed silent: **3.19x** separates the broken arm from the nearest healthy one, the reject threshold cleared all three healthy arms, and the destroyed one did not pass. **It would have aborted the Wan run on the first image instead of the fourth.** The mechanism shows raw, not only in the ratio: on the broken arm the prompt moves the latent **0.075** while the seed moves **0.250** — the model generates from noise and ignores what is asked. The `> 0.8` healthy prediction missed by 1.1% and **the threshold was not moved**; a threshold changed after seeing the number it was meant to classify is a description, not a guard. The reject side rests on **one** genuinely broken arm, and the ratio is noisy in absolute value (Hunyuan's two seeds gave 1.87 and 3.85), so what counts is distance from the threshold, not the second decimal.

Two things layer 1 established on its first pass. **The Z-Image card had 0.1241 on the wrong row** — it belongs to `zimage-v2-w4a4` (170 convrot), not to the mixed build, which is 0.0774; confirmed across nine independent calibrations, corrected the same day. The band's `tolerado` value is unchanged, it just gained an owner, and it is the *most aggressive* build measured. And **the calibration seed moves the median 2-6%**: the same Wan checkpoint measures 0.051807 or 0.054631 depending on which calibration you use. Smaller than the band's own 45% width, so the per-model line survives — but a four-decimal number from one calibration claims precision this bench does not have, so the spread now travels beside it.

## The LTX card measured half the model, and the half it skipped ranks the arms the same way

LTX 2.x generates **video and audio in one latent**. Until 2026-09-13 `tools/ltx25_video.py`
decoded the video branch only — its own docstring said so — and the published LTX 2.5 card compared
three transformers on 249 frames and zero audio samples. The owner called it: *"um gerador de video
como o LTX nao gera somente imagens, ele gera imagem e audio ao mesmo tempo, entao voce tem que
comparar os dois."* `tools/ltx_video.py` now decodes both (`LTXVAudioVAEDecode` on the second
output of `LTXVSeparateAVLatent`) and writes PNGs, a FLAC and an MP4 with the track;
`tools/compara_av.py` measures both branches against a reference with silence and same-RMS white
noise as controls. Re-rendered, same seed, three arms:

```
arm                   MAE   PSNR   SSIM  | log-mel L1   SNR      lag    spec.conv  RMS
int8 Lightricks      4.10  29.71  0.941  |   0.041     11.2 dB   0 ms    0.124   -38.8 dBFS
W4A8 ours            7.81  25.39  0.895  |   0.120      3.4 dB   0 ms    0.311   -38.3 dBFS
control: silence                        |   6.980      0.0 dB
control: white noise, same RMS          |   1.471     -3.0 dB            1.097
```

Audio ranks the arms as the picture does and by a wider margin (1.9x further in frames, 2.9x in
sound); neither arm changed level or slid in time. **The re-render came back pixel-identical to
the first run in all three arms** (MAE 0.0 on frames 1/63/125/187/249), across a server restart
and, for BF16, a different disk — the PNG hashes differ only by the embedded workflow metadata.

**LTX 2.3 distilled 1.1, measured 2026-09-14 on the same protocol at 8 steps, every arm on the
same saved conditioning, reference BF16 through the lossless GGUF container:**

```
arm                            GiB    sampler 8 steps     MAE   PSNR   SSIM  | log-mel  conv   lag    level
BF16 (GGUF, partial load)     39.15   8.05 s/it (66 s)     --     --     --  |   --      --     --   -23.5
GGUF Q6_K (third party)       16.55   5.04 s/it (40 s)   3.59  29.00  0.941 |  0.163  0.419  0 ms   -24.1
W4A8 ours (transformer 11.66) 15.51   2.30 s/it (18 s)  10.39  21.89  0.829 |  0.163  0.424  0 ms   -24.0
W4A4 ours (control)           14.31   1.59 s/it (12 s)  14.45  20.36  0.734 |  0.281  0.491  0 ms   -25.8
controls (log-mel): silence 8.460, white noise at the reference's RMS 1.748
```

W4A8 is a usable 2.3 (P1); W4A4 does not break and is worse (P2 — fourth family that tolerates
A4); the 6-bit GGUF is 2.9x closer in the frames and 2.2x slower per step (P3); **the sound does
not separate W4A8 from Q6_K** (0.163 against 0.163) while the picture puts them 3x apart — on 2.5
the sound ranked the arms with a wider margin than the picture, on 2.3 it ties the first two (P4
not refuted, with a tie). **The tie is the metric saturating, measured the same night:** at 3 steps
(tail of the same schedule, same conditioning) Q6_K stays in phase with the reference (log-mel
0.063, waveform SNR +8.4 dB) and W4A8 does not (0.223, −1.1 dB) — at 8 steps both had drifted out
of phase (SNR −1 dB each) and landed on the same log-mel floor. Caveat: at 3 steps the 2.3 renders a
different, degraded scene in every arm, so that run answers only the mechanism
(`bench/criterio_fechamento_2026-09-14.md`, G). P5 was untestable as written (no DisTorch BF16 arm
ever ran) and P6 refuted (`bench/criterio_ltx23.md`). The "whole run" wall-clock ranked the arms by
which disk they were read from (BF16 GGUF from local C: 155 s; W4A8 from the W: share 208 s) — one
more reason that column is not a speed. Published with MP4/FLAC proofs, the identity control and
the broken-saver negative: **https://huggingface.co/JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot**

**And the BF16 arm killed the server once — then the 2.3 work killed it three more times, same
signature.** `Windows fatal exception: access violation` in `torch/storage.py __getitem__` under
`comfy/utils.py:136` (`f.get_tensor(k)` inside `load_torch_file`) — the page-in of a memory-mapped
39–43 GiB safetensors. First from D: with ~24 GiB RAM free; the retry from W: under a second name
(hardlink, same inode) rendered in 769.6 s with 40 GiB free. **This file first recorded W: as "a
local disk". It is not**: `net use` lists `W: \\192.168.3.40\zfe`, an SMB share like D: and P:;
the claim came from a grep of `net use` that only looked for D:. So every BF16 load here was an
mmap over SMB — one survived, four died (32–44 GiB free, no correlation with RAM), while a bare
Python process paged the same 39 GiB file through in 4 s. The redirector was the suspect for a few
hours; the paragraph after this one is the measurement that replaced it. The only local NTFS
volumes are C: and F:. `folder_paths` returns the **first** yaml root that has the name, so yaml order decides which
*volume* a 39 GiB mmap comes from, and no log says which. Proofs on the Hub: `av/*.mp4`,
`av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.

**And the redirector was the wrong suspect too — measured 2026-09-14.** The same two-view
mapping that `safe_open` performs, done by hand in a bare Python process without ComfyUI, dies the
same way on the local C: drive as on W:. What was measured on this 39.13 GiB file, with the system
commit counters read before and after each call (`GetPerformanceInfo`):

```
call                                                       commit charge
safetensors.safe_open(framework="pt")                      +40.8 GiB at open, before any tensor is read
   (two copy-on-write views of the same file: memmap2 for the header and
    torch.UntypedStorage.from_file(shared=False) for the data; +80.2 GiB while both are alive)
torch.empty of the model's parameters                      +40.7 GiB more
read-only mmap (what numpy and the GGUF loader use)          0
UntypedStorage.from_file(shared=True)                        0
```

Windows charges a copy-on-write view its whole size at mapping time, so ComfyUI's normal
safetensors path needs twice the file in commit just to open it and three times to build the
model. This machine's commit limit was 124.8 GiB (63.6 GiB of RAM plus a 61 GiB system-managed
pagefile) when this was measured and is **98.72 GiB as of 2026-09-20**, because the pagefile shrank
to 35.07 GiB -- so the trap below is tighter now than the numbers in it suggest. The old figure, for
the record: 124.8 GiB (63.6 GiB of RAM plus a 61 GiB system-managed
pagefile) with about 70 GiB already committed by other processes. When the charge forces the
pagefile to grow, the new view sometimes comes back with the limit raised but the charge not
taken, and the first read through it is an access violation in `torch/storage.py __getitem__` —
the server's exact signature (`safe_open` + first `get_tensor` on W:, 3 of 3 runs; the hand-made
two-view mapping on C:, 1 of 1; the same calls survive on other runs, which is why one BF16 load
in seven succeeded). The two deaths inside `nn.Linear.__init__` — the model's `torch.empty` —
carry the same signature and were not reproduced in isolation. So the list above stays, with a
different reading: rows 3 to 5 died of commit, not of the network. The LTX 2.3 reference was
rendered without the safetensors reader at all: the same BF16 weights, bit for bit, in a GGUF
container read through a read-only memmap (`tools/safetensors_to_gguf_bf16.py`; bytes,
dequantized weight and Linear output verified identical on 12 sampled layers with
`tools/probe_gguf_bf16_equivalence.py`).

**Rule that follows, for this machine:** never open a safetensors bigger than about half the free
commit through ComfyUI's normal reader (`safe_open` costs 2x the file; the model another 1x). For a
BF16 reference of a 20 B+ model, convert it losslessly with `tools/safetensors_to_gguf_bf16.py`
and load it with `UnetLoaderGGUF`; keep the text encoder out of the process with
`tools/ltx_video.py --encode-only` / `--cond-from` (saved conditioning costs nothing). The pagefile
is the owner's, not ours: `?:\pagefile.sys`, system-managed, 61.2 GiB on C: at the time of writing,
and every death above needed it to grow.

**And `LTXVSaveConditioning` is not a way to keep the text encoder out of the process for LTX 2.3 —
measured 2026-09-14.** ComfyUI-LTXVideo's saver keeps the tensor and an attention mask. The LTX 2.3
encoder returns `{"unprocessed_ltxav_embeds": True}` beside the tensor
(`comfy/text_encoders/lt.py:201-204`), and the model applies `caption_projection` and the embeddings
connectors only when that key arrives (`comfy/model_base.py:1185` →
`comfy/ldm/lightricks/av_model.py:583`). Loaded back through `LTXVLoadConditioning` the key is gone,
the 6144-wide context passes the "already processed" width check and goes raw into the
cross-attention: the same W4A8 model, same seed, renders brown noise with noise for sound — **MAE
75.9, SSIM 0.19, log-mel 1.05** against the live encoder (`bench/ltx23/cond_identity_ltxv_saver/`).
Four LoRA renders and one BF16 attempt were made on that conditioning before the identity control
caught it; they stay under `bench/ltx23/*_ltxv_saver/` as what the wrong tool produces and decide
nothing. **The identity control is not optional**: a render on saved conditioning that was never
compared with the live path is a render of an unknown prompt. The replacement keeps every option —
`tools/ltx_encode_lowcommit.py` (bare process, zero-commit reader, float32 tensor, options as
tensors and JSON metadata) and `VoidLoadConditioningFull` in `custom_nodes/comfy-void-stage-tools`,
which refuses a file without them; `tools/ltx_video.py --cond-from` uses that node. The encoder run
on the 3080 Ti differs from the server's 3090 encode by rel-L2 1.0e-3 (87 % of elements bit-equal
in bf16, max |Δ| 1.0 on a [−148, 294] range): not bit-identical, and said so where it is used.

**"Per-step time depends on residency by 3x" sat in this file for an hour and was wrong — the
field it came from is not per-step time.** `tools/ltx_video.py` writes `s_por_passo` and
`s_por_quadro` as the whole run's wall-clock divided by steps or frames: model load, text-encoder
load and encode, sampling, both VAE decodes and muxing, in one number. The per-step instrument is
the sampler's own tqdm bar in the server log, and it says the same W4A8 249-frame render sampled
**8 steps in 18 s (2.30 s/it) with the encoder live and 8 steps in 18 s (2.30 s/it) with saved
conditioning**. The 638 s against 208 s of wall-clock was loading the 22.7 GB encoder over SMB and
encoding once; residency changed nothing the sampler could see. Same instrument, same protocol,
the other 2.3 arms: W4A4 **1.60 s/it**, GGUF Q6_K **5.12 s/it** (dequantized math), all 249 frames
at 512 px on the 3090. The 2.5 card's "s/frame" column had the same defect — its 3 steps sample in
about 25 s and the runs took 400–800 s — and now says so. **Rule: a speed number from
`ltx_video.py`'s JSON is the wall-clock of a run; a per-step number comes from the progress bar,
or from a tool that times the sampler alone.** The JSON now carries a `nota_tempo` field saying
exactly that.

## The closing round of 2026-09-14: the six conversions nobody had measured at the output

Counted first (`.scratch/sidecars_2026-09-14.txt`): **53** `.quant.json` sidecars across the four
yaml roots, 25 with their weights still on disk, 44 verified at the output, **6 never verified**.
The criterion for all six was written before any number (`bench/criterio_fechamento_2026-09-14.md`,
A/C/D/E/G) and the verdicts sit in the same file with the hour. What each one taught:

- **A — the factory Gemma 3 12B in W4A8 as the LTX 2.3 encoder is usable, and releasing the two
  text-encoder locks on it is free.** Conditioning rel-L2 0.043 from BF16 on the quantized-math
  path and 0.042 on the dequantized one — they differ by 0.001, in opposite directions on the
  positive and negative prompt — and the W4A8 transformer on that conditioning lands **MAE 6.75 /
  SSIM 0.894 / log-mel 0.087** from the BF16-encoder render: same scene, under the transformer's own
  quantization distance (10.39). Published: **https://huggingface.co/JoaoZaokk/Gemma-3-12B-it-W4A8-ConvRot**.
  Open, and written on both Gemma cards: the 2026-08-31 monkeypatch measurement said releasing
  W4A8 added 0.18 on the encoder's raw output; the flag measures 0.001 on the projected
  conditioning. Different tensor, different instrument; not reconciled.
- **C — on LTX 2.5, `int8_tensorwise` + ConvRot reproduces Lightricks' int8 (MAE 4.19 against
  4.10, log-mel 0.043 against 0.041), and the same int8 WITHOUT the rotation lands 2x farther in
  the frames (8.23) — farther than our W4A8 (7.81), which keeps the rotation and drops the weight to
  4 bits.** In the sound the unrotated int8 still beats W4A8 by 2x (0.060 against 0.120). The
  criterion's reading "int8 worse than 4 bits = broken build" did not hold: same scene, SSIM 0.889
  against 0.895, per-frame ranges overlapping. On this model the rotation is a term of the same
  size as the weight bit-width. Sampler from the progress bar: int8 2.02–2.12 s/it, BF16 8.48 s/it
  (3 steps, 512 px); the "whole run" column ranked the arms by disk again. Both builds and the
  proofs went to the 2.5 card, the unrotated one labelled as the measured negative.
- **D — the three heretic Gemma builds against the heretic BF16: W4A8 keeps the scene (5.45 MAE);
  both W4A4 builds render a coherent, well-lit, DIFFERENT scene — daylight where the prompt said
  dusk — at 29.8 and 38.1 MAE.** Conditioning: W4A8 0.042 < SmoothQuant 0.158 < ConvRot 0.221
  released, 0.043 < 0.082 < 0.111 locked — the predicted order on both paths; smoothing pays 1.4x
  and does not rescue (the 2026-09-01 open question, closed). **Releasing the locks costs 2x on
  W4A4 and 0.001 on W4A8**: the 4-bit activation is the term that hurts. The abliterated BF16 sits
  0.10 from the factory BF16 and keeps the scene (MAE 9.34), so the flip is somewhere between 0.10
  and 0.16 of rel-L2 on this prompt — one prompt, one seed. The W4A4 encoder weights stay off the
  Hub; their proofs and sidecars are on the heretic card.
- **G — the audio tie explained** (in the LTX section above: saturation under phase
  decorrelation, shown at 3 steps).
- **Two instrument defects found on the way.** `tools/sampler_tempo_do_log.py` read only
  `Prompt executed in X seconds`; ComfyUI switches to `HH:MM:SS` above 600 s (`main.py:400`), so
  every long run showed as `(sem fim)` — that is how the 2.5 BF16's 8.48 s/it (761 s wall) had gone
  missing. And the lazy-load dtype guard asked whether the file disagreed with itself, when the
  failing comparison is file against compute dtype (the capybara paragraph in the dynamic-VRAM
  section above).

- **E — `capybara_v0.1` in W4A8 is usable**: latent divergence 0.1439 from its own BF16 against
  0.7072 for its W4A4 on the same prompt and seed; the same apple, a little softer, stem lost. The
  reconversion came out **byte-identical** to the 2026-09-01 build across two cards (sha256
  `4317156b…`, 8,847,634,096 B). Its per-step from that single ladder run (6.56 s against 1.42 for
  BF16, after a 15.5 GiB reference in the same process) is not reported as a speed. Uploaded to
  the Hunyuan repo with both images.
- **The W4A4 encoders on stock ComfyUI's locked path keep the scene** — ConvRot 16.4 MAE,
  SmoothQuant 27.4, both dusk — where the released path lost it (38.1 / 29.8, daylight), so the flip
  sits between rel-L2 0.11 (locked ConvRot, kept) and 0.16 (released SmoothQuant, lost). The render
  order on the locked path (ConvRot closer) inverts the conditioning order (SmoothQuant closer):
  one seed, noted, not explained — conditioning distance orders formats coarsely and does not
  order two neighbouring W4A4 builds, the per-layer-error lesson again, now on the encoder. Nothing
  recommends either build (3–5x the W4A8's distance for 8 % less memory and no speed on that
  path); the weights stayed off the Hub at first and the proofs went up
  (`bench/ltx23/encoder_heretic_locked/`). **Then the owner said to upload them** (05:10), so they
  are on the heretic card under `w4a4/`, labelled with the scene change beside the file.

Published in this round: Krea 2 Turbo W4A4 (gated, licence terms met), Qwen3-VL 4B W4A8, the
factory Gemma W4A8 (new repo), the two 2.5 int8 builds (2.5 repo), capybara W4A8 (Hunyuan repo),
the heretic W4A4 proofs and sidecars (heretic repo); the 2.5, heretic, Hunyuan and 2.3 cards
rewritten. Still open: the monkeypatch-against-flag discrepancy on what releasing costs W4A8; the
capybara W4A8 speed; one prompt and one seed per arm everywhere here.
## LoRA over a quantized weight: it is a requantization, and that is measured

The owner asked on 2026-09-13 whether LoRAs "work the way they should" on the quantized builds. The
only prior measurement here (2026-08-19) answered a narrower question — the native kernel still
fires with a LoRA applied, 680/680. **Whether the LoRA arrives intact in the weight had never been
measured.** Traced first, then measured on the real path (`tools/probe_lora_requant.py`):

`LoraLoaderModelOnly` over a `QuantizedTensor` does not keep the LoRA as a separate branch.
`ModelPatcher.patch_weight_to_device` (`comfy/model_patcher.py:899`) runs `convert_weight` →
`W.dequantize()` (`comfy/ops.py:1449`), adds the delta in `lora_compute_dtype` (fp16 on this
card), then `set_weight` → `W.requantize_from_float(W', scale="recalculate",
stochastic_rounding=seed)` (`comfy/ops.py:1455-1457`). **Dequantize, add, requantize to the same
4 bits.** The kernel is unchanged because the weight comes back in the same layout; the delta, which
is usually smaller than the 4-bit grid step, only survives in expectation.

Measured per layer on the real path, `err` = `‖W − W_bf16‖/‖W_bf16‖` (weight-space, **not** the
activation-space `err_w4a4` of the band table above — the two do not compare):

```
model, format              LoRA                       |δ|/|W|   survival  cosine  noise/LoRA   err: before -> no-op requant -> with LoRA
Z-Image v2  W4A4 cg256     RealisticSnapshot r32       0.095     1.000     0.51     1.7x        0.157 -> 0.163 -> 0.238   (1.45x)
Krea2 Turbo W4A4 cg256     krea2 turbo LoRA r64        0.0086    1.000     0.14     7.2x        0.162 -> 0.168 -> 0.176   (1.07x)
Wan 2.2 5B  W4A8           a 14B LoRA (wrong model)    0 (shape fails)  -   -       -           0.0731 -> 0.0835          (1.14x, NO LoRA applied)
LTX 2.5 22B W4A8           ltx2-squish (an LTX 2.0 LoRA) 0.013-0.086, ZERO in 16/24  0.956  0.39   3.1x    0.0731 -> 0.0837 -> 0.0837 (zero) .. 0.114
LTX 2.5 22B W4A8           LTX23 Product Commercial r16 0.0021    0.909     0.05    19x         0.0731 -> 0.0836 -> 0.0838   (1.15x)
LTX 2.3 22B W4A8           LTX23 Product Commercial r16 0.0020    0.908     0.05     20x        0.0731 -> 0.0837 -> 0.0838   (1.15x)
Qwen-Edit 2511 W4A8        Lightning 4-step r64         0.0005    0.858     0.01    103x        0.0731 -> 0.0836 -> 0.0835   (1.14x)
```

**And the output contradicts the weight, which is the finding.** Rendered the same day
(`bench/qwen_edit_lora/grade_lightning_4passos.png`, criterion R1-R4 written first in
`bench/criterio_lora.md`): at 4 steps without the LoRA both INT8 and W4A8 **fail** — no scarf, the
sign stays OPEN, a speckled apple-pear hybrid, oversharpened texture (the control that had to
fail, failed); with the LoRA, INT8, W4A8 merged and W4A8 bypass **all obey all three instructions**;
merged vs bypass differ by 1.26 in the untouched region against 3.5-3.9 between either and INT8.
The layer where the LoRA looked most buried (86% survival, noise 103x the delta) is the one whose
output is intact. **Weight-space per-layer numbers rank and alarm; they do not decide** — the same
lesson this file already records for per-layer error across formats. Caveat: a 4-step LoRA changes
the whole regime and is the most robust kind; a subtle style LoRA was not tested at the output.

On LTX 2.5 W4A8 with `ltx2-squish` (49 frames, same seed, no trigger word in the prompt — a design
hole, so this measures the cost of loading the LoRA, not its effect): merged and bypass land on the
**same** composition and sit 6.3 MAE apart, against 15-18 from the no-LoRA reference and 28 from a
seed change; neither moved the audio level or timing, while the other seed moved both (RMS -28 vs
-20 dBFS, lag +78 ms). The requantization noise perturbs the output by a third of what the LoRA does
and a quarter of what a seed does. `bench/ltx25/lora/`.

**LTX 2.3 W4A8 with its own LoRA (`LTX23_Product_Commercial`, trigger `srx_commercial`), measured
at the output on 2026-09-14 on valid conditioning, two rounds of 49 frames.** With the trigger,
merged and bypass both execute the prompt's "rotating slowly" (motion 4.6 against the no-LoRA
reference's 1.25) and draw a different headphone: 41 MAE from the same-seed reference, against 57
for a seed change — the LoRA still moves less than a seed (R7 refuted, as on 2.5), and R9 was
undecidable because a commercial prompt already renders as a commercial without the LoRA. Merged
and bypass sit 7.5 apart (5.1 without the trigger) against 23–41 from the reference — a fifth of
the LoRA's effect, the third round with that proportion (R8 confirmed). Bypass lowered the audio
level both times (−2.2 / −4.7 dB; merged −0.1 / −2.6) and costs 24 % per step (0.68 against 0.55
s/it, from the sampler's progress bar); merged costs nothing. The same LoRA was not rendered on the
BF16 original, so "same effect as unquantized" is not measured. `bench/criterio_lora.md`,
`bench/ltx23/lora*/`.

The two LTX rows add two things. `ltx2-squish` ships **all-zero `lora_B` for 768 of its 1152
matrices** (every audio and cross-modal attention family, read from the file); ComfyUI matches the
key, applies a zero delta and requantizes the layer anyway, so two thirds of the layers that LoRA
names pay the +14% for nothing. And on the `asym_w4a8_int8` codebook layout the survival is **not**
1.000 — 0.956 and 0.909 — a small delta loses 5-9% in the requantization, a bias the convrot W4A4
rows do not show; with |δ|/|W| = 0.002 the added noise is 19x the LoRA itself.

Three things that hold across the rows. **The LoRA is there, in expectation** — survival 1.000 to
three decimals, stochastic rounding has no bias. **What lands in the weight is the delta plus noise
larger than the delta** — cosine 0.51 and 0.14; the noise grows with the LoRA's own magnitude
(each entry moves with probability ∝ |δ|/step), which is why the √2 "independent noise" prediction
written before the run held on Z-Image (1.45x) and was refuted on Krea2 (1.07x). **Requantization
is not idempotent**: a no-op patch already costs 0.157 → 0.163 (convrot) and 0.0731 → 0.0835
(asym W4A8 with codebook).

The Wan row is the trap that only exists on quantized models. A LoRA from another architecture
matched 300 keys **by name**, failed every `calculate_weight` on shape, and ComfyUI logged
`ERROR lora ... shape` and **continued** — delta zero, weight requantized anyway, model 14% worse in
weight error with nothing applied. On a BF16 model the same failure rewrites the weight unchanged
and is harmless. The only evidence is one log line per layer.

`LoraLoaderBypassModelOnly` (`comfy_extras/nodes_lora_debug.py`, labelled "for debugging") keeps the
delta as a BF16 low-rank branch in the forward and never touches the weight, so it has none of this
noise. Whether the noise is *visible* is a render question; the criterion for those renders was
written before running them in `bench/criterio_lora.md` (R1–R6), with the control that has to fail
— Qwen-Image-Edit at 4 steps **without** its Lightning LoRA — named first.

Not covered: one strength (1.0), one LoRA per model, a sample of layers; nothing on text-encoder
LoRAs; the bypass loader is measured only at the output, since by construction it leaves the
weight alone.

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
