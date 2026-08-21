# Inventário auditável da bancada

Gerado em 2026-08-21, ticket `.scratch/estado-entregavel/issues/12-inventario-da-bancada.md`.

**Como ler este documento**: cada afirmação abaixo é `[EXECUTADO]` (comando rodou agora, saída
colada) ou `[LIDO]` (inferido de código-fonte ou de um arquivo gerado por outra rodada, não
verificado nesta rodada por execução). Nenhum modelo foi carregado, nenhuma GPU foi tocada, nenhum
`quant_audit.py` foi rerodado — a seção de modelos reusa `quantization_inventory.{json,md}` como
pedido pelo ticket.

---

## 1. ENV

### 1.1 Stack fixado — `[EXECUTADO]` via `python_embeded\python.exe -s -m pip list`, agora

| Pacote | Versão instalada |
|---|---|
| torch | `2.13.0+cu130` |
| torchvision | `0.28.0+cu130` |
| torchaudio | `2.11.0+cu130` |
| comfy-kitchen | `0.2.31` |
| nunchaku | `1.2.1+cu13.0torch2.11` |
| spas_sage_attn | `0.1.0+cu130torch2.9.0andhigher.post4` |
| sageattention | `2.2.0+cu130torch2.10.0andhigher.post6` |
| flash_attn | `2.8.4` |
| triton-windows | `3.7.1.post27` |
| safetensors | `0.8.0` |
| diffusers | `0.38.0` |
| transformers | `5.14.1` |
| gguf | `0.19.0` |

Confronto com `CLAUDE.md` (que o próprio commit `33331a8` já corrigiu antes desta rodada): torch,
torchvision, torchaudio, comfy-kitchen, nunchaku e spas_sage_attn batem exatamente com o `pip list`
acima. `CLAUDE.md` não lista `sageattention` nem `flash_attn` na tabela de ambiente (só menciona a
versão do sageattention em texto corrido, na nota da DLL) — ambos batem com o texto corrido quando
conferidos.

### 1.2 `cudart64_12.dll` — `[EXECUTADO]` via `find . -iname "cudart64*.dll"`, agora

```
./python_embeded/Lib/site-packages/torch/lib/cudart64_13.dll
./venvs/ultravox311/Lib/site-packages/torch/lib/cudart64_12.dll
./venvs/ultravox311/Lib/site-packages/torchvision/cudart64_12.dll
./_backup_20260816_preupgrade/python_embeded/Lib/site-packages/torch/lib/cudart64_12.dll
./_backup_20260816_preupgrade/python_embeded/Lib/site-packages/torch/lib/cudart64_13.dll
```

Confirma o que `CLAUDE.md` já registrou: o `python_embeded` ativo (o que os comandos deste
documento usam) só tem `cudart64_13.dll`. As três ocorrências de `cudart64_12.dll` restantes estão
em `venvs/ultravox311` (venv separada, não é ComfyUI) e em `_backup_20260816_preupgrade` (backup
congelado do stack pré-upgrade). Nenhuma delas está no caminho do ComfyUI em uso. Não peguei
timestamp de instalação de novo — isso já está registrado em `CLAUDE.md` e não mudou.

### 1.3 O que está preso a quê — `[LIDO]`, do próprio `CLAUDE.md`, não reconferido por execução nesta rodada

- `comfy-kitchen 0.2.31` é o registry que resolve se `convrot_w4a4_linear` e
  `quantize_convrot_w4a4_weight` caem em `comfy_kitchen.backends.cuda.*` (native) ou caem em outro
  lugar (fallback deque/eager). O preflight que checa isso mora em `quant_w4a4.py` e
  `quant_w4a8.py`/`quant_mixed.py` (ver seção 3); não rodei esse preflight agora porque ele
  importa `comfy.quant_ops`, que é uma carga de módulo pesada e, dependendo do caminho, pode tocar
  CUDA — fora da lista de comandos permitidos desta rodada.
- `nunchaku 1.2.1+cu13.0torch2.11` é um build de torch 2.11 rodando sob torch 2.13 instalado —
  funciona por compatibilidade binária, não por match de versão. Precisa de
  `--disable-dynamic-vram` no ComfyUI (regra em `CLAUDE.md`, verificada lá por execução real
  2026-08-18; não re-executei).
- `spas_sage_attn` e `sageattention` são builds `+cu130` que linkam `torch_cuda.dll` em vez de
  `cudart` diretamente — é por isso que a DLL cu126 antiga parou de ser necessária. Fato já
  registrado em `CLAUDE.md`, com a ressalva explícita de que "import bem-sucedido" não é prova de
  execução de kernel; o teste real é `_check_accel.py`, que não rodei nesta sessão (toca GPU,
  proibido pela lista de comandos).

### 1.4 O que este documento **não** cobre em ENV

Não rodei `_check_accel.py` (kernel real), não rodei o preflight de `comfy_kitchen.registry`, não
conferi timestamps de instalação de nenhum pacote além do que já estava em `CLAUDE.md`. `pip list`
prova o que está instalado agora, não prova que o kernel executa — essa é a mesma distinção que o
próprio `CLAUDE.md` já faz para a DLL.

---

## 2. NODES

### 2.1 Contagem — `[EXECUTADO]` agora, recontada, não copiada de `CLAUDE.md`

`ls ComfyUI/custom_nodes/` (sem `-a`) dá **66 entradas / 64 diretórios** — bate exatamente com o
que `CLAUDE.md` afirma. Mas `find` (que enxerga ocultos) acha uma **67ª entrada**: `.disabled/`,
um diretório vazio e oculto que `ls` sem `-a` não mostra. `CLAUDE.md` não erra o número que
declara — só não menciona esse diretório oculto, porque não usou `-a`. Registrando aqui porque
"recontar na hora" pegou algo que a contagem anterior não tinha como pegar do jeito que foi feita.

Dos 67:
- **65 diretórios** (incluindo `.disabled/` vazio e `__pycache__/`, que não são pacotes de nó).
- **2 arquivos soltos**: `example_node.py.example` (template, não é nó ativo) e
  `websocket_image_save.py` (nó de exemplo que o próprio ComfyUI distribui, ativo).
- Descontando `.disabled/` e `__pycache__/`, sobram **63 diretórios que são pacotes de nó de
  fato**.

### 2.2 Quantos têm `.git` próprio — `[EXECUTADO]` agora, checagem direta, não assumida de `CLAUDE.md`

`CLAUDE.md` afirma "Each of the 66 entries ... is its own repo". Testei isso agora
(`test -d <pacote>/.git` para os 63 diretórios de pacote):

- **31 têm `.git` próprio** (são de fato repos git independentes).
- **32 não têm** `.git` — são diretórios simples, instalados por outro caminho (zip, cópia manual,
  ou `.git` removido depois do clone). Entre os 32 sem `.git`: `comfy-quant-preflight` (nosso, ver
  2.3), `comfy_convrot_native`, `rgthree-comfy`, `comfyui-impact-pack`, `ComfyUI-WanVideoWrapper`,
  `whiterabbit`, `comfyui-rogala`, e outros 25.

Ou seja: a afirmação "cada entrada é seu próprio repo" está errada para quase metade dos pacotes
instalados nesta bancada. Isso é uma correção a `CLAUDE.md`, não deste ticket — deixo registrado
aqui e não editei `CLAUDE.md` (fora do escopo de arquivos permitidos desta rodada).

### 2.3 Quais são nossos — `[EXECUTADO]`, `git -C . ls-files | grep custom_nodes`, agora

```
custom_nodes/comfy-quant-preflight/__init__.py
custom_nodes/comfy-quant-preflight/checks.py
custom_nodes/comfy-quant-preflight/test_checks.py
```

Só **`comfy-quant-preflight`** é rastreado pelo repo raiz — é o único pacote "nosso" pela definição
do ticket (rastreado pelos `git ls-files` do repo raiz). `comfy_convrot_native` tem nome que sugere
ligação com o projeto ConvRot, mas **não é rastreado**: `git check-ignore -v` confirma que ele cai
na regra `.gitignore:5:/*` (o allowlist do repo raiz) sem exceção — está no disco, não está no
git. Se ele é "nosso" no sentido de autoria, não é "nosso" no sentido rastreado; marco como
**não-rastreado, procedência de autoria não verificada nesta rodada**.

### 2.4 O que o workflow de aceitação `LTX25-int8-acceptance-v2.json` realmente usa — `[EXECUTADO]`, parse do JSON + do fixture, agora

`tools/fixtures/object_info_ltx25.json` tem exatamente **16 classes**. Carreguei o workflow real
(`ComfyUI/user/default/workflows/LTX25-int8-acceptance-v2.json`, 18 nós no grafo) e extraí o
conjunto de `type` usado — os dois conjuntos batem **exatamente**, os mesmos 16 nomes:

```
CFGGuider, CLIPLoader, CLIPTextEncode, EmptyLTXVLatentVideo, KSamplerSelect,
LTXVConcatAVLatent, LTXVConditioning, LTXVEmptyLatentAudio, LTXVSeparateAVLatent,
ManualSigmas, RandomNoise, SamplerCustomAdvanced, SaveImage, UNETLoader, VAEDecode, VAELoader
```

Cruzei cada uma com o campo `python_module` do próprio fixture: **as 16 vêm de `nodes` ou de
`comfy_extras.*`** — ou seja, do **core do ComfyUI**, nenhuma de um pacote em `custom_nodes/`.

**Achado central desta seção**: o workflow de aceitação `-v2` não usa nenhum dos 63 pacotes de nó
instalados. Zero. Isso não é "poucos nós usados" — é nenhum.

Contexto (não pedido pelo ticket, mas achado ao lado): o workflow irmão sem `-v2`
(`LTX25-int8-acceptance.json`, também 18 nós) usa duas classes que **não** existem no fixture:
`CLIPLoaderMultiGPU` e `UNETLoaderDisTorch2MultiGPU`. `grep` confirma que essas duas classes vêm de
`ComfyUI-MultiGPU` (achadas em `ComfyUI-MultiGPU/ci/example_workflows_api/*.json`); não achei
ocorrência em `ComfyUI-LTX2-MultiGPU`.

### 2.5 Órfãos — `[EXECUTADO]` para o que foi checado, com o limite de cobertura dito explicitamente

Pela definição estrita do ticket (nosso / usado pelo `-v2` / órfão): dos 63 pacotes, **1 é nosso**
(`comfy-quant-preflight`) e **0 são usados pelo `-v2`** (seção 2.4) — logo, pela leitura mais
literal, os outros **62 não são usados por esse workflow específico**.

Isso superestimaria "órfão" se eu parasse aí: `ComfyUI-MultiGPU` claramente não é órfão (usado pelo
workflow irmão, seção 2.4), e pacotes como `ComfyUI-LTXVideo`, `ComfyUI-nunchaku`,
`ComfyUI-GGUF`, `ComfyUI-VideoHelperSuite` aparecem por nome em outros workflows salvos (`Video-
LTX2_MultiGPU.app.json`, `TXT2IMG-ZIMG.nunchaku.json`, etc.) — **não conferi isso por parsing**,
só por familiaridade com os nomes dos arquivos, o que é exatamente o tipo de afirmação "traçada"
que este projeto pede para não fazer. Não vou marcar nenhum pacote como "órfão confirmado" com
base numa suposição.

**O que fiz**: parseei `class_type`/`type` só dos dois workflows `LTX25-int8-acceptance*.json`
(2 de 28 arquivos `.json` na pasta `ComfyUI/user/default/workflows/`).
**O que não fiz**: não parseei os outros 26 arquivos de workflow salvos, então não posso listar
órfãos verdadeiros (pacotes não usados por *nenhum* workflow salvo) sem inventar dado. A tabela
abaixo reporta o que cada pacote é, sem inventar um veredito de "órfão" para os 62 não checados.

| Pacote | `.git` próprio | Rastreado pelo repo raiz | Usado por `LTX25-int8-acceptance-v2.json` |
|---|---|---|---|
| comfy-quant-preflight | não | **sim (nosso)** | não |
| ComfyUI-MultiGPU | sim | não | não no `-v2`; **sim** no `LTX25-int8-acceptance.json` irmão |
| (demais 61 pacotes) | ver 2.2 | não | não confirmado neste ticket para nenhum workflow além dos dois LTX25 |

---

## 3. MODELOS CONVERTIDOS

Ponto de partida pedido pelo ticket: **não rodei `quant_audit.py`** (percorre ~600 GiB). Reusei
`quantization_inventory.json` / `.md` como já existem.

- `quantization_inventory.md` / `.json`: `generated_at` = `2026-08-19T00:41:16.399314-03:00`
  (**dois dias atrás** da data deste documento — `[EXECUTADO]`, lido do próprio JSON agora).
  Cobre `ComfyUI/models` (159 arquivos, 654 561 491 511 bytes / ~654 GiB) — **não cobre
  `D:/ComfyUI-Models/`**, que é montado por `extra_model_paths.yaml` e fica fora do `models_root`
  que o `quant_audit.py` percorreu.

### 3.1 Achados por busca de nome — `[EXECUTADO]`, `find` em `ComfyUI/models` e `D:/ComfyUI-Models`, agora

**Em `ComfyUI/models`** (dentro do que `quantization_inventory` já cobre):

| Arquivo | Sidecar `.quant.json`? |
|---|---|
| `diffusion_models/capybara_v0.1_w4a8.safetensors` | sim |
| `diffusion_models/hv15_w4a8.safetensors` | sim |
| `diffusion_models/zimage-v2-w4a4.safetensors` | sim |
| `text_encoders/gemma_3_12B_it_heretic_w4a8.safetensors` | sim |

**Em `D:/ComfyUI-Models/`** (fora do que `quantization_inventory` cobre):

| Arquivo | Sidecar `.quant.json`? |
|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16_int8.safetensors` | sim |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors` | sim |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors` | sim |
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | **não** |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | **não** |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | **não** |
| `diffusion_models/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | **não** |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | **não** |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | **não** |

Nota: além dos arquivos convertidos, há um outro achado por nome no `find` de `D:` que **não** é
um checkpoint — `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors.metadata` e equivalentes moram em
`D:/ComfyUI-Models/.cache/huggingface/download/...`. São metadados de cache do `huggingface_hub`
(ETag + hash), não manifestos de conversão; não têm campo de ferramenta/parâmetro e não contam
como sidecar.

### 3.2 Com sidecar — ferramenta e parâmetros, direto do `.quant.json` — `[EXECUTADO]`, `cat` de cada sidecar, agora

| Arquivo | `quantization` (campo do sidecar) | `layout`/`convrot` | `group_size` | `comfy_kitchen` | Ferramenta (ver 3.3) |
|---|---|---|---|---|---|
| `capybara_v0.1_w4a8` | `asym_w4a8_int8` | `AsymW4A8Int8Layout`, `convrot_groupsize: 256` | 16 | 0.2.31 | `tools/quant_w4a8.py` |
| `hv15_w4a8` | `asym_w4a8_int8` | `AsymW4A8Int8Layout`, `convrot_groupsize: 256` | 16 | 0.2.31 | `tools/quant_w4a8.py` |
| `gemma_3_12B_it_heretic_w4a8` | `asym_w4a8_int8` | `AsymW4A8Int8Layout`, `convrot_groupsize: 256` | 16 | 0.2.31 | `tools/quant_w4a8.py` |
| `zimage-v2-w4a4` | `mixed convrot_w4a4 / asym_w4a8_int8` | por camada (170 camadas em `convrot_w4a4`, 0 em `asym_w4a8_int8`, 0 em bf16), `convrot_groupsize: 256` | 16 | 0.2.31 | `tools/quant_mixed.py` (calibrado com `calibrate_activations.py`, ver campo `calibration`) |
| `ltx-2.5-22b-distilled-transformer-bf16_int8` | `int8_tensorwise` | `convrot: false`, `convrot_groupsize: 256` | — | 0.2.31 | `tools/quant_int8.py` |
| `ltx-2.5-22b-distilled-transformer-bf16_int8_convrot` | `int8_tensorwise+convrot` | `convrot: true`, `convrot_groupsize: 256` | — | 0.2.31 | `tools/quant_int8.py` |
| `ltx-2.5-22b-distilled-transformer-bf16_w4a8` | `asym_w4a8_int8` | `AsymW4A8Int8Layout`, `convrot_groupsize: 256` | 16 | 0.2.31 | `tools/quant_w4a8.py` |

Parâmetros adicionais de destaque:
- `zimage-v2-w4a4`: calibrado com 2 prompts, seeds `[1234, 5678]`, 8 steps, cfg 1.0, sampler
  `euler`, scheduler `simple`, 128 rows reservoir, 4 runs, 170 camadas — todos os 170 `never_ran`
  vazio (nenhuma camada ficou sem medição). `preserved_tensors: 283`.
- Os três arquivos LTX 2.5 em `D:` compartilham a mesma origem
  (`\\192.168.3.68\estoque\ComfyUI-Models\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16.safetensors`,
  42 018 190 584 bytes) — são três conversões diferentes do **mesmo** arquivo-fonte, com
  `conversion_seconds` de 777.4 s, 768.9 s e 671.6 s respectivamente (uma corrida cada, não
  repetida — não posso dar min-max porque só há uma amostra por arquivo no sidecar).

### 3.3 De onde veio a correspondência ferramenta ↔ arquivo — `[LIDO]`, comparação de assinatura de campo, não confirmada por log de execução

Não existe, nesta bancada, um log de terminal amarrando "rodei o comando X às HH:MM e ele escreveu
o arquivo Y". A correspondência da tabela 3.2 vem de comparar os campos que cada `.quant.json`
carrega com o código-fonte de cada ferramenta em `tools/`:

- `tools/quant_w4a8.py` grava literalmente `"quantization": "asym_w4a8_int8"`,
  `"layout": "AsymW4A8Int8Layout"`, `"symmetric": True`, `"codebook": not args.no_codebook` —
  bate campo a campo com os quatro arquivos marcados `quant_w4a8.py` acima.
- `tools/quant_int8.py` grava `f"int8_tensorwise{'+convrot' if args.convrot else ''}"` e o campo
  `"quantized_on": args.device` — bate com os dois arquivos `bf16_int8*`.
- `tools/quant_mixed.py` grava `"mixed convrot_w4a4 / asym_w4a8_int8"` mais os campos
  `"selection"`, `"layer_counts"`, `"calibration"` — só `zimage-v2-w4a4.quant.json` tem essa
  assinatura completa.
- `tools/quant_w4a4.py` (puro ConvRot W4A4) grava `"quantization": "ConvRot W4A4"` e
  `"layout": "TensorCoreConvRotW4A4Layout"` — **nenhum sidecar encontrado nesta bancada tem essa
  assinatura**. Não há conversão puramente `quant_w4a4.py` sobrevivendo no disco hoje (ver 3.5).
- `tools/quant_w4a4_smooth.py` grava `"ConvRot W4A4 + SmoothQuant"` — também não encontrado.

Reforço parcial (não prova, mas corrobora) em `W4A4_PROGRESS.md:455`: "Converted the same source
with `tools/quant_w4a8.py` to comfy-kitchen's `asym_w4a8_int8` format" — registro contemporâneo à
conversão, escrito por quem rodou o comando, não por esta rodada.

### 3.4 Sem sidecar — `[EXECUTADO]` para a busca por nome; `[LIDO]` para a origem via `fetch_*.py`

Os seis arquivos de 3.1 sem `.quant.json` **não foram produzidos pelos conversores locais** —
achei `.metadata` de cache do `huggingface_hub` com o mesmo nome de arquivo em
`D:/ComfyUI-Models/.cache/huggingface/download/...`, e os nomes batem exatamente com listas
hard-coded em dois scripts de download:

- `tools/fetch_ltx25.py`, `REPO = "Lightricks/LTX-2.5"`, baixa por `hf_hub_download`:
  - `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
  - `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
  - `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
  - `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors`
- `tools/fetch_minimax_h3.py`, baixa por `hf_hub_download`:
  - de `starsfriday/MiniMax-H3-w4a8`: `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`
  - de `Comfy-Org/MiniMax-H3`: `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
    (mais dois VAEs do mesmo repo, não quantizados, fora do escopo desta seção)

Ou seja: **procedência conhecida, mas é procedência de download, não de conversão local** — esses
seis arquivos chegaram já quantizados de fora (Lightricks e terceiros na HF), nenhuma ferramenta
deste repo os produziu. Isso não é "procedência desconhecida" no sentido do ticket (o ticket pede
essa frase só quando a origem realmente não dá para determinar); é procedência determinada e
diferente de conversão local. Não abri esses seis arquivos (nem `inspect_quant.py`, que só lê
header, foi rodado neles nesta seção — ver 3.6 sobre o que faltou).

### 3.5 O que o log menciona e não achei no disco — `[EXECUTADO]` busca, `[LIDO]` a menção original

`W4A4_PROGRESS.md` (linha ~448) cita quatro arquivos como "estruturalmente válidos... mantidos
como fixtures": `hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot`, `hv15_w4a4_g64`, `hv15_w4a4_g16`,
`capybara_v0.1_w4a4_convrot`. Busquei os quatro por nome em `ComfyUI/models` e em
`D:/ComfyUI-Models` agora — **nenhum apareceu**. Não afirmo que foram apagados (apagar é proibido
nesta rodada e eu não apaguei nada; posso simplesmente não ter procurado no lugar certo, ou eles
nunca existiram fora do host onde o log foi escrito). Registro só a ausência no que procurei.

### 3.6 O que esta seção **não** cobriu

- Não rodei `quant_audit.py` de novo (proibido pelo ticket) — a lista de 159 arquivos em
  `ComfyUI/models` tem dois dias, não é a foto de agora.
- Não há um `quant_audit.py` equivalente para `D:/ComfyUI-Models/` — os nove arquivos da tabela
  3.1 vieram só de `find` por padrão de nome, não de uma varredura completa de metadados. Pode
  haver outros arquivos quantizados em `D:` com nome que não bate nenhum dos padrões buscados
  (`w4a4`, `w4a8`, `nvfp4`, `int8`, `convrot`).
- Não rodei `inspect_quant.py` em nenhum dos 13 arquivos encontrados — a tabela 3.2 vem só dos
  sidecars, não do header do próprio `.safetensors`. Isso significa que não confirmei que o
  conteúdo do arquivo bate com o que o sidecar promete; só que o sidecar existe e diz o que diz.
- Correspondência ferramenta↔arquivo em 3.2/3.3 é inferência por assinatura de campo, não prova
  por log de execução — sinalizado explicitamente linha a linha, não só aqui.

---

## Ticket a graduar (não fiz a mudança, só registrando)

`tools/quant_audit.py` hoje lê metadados de `ComfyUI/models` mas não tem campo para carregar
procedência (nem sidecar `.quant.json`, nem raiz `D:/ComfyUI-Models/`). Estender o schema de
`quantization_inventory.json` com um campo tipo `provenance: {tool, params, sidecar_path}` por
modelo, lido do `.quant.json` quando existir, evitaria que este documento precise ser mantido à
mão toda vez que um novo modelo for convertido. Não mudei a ferramenta nesta rodada — fora de
escopo do ticket 12.

## Recontagem do repo raiz — `[EXECUTADO]` agora

`git -C F:/COMFY_PORTABLE ls-files | wc -l` = **93** hoje. `CLAUDE.md` diz 89 ("as of 2026-08-21" —
mesma data deste documento) e o próprio texto do ticket 12 diz "88 rastreados". Nenhum dos dois
bate com a contagem de agora — sinal de que o repo mudou entre a escrita de cada um desses números
e esta rodada (há uma mudança não commitada em `tools/comfy_run_workflow.py` pendente no
`git status`, e possivelmente outras sessões commitando em paralelo, conforme o próprio ticket
avisa: "há outros dois agentes trabalhando AGORA"). Não investiguei a causa da diferença — fora de
escopo deste ticket; só recontei em vez de copiar o número de um documento, como a disciplina do
projeto pede.
