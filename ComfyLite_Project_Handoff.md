# ComfyLite — Project Handoff

## 1. Visão

**ComfyLite** é uma interface desktop leve e bonita para descoberta, instalação, resolução e execução de workflows de IA usando o ecossistema ComfyUI como backend/runtime, sem obrigar o usuário a trabalhar diretamente com o editor de nodes.

O objetivo não é substituir o ecossistema do ComfyUI. É criar uma camada superior que:

- entende o hardware disponível;
- cataloga modelos, LoRAs, VAEs, encoders e workflows;
- descobre workflows e modelos em múltiplas fontes;
- resolve automaticamente dependências;
- escolhe quantizações e variantes compatíveis;
- otimiza VRAM, attention e multi-GPU;
- executa o workflow;
- mostra imagem, vídeo e áudio diretamente na aplicação;
- expõe tudo a agentes externos via MCP.

Nome do projeto: **ComfyLite**.

---

## 2. Princípio central

A aplicação deve tratar o ComfyUI como **runtime/headless engine**, não necessariamente como interface.

Pipeline conceitual:

```text
ComfyLite GUI
    ↓
Capability / Workflow Planner
    ↓
Workflow Resolver
    ↓
Resource + Device Planner
    ↓
Comfy Runtime / Direct Runtime
    ↓
GPU(s)
```

Para workflows conhecidos e otimizáveis:

```text
load → use → return
```

A aplicação deve controlar a residência dos componentes na GPU conforme a necessidade real do grafo:

- text encoder;
- DiT / UNet / transformer;
- VAE;
- CLIP Vision;
- ControlNet;
- adapters;
- LoRAs;
- outros módulos.

Políticas sugeridas:

- `FAST`: mantém componentes quentes enquanto houver VRAM.
- `BALANCED`: decide conforme próximo uso/custo de reload.
- `MAKE IT FIT`: offload agressivo após o último uso.

---

## 3. Hardware alvo inicial

Hardware prioritário:

```text
RTX 3090    24 GB
RTX 3080 Ti 12 GB
Compute Capability: SM86
```

A aplicação deve suportar:

- single GPU;
- multi-GPU;
- placement por componente;
- layer splitting;
- CPU offload;
- auto-split;
- quantizações;
- benchmark por configuração.

Não tratar 3090 + 3080 Ti simplesmente como uma única GPU de 36 GB.

O planner deve preferir:

1. component/stage placement;
2. depois layer splitting;
3. depois CPU offload.

Exemplo:

```text
GPU0 / 3090
  transformer principal
  latents
  sampler

GPU1 / 3080 Ti
  text encoder
  VAE
  CLIP Vision

CPU
  cold weights
```

Quando necessário, usar split de layers entre GPUs.

---

## 4. Inventory Scanner

Ao abrir o ComfyLite, escanear automaticamente:

### Hardware

- GPUs;
- VRAM;
- compute capability;
- driver;
- CUDA;
- PyTorch;
- largura/link PCIe quando possível;
- RAM;
- storage.

### Modelos locais

Detectar e catalogar:

- checkpoints;
- diffusion models;
- transformers;
- GGUF;
- safetensors;
- VAEs;
- text encoders;
- CLIP;
- ControlNet;
- LoRAs;
- adapters;
- upscalers;
- audio models;
- video models.

Para cada artefato:

```text
id
path
hash
architecture
role
format
quantization
weight_precision
activation_precision
size
base_model
compatible_models
compatible_workflows
metadata
source
tested_configs
benchmark_profiles
```

Banco sugerido: SQLite local.

---

## 5. Quantization-aware Model Downloader

O downloader não deve tratar um modelo apenas pelo nome.

Ele deve entender variantes.

Exemplo:

```text
Model: ExampleModel

FP16
BF16
FP8
W8A8
W8A16
W4A8
W4A16
NVFP4
GGUF Q8
GGUF Q6
GGUF Q5
GGUF Q4
...
```

Separar explicitamente:

```text
weight quantization
activation quantization
KV/cache precision quando aplicável
```

Exemplo:

```text
Weights: W4
Activations: A8
→ W4A8
```

A aplicação deve permitir:

```text
[ Auto ]
[ Best Quality ]
[ Fastest ]
[ Lowest VRAM ]
[ Manual ]
```

O `Auto` deve considerar:

- GPU;
- arquitetura;
- kernel disponível;
- VRAM;
- RAM;
- runtime;
- modelo;
- workflow;
- quantização instalada;
- benchmark local.

Se o usuário já possuir a variante necessária, não baixar novamente.

---

## 6. Sources / Adapters

Não existe apenas um "parser".

Criar uma interface de **Source Adapter**.

```text
SourceAdapter
 ├─ search()
 ├─ fetch_metadata()
 ├─ list_files()
 ├─ resolve_download()
 ├─ detect_workflows()
 └─ normalize_metadata()
```

Adapters iniciais:

### Hugging Face

Funções:

- pesquisar modelos;
- pesquisar repositórios;
- enumerar arquivos;
- identificar variantes/quantizações;
- localizar workflow JSON;
- localizar README;
- extrair metadata;
- encontrar LoRAs;
- encontrar dependências;
- baixar arquivos selecionados.

### GitHub

Funções:

- pesquisar repositories;
- procurar workflows JSON;
- procurar exemplos;
- procurar custom nodes;
- procurar código relacionado ao modelo;
- detectar requirements;
- localizar releases;
- localizar assets.

### CivitAI

Funções:

- pesquisar modelos;
- pesquisar LoRAs;
- pesquisar workflows;
- baixar assets;
- ler metadata;
- identificar base model;
- identificar recursos utilizados pelo workflow;
- usar token opcional do usuário.

### CivitAI / fontes NSFW

Quando possível, permitir modo de pesquisa:

```text
metadata/workflow search
NSFW media hidden
```

O objetivo é aproveitar workflows e metadata úteis sem obrigar o agente/modelo de busca a processar thumbnails ou conteúdo explícito desnecessário.

Se uma fonte/proxy/API oferecer filtragem de conteúdo, usar isso como preferência.

### Outras fontes

Arquitetura deve permitir novos adapters sem alterar o resolver central.

---

## 7. Workflow Searcher

O usuário deve poder pesquisar por intenção:

```text
"image edit"
"LTX image to video 12GB"
"Wan video 24GB"
"Qwen image edit low VRAM"
"workflow LTX multi GPU"
"workflow W4A8"
```

O Workflow Searcher consulta:

```text
local catalog
Comfy templates
Hugging Face
GitHub
CivitAI
outras fontes configuradas
```

Resultado normalizado:

```text
WorkflowCandidate

name
source
source_url
workflow_file
capabilities
required_models
required_nodes
quantizations
estimated_vram
supported_hardware
media_examples
metadata
trust_score
compatibility_score
```

---

## 8. Search / Retrieval Agent opcional

Além dos adapters determinísticos, pode existir um agente LLM barato para interpretar páginas e resultados difíceis.

Nome sugerido:

**Retrieval Agent** ou **Workflow Research Agent**.

Não chamar isso de parser.

Providers configuráveis:

```text
OpenRouter
OpenRouter Free
Xiaomi MiMo
OpenAI-compatible endpoint
local model
outros
```

O usuário fornece:

```text
base URL
API token
model
```

O agente pode:

- interpretar README ruim;
- entender qual arquivo precisa ser baixado;
- descobrir qual workflow corresponde ao modelo;
- relacionar aliases;
- identificar versão correta;
- interpretar instruções de instalação;
- encontrar dependências ocultas;
- comparar forks;
- sugerir alternativas.

O agente NÃO substitui os adapters.

Arquitetura:

```text
Source Adapter
      +
Retrieval Agent
      ↓
Normalized Result
```

---

## 9. Workflow Resolver

O resolver recebe um workflow e responde:

```text
READY
ou
MISSING DEPENDENCIES
```

Ele deve detectar:

- nodes ausentes;
- custom nodes;
- modelos ausentes;
- VAEs;
- encoders;
- LoRAs;
- ControlNets;
- adapters;
- Python packages;
- versões incompatíveis;
- quantizações incompatíveis;
- caminhos incorretos;
- nodes renomeados;
- modelos equivalentes já instalados.

Exemplo:

```text
Workflow: LTX Image → Video

Missing:
  LTX transformer
  text encoder
  VAE
  custom node X

Available locally:
  compatible transformer W4A8
  compatible VAE

Suggested:
  use local W4A8
  download encoder
  install node X

[ Resolve ]
```

---

## 10. Workflow Importer

Aceitar:

- JSON do ComfyUI;
- workflow embedded em PNG quando suportado;
- workflow encontrado em HF;
- workflow encontrado em GitHub;
- workflow encontrado no CivitAI;
- drag-and-drop;
- URL.

Sempre preservar:

```text
original workflow
```

Criar separadamente:

```text
ComfyLite optimized derivative
```

Nunca destruir ou sobrescrever silenciosamente o original.

---

## 11. Workflow Optimization

Depois de resolver o workflow:

```text
original graph
    ↓
normalize
    ↓
capability analysis
    ↓
resource planning
    ↓
optimized workflow
```

Possíveis otimizações:

- trocar loader por variante compatível;
- selecionar quantização;
- ativar multi-GPU;
- ajustar offload;
- escolher attention backend;
- retirar nodes redundantes;
- ajustar VAE placement;
- controlar text encoder residency;
- cache quando vantajoso;
- liberar memória após last-use.

---

## 12. Attention Autotuner

Não "empilhar" Flash + Sage + Triton na mesma operação.

Criar um dispatcher/autotuner que escolha o melhor backend compatível.

Candidates:

```text
PyTorch SDPA
FlashAttention
SageAttention
Triton-based implementations
xFormers
model-specific kernels
experimental kernels instalados
```

Para SM86, levar em conta implementações adequadas a Ampere.

O app executa microbenchmarks e validação.

Database:

```text
model
block/type
shape
sequence length
head dimension
dtype
GPU
kernel
latency
VRAM
status
```

Exemplo:

```text
LTX2 / SM86 / shape X

Sage/Triton   PASS   1.00x
Flash         PASS   1.07x
SDPA          PASS   1.18x
```

Depois usar automaticamente o vencedor.

---

## 13. Multi-GPU Planner

Criar `DevicePlanner`.

Entradas:

```text
workflow
model sizes
activation estimates
GPU VRAM
PCIe topology
quantization
resolution
frames
batch
runtime
```

Saída:

```text
placement plan
```

Estratégias:

```text
component placement
stage placement
layer split
CPU offload
hybrid
```

A integração pode aproveitar ferramentas já instaladas, inclusive soluções de multi-GPU/layer distribution, em vez de reinventar tudo.

---

## 14. LoRA Catalog

Escanear todas as LoRAs locais.

Guardar:

```text
name
hash
source
base architecture
target modules
recommended strength
compatible models
description
tags
preview images
tested configs
```

Quando possível, buscar metadata original em:

- Hugging Face;
- CivitAI;
- GitHub;
- sidecar local.

### LoRA Analyzer

Função opcional:

```text
Analyze LoRA
```

Gerar matriz:

```text
base
0.3
0.5
0.7
1.0
```

Guardar previews e resultado para consulta futura.

---

## 15. MCP Server

ComfyLite deve expor um MCP com acesso amplo ao aplicativo.

Objetivo: permitir que Codex/Claude/outro agente consiga investigar e corrigir problemas sem o usuário precisar navegar manualmente por arquivos e nodes.

Ferramentas MCP sugeridas:

```text
inventory.list_models
inventory.inspect_model
inventory.list_loras
inventory.list_workflows

workflow.load
workflow.inspect
workflow.validate
workflow.resolve
workflow.optimize
workflow.run
workflow.stop

search.models
search.workflows
search.github
search.huggingface
search.civitai

download.model
download.workflow
download.dependency

runtime.status
runtime.logs
runtime.vram
runtime.devices
runtime.loaded_models

benchmark.run
benchmark.compare

app.get_config
app.update_config
```

O agente deve conseguir:

1. receber erro;
2. ler workflow;
3. inspecionar nodes;
4. ler logs;
5. descobrir dependency;
6. buscar solução;
7. baixar/configurar;
8. testar;
9. devolver resultado.

Exemplo:

```text
Workflow falhou
     ↓
Codex via MCP
     ↓
inspect workflow
     ↓
inspect logs
     ↓
missing LoaderXYZ
     ↓
search node
     ↓
install dependency
     ↓
test workflow
     ↓
PASS
```

Alterações destrutivas ou instalação de código externo devem continuar sujeitas à política de autorização definida pelo usuário.

---

## 16. GUI

Evitar Tkinter.

Stack sugerida:

```text
Tauri
Svelte
Python worker/runtime
```

Objetivo:

- leve;
- bonita;
- rápida;
- desktop;
- sem browser Comfy obrigatório;
- sem Electron se não necessário.

Tela principal:

```text
Generate

[ Image ]
[ Image Edit ]
[ Inpaint ]
[ Video ]
[ Image → Video ]
[ Video → Video ]
[ Audio ]
[ Upscale ]
```

Após escolher tarefa:

```text
Recommended workflows
Installed
Available for download
Compatible
Estimated VRAM
Estimated speed
```

---

## 17. Media Workspace

A aplicação deve mostrar o resultado, não apenas imprimir path.

Imagem:

- preview;
- zoom;
- before/after;
- save;
- metadata.

Vídeo:

- player;
- timeline;
- frame extraction;
- metadata.

Áudio:

- player;
- waveform;
- metadata.

Image editing/inpainting:

- brush;
- erase;
- mask overlay;
- invert mask;
- blur/feather;
- crop;
- zoom.

---

## 18. Compatibility Runtime vs Optimized Runtime

Dois modos:

### Compatibility Mode

Executa o workflow praticamente como ComfyUI stock/headless.

Objetivo:

```text
máxima compatibilidade
```

### Optimized Mode

Usa planner do ComfyLite:

```text
VRAM lifecycle
quantization
attention autotuner
multi-GPU
placement
benchmark profiles
```

Fluxo:

```text
workflow
   ↓
todos os componentes conhecidos?
   ├─ yes → Optimized Runtime
   └─ no  → Compatibility Runtime
```

---

## 19. Test-before-trust

Nenhuma otimização deve ser considerada válida apenas porque "deveria funcionar".

Criar `ProbeRunner`.

Teste mínimo:

```text
load
sanity inference
check NaN
check Inf
check dimensions
record latency
record peak VRAM
unload
verify release
```

Para attention:

```text
baseline
candidate
numeric comparison
benchmark
PASS / DEGRADED / FAIL
```

Para multi-GPU:

```text
candidate split
run
OOM?
latency
VRAM
quality check
store result
```

---

## 20. Security / Trust

Modelos e dados são diferentes de código executável.

Política recomendada:

### Auto-download permitido

- safetensors;
- GGUF;
- JSON workflow;
- metadata;
- previews.

### Requer autorização

- Python custom node;
- pip package;
- executable;
- script;
- arbitrary repository code.

Adicionar trust information:

```text
source
repo
author
downloads
hash
signature when available
last update
permissions
code-install required?
```

---

## 21. Configuração de credenciais

Tela `Sources & Providers`.

```text
Hugging Face
  token optional

GitHub
  token optional

CivitAI
  API token optional

OpenRouter
  API key
  model

Xiaomi MiMo
  endpoint
  API key
  model

Custom OpenAI-compatible
  base URL
  API key
  model
```

Secrets nunca devem ser gravados em texto puro em config exportável.

Usar credential store do sistema operacional quando possível.

---

## 22. MVP / Roadmap

### Phase 0 — Foundation

- repo;
- config;
- logging;
- SQLite;
- hardware scanner;
- Comfy installation detection.

### Phase 1 — Inventory

- models;
- LoRAs;
- workflows;
- hashes;
- architecture detection;
- quantization detection.

### Phase 2 — GUI

- Tauri/Svelte shell;
- inventory browser;
- generation page;
- settings.

### Phase 3 — Workflow Runtime

- import workflow;
- execute via Comfy backend;
- progress/events;
- image/video/audio viewer.

### Phase 4 — Workflow Resolver

- missing nodes;
- missing models;
- dependency graph;
- local alternatives.

### Phase 5 — Source Adapters

- Hugging Face;
- GitHub;
- CivitAI;
- workflow search.

### Phase 6 — Downloader

- quantization aware;
- resumable download;
- checksum;
- destination routing;
- dependency install.

### Phase 7 — Retrieval Agent

- OpenRouter;
- MiMo;
- custom endpoint;
- README/repository reasoning.

### Phase 8 — MCP

- inventory;
- workflow;
- logs;
- search;
- resolver;
- runtime;
- downloader.

### Phase 9 — VRAM Scheduler

- liveness analysis;
- load/use/return;
- FAST;
- BALANCED;
- MAKE IT FIT.

### Phase 10 — Multi-GPU

- device planner;
- component placement;
- layer splitting;
- CPU offload;
- auto-profile.

### Phase 11 — Attention Autotuner

- candidate discovery;
- benchmark;
- validation;
- persistent profiles.

### Phase 12 — LoRA Intelligence

- metadata;
- source lookup;
- compatibility;
- analyzer;
- previews.

### Phase 13 — Optimized Runtime

- known workflow compiler;
- direct execution where useful;
- optimized graph generation.

---

## 23. Non-goals iniciais

Não tentar no MVP:

- reimplementar todo o ComfyUI;
- converter qualquer custom node arbitrário para Python otimizado;
- instalar silenciosamente código encontrado na internet;
- suportar todo modelo existente;
- criar um novo inference framework do zero;
- resolver performance apenas com hardcoded model-name checks.

Evitar arquitetura:

```python
if model.startswith("wan"):
    ...
elif model.startswith("ltx"):
    ...
elif model.startswith("flux"):
    ...
```

Preferir capacidades, metadata e adapters.

---

## 24. Resultado final desejado

Experiência pretendida:

```text
User:
"Quero image-to-video com esse modelo."

ComfyLite:
- detecta hardware;
- encontra modelos locais;
- encontra workflows;
- encontra workflows externos quando necessário;
- compara variantes;
- escolhe quantização;
- resolve nodes;
- baixa dependências;
- planeja 3090 + 3080 Ti;
- escolhe attention;
- testa;
- executa;
- mostra vídeo.
```

Quando alguma coisa quebrar:

```text
User:
"Resolve isso."

Agent via MCP:
- acessa estado completo do ComfyLite;
- lê workflow;
- lê logs;
- inspeciona inventory;
- pesquisa HF/GitHub/CivitAI;
- corrige resolução;
- testa;
- retorna o workflow funcional.
```

Esse é o núcleo do **ComfyLite**.
