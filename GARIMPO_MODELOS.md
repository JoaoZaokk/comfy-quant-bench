# Garimpo de modelos — fontes, endpoints e regras de parse

Playbook de caça a modelos e quantizações. A primeira parte é o material original; a segunda
(**Verificado nesta máquina**) corrige e completa o que foi testado de fato em 2026-08-18, com os
comandos que funcionaram.

---

## FONTES GERAIS — TODAS CATEGORIAS

### HUGGINGFACE (primária)

| uso | endpoint | formato |
|---|---|---|
| lista+busca | `https://huggingface.co/api/models?search={q}&limit=100&sort=downloads&direction=-1` | JSON |
| por autor | `https://huggingface.co/api/models?author={org}&sort=downloads&direction=-1` | JSON |
| por tag | `https://huggingface.co/api/models?other={tag}` — ex: `other=SVDQuant`, `other=gguf` | JSON |
| por tarefa | `https://huggingface.co/api/models?filter={task}` — ex: `filter=text-to-speech` | JSON |
| trending | `https://huggingface.co/models?sort=trending` | scrape |
| papers | `https://huggingface.co/api/daily_papers` | JSON |
| coleções | `https://huggingface.co/collections?search={q}` | scrape |

Parse: `jq '.[] | {id, downloads, likes, lastModified, tags}'`; grepar tags por `base_model:` e
`quantized`.
Regra: downloads < 100 = quant morto ou mirror lixo; sempre conferir `lastModified`.

### MODELSCOPE (CN — primária p/ Wan, Z-Image, Qwen, FunASR)

| uso | endpoint |
|---|---|
| REST | `https://modelscope.cn/api/v1/models?Query={q}&PageSize=50` (se 400 → SDK) |
| SDK | `from modelscope.hub.api import HubApi; HubApi().list_models(...)` |
| intl | `https://www.modelscope.ai/models` (mesma API) |

Espelho oficial `nunchaku-tech` fica aqui quando o HF atrasa.

### CIVITAI (checkpoints / LoRA de difusão)

`https://civitai.com/api/v1/models?query={q}&limit=100&sort=Most+Downloaded&types=Checkpoint,LORA`

Paginação: `metadata.nextCursor` → `?cursor=`.
Parse: `modelVersions[].files[]`, filtra `.safetensors`; identifica fp8/gguf/svdq pelo nome do
arquivo.

### GITHUB

| uso | endpoint |
|---|---|
| repos | `https://api.github.com/search/repositories?q={q}&sort=stars&order=desc` (10/min sem token) |
| topics | `https://github.com/topics/{t}?o=desc&s=stars` — ex: comfyui, quantization, tts |
| code | `gh search code "{q}" --owner X` (CLI, precisa login) |
| forks | `https://api.github.com/repos/{o}/{r}/forks?sort=updated` ← acha fork vivo (lição yannickcruz) |

### GITEE / CN GIT

- `https://gitee.com/api/v5/search/repositories?q={q}` (JSON, token ajuda)
- `https://search.gitee.com/?q={q}` (scrape)
- `ai.gitee.com` (模力方舟) — mirrors de modelos CN
- GITCODE (CSDN): `https://gitcode.com/search?q={q}` (scrape)
- CNB.COOL: `https://cnb.cool` — mirrors `ai-models/*`

### ARXIV / PAPERS

- `http://export.arxiv.org/api/query?search_query=all:{q}&max_results=20` (XML/Atom)
- `https://api.semanticscholar.org/graph/v1/paper/search?query={q}` (JSON)
- `https://api.openreview.net/notes/search?query={q}` (JSON)
- `alphaxiv.org` — discussão de paper

---

## LLM

Autores no HF para varrer (GGUF/GPTQ/AWQ): `bartowski`, `mradermacher`, `unsloth`,
`hugging-quants`, `TheBloke` (arquivo), `MaziyarPanahi`, `bullerwins`, `SanjiWatsuki`,
`QuantStack`, `LoneStriker`.

Busca: `api/models?search={nome}&sort=downloads` + grep nas tags por `gguf|gptq|awq|fp8|mxfp4`.

- OLLAMA: `https://ollama.com/search?q={q}` (scrape; sem API pública de busca)
- vLLM/SGLang: `docs.vllm.ai` — conferir se o quant roda **antes** de baixar

## VLM

Tarefa: `api/models?filter=image-text-to-text&sort=downloads`
Orgs: OpenBMB (MiniCPM-V), Qwen (Qwen3-VL), OpenGVLab (InternVL), liuhaotian (LLaVA),
vikhyatk (moondream), google (paligemma).
Quants GGUF: mesmos autores de LLM.

## STT

Tarefa: `api/models?filter=automatic-speech-recognition&sort=downloads`

- whisper.cpp: `hf.co/ggerganov/whisper.cpp` (ggml bins, releases + HF)
- faster-whisper: `hf.co/Systran` (distil-large-v3, whisper-large-v3-turbo)
- CN (ModelScope): `iic/SenseVoiceSmall`, `iic/paraformer-zh` (FunASR), org `funasr`
- outros: `UsefulSensors/moonshine`, `nvidia/canary-1b`, deepgram (nova), `openai/whisper-large-v3`

## TTS / VOZ

Tarefa: `api/models?filter=text-to-speech&sort=downloads`

| projeto | onde |
|---|---|
| fish-speech | `hf.co/fishaudio` (S1/S2) |
| cosyvoice | `hf.co/FunAudioLLM` + ModelScope `iic/CosyVoice-300M-SFT` |
| f5-tts | `hf.co/SWivid/F5-TTS` |
| kokoro | `hf.co/hexgrad/Kokoro-82M` |
| chatterbox | `hf.co/ResembleAI/chatterbox` (+ -turbo, multilingual) |
| indextts | `hf.co/IndexTeam/Index-TTS-2` |
| gpt-sovits | github `RVC-Boss/GPT-SoVITS` (pesos no release/HF) |
| piper/vits | `hf.co/rhasspy/piper-voices` (ONNX, CPU-friendly) |
| melo/xtts | `hf.co/myshell-ai/MeloTTS`, `coqui/XTTS-v2` |
| RVC | github `RVC-Project` + busca HF "rvc" (`.pth`/`.index`) |

## MÚSICA / ÁUDIO GEN

- ace-step: github `ace-step/ACE-Step-1.5` + `hf.co/ACE-Step`
- stable audio: `hf.co/stabilityai/stable-audio-3-medium` (+ `-optimized`)
- musicgen: `hf.co/facebook/musicgen-large` (audiocraft)
- busca HF "music generation" com `sort=trending`

## DIFUSÃO IMAGEM

- Oficiais: `black-forest-labs`, `Qwen`, `Tongyi-MAI` (Z-Image), `HiDream-ai`, `stabilityai`,
  `RunDiffusion`
- Quants nunchaku/svdq: `nunchaku-ai`, `mit-han-lab`, `nunchaku-tech` (MS), `tonera`, `QuantFunc`,
  `ryg81`, `lite-infer`
- Quants SDNQ: `Disty0` (`/collections/Disty0/sdnq`)
- GGUF de difusão: `city96`, `QuantStack`
- fp8/fp16 wrappers: `Kijai`, `drbaph`, `Comfy-Org`
- lightx2v: quants Wan/Z-Image low-vram (HF + MS)
- Índices: `hf.co/models?other=SVDQuant`, `collections/mit-han-lab/svdquant`

## VÍDEO

- Oficiais: `Wan-AI`, `tencent` (HunyuanVideo/1.5), `Lightricks` (LTX), `SkyworkAI` (SkyReels),
  `THUDM` (CogVideoX), `MiniMax` (MiniMax-H3), `Glanty` (Capybara)
- Quants GGUF: `QuantStack` (Wan), `city96`
- Quants SDNQ: `Disty0` (Wan / LTX-2)
- Índices: `collections/ryg81/svdq-int4`

## 3D

`tencent/Hunyuan3D-2(.1)`, `microsoft/TRELLIS(.2)`, `StabilityAI/TripoSR`, `VAST-AI`.
Quants: quase nada — varrer busca `hunyuan3d fp8|gguf`.

## MULTIMODAL MISTO

Tarefa: `filter=any-to-any`, `image-text-to-text`, `text-to-audio`
`microsoft/Phi-4-multimodal`, `Qwen/Qwen2.5-Omni`, `google/gemma-3n`, `nvidia/personaplex`

## COMUNIDADE CN

`site:blog.csdn.net {q}` · `site:zhuanlan.zhihu.com {q}` ·
`https://search.bilibili.com/all?keyword={q}` (descrição de vídeo tem link de modelo) ·
`site:juejin.cn {q}` · `https://www.baidu.com/s?wd={q}` (acha o que o Google não indexa)

## COMUNIDADE EN

Reddit: r/LocalLLaMA, r/StableDiffusion, r/comfyui — `{inst}.json?q=` (rate-limit; fallback redlib)
Patreon: `spooknik`, `TheLocalLab` (quants svdq exclusivos, pago)
Discord / grupos de FB: não parseável — usar só como dica de existência, depois caçar o repo

## DICIONÁRIO DE KEYWORDS (EN + CN)

| eixo | termos |
|---|---|
| quant | quantized gguf gptq awq fp8 fp4 int4 int8 svdq sdnq nunchaku mxfp4 nvfp4 量化 转换 |
| perf | accelerate speedup 加速 显存 优化 offload 部署 实测 |
| pacote | 整合包 便携 一键 (launcher/整合包 CN = fonte de wheel pré-compilada) |
| erro | fork fix 修复 兼容 (acha fork vivo) |

## REGRAS DE PARSE

1. JSON primeiro (HF / Civitai / GH / arXiv / SS) → `jq`; scrape só quando não há API
   (ollama, bilibili, gitee-search).
2. Sempre `sort=downloads`. Estrela mente, download não.
3. Dedupe de mirrors: mesma tag `base_model` + mesmo nome de arquivo = 1 modelo; fica o de mais
   downloads.
4. Conferir `lastModified` contra hoje — repo parado 12 meses = risco de quebrar no teu torch
   (lição do triton).
5. **Nome de arquivo vale mais que descrição**: extrair o formato do nome
   (`svdq-int4_r32-*`, `-fp8.safetensors`, `-Q4_K_M.gguf`, `SDNQ-uint4-svd-r32`).
6. Gate de GPU local: `sm_86` → descartar `*_fp4*` e `nvfp4`; aceitar int4 / gguf / fp8 / sdnq.
7. Rate limit: GH sem token 10/min → usar token; HF tranquilo; Civitai por cursor; ModelScope SDK.
8. Guardar `{id, downloads, lastModified, tags, arquivos[]}` num índice próprio — re-runs diários
   acham quant novo em 24h.

---

# Verificado nesta máquina (2026-08-18)

O que foi testado de fato, com o comando que funcionou. Onde diverge do material acima, o que vale
é isto — foi executado, não citado.

## ModelScope: o endpoint REST da lista está errado

`GET /api/v1/models?Query={q}` devolve **`404 page not found`**. E `POST /api/v1/dolphin/models`
devolve corpo que não parseia. O que funciona é **PUT**:

```bash
curl -s -X PUT "https://modelscope.cn/api/v1/dolphin/models" \
  -H "Content-Type: application/json" \
  -d '{"PageSize":40,"PageNumber":1,"SortBy":"Default","Name":"nunchaku"}'
```

Resposta: `.Data.Model.Models[]` com `Path` (org) e `Name` (repo), mais `.Data.Model.TotalCount`.

Listar arquivos de um repo:

```bash
curl -s "https://modelscope.cn/api/v1/models/{org}/{repo}/repo/files?Revision=master"
# -> .Data.Files[] com .Name e .Size
```

Baixar um arquivo:

```bash
curl -L "https://modelscope.cn/api/v1/models/{org}/{repo}/repo?Revision=master&FilePath={arquivo}"
```

**Throughput medido: ~1,8 MB/s**, contra 15–20 MB/s do Hugging Face no mesmo momento. Regra
prática: achar no ModelScope, **baixar no HF** — quase tudo que está lá tem espelho, e sem gate.

## Hugging Face: tamanho só vem com `blobs=true`

Sem o parâmetro, `siblings[]` traz só nomes. Com ele, vem `size`:

```bash
curl -sL "https://huggingface.co/api/models/{org}/{repo}?blobs=true"
# -> .siblings[] com .rfilename e .size ; .gated e .private no topo
```

Checar `gated` e `private` **antes** de assumir que precisa de token. Os repos de quant de
terceiros (`tonera`, `QuantFunc`) estavam todos `gated: False, private: False`.

Mas cuidado para não juntar duas perguntas diferentes:

| pergunta | responde |
|---|---|
| preciso de token para **ter acesso**? | `gated` / `private` |
| por que baixa **rápido**? | token configurado + backend Xet |

Esta máquina tem token de HF em `~/.cache/huggingface/token` e `HF_TOKEN` no ambiente, mais o
diretório `~/.cache/huggingface/xet/` — o armazenamento novo do HF, com dedup e CDN. É daí que
vêm os 15–20 MB/s. Um repo aberto continua acessível sem token, mas não necessariamente na mesma
velocidade.

Para o ModelScope o token **não** foi testado aqui. Os 1,8 MB/s medidos foram anônimos, então
não dá para afirmar que aquele é o teto da plataforma — pode ser limite de sessão não
autenticada. Autenticação é via SDK:

```python
from modelscope.hub.api import HubApi
HubApi().login('<token>')
```

Se o ModelScope for a única fonte de algo (caso do `QuantFunc` antes de o espelho no HF ser
encontrado), vale testar com token antes de concluir que é lento.

## Nomear pelo arquivo, não pelo repo

Confirmado na prática, e vale mais do que a regra 5 sugere. Casos reais:

- `qwen_image_layered_bf16` **não** é `qwen-image-edit-2509`, apesar de os dois serem "qwen image".
  Trocar um pelo outro produz workflow que roda e faz a coisa errada.
- `tonera/FLUX.2-klein-9B` e `-4B` são modelos diferentes; peguei o errado por não olhar o número.

Heurística que funcionou: reduzir o nome a tokens, descartar ruído de precisão
(`bf16 fp16 fp8 e4m3fn scaled mixed int8 gguf`) e exigir que o alvo seja subconjunto do candidato.
Implementado em `tools/swap_to_nunchaku.py`.

## Gate de GPU: não basta o arquivo ser int4

`sm_86` descarta `fp4`/`nvfp4` — isso a regra 6 já diz. Mas há **dois gates a mais**, e os dois
me pegaram hoje:

1. **Precisa existir loader.** `nunchaku-sdxl` e `nunchaku-sana` têm INT4 publicado e o
   `ComfyUI-nunchaku` instalado não tem nó para nenhum dos dois — só Flux, QwenImage e ZImage.
   Kernel existe, nó não, modelo é inútil.
2. **Loader existir não significa cobrir a família.** `NunchakuFluxDiTLoader` existe e **não**
   carrega FLUX.2: zero menção a `flux2`/`klein` no nó e no pacote `nunchaku 1.2.1`.

Verificação barata antes de baixar:

```bash
grep -oE '"Nunchaku[A-Za-z0-9]+"' ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py | sort -u
grep -rniE "flux2|sdxl|sana" ComfyUI/custom_nodes/ComfyUI-nunchaku/
```

## Formato ≠ aceleração

Distinção que decide se vale baixar, medida aqui:

| formato | 4 bits no disco | 4 bits no cálculo | efeito |
|---|---|---|---|
| SVDQuant (nunchaku) | sim | **sim** (`gemm_w4a4`) | **2,27x** medido no Z-Image |
| GGUF Q8/Q4 | sim | não (dequant p/ bf16) | só VRAM |
| SDNQ uint4+svd | sim | não | só VRAM |
| fp8 em Ampere | sim | não (sem unidade fp8 na sm_86) | só VRAM |

Se o objetivo é velocidade e não só caber na placa, só a primeira linha serve.

## O original costuma estar junto do quant

`tonera/Beyond_Reality_Zimage_v2_svdq` publica o `transformer/` bf16 (11,47 GiB) ao lado do
`svdq-int4_r32`. Antes de investir em reconstruir peso a partir do quantizado, **listar o repo
inteiro** — o bf16 pode estar ali, e vem com a qualidade original em vez da do INT4.

## Matar download em background não mata o filho

`TaskStop` (ou matar o shell) encerra o wrapper, **não** o `curl` que ele lançou. Aconteceu aqui:
três `curl` do ModelScope continuaram vivos depois de a fila ser "morta", competindo por banda
com os downloads novos — e dois deles iam **sobrescrever arquivos já completos** no `mv .part`
final, trocando um arquivo bom por outro baixado pela metade da velocidade.

Depois de matar uma fila de download, conferir:

```powershell
Get-CimInstance Win32_Process -Filter "Name='curl.exe'" |
  Select-Object ProcessId, CommandLine
```

e limpar os `.part` órfãos antes de recomeçar.

## Um loader por família, e a assinatura muda

Não basta o nó existir. No `ComfyUI-nunchaku 1.2.1` os três loaders de DiT têm assinaturas
diferentes:

| loader | argumentos |
|---|---|
| `NunchakuZImageDiTLoader` | `model_name` |
| `NunchakuQwenImageDiTLoader` | `model_name, cpu_offload, num_blocks_on_gpu, use_pin_memory` |
| `NunchakuFluxDiTLoader` | `model_path, attention, cache_threshold, cpu_offload, device_id, data_type` |

O de FLUX traz `cache_threshold`, que é o first-block cache do próprio nunchaku. Deixar ligado
mistura ganho de cache com ganho de quantização em qualquer medição.

## Build de terceiro pode divergir em detalhe invisível no nome

`tonera/Qwen-Image-Edit-2511-Lightning-Nunchaku` tem nome, tamanho e formato plausíveis e **não
carrega**. O `QuantFunc/Nunchaku-Qwen-Image-2512`, mesma categoria de publicador, carrega e roda
idêntico ao oficial.

A diferença está numa camada só, e o nome do arquivo não a mostra. As camadas de modulação
(`img_mod`/`txt_mod`) **não** são SVDQuant nos builds que funcionam: são AWQ W4A16. Medido nos
três arquivos que existem aqui:

| checkpoint | sufixos em `img_mod`/`txt_mod` | formato |
|---|---|---|
| `svdq-int4_r128-qwen-image-edit-2509` (funciona) | `qweight` I32 + `wscales` + `wzeros` | AWQ W4A16 |
| `QuantFunc` 2511 `balance_int4` (funciona) | `qweight` I32 + `wscales` + `wzeros` | AWQ W4A16 |
| `tonera` 2511-lightning (**morto**) | `qweight` I8 + `smooth_factor` + `smooth_factor_orig`, mais 4 `weight` em F16 | SVDQuant |

Ou seja: a tonera quantizou **as 116 camadas de modulação como SVDQuant onde o loader exige
AWQ**, e ainda deixou 4 sem quantizar. Dizer "deixou em F16" descreve só os 4.

Checagem barata antes de baixar 13 GiB — a presença de `wzeros` é a assinatura do AWQ, e a de
`smooth_factor` a do SVDQuant:

```python
import json, struct
with open(path,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
mod = {k.rsplit('.',1)[-1] for k in h if '.img_mod.' in k}
print(sorted(mod))   # {'bias','qweight','wscales','wzeros'} -> carrega
                     # {'bias','qweight','wscales','smooth_factor',...} -> não carrega
```

Isso é lido do header, sem baixar o arquivo, via `hf_hub_download` de um range ou via API.

## Escolher a variante pelo rank que o slot já usa, não pelo adjetivo

Publicadores rotulam as variantes como *best_quality* / *balance* / *ultimate_speed*, o que soa
como a escolha relevante. Não é: o que muda é o **rank** do ramo de baixa dimensão (256 / 128 /
32). Trocar um `r128` por um `r128` é troca de um arquivo. Trocar por `r32` muda junto o ponto de
qualidade/velocidade, e aí a comparação A/B passa a medir duas coisas ao mesmo tempo.

`QuantFunc/Nunchaku-Qwen-Image-EDIT-2511` publica os seis cruzamentos (3 ranks × int4/fp4). Numa
3090 (sm_86) só os três `int4` interessam — FP4 é Blackwell.

## Conferir o tamanho contra o manifesto antes de instalar

O listing da HF já traz `size` por arquivo. Baixar para um nome temporário, comparar o tamanho
com o do manifesto e só então renomear custa nada e elimina a classe inteira de "download
truncado que parece modelo". Vale mais aqui do que em geral: um `.safetensors` truncado abre o
header sem erro e só quebra na hora de ler o tensor.

## Comparar dois modelos exige o ajuste de cada um

Rodar dois checkpoints no mesmo `steps`/`cfg` mede o ajuste, não o modelo. Qwen-Image **base** em
8 steps e cfg 1.0 sai cru e parece quantização ruim; em 20 steps e cfg 4.0, mesmo arquivo e mesma
seed, sai nítido. Modelo distilled/Lightning quer poucos steps e cfg 1; base quer o contrário.
