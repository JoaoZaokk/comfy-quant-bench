# NOVO — rodada 1, 2026-08-30

Três agentes em paralelo: ModelScope, Gitee, HuggingFace. Cada linha abaixo com achado foi
**reconferida por quem orquestrou**, chamando o endpoint de detalhe direto — os agentes
marcaram coisas como CONFERIDO admitindo, no mesmo parágrafo, não ter lido metadado.

## O que muda uma decisão desta bancada

### 1. LTX-2.5 já existe quantizado, e em formato misto 4/8

`joeygambino/LTX-2.5-Quantized` — HuggingFace, 4625 downloads, mexido 2026-08-21.

```
LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors
LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors
LTX25-distilled-DiT-comfy-nvfp4.safetensors
LTX25-distilled-DiT-comfy-int8.safetensors
LTX25-distilled-DiT-Q2_K.gguf .. Q8_0.gguf
```

**`mix4x8` é o mesmo conceito do `tools/quant_mixed.py`**: 4 bits e 8 bits misturados por
camada, num único arquivo, com o prefixo `comfy-`. Duas variantes de tamanho sugerem dois
pontos de corte diferentes na mesma decisão que este repo toma por medição de erro.

- **CONFERIDO:** existência, contagem de arquivos (21), tags, `base_model: Lightricks/LTX-2.5`
- **NÃO conferido:** como a mistura foi decidida (medida? heurística? crest factor?), se
  carrega no ComfyUI daqui, se o `comfy_quant` por camada é o mesmo formato.

### 2. Quatro W4A4 de vídeo/difusão no ModelScope

Todos verificados por `GET /api/v1/models/<owner>/<nome>` (controle: Qwen, 7,1 M downloads).

| modelo | downloads | por que interessa |
|---|---|---|
| `XXXXinXXXXX/Wan22-i2v-w4a4` | 1439 | maior adoção do grupo; W4A4 em image-to-video |
| `ApacheOne/Wan2.2-Animate-2-14B-OrbitQuant-W4A4` | 38 | **OrbitQuant** é método que esta bancada não conhece |
| `ultranationalism/Z-Image-Turbo-SVDQuant-NVFP4` | 35 | **Z-Image** é justamente o modelo em que o `quant_mixed.py` foi calibrado aqui |
| `ModelsLab/MiniMax-H3-svdquant-nvfp4_r32` | 2 | MiniMax-H3 é o que a 0.34 trouxe |

- **CONFERIDO:** os quatro existem, com downloads e data de criação reais.
- **NÃO conferido:** nada do conteúdo. Não sei o kernel, não sei se roda em sm86, não sei
  se `OrbitQuant` é ConvRot com outro nome ou coisa distinta, não sei se algum carrega no
  ComfyUI. **Três dos quatro são NVFP4, que é formato nativo de Blackwell** — se ele não
  tiver caminho em Ampere, esta tabela inteira é inaplicável nesta placa. É a primeira
  pergunta da rodada 2.

### 3. Alguém mais faz W4A4 com calibração em ativação real

`AlperKTS/Krea-2-SVDQuant-ComfyUI` — 4830 downloads, tags `w4a4, int4, svdquant, comfyui`,
125 arquivos, incluindo `calibration/krea2_act_stats_base.safetensors` e `_turbo`.

Estatística de ativação empacotada junto do modelo é exatamente o que o
`tools/calibrate_activations.py` produz aqui. É o primeiro caso externo comparável.

Menor, mesma família: `Patil/krea-turbo-svdquant` (39 downloads, `svdquant_config.json`).

## O que NÃO existe

Com consulta de controle validada:

- **`gitee.com/comfyui-cn` e `gitee.com/ComfyUI-Extensions` não existem** — 404 em API
  (`/api/v5/orgs/`, `/api/v5/users/`) e em HTML. Controle `mindspore` → 200 nos dois.
  Os dois links da lista original são becos.
- Gitee: 9 achados descartados por serem espelho declarado de GitHub/HF. Um único original
  chinês, `deeprd/Wan2GP` (0 estrelas, FP8 Scaled, não W4A4).

## O erro da rodada 1, registrado

Um agente escreveu *"LTX-Video quantizado → 0, controle validado"* e no mesmo relatório
entregou `LTX-2.5-Quantized`. A busca não estava cega e o controle estava certo — **o
modelo mudou de nome.** Ele segurou fixo o eixo que era a variável.

Regra nova na seção 1.2 do `AGENTS.md`: toda ausência lista os apelidos tentados.

Segundo erro: os três agentes escreveram `NOVO.md` por cima uns dos outros, e sobrou o do
último parecendo o total. Agora só quem orquestra escreve este arquivo.

## Não coberto nesta rodada

Conteúdo de qualquer arquivo — nenhum peso foi baixado, por regra. Nenhum teste de carga no
ComfyUI. Nenhuma GPU tocada: o lock estava com a sessão irmã (`glm-w8a8-test`), medido em
atividade real (5 MiB → 12 GiB, 29% util ao longo de 50 s).
