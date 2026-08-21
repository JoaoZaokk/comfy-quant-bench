# Inventário auditável de env, nodes e modelos convertidos

Type: task
Status: resolved

## Question

O escopo deste mapa inclui env, aplicativos, nodes e modelos convertidos — coisas que
o git **não** rastreia. Modelo convertido é artefato de 20 GB, não código; pacote de
node é repo de terceiro. Nenhum deles entra como alvo de revisão, mas todos entram
como **estado auditável da bancada**.

Levantar e escrever:

- **Env**: versões do stack fixado (Torch 2.13.0+cu130, comfy-kitchen 0.2.23,
  nunchaku 1.2.1, spas_sage_attn), e o que está preso a quê. Há fatos frágeis já
  documentados — a `cudart64_12.dll` copiada lado a lado, sem a qual Sage e FlashVSR
  não sobem.
- **Nodes**: 66 pacotes instalados. Quais são nossos (três arquivos estão nos 88
  rastreados), quais estão órfãos, quais o workflow de aceitação realmente usa.
- **Modelos convertidos**: quais existem, **qual conversor produziu cada um** e com
  que parâmetros. Hoje há `int8_convrot`, `w4a8`, `nvfp4`, `int8` do LTX 2.5 em `D:`,
  e a procedência de cada um não está num lugar só.

## Reusar, não criar

`tools/quant_audit.py` já percorre `ComfyUI/models` e reescreve
`quantization_inventory.{json,md}` a partir de metadados. A parte de modelos deste
ticket deve **estender** isso, não escrever um segundo inventário.

## Critério de fechamento

Fecha quando existir um documento que responda, para cada modelo convertido, qual
ferramenta e quais parâmetros o produziram; e uma lista de nodes marcando nossos,
usados e órfãos.

Não fecha por decisão escrita: é levantamento, e ou existe ou não existe.

## Resolução

Criado `F:\COMFY_PORTABLE\INVENTARIO_BANCADA.md`, três partes:

- **ENV**: `pip list` real (executado agora) confronta com `CLAUDE.md` — Torch/torchvision/
  torchaudio/comfy-kitchen/nunchaku/spas_sage_attn batem exatamente. `cudart64_12.dll` confirmado
  ausente do `python_embeded` ativo (só existe em `venvs/ultravox311` e no backup pré-upgrade).
- **NODES**: recontagem ao vivo deu 66 entradas/64 diretórios não-ocultos (bate com CLAUDE.md),
  mas achei uma 67ª entrada oculta (`.disabled/`, vazio) que `CLAUDE.md` não menciona. Descobri
  também que a afirmação "cada entrada é seu próprio repo" é falsa para quase metade: 31 de 63
  pacotes têm `.git` próprio, 32 não têm. Só `comfy-quant-preflight` é rastreado pelo repo raiz
  (nosso). Cruzei o workflow `LTX25-int8-acceptance-v2.json` (18 nós) com o fixture de 16 classes:
  as 16 batem exatamente e **todas vêm do core do ComfyUI** — o workflow de aceitação `-v2` não usa
  nenhum dos 63 pacotes de `custom_nodes/`. Não classifiquei os 62 restantes como "órfãos" porque
  só cruzei 2 dos 28 workflows salvos; documento diz isso explicitamente em vez de inventar.
- **MODELOS CONVERTIDOS**: reusei `quantization_inventory.{json,md}` (gerado 2026-08-19, não
  rerodado) e busquei por nome em `ComfyUI/models` e `D:/ComfyUI-Models`. 7 arquivos com
  `.quant.json` (4 em `ComfyUI/models`, 3 em `D:`) — ferramenta inferida por assinatura de campo
  contra o código-fonte de `tools/quant_w4a8.py`, `quant_int8.py`, `quant_mixed.py` (marcado
  explicitamente como inferência, não prova por log). 6 arquivos em `D:` sem sidecar, rastreados
  até `tools/fetch_ltx25.py`/`fetch_minimax_h3.py` — baixados já quantizados de HuggingFace
  (`Lightricks/LTX-2.5`, `starsfriday/MiniMax-H3-w4a8`, `Comfy-Org/MiniMax-H3`), não produzidos
  localmente. Quatro arquivos que `W4A4_PROGRESS.md` cita como mantidos ("fixtures") não foram
  encontrados no disco nesta busca.

Comando que provou cada parte está citado inline no próprio `INVENTARIO_BANCADA.md`, seção por
seção, marcado `[EXECUTADO]` ou `[LIDO]`.

**O que ficou sem cobertura** (dito também dentro do artefato, seção 3.6 e 2.5):
- Nenhum modelo foi carregado nem `quant_audit.py` rerodado (proibido pela rodada).
- `D:/ComfyUI-Models/` não tem varredura de metadados equivalente à de `ComfyUI/models` — os 9
  arquivos ali vieram só de busca por padrão de nome, pode haver outros não capturados.
- Correspondência ferramenta↔sidecar é inferência de campo, não confirmada por log de execução
  real (não existe log de terminal amarrando comando a arquivo nesta bancada).
- Só 2 dos 28 workflows salvos foram cruzados contra pacotes de nó — não há veredito de "órfão"
  para os 62 pacotes restantes, só o que cada um é (rastreado/não, `.git` próprio/não).
- Recontagem do repo raiz deu 93 arquivos rastreados agora, não os 89 (`CLAUDE.md`) nem 88 (texto
  deste ticket) — divergência registrada, causa não investigada (fora de escopo).
- Achado de graduação (não implementado): `quant_audit.py` não tem campo de procedência; sugestão
  de schema (`provenance: {tool, params, sidecar_path}`) descrita no artefato, sem mudar a
  ferramenta.
