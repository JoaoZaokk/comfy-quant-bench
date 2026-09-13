# O que explodiu em vídeo no HuggingFace — varredura de 2026-09-12

Duas varreduras independentes, modelos diferentes, prompts diferentes, rodadas em paralelo: uma
olhando **7 dias** por `trendingScore`, outra olhando **30 dias** por `downloads`. Elas
convergiram, e a convergência é o achado principal.

Tudo abaixo vem de campos da API do Hub (`hub_repo_search`, `hub_repo_details`, `hf_fs find`).
**Nenhum número aqui foi lido de README**, e nenhum modelo foi executado — isto é um mapa de
adoção, não de qualidade.

---

## O achado: uma base só domina a semana

**Praticamente toda a atividade de vídeo dos últimos 7 dias é derivada de
`MiniMaxAI/MiniMax-H3`** (`lastModified` 13 ago 2026, portanto a base em si está *fora* da
janela). Dezenas dos repos "novos" são LoRAs, merges e quantizações dessa mesma base, criados a
horas de distância uns dos outros por contas diferentes — padrão de repack em massa, não de
lançamentos independentes.

Isso importa para esta bancada porque **o MiniMax H3 já está no disco aqui** em três arquivos
(`MiniMax_H3_FL2VA_...`, `MiniMax_H3_Ref2VA_...`, e o `minimax_h3_..._w4a8_convrot` do
Winnougan), e já foi medido: é o checkpoint onde `w4a4_int4mm_layers: 0` e o INT8 venceu o nosso
W4A4 por 1,40x em 60 de 60 comparações.

## Tabela, por tração medida

| repo | tarefa | downloads | likes | atualizado | maior arquivo, bytes exatos | gated |
|---|---|---|---|---|---|---|
| `lightx2v/Minimax-h3-Turbo` | image-to-video | 1,4 M | 913 | 10 set 2026 | `minimax_h3_fl2v_turbo_4step_v1.1_768p_fp8` — 34.023.114.720 | não |
| `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` | text-to-video | 271,9 K | 300 | 27 ago 2026 | shard 1 de 14 — 5.331.008.552 | não |
| `WarmBloodAban/Minimax-h3_Singularity` | image-to-video | 114,1 K | 347 | **12 set 2026** | `..._ref2va_v1.3_int8` — 34.004.507.622; podado — 20.967.647.456 | não |
| `ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8` | text-to-video | 7,0 K | 7 | 10 set 2026 | `ltx25_uncensored_v1.1-Q8_0.gguf` — 22.334.842.304 | não |
| `coolthor/MiniMax-H3-pruned-NVFP4` | image-text-to-video | 2,4 K | 21 | 12 set 2026 | `minimax_h3_fl2va_pruned_nvfp4` — 12.528.636.800 | não |
| `AlayaLab/Evoke-Turbo` | text-to-video | 35 | 0 | 11 set 2026 | shard 4 de 6 — 9.944.020.376 (soma 57.249.822.376) | não |
| `Yi30/wan2.2-ti2v-w4a8-svd-nvidia` | — | 0 | 0 | 11 set 2026 | quantização W4A8/SVDQuant do Wan2.2 | não |

## O negativo que vale mais que a tabela

**`wanvideo/wan-3-0-video` NÃO é um lançamento do Wan 3.0.** O repo tem a tag `wan-3.0` e a data
certa (4 set 2026), e a tentação era reportá-lo. Foi conferido por `hf_fs find` em vez de assumido:
o repositório contém **apenas** `.gitattributes` (1.519 B) e `README.md` (4.390 B). **Zero pesos,
zero config, zero tag de biblioteca.** Não existe Wan 3.0 publicado.

Quem procurar "o Wan mais novo" vai esbarrar nisso. O mais novo de verdade, lido direto do autor
`Wan-AI` ordenado por `lastModified`, é:

```
Wan2.2-Animate-2-14B-Diffusers            criado 6 ago 2026, modificado 13 ago 2026
Wan2.2-Animate-2-14B-Distilled-Diffusers  criado 6 ago 2026, modificado 13 ago 2026
Wan2.2-Animate-2-14B                      criado 14 jul 2026, modificado  9 ago 2026
Wan-Dancer-14B                            criado 10 jul 2026, modificado 17 jul 2026
```

## O que NÃO foi coberto

- Nenhum dos dois agentes passou da primeira página de 50 por combinação de filtro/ordenação.
- `trendingScore` é métrica opaca e não documentada do HF. É reportada como o número que a API
  devolve, **não** como taxa de crescimento verificada.
- Não há delta "downloads de 7 dias atrás contra hoje". O ranking mistura *estar na janela* com
  *contagem absoluta atual*, que não é a mesma coisa que explodir.
- Não se sabe se o campo `downloads` desta API é acumulado ou janela móvel — não foi verificado.
- Filtros `video-to-video`, `image-text-to-video` e `audio-to-video` não tiveram busca dedicada.
- **Nenhum modelo foi executado.** Nada aqui diz que algum deles é bom.
- A aritmética de "4 bits ≈ BF16÷4" que um dos agentes usou para prever encaixe em 24 GB é regra
  de bolso, **não medição** — nesta bancada o Qwen Edit 2511 saiu de 38,05 GiB para 10,79 GiB
  (3,53x, não 4x), e o mesmo modelo em W4A4 deu 9,60 GiB (3,96x). Não usar essa divisão como fato.

## Consequência para o plano

O candidato com sinal de tração mais forte e custo de teste mais baixo é o
`Minimax-h3_Singularity` **podado**, 20.967.647.456 B (19,53 GiB), que cabe na 3090 sozinho. Mas a
base dele já é conhecida desta bancada e já perdeu para o INT8 em 60 de 60 comparações por camada,
então o valor de testá-lo é medir o *fine-tune*, não o formato.
