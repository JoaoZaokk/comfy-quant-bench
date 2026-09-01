# Handoff — 2026-09-01, noite

Continuação do `HANDOFF_2026-09-01.md`. Aquele fechou a investigação; este fechou a **publicação**.

---

## 1. Quatro repos no ar no HuggingFace, 39,2 GiB

| repo | pesos | licença | papel |
|---|---|---|---|
| [Beyond-Reality-Z-Image-v2-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Beyond-Reality-Z-Image-v2-W4A4-ConvRot) | 2 · 6,24 GiB | apache-2.0 | **o caso bom** — 4 bits funciona e é mais rápido que BF16 |
| [Qwen3-4B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen3-4B-W4A4-ConvRot) | 1 · 2,42 GiB | apache-2.0 | text encoder, 7,49 → 2,42 GiB |
| [Wan2.1-VACE-1.3B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Wan2.1-VACE-1.3B-W4A4-ConvRot) | 3 · 6,34 GiB | apache-2.0 | a linha por modelo + armadilha `vace_strength` |
| [HunyuanVideo-1.5-720p-T2V-Quantized](https://huggingface.co/JoaoZaokk/HunyuanVideo-1.5-720p-T2V-Quantized) | 3 · 24,22 GiB | **tencent** | W4A8 usável, 0,1837 granulada, 0,2147 destruída |

Todos seguem o desenho do dono: **o build quebrado sobe rotulado, ao lado do que funciona**. GitHub
(`comfy-quant-bench`) aponta para os quatro; os quatro apontam de volta.

**Ferramenta:** `tools/hf_publicar.py`. Card primeiro (barato, dá o link), pesos depois. Um repo com
README e sem peso é recuperável; peso sem README não explica nada.

## 2. Licença: o único portão, e ele está resolvido

Conferido **na fonte**, não de memória:

```
Qwen/Qwen3-4B                        apache-2.0
Wan-AI/Wan2.1-VACE-1.3B              apache-2.0
Tongyi-MAI/Z-Image-Turbo             apache-2.0   (base do Beyond Reality)
Nurburgring/BEYOND_REALITY_Z_IMAGE   apache-2.0   + Civitai 1090420 confirma
Glanty/Capybara                      mit
tencent/HunyuanVideo-1.5             other        <- Community License
```

**Tencent não é permissiva, e crédito sozinho não cumpre.** Li os termos. O repo carrega as quatro
obrigações: cópia integral do acordo (s.3a), aviso destacado de modificação tensor-a-tensor (s.3b),
o texto literal exigido (s.3d) e não-afiliação (s.3e). E o topo do card avisa que **o acordo não
vale na UE, Reino Unido e Coreia do Sul**, além do limite de 100 M de usuários mensais.

**`capybara_v0.1` ficou de fora de propósito.** Glanty declara MIT, mas o arquivo é arquitetura
HunyuanVideo 1.5 — se a licença da Tencent viaja por cima disso é pergunta para o autor dele. Fica
medido e não redistribuído, e o README público diz isso.

## 3. Erro meu, publicado em quatro lugares, corrigido no mesmo dia

`capybara_v0.1` foi tratado como família Z-Image, e seu `0,2163` virou o **teto do Z-Image**.

```
capybara_v0.1     1364 tensores  54 double_blocks  byt5_in    16 653 435 264 bytes
hunyuanvideo1.5   1361 tensores  54 double_blocks  byt5_in    16 653 368 128 bytes   (67 KiB!)
z-image v2         453 tensores   0 double_blocks  layers
```

É Hunyuan, ~13 B. A tabela vira:

| modelo | params | tolerado | não tolerado |
|---|---|---|---|
| Wan 2.1 VACE | 1,3 B | 0,0546 | 0,0793 |
| Z-Image v2 | ~6 B | 0,1241 | **nunca medido** |
| HunyuanVideo 1.5 (família) | ~13 B | 0,1837 | 0,2147 **e** 0,2163 |

A evidência não some — muda de fila e **fortalece** o Hunyuan (duas quebras independentes). O que
some é a faixa do Z-Image, agora honestamente vazia. A monotonia do tolerado
(0,0546 < 0,1241 < 0,1837) não dependia da célula errada e sobrevive.

Os 432 módulos que o perfil `hunyuan_video_15` casou no capybara diziam isso desde sempre.

Segunda correção junto: o finetune Beyond Reality é do **Nurburgring**, não da tonera — ela fez uma
SVDQuant independente do mesmo modelo, creditada como tal.

## 4. Wan 2.1 VACE — terceira família (o trabalho da madrugada)

Mediana `err_w4a4` **0,1602**, a faixa do meio que nenhum modelo tinha ocupado, onde a regra previa
"correta, granulada". **Saiu destruída**, três sementes. Previsão escrita antes: errada.

Custou quatro renderizações porque **a referência FP16 também estava quebrada**. Causa, um eixo
variado: `WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) preenche `vace_frames` com
zeros, passa por `process_latent_in` (zero vira **não** nulo), concatena máscara de uns e aplica em
`vace_strength=1.0`. Qualquer checkpoint VACE em workflow T2V comum sai destruído, sem erro.
`quality_ladder.py` ganhou `--vace-strength`; `tools/probe_vace_strength.py` varia esse eixo.

`--shift` foi adicionado numa hipótese que a medição recusou (o patch aplica, mas o scheduler
`simple` não consulta shift — latente bit-idêntico). Mantido, com o comentário corrigido.

## 5. Próximo passo, já aprovado por ele

**Construir a camada 1 do avaliador em lote** (`tools/avaliar.py`). Ele disse "sim, isso vai ser
útil". Desenho combinado:

**Roda desacompanhado, gera `.json` por checkpoint + `.md` digest ordenado por "olha aqui primeiro".**

Três níveis, para poder rodar "em tudo" sem queimar GPU:
- `--rapido` (default, sem GPU): cabeçalho, contagem de formatos, metadata, sidecar, razão de tamanho
- `--dispatch` (GPU): `probe_quant_dispatch --forward-only`, forwards quantizados vs `dequantize`
- `--erro` (GPU, precisa de calibragem): erro por camada

**As três guardas** — é o que falta escrever:
1. **saúde do braço de referência** — o arquivo NÃO quantizado produz latente dentro da faixa da
   família? (`607,6` fora vs `1543,2` dentro no Wan: 2,5x, gritante). **Esta teria abortado a
   rodada do Wan na primeira imagem em vez da quarta.**
2. **dispatch** — `dequantize > 0` com formato quantizado → economiza VRAM e não roda 4 bits
3. **erro fora da banda conhecida daquele modelo** → é aqui que a linha por modelo vira ferramenta

**Vereditos de um conjunto fechado, e `APROVADO` não está nele:**
`REPROVADO` / `OLHAR` (com motivo) / `SEM VEREDITO`.

Por quê: medido nesta bancada, nenhum corte em erro por camada ou em divergência separa usável de
inutilizável — 0,1837 correta contra 0,2147 destruída, e 0,7173 boa contra 0,8255 destruída, 15% de
distância nos dois eixos. **O número reprova sozinho, aponta onde olhar e prevê antes de gastar GPU.
Não aprova.** Quase tudo já existe em `verify_w4a4`, `probe_quant_dispatch`, `quant_mixed`,
`quality_ladder`, `metricas_imagem` — falta o orquestrador e as guardas.

## 6. Aberto

| item | estado |
|---|---|
| `tools/avaliar.py` camada 1 | aprovado, não começado |
| `--sem-pensar` na bateria de texto | **mente** — modelos ainda emitem `<think>`. Consertar ou remover |
| `tools/probe_piso_de_ruido_do_prompt.py` | escrito, **nunca rodado** |
| Migrar os sete conversores para `_conversion.py` | ticket 08 só fecha aí |
| Teto do Z-Image | célula vazia; ninguém tentou acima de 0,1241 |
| Perfis novos (Flux, Wan 2.2, LTX 2.5, Qwen-Image) | cada um é trabalho novo |
| Gemma W4A8 no HF | é memory-only (336 dequantize), sem referência BF16 — card fraco |

## Estado de fechamento

```
git raiz          limpo
comfy-quant-bench limpo, main em sincronia com origin
GPU 0             livre, lock livre
```

Commits da sessão, raiz: `7628413` (Wan), `079c615` (hf_publicar), `a846629` (cards),
`cc659ac` (notices Tencent), `d38c454` (escada vs pesos), `f757f54` (correção capybara).
`comfy-quant-bench`: `558cf9f`, mais o commit das duas correções.
