# Resultados Qwen-Image-2.1 — rodada autônoma 26/09 (atualizado conforme sai)

## Fase 1 — bateria base (1024², 25 passos, euler/simple, cfg 1, TE qwen3vl_8b_w4a8, RTX 3090, --disable-dynamic-vram)
Métricas contra bf16 da mesma seed/prompt, 12 imagens por DiT (`fase1_metricas.json`):

| DiT | it/s (mediana) | s/imagem quente | pesos na VRAM | MS-SSIM médio (mín) | PSNR médio | SSIM | grão/ref |
|---|---|---|---|---|---|---|---|
| bf16 Comfy-Org | 1,05 | 25,8 | ~13,6 GB | — | — | — | 1 |
| bf16 junto dos shards (MLP separada) | 1,05 | 25,8 | ~13,6 GB | 1 (12/12 pixel a pixel iguais) | ∞ | 1 | 1,00 |
| int8 convrot Comfy-Org | 2,37 | 11,8 | 6,9 GB | 0,988 (0,952) | 36,2 | 0,983 | 1,00 |
| mixed W4A8 NidAll | 2,01 | 13,6 | 4,0 GB | 0,937 (0,843) | 25,5 | 0,923 | 0,96 |
| int4 SVDQ r128 mesmertech (nosso loader) | 2,44 | 13,4 | ~4 GB | 0,893 (0,804) | 22,0 | 0,868 | 1,00 |

- A MLP fundida da Comfy-Org (`gate_up`) e a separada do diffusers dão a MESMA imagem, bit a bit, nas 12.
- s/imagem inclui VAE e overhead; o int4 tem it/s maior que o int8 mas s/imagem maior (o patcher move o modelo inteiro).

### Avaliação cega MiMo V2.6 Pro (`mimo_fase1/resumo.json`)
| DiT | geral | artefatos | aderência | texto | posição média no ranking (1–5) |
|---|---|---|---|---|---|
| bf16 | 7,75 | 8,33 | 7,83 | 9,0 | 2,27 |
| bf16junto (mesma imagem) | 8,00 | 8,50 | 7,92 | 8,75 | 2,27 |
| int8 | 7,83 | 8,42 | 7,67 | 9,5 | 2,91 |
| mixed | 7,50 | 8,17 | 8,00 | 9,25 | 3,82 |
| int4 | 7,42 | 7,75 | 7,50 | 8,5 | 3,73 |

- **Ruído do juiz**: mesma imagem avaliada 2× às cegas → diferença média de 0,75 na nota geral (máx 2), 0,61 em
  todos os critérios. Diferenças de nota entre builds (≤ 0,6) estão DENTRO do ruído por imagem; só o ranking agregado
  separa: bf16 ≈ int8 > int4 ≈ mixed.
- O juiz acertou "bf16 = bf16junto são idênticas" em só 6/12 rankings e declarou idênticas imagens diferentes; uma
  vez pôs as duas cópias em 1º e 4º. Conclusão: MiMo serve para defeito grosso (int4 neon, contagem), não para
  ordenar builds próximos. Minhas observações visuais em `observacoes_visuais.md` (não cegas).

## Fase 2 — nossas quantizações e erro por camada
- Perfil `qwen_image21` nos conversores (192 lineares, a seleção da Comfy-Org/NidAll).
- **Achado (bug nosso)**: os conversores passavam o peso BF16 direto ao quantizador; a rotação ConvRot em BF16 muda
  ~8% dos códigos int8 (±1–2) e 2,1% dos W4A8. Comfy-Org e NidAll quantizam do FP32: com FP32 nosso quantizador
  reproduz o arquivo deles em 99,99999% (int8) e 99,9998% (W4A8) dos códigos; o codebook W4A8 é o mesmo. Corrigido
  em quant_int8/w4a8/w4a4/mixed (`.float()` antes do quantizador; dtypes de saída iguais). Os primeiros builds
  (rotação BF16) ficam como controle; os `_f32` são os corrigidos (fase 3). O `_mixed` já saiu com a correção.
- Também corrigido em `calibrate_activations.py`: o lado do latente era `size // 8` fixo; o Qwen 2.1 é /16 (calibraria
  com 4× os tokens) e o LTX /32. Conferido: 4097 linhas por chamada = 4096 tokens de 1024² + 1.
- Calibração: 6 prompts DIFERENTES dos da bateria (held-out), seed 42, 25 passos, 256 linhas/camada, 165 s.
- Mixed (quant_mixed, promote 0,15): 120 W4A4 + 72 W4A8, 3,61 GiB. Promovidos: todos os `attn.to_out.0` e
  `attn.to_v` piores (W4A4 até 0,34 → W4A8 0,08); `gate_up` inteiro fica W4A4 (0,12).

### Erro de saída por camada (rel-RMSE, ativações reais da calibração, kernels reais) — `fase2/erro_por_camada.json`
| build | média | pior | to_q | to_k | to_v | to_out | gate_up | mlp.out |
|---|---|---|---|---|---|---|---|---|
| int8 Comfy-Org | 0,0083 | 0,019 | 0,005 | 0,005 | 0,011 | 0,013 | 0,007 | 0,008 |
| nossa int8 (rot. BF16) | 0,0087 | 0,020 | 0,006 | 0,005 | 0,012 | 0,014 | 0,008 | 0,009 |
| W4A8 NidAll | 0,0424 | 0,087 | 0,027 | 0,024 | 0,062 | 0,061 | 0,040 | 0,042 |
| nossa W4A8 (rot. BF16) | 0,0424 | 0,087 | idem | | | | | |
| int4 SVDQ r128 mesmertech | 0,0793 | 0,158 | 0,043 | 0,038 | 0,101 | 0,108 | 0,079 | 0,107 |
| nosso mixed W4A4+W4A8 | 0,0831 | 0,150 | 0,081 | 0,072 | 0,067 | 0,070 | 0,122 | 0,087 |
| nossa W4A4 pura (rot. BF16) | 0,1398 | 0,341 | 0,081 | 0,072 | 0,193 | 0,230 | 0,122 | 0,141 |

- O defeito BF16 custa ~5% de erro na int8 (0,0087 × 0,0083); na W4A8 some no arredondamento de 4 bits.
- O ramo low-rank r128 do SVDQuant corta o erro da W4A4 pura quase pela metade (0,079 × 0,140).
- Nosso mixed empata com o SVDQ em média (0,083 × 0,079) e tem pior caso menor (0,150 × 0,158), sem low-rank, mas
  com 72 camadas em W4A8 (mais lentas que W4A4 no Ampere). Render decide (fase 3).

## Shift a 2048² (issue #16447) — int8 Comfy-Org, 25 passos, 3 prompts × 2 seeds
- ~82 s/imagem a 2048² (decode tiled). Mesma composição nos dois braços (mesmo ruído inicial).
- MiMo cego: ranking padrão 4 × 2 mu131 ("detalhe mais fino"); notas gerais mu131 8,67 × 8,50 (dentro do ruído de
  0,75); texto mu131 8 × 6 (no p0_s7 o padrão fragmentou "2.1").
- Eu (não cego, `folhas_shift/`): mu131 um pouco mais limpo e escuro, menos ruído nos escuros; padrão com mais
  detalhe fino (galhos na névoa) e mais ruído. Nenhum defeito grosso no padrão.
- Conclusão: com int8 a 25 passos, o mu fixo 0,69 a 2048² **não produz defeito visível**; a troca é detalhe × limpeza.
  O relato da issue (40 passos, BF16, "claramente melhor") não se reproduziu aqui como vitória clara. Não recomendo
  aplicar o PR #16553 só por qualidade sem teste a 40 passos e BF16; o `shift_terminal` não foi testado.

## Fase 3 — builds corrigidos (FP32 na entrada)
- Nossa int8 `_f32` × int8 Comfy-Org: códigos 99,99995% iguais; 0,4% das escalas diferem em 1 ulp (≤ 2e-7
  relativo; CUDA × CPU). **Mesmo quantizador** — o int8 da Comfy-Org é reproduzível pela nossa ferramenta.
- Erro por camada (`fase2/erro_por_camada_f32.json`): nossa int8 `_f32` 0,0083 = Comfy-Org 0,0083; nossa W4A8
  `_f32` 0,0424 = NidAll 0,0424 (idênticos até a 4ª casa por tipo de camada). W4A4 `_f32` 0,1398 = versão BF16:
  no 4 bits o arredondamento domina e a rotação em BF16 não aparece.
- Ou seja: com a correção, "nossa quant × a do Comfy" é o mesmo arquivo na prática. A diferença que sobra entre
  publicações é de FORMATO/seleção (int8 × W4A8 × SVDQ × W4A4/mixed), não de algoritmo de arredondamento.

### Bateria das nossas (fase 3; `fase3_metricas.json`, folhas `folhas_fase3/`)
| DiT | it/s | s/img | pesos | MS-SSIM vs bf16 (mín) | PSNR | SSIM | grão |
|---|---|---|---|---|---|---|---|
| int8 Comfy-Org | 2,37 | 11,8 | 6,9 GB | 0,988 (0,952) | 36,2 | 0,983 | 1,00 |
| nossa int8 `_f32` | 2,40 | 11,7 | 6,8 GB | 0,995 (0,987) | 38,8 | 0,989 | 1,00 |
| W4A8 NidAll | 2,01 | 13,6 | 4,0 GB | 0,937 (0,843) | 25,5 | 0,923 | 0,96 |
| nossa W4A8 `_f32` | 2,01 | 13,5 | 4,0 GB | 0,932 (0,842) | 25,0 | 0,919 | 0,96 |
| int4 SVDQ mesmertech | 2,44 | 13,4 | ~4 GB | 0,893 (0,804) | 22,0 | 0,868 | 1,00 |
| nosso mixed W4A4/W4A8 | 2,82 | 9,9 | 3,7 GB | 0,865 (0,775) | 21,0 | 0,835 | 0,99 |
| nossa W4A4 `_f32` | **3,26** | **8,7** | 3,6 GB | 0,821 (0,695) | 20,2 | 0,725 | 1,03 |

- **Piso de ruído da métrica**: nossa int8 e a da Comfy-Org são o mesmo arquivo na prática (códigos 99,99995%), e
  mesmo assim as imagens diferem (MS-SSIM 0,955–0,9995 entre elas; PSNR 25–45). No p5 (maçãs) a trajetória da
  Comfy-Org diverge (a "meia maçã") e a nossa não — perturbação de 1 ulp muda a imagem. Diferenças de média de
  MS-SSIM ≲ 0,006 entre builds são sensibilidade caótica, não qualidade. (bf16 × bf16junto deram 12/12 idênticas,
  então o runtime é determinístico: é o peso, não o kernel.)
- **Velocidade**: W4A4 nativo é o mais rápido no Ampere: 3,26 it/s = 1,36× a int8 e 1,62× a W4A8 (e 1,34× o int4
  SVDQ do Nunchaku, que tem o ramo low-rank e o patcher de modelo inteiro). O mixed fica no meio (2,82).
- **Qualidade (eu, não cego)**: o **traço duplo no neon** aparece em TODO build com ativação de 4 bits (int4 SVDQ,
  nosso mixed, nossa W4A4 — a pior, "IMAGE2.1" borrado) e em nenhum W4A8/int8 → o artefato vem da quantização da
  ATIVAÇÃO a 4 bits (por token), não do peso de 4 bits (W4A8 também tem peso de 4 bits e está limpo).
  Pele: nossa W4A4 áspera/"crocante" (grão 1,03), mixed um pouco; o SVDQ, com low-rank, segura bem a pele mas não o
  neon.
- Placar: int8 (qualquer das duas) ≈ bf16 com 2,3× de velocidade; W4A8 limpo com 4 GB; 4 bits de ativação ainda não
  está pronto para neon/pele — nem o nosso nem o da mesmertech.

## Runtime — dynamic VRAM (modo padrão do .bat do dono) × --disable-dynamic-vram
- **Crash reproduzido (1ª tentativa, 26/09 20:34)**: ComfyUI 0.37.4 + comfy-aimdo 0.5.5, dynamic VRAM, Qwen 2.1 BF16
  lido de `\NAS\purple` (P:). O modelo é "staged" (13.571 MB), fica 1:53 em "Model Initializing" e o aimdo
  loga `hostbuf_read_file_slice: device copy failed result=2 ... size=67108864` (result 2 = cudaErrorMemoryAllocation)
  e o processo ABORTA (`Fatal Python error: Aborted`, pilha em `reset_prefix_cache`/`model_patcher.cleanup`).
  Não há fallback: o servidor inteiro cai. Com --disable-dynamic-vram o mesmo grafo roda (fase 1).
  Hipóteses a separar: (a) leitura direta de arquivo em compartilhamento de rede pelo aimdo; (b) pressão de VRAM com
  o TE (6 GB staged) ainda residente; (c) o cache KV do prefixo pedindo memória (2f7c6d47 mexe nisso).
- `--disable-dynamic-vram`, boot novo (runtime base, 20:42): as 9 imagens (bf16/int8/mixed × 3 prompts) saem
  **bit a bit iguais** às da bateria da tarde, e o it/s se repete (bf16 1,02–1,08; int8 2,36–2,39). Determinístico.
- Correção de ferramenta: `.scratch/roda_eros_2gpu.py` esperava o servidor para sempre; agora tem prazo de 10 min
  e `PULAR_GRAFOS` para pular grafos de servidor sabidamente morto (usado para destravar o resto do braço dyn).

### MiMo cego nas nossas (`mimo_fase3/resumo.json`, 12 imagens por build)
| build | geral | artefatos | aderência | texto | anatomia |
|---|---|---|---|---|---|
| bf16 | 8,08 | 8,50 | 8,08 | 8,5 | 8,0 |
| nossa W4A8 `_f32` | 8,00 | 8,33 | 8,25 | 8,5 | 7,6 |
| nossa int8 `_f32` | 7,92 | 8,50 | 8,33 | 9,25 | 7,9 |
| int4 SVDQ mesmertech | 7,33 | 7,75 | 7,58 | 9,0 | 7,3 |
| nosso mixed | 6,92 | 7,50 | 7,08 | 7,25 | 7,3 |
| nossa W4A4 `_f32` | **6,08** | **6,17** | 7,08 | **5,5** | 6,7 |
- Notas batem com as métricas e com o que eu vi: int8/W4A8 ≈ bf16 (dentro do ruído de 0,75); os builds com ativação
  de 4 bits caem, a W4A4 pura bem além do ruído (−2,0 no geral, −3,0 no texto). O ranking de 6 imagens saiu
  incoerente (bf16 com posição média 3,55; int4 "1º" 5×) — com 6 imagens o juiz não ordena; só as notas servem.

## Fase 4 — escalas refinadas por mínimos quadrados (`tools/refina_escalas.py`, novo)
Escala por linha ótima em forma fechada com códigos fixos, sobre metade das linhas de calibração; mede na outra
metade (held-out). "Só escalas" do QAT klein resolvido camada a camada, sem gradiente pela rede.
| build | camadas trocadas | erro held-out antes | depois | Δ |
|---|---|---|---|---|
| mixed | 186/192 | 0.0831 | 0.0815 | -2.0% |
| w4a4_convrot_f32 | 192/192 | 0.1397 | 0.1357 | -2.8% |
| w4a8_f32 | 186/192 | 0.0424 | 0.0414 | -2.3% |
- Ganho real e pequeno (2–3%): o erro vem do ARREDONDAMENTO dos códigos, não da escala. Render decide se aparece.
- Render dos refinados (`fase4_metricas.json`, `folhas_fase4/`): MS-SSIM mixed 0,865→0,857, W4A4 0,821→0,818, W4A8
  0,932→0,929 (dentro do piso de ruído); SSIM W4A4 0,725→0,766 e grão 1,03→0,94 (menos ruído fino). **Visualmente
  nada muda de verdade**: pele da W4A4 continua enrugada/áspera, neon continua com traço duplo no mixed e na W4A4.
  **Negativo**: refinar escala não conserta ativação de 4 bits. It/s igual (mesmo formato).
- Por que: a W4A8 (mesmo peso de 4 bits, ativação de 8) tem erro 0,042; a W4A4, 0,140. O que falta é a ATIVAÇÃO
  (4 bits por token, com outliers de canal — crest p99 até ~87 no `img_mlp.out`). No Qwen 2.1 as entradas das
  lineares vêm de LayerNorm SEM afim + modulação, e de saídas de atenção/SwiGLU: não há onde dobrar um fator de
  SmoothQuant sem mudar o runtime. O que resolve é um ramo low-rank (SVDQuant, como o da mesmertech) ou escala por
  canal em runtime — trabalho de kernel/layout no comfy-kitchen, não de conversor.

## Fase 5 — mixed conservador (promote 0,10) — `fase5_metricas.json`, `folhas_fase5/`
- Composição: W4A4 só em to_q (31/32), to_k (30/32) e 7 avulsas; W4A8 em to_v, to_out, gate_up (30/32) e mlp.out
  (29/32). 68 W4A4 + 124 W4A8, 3,83 GiB, pior W4A4 restante 0,098.
- Render: **2,21 it/s** (W4A8 2,01; int4 SVDQ 2,44; int8 2,40), MS-SSIM 0,917 (W4A8 0,932; int4 0,893; mixed
  0,865), SSIM 0,894, PSNR 24,1.
- **Visual: o neon volta a ter traço simples e a pele fica como a da W4A8.** O mixed 0,15 (gate_up 32/32 e mlp.out
  18/32 em W4A4) tinha o traço duplo; o 0,10 (gate_up e mlp.out em W4A8) não. → o artefato de neon vem da ativação
  de 4 bits na MLP (gate_up / mlp.out), não da atenção. Q/K em W4A4 não aparecem no render.
- Posição: W4A8-com-Q/K-em-W4A4 = qualidade de W4A8, +10% de velocidade, 3,8 GB. A int8 continua mais rápida e
  mais fiel (2,40 it/s, MS-SSIM 0,99) ao custo de 6,8 GB — no 3090 (24 GB) a int8 é a escolha; os 4 bits servem
  para caber em placa menor ou junto de outros modelos.
- MiMo cego (`mimo_fase5/`, 4 builds): geral bf16 8,00 · W4A8 7,83 · **mixed 0,10 7,83** · mixed 0,15 7,17; texto
  9,5 · 8,5 · 8,25 · 6,25; posição média no ranking 1,92 · 1,92 · 2,5 · 3,67 (com 4 imagens o ranking foi coerente).
  Confirma: mixed 0,10 = qualidade de W4A8.
- **Isolamento do crash (21:28–21:59)**: dynamic VRAM com int8 e mixed RODA — 6/6 imagens bit a bit iguais às do
  modo disable, mesmo it/s (int8 2,31–2,49; mixed 2,02–2,11) — mas a 1ª imagem leva 35–70 s a mais em "Model
  Initializing" (o aimdo lê os pesos do NAS sob demanda). O BF16 derrubou o processo de novo (2/2), mesmo erro do
  aimdo. Com os commits da master (2f7c6d47 + 1d61dcc3) aplicados: **mesmo crash** no BF16 (3/3 no total);
  int8/mixed iguais bit a bit e sem ganho de velocidade, em dynamic e em disable.
- Commits da master no modo disable: int8/mixed idênticos; **BF16 muda um pouco** (PSNR 34,9–46,6, MS-SSIM
  0,990–0,999 contra a 0.37.4). Hipótese: o 2f7c6d47 muda a decisão de onde/se guardar o cache KV do prefixo quando a
  VRAM está no limite (BF16 13,5 GB no 3090), e prefixo em cache × recalculado muda o último bit. Não logado.
  Conclusão: os dois commits NÃO valem adoção agora (sem ganho medido, sem conserto do crash). Revertidos; checkout
  conferido (só os 5 patches locais de antes).
- **Causa isolada (22:02)**: o MESMO BF16 copiado para disco local (`C:\ComfyBench_local`, cópia temporária) roda em
  dynamic VRAM: `fast_disk=True`, "Model Initialization" em 3 s (contra 1:53 e crash pelo NAS), 1,09 it/s, imagem
  bit a bit igual à do modo disable. → o crash é o caminho de **leitura por rede** do aimdo 0.5.5
  (`fast_disk=False`, `hostbuf_read_file_slice`) com modelo grande: falha a cópia para a placa e o processo aborta
  em vez de cair no carregamento normal. Modelos menores (int8 6,9 GB, W4A8 4 GB) passam pelo NAS, só mais devagar
  na 1ª imagem.
- Recomendação para o dono: com modelos grandes em `P:`/`D:` (compartilhamentos de rede), usar
  `--disable-dynamic-vram` ou manter o DiT em disco local. Vale reportar upstream (comfy-aimdo): abortar o processo
  num erro de cópia é o defeito; o esperado seria erro recuperável/fallback.
- A cópia local de 14,2 GB ficou em `C:\ComfyBench_local\diffusion_models\` (não apaguei; pode remover).
- Upstream: Comfy-Org/ComfyUI#16223 tem o MESMO erro (`hostbuf_read_file_slice device copy failed result=2`), mas lá
  a causa achada foram variáveis `CUDA_VISIBLE_DEVICES`/`GPU_DEVICE_ORDINAL` persistentes com duas instâncias.
  Aqui não há nenhuma delas persistente (User/Machine vazias), é UMA instância, e o disco local passa: é outro
  gatilho (leitura por rede). Dado novo para a issue — comentar lá é escrita externa, fica para o dono autorizar.

## Velocidade a 2048² (resolução nativa; 1 imagem por DiT, retrato seed 42, 25 passos, 22:10)
| DiT | s/it | × bf16 | s/imagem (inclui carga e decode tiled) |
|---|---|---|---|
| bf16 | 4,50 | 1,00 | 365 (1ª do boot, com TE) |
| int8 Comfy-Org | 2,66 | 1,69 | 168 |
| nossa W4A8 | 2,86 | 1,57 | 150 |
| nosso mixed 0,10 | 2,78 | 1,62 | 145 |
| nossa W4A4 | **2,13** | **2,11** | 114 |
| int4 SVDQ | 2,68 | 1,68 | 142 |
- A 2048² a atenção (não quantizada) pesa mais: o ganho de 1024² (int8 2,3×) cai para 1,7×, e o mixed 0,10 deixa de
  ganhar da int8 (2,78 × 2,66 s/it). A W4A4 segue a mais rápida, mas com os artefatos já descritos.
