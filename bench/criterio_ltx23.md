# Critério escrito ANTES de medir: LTX 2.3 distilled 1.1 em 10 s com áudio, e o áudio do LTX 2.5

Escrito 2026-09-13 às 17:45, com o braço BF16 do 2.5 renderizando e nenhum número de áudio ainda
existente nesta bancada. O que está abaixo é aposta; o que vier depois é medição, e a diferença
entre os dois fica registrada aqui.

## Estado no momento da escrita

- LTX 2.3 distilled 1.1 BF16: 46.149.345.334 B em `P:/ComfyBench/checkpoints/`, checkpoint único.
- `_w4a8`: 16.651.422.654 B, 1440 camadas `asym_w4a8_int8`, 4507 tensores preservados.
- `_w4a4`: **ainda não convertido** (o conversor não tinha perfil LTX; ganhou um, idêntico ao do
  W4A8 — mesmas 1440 camadas — e roda depois do braço BF16 do 2.5 para não competir por RAM).
- Adversário de terceiro: `ltx-2.3-22b-distilled-1.1-Q6_K.gguf` (ComfyUI-GGUF, 6 bits, matemática
  dequantizada).
- Encoder fixo entre braços: `gemma_3_12B_it.safetensors` (Comfy-Org, BF16, o de fábrica),
  **não** o `heretic`. VAEs e projeção de texto sempre do arquivo BF16 (`--aux-checkpoint`).
- Protocolo igual ao do 2.5: 249 quadros a 25 fps, 512 px, semente 1234, mesmo prompt, cfg 1,
  euler; **8 passos** (sigmas do template oficial) em vez dos 3 do 2.5 destilado.

## Previsões — LTX 2.3

**P1 — W4A8 presta, em vídeo e em áudio.** MAE de vídeo contra o BF16 entre 5 e 15 (o 2.5 deu
7,81 a 3 passos; 8 passos dão mais oportunidade de a trajetória divergir, então a faixa é larga
para cima). Áudio: L1 de log-mel **pelo menos 2x menor** que o controle de ruído branco, e sem
silêncio (fração < 5%). Refuta: vídeo destruído, ou áudio mudo, ou L1 dentro de 1,5x do ruído.

**P2 — W4A4 funciona no LTX, como no 2.5 do riftcast, e é pior que o W4A8.** O LTX é a
terceira família onde o W4A4 *não* quebrou (Z-Image, Krea2, LTX 2.5 riftcast), contra três onde
quebrou (Qwen Edit, Qwen base, Wan 2.2). Aposta: vídeo coerente, MAE maior que o do W4A8.
Refuta: estática, borrão ou granulado colorido — e aí o LTX 2.3 entra na lista das que quebram,
e a linha "modelos de 2025 aguentam A4" perde força.

**P3 — o GGUF Q6_K de terceiro é mais fiel que o nosso W4A8** (MAE menor, log-mel menor), pelo
mesmo motivo que o int8 oficial venceu em cinco famílias: 6 bits de peso e matemática em alta
precisão. E é **mais lento por quadro** que o W4A8 residente, porque dequantiza. Refuta: nosso
W4A8 mais fiel que o Q6_K — seria a primeira vez nesta bancada.

**P4 — vídeo e áudio ordenam os braços do mesmo jeito.** Saem do mesmo latente, com os mesmos
pesos; o ranking por MAE de vídeo e o ranking por L1 de log-mel coincidem. Refuta: um braço melhor
em vídeo e pior em áudio — aí a quantização machuca os dois ramos de forma diferente, o que seria
um achado.

**P5 — velocidade.** W4A8 residente pelo menos 1,5x mais rápido por quadro que o BF16 em
DisTorch (o 2.5 deu 1,95x). Refuta: < 1,2x.

**P6 — o BF16 do 2.3 roda por DisTorch `cpu,40gb` lendo de P: sem derrubar o servidor**, agora
com 40 GiB de RAM livre e sem conversão concorrente. O 2.5 derrubou o servidor lendo de D: com
24 GiB livres (`access violation` em `torch/storage.py __getitem__`, dentro do mmap do
`load_torch_file`). Refuta: derruba de novo — e aí o mmap sobre SMB é o suspeito, não a RAM.

## Previsões — áudio do LTX 2.5 (os três braços, mesma semente)

**A1** O int8 da Lightricks fica mais perto do BF16 também no áudio (L1 de log-mel menor que o
do nosso W4A8), na mesma direção do vídeo (4,10 contra 7,81 de MAE na rodada anterior).
Refuta: W4A8 mais perto no áudio.

**A2** Nenhum braço muda de *nível*: RMS dos três dentro de ±3 dB, fração de silêncio < 5% em
todos. Refuta: um braço 6 dB abaixo ou mudo — degradação do ramo de áudio que o vídeo não mostra.

**A3** Lag de correlação cruzada dentro de ±20 ms nos dois braços (mesma semente, mesmo VAE de
áudio, mesma grade temporal). Refuta: lag maior — o braço quantizado deslocou o áudio no tempo.

## Vereditos (escritos depois de medir; a data de cada um está na linha)

**P6 — REFUTADA, e a cláusula de refutação também errou (2026-09-14 01:15).** O BF16 do 2.3 nunca
chegou a rodar pelo caminho previsto: seis tentativas pelo leitor de safetensors do ComfyUI mataram
o servidor (quatro em `load_torch_file`, uma no `torch.empty` do modelo, uma no `HostBuffer` do
dynamic VRAM), com 24 a 44 GiB de RAM livre e tanto de W: (SMB) quanto de C: (NTFS local). A
cláusula dizia "e aí o mmap sobre SMB é o suspeito, não a RAM" — os dois suspeitos estavam errados.
Medido em processo nu, com os contadores de commit lidos em volta de cada chamada:
`safe_open(framework="pt")` cobra **+40,8 GiB de commit ao abrir** (duas views copy-on-write, +80,2
enquanto as duas vivem) e o `torch.empty` do modelo mais +40,7; mmap somente-leitura cobra zero. O
limite de commit da máquina é 124,8 GiB com ~70 já tomados; quando a cobrança força o pagefile a
crescer, o primeiro toque na view às vezes é o `access violation` em `torch/storage.py __getitem__`
— reproduzido em C: sem ComfyUI (`W4A4_PROGRESS.md` parte 51). A referência BF16 foi renderizada por
outro caminho: os mesmos bytes num GGUF sem perda (`tools/safetensors_to_gguf_bf16.py`) lidos por
`UnetLoaderGGUF`, equivalência 12/12 camadas (`tools/probe_gguf_bf16_equivalence.py`). Isso muda a
P5: "BF16 em DisTorch" deixa de existir como braço; a velocidade do BF16 reportada é a do GGUF
streamado, e não se compara com o W4A8 residente.

**P1 — CONFIRMADA (2026-09-14 01:46, `bench/ltx23/av/`).** W4A8 MAE 10,39 [9,94–10,99], dentro da
faixa 5–15; log-mel 0,163 contra 1,748 do ruído branco (10,7x menor, ≥ 2x); silêncio 0 %.

**P2 — CONFIRMADA.** W4A4 coerente (mesmo farol, rochas, ondas, aves; `bench/ltx23/av/contato_av.png`)
e pior que o W4A8 nos dois ramos: 14,45 contra 10,39; 0,281 contra 0,163; nível −2,3 dB. O LTX 2.3
entra na lista das famílias que aguentam A4 (Z-Image, Krea2, LTX 2.5, agora 2.3), contra Qwen Edit,
Qwen base e Wan 2.2.

**P3 — CONFIRMADA no vídeo, EMPATE no áudio.** Q6_K MAE 3,59 contra 10,39 (2,9x mais fiel), mas
log-mel 0,163 contra 0,163 e convergência espectral 0,419 contra 0,424 — o som não separa. E mais
lento por passo: 5,04 contra 2,30 s/it (2,2x), como previsto, porque dequantiza.

**P4 — NÃO REFUTADA, com empate.** Vídeo ordena Q6_K < W4A8 < W4A4; áudio ordena Q6_K = W4A8 <
W4A4. Nenhum braço inverteu; o áudio não distingue os dois primeiros. No 2.5 o som ordenava com
margem maior que o quadro; no 2.3 não separa — não sei por quê, e fica registrado assim. Hipótese
barata para depois: 8 passos contra 3 (SNR de onda ≈ −1 dB em todos = fase descorrelacionada; o
log-mel pode estar num piso); teste = a mesma comparação a 3 passos.

**P5 — NÃO TESTÁVEL como escrita.** O braço "BF16 em DisTorch" nunca existiu (P6). O que existe, no
instrumento certo (barra do sampler, não o `s_por_quadro` — ver `W4A4_PROGRESS.md` parte 51): W4A8
residente 2,30 s/it contra 8,05 s/it do BF16 por GGUF parcialmente carregado (20,7 GB na placa,
19,6 GB descarregados) = 3,5x; isso mede streaming de peso + kernel, não só kernel. O "1,95x do
2.5" citado na previsão era parede da corrida inteira (carga inclusa) e não vale como velocidade.

**Controle de identidade do condicionamento (01:13):** MAE 1,74 [1,46–2,08], PSNR 35,75, SSIM 0,977,
log-mel 0,074, lag 0 ms, contra o render com encoder vivo — passou; o rodapé de ~2 MAE é comum aos
quatro braços e menor que toda distância da tabela.

**Nota que a fila h/i ensinou (2026-09-14 01:00):** o condicionamento salvo por
`LTXVSaveConditioning` NÃO é o prompt no LTX 2.3 — perde `unprocessed_ltxav_embeds`, o modelo pula
`caption_projection` e renderiza ruído (controle de identidade: MAE 75,9 contra o encoder vivo,
`bench/ltx23/cond_identity_ltxv_saver/`). Os braços desta previsão só valem se o controle de
identidade do condicionamento novo (`tools/ltx_encode_lowcommit.py` + `VoidLoadConditioningFull`)
der distância pequena contra o render com encoder vivo — esse número entra aqui junto com P1–P5.

## O que NÃO vai estar coberto, já sabido

Uma semente, um prompt, 512 px, uma placa. Nenhuma métrica perceptual de áudio validada aqui.
Sincronia áudio-vídeo não medida. O text encoder do 2.3 fica em BF16 em todos os braços; a
versão W4A8 dele existe (`gemma_3_12B_it_w4a8`, 8,31 GiB, 336 camadas) e entra no card como
peça da suíte, com a advertência das duas travas do ComfyUI — memória sem velocidade — que já
está medida para o Gemma heretic e não foi remedida no stock.
