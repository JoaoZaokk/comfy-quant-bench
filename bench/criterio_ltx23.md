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

## O que NÃO vai estar coberto, já sabido

Uma semente, um prompt, 512 px, uma placa. Nenhuma métrica perceptual de áudio validada aqui.
Sincronia áudio-vídeo não medida. O text encoder do 2.3 fica em BF16 em todos os braços; a
versão W4A8 dele existe (`gemma_3_12B_it_w4a8`, 8,31 GiB, 336 camadas) e entra no card como
peça da suíte, com a advertência das duas travas do ComfyUI — memória sem velocidade — que já
está medida para o Gemma heretic e não foi remedida no stock.
