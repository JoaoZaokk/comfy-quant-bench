# QAT por bloco do Qwen-Image-2.1 para W4A4 — critério (escrito antes de rodar, 27/09 ~02:40)

Autorização: o dono pediu "roda o QAT no Colab" (27/09). Treino na 3090 continua vetado; roda no Colab.

## O que é
`tools/colab_qat_qwen21/qat_qwen21_blocos.py`: reconstrução bloco a bloco (estilo BRECQ/OmniQuant) com W4A4 ConvRot
simulado em peso E ativação, professor = o próprio BF16 da Comfy-Org. Para cada um dos 32 blocos, as 6 lineares
aprendem a imitar o bloco BF16 com entrada vinda dos blocos já quantizados. Peso mestre FP32; códigos exportados no
formato nativo `convrot_w4a4` (os mesmos 192 tensores da nossa W4A4 `_f32`). Bloco só aceita o treino se o erro de
validação cair; senão fica o RTN.

Dados: 48 prompts de treino (texto/neon/emissivo, pele, mãos, paisagem, produto, ilustração; nenhum da bateria),
trajetórias do professor na própria VM (25 passos euler/simple, shift 0,69, 1024²), 4 passos por trajetória → 192
amostras (16 de validação). Holdout fim a fim = os 6 prompts da bateria de 26/09.

Smoke local na CPU (modelo minúsculo aleatório): códigos idênticos ao quantizador do comfy-kitchen, escalas exatas;
Linear simulada × kernel eager rel 3e-3 (o eager multiplica em BF16); treino reduziu validação e fim a fim.

## Previsões e decisão
- Q1: erro de validação por bloco cai em todos ou quase todos os blocos (o treino ajusta ao alvo).
- Q2: erro fim a fim do latente final no holdout cai contra o RTN (mesma sementes).
- Q3 [DECIDE, render local]: na bateria (6 prompts × 2 seeds), a W4A4 QAT fica visivelmente melhor que a nossa W4A4
  `_f32` (RTN) no neon e na pele. Vitória se o traço duplo do neon some ou reduz claramente em p0 (2 seeds) E a pele
  do p1 deixa de ficar áspera, sem regressão de composição/contagem maior que a do RTN. Métricas contra bf16
  (MS-SSIM/PSNR) e MiMo cego (notas; ruído 0,75) como apoio. Velocidade tem de ser a mesma da W4A4 (3,26 it/s):
  mesmo formato.
- Aposta: Q1 e Q2 sim; Q3 parcial — a pele melhora, o neon talvez não some (a ativação continua 4 bits por token).

## Não coberto
Uma rodada, uma semente de embaralhamento; 192 amostras é pouco para 32 blocos × 200 M parâmetros (risco de
sobreajuste; a validação por bloco é a guarda). Q/K/V/out e MLP todos em W4A4 (alvo = a W4A4 pura, a mais rápida).

## Resultado (27/09, ~04:40)
- VM: Colab G4 (RTX PRO 6000, 95 GB), ~45 min (2301 s de script: 271 s de captura, ~50 s por bloco, holdout).
  Checkpoint privado em HF `JoaoZaokk/qwen21-w4a4-qat`; local em `P:\ComfyBench\diffusion_models\
  qwen_image_2.1_bf16_w4a4_qat.safetensors` (3,77 GB). `verify_w4a4 --structural-only`: PASS (192 convrot_w4a4,
  bytes preservados iguais à fonte).
- **Q1 confirmada**: 32/32 blocos aceitos; erro de validação por bloco −8,5% em média (−1,9% a −51,5%).
- **Q2 fraca**: holdout fim a fim 0,3109 → 0,3071 (−1,2%); 3 prompts melhoram, 3 pioram — a métrica de latente final
  é dominada pela divergência caótica da trajetória.
- **Q3 (render, 3080 Ti, RTN e QAT na mesma placa, 12 imagens cada; `fase_qat_metricas.json`, `folhas_qat/`)**:
  MS-SSIM médio contra bf16 0,818 → **0,843** (mín 0,690 → 0,740), SSIM 0,722 → 0,791, grão 1,02 → 0,93; QAT vence
  7/12. Visual: **pele** craquelada → lisa, próxima do BF16; **maçãs** (RTN inventa 5 maçãs + 3 peras e posteriza) →
  composição do BF16, textura limpa; **perfume** mais fiel. **Neon**: misto (s7 mais perto do BF16, s42 pior); o
  traço duplo NÃO some. Aposta confirmada: pele melhora, neon não.
- Velocidade: igual ao RTN (mesmo formato). A 3080 Ti caiu de ~3,2 para ~2,8 it/s no meio da bateria do RTN e ficou
  assim (ambiental, não do QAT).
- Próximo, se o dono quiser: mais passos por bloco (o bloco 5 ainda caía em 300), mais amostras com texto/neon, e o
  mesmo QAT sobre o mixed 0,10 (só Q/K em W4A4) — o neon é MLP, e ali o W4A8 já resolve.
- MiMo cego (`qwen21_2026-09-26/mimo_qat/`, 12 imagens, ruído 0,75): QAT × RTN — artefatos **7,67 × 6,67**, detalhe
  7,33 × 6,50, anatomia 7,20 × 6,56, aderência 6,83 × 6,75, **texto 4,5 × 6,5 (pior)**, geral 6,67 × 6,50 (empate);
  ranking com bf16: bf16 1,25 · RTN 2,25 · QAT 2,5. Bate com o visual: o QAT tira artefato e conserta pele/textura,
  mas escreve pior. Veredito: ganho real e parcial; não substitui o mixed 0,10 (neon e texto limpos com W4A8 na MLP).
