# W8A8 sem rotação, W8A16 e W4A16 do Qwen-Image-2.1 — critério (escrito antes de medir, 27/09)

Pedido do dono: "testa o w8a8, w8a16 e w4a16. Depois vemos se sobe ou nao." Sem upload até decisão.

## Builds
- **W8A8 rowwise** (`qwen_image_2.1_bf16_int8.safetensors`): `tools/quant_int8.py --no-convrot`, int8 por linha
  no peso e na ativação (layout `int8_tensorwise`, kernel int8 nativo), SEM a rotação Hadamard. A nossa "int8 ConvRot"
  já medida é o W8A8 COM rotação; este braço mostra o que a rotação compra.
- **W8A16** (`qwen_image_2.1_bf16_Q8_0.gguf`): GGUF Q8_0 (int8, escala fp16 por bloco de 32), `tools/quant_gguf.py`
  (novo). Carregado pelo ComfyUI-GGUF (`UnetLoaderGGUF`), que desquantiza o peso para BF16 antes de cada matmul:
  ativação BF16, conta BF16.
- **W4A16** (`qwen_image_2.1_bf16_Q4_1.gguf`): GGUF Q4_1 (uint4 com escala e mínimo fp16 por bloco de 32).
  Q4_K_M (o mais usado pela comunidade) não dá para gerar sem o llama.cpp (gguf-py não implementa a quantização
  K); Q4_1 é o W4A16 assimétrico mais próximo que dá para gerar aqui.
- Mesmas 192 lineares de todos os outros builds; o resto fica BF16. Quantização a partir do FP32.

## Medição
Mesma bateria (6 prompts × seeds 42/7, 1024², 25 passos, euler/simple, cfg 1), RTX 3090, `--disable-dynamic-vram`,
um boot. Métricas contra o BF16 de 26/09 (runtime determinístico: bit a bit igual entre boots). Controle: rerender da
nossa int8 ConvRot p0_s42 no mesmo boot, tem de sair bit a bit igual à de 26/09.

## Previsões
- W8A16 Q8_0: fidelidade ≥ int8 ConvRot (bloco de 32 com escala própria é mais fino que escala por linha, e a
  ativação não é quantizada) → MS-SSIM ≥ 0,99. Velocidade: ~BF16 ou abaixo (desquantizar + matmul BF16), ~1,0 it/s.
  VRAM de pesos ~7,2 GB.
- W4A16 Q4_1: fidelidade ENTRE W4A8 e int8 — o peso de 4 bits é o mesmo tipo de erro da W4A8, mas sem erro de
  ativação; e sem o traço duplo no neon (o artefato vem da ativação de 4 bits na MLP). Velocidade ~BF16 ou abaixo.
  ~4,3 GB.
- W8A8 rowwise: velocidade = int8 ConvRot (mesmo kernel sem a rotação: talvez um pouco mais rápido); fidelidade
  pior que a ConvRot (outliers de canal na ativação sem rotação), quanto pior é o que se mede.

## Decisão
Recomendar algum para subir só se trouxer algo que as versões já publicadas não trazem: ex. W4A16 com qualidade de
W4A8 ou melhor em ~4 GB (útil para quem não tem kernel int4/int8), ou W8A16 mais fiel que a int8. Velocidade abaixo
do BF16 é aceitável para weight-only (o ganho é VRAM), mas tem de ser declarado.

## Resultado (27/09 ~14:45)
Ver W4A4_PROGRESS Parte 62 e `qwen21_2026-09-26/fase_pesos_so_metricas.json`. Previsões: W8A16 ≥ int8 ConvRot — NÃO (0,987 × 0,995, um outlier de trajetória); W4A16 entre W4A8 e int8 — NÃO (0,912, abaixo da W4A8), mas sem traço duplo — SIM; W8A8 rowwise pior que ConvRot — SIM (0,976). Velocidades: GGUF abaixo do BF16 (0,94/0,83 it/s), W8A8 rowwise 2,35.
