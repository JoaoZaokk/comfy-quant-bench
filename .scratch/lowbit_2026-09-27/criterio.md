# Critério: loader nativo low-bit (Bonsai ternário/binário, packs gemlite/MLX/unpacked)

Escrito antes de qualquer número do loader. 2026-09-27.

## O que já foi medido (sondas desta sessão, CPU, sem GPU)

- gemlite int2 e int1: `w = q*s + z`, q desempacotado intercalado LSB-first ao longo de K,
  pack transposto (K/r, N), s/z fp32 (K/128, N). Reproduz o unpacked **bit a bit** (fração idêntica
  1,000000, relL2 0) em 2 camadas por variante; as outras 7 combinações de ordem/fórmula falham
  (0-34% idênticos), o que serve de controle negativo da sonda.
- MLX 2bit/1bit contra unpacked: relL2 0 em 8/8 camadas (bench/bonsai_packs_resultado_2026-09-22.md).

## Desenho

Um layout só, `LowBitAffineLayout` (formato `lowbit_affine`): qdata uint8 (N, K*bits/8) LSB-first ao
longo de K; scale e zero (N, K/G) no dtype da fonte; `W = code*scale + zero`. bits e G derivados dos
shapes. Registro pelo custom node, sem editar arquivo do core. Peso fica empacotado na VRAM e no
offload; matmul BF16 depois de desquantizar (A16): **velocidade no máximo igual ao BF16**; o ganho
esperado é VRAM e banda de offload.

## Previsões e critérios de aceite (escritos antes)

P1. Desquantização do layout, nas 100 camadas de cada um dos 6 arquivos (2 modelos x 3 packs),
    igual bit a bit ao unpacked do mesmo modelo. Controle: ternário contra unpacked binário tem de
    falhar (fração idêntica < 5%).
P2. Kernel Triton == referência torch bit a bit (mesma conta fp32, mesmo arredondamento final).
P3. Loader real do ComfyUI: o MODEL sai do nó, 100 camadas em `LowBitAffineLayout`, 0 camadas
    quantizadas perdidas; nenhuma chave sobrando.
P4. Render klein-4B do Bonsai ternário pelo loader (pack MLX) vs o mesmo modelo convertido para BF16
    (`klein4b_braco2_bonsai_ternario_bfl.safetensors`), mesmo seed/prompt/sampler: imagens idênticas ou
    MS-SSIM >= 0,999 (os pesos desquantizados são bit a bit os mesmos; diferença só pode vir de ordem
    de operação). Se cair abaixo, o loader está errado, não "perdeu qualidade".
P5. VRAM de pesos do DiT: ternário <= 45% do BF16, binário <= 35% (conta: 2,5/16 e 1,5/16 bit por peso
    nas 100 camadas + ~0,39 GB densos). Medir `torch.cuda.memory_allocated` depois do load completo.
P6. Velocidade (3090, 1024², 4 passos, aquecida, >= 3 repetições): loader entre 0,85x e 1,0x do BF16.
    Abaixo de 0,85x = desquantização custando demais; registrar como negativo.
P7. Offload: com `offload=ram` forçado, render completa e imagem idêntica à P4.
P8. CPU: 1 camada roda no caminho torch e bate com a referência (render em CPU não é aceite, só smoke).

## Fora do escopo desta rodada

- Kernel W2A8/W1A8 com ativação int8 (ganho de velocidade real). Só depois de P1-P7.
- LoRA sobre peso low-bit.
- Arc A770: sem placa nesta máquina; só análise.
