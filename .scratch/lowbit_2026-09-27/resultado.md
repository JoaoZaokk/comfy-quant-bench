# Resultado: loader nativo low-bit (Bonsai ternário/binário), 2026-09-27

Critério em `criterio.md`, escrito antes. RTX 3090 (cuda:0) salvo onde indicado; klein-4B, 1024², 4 passos,
euler/simple, cfg 1, 2 prompts x 2 sementes (11, 12) por braço; it/s = mediana das 3 execuções quentes.

| P | previsão | medido | veredito |
|---|---|---|---|
| P1 | 100 camadas x 6 arquivos bit a bit ao unpacked; controle falha | 600/600 idênticas; controle ternário vs unpacked binário 0/100 (1,37% dos elementos) | confirmada |
| P2 | Triton == torch bit a bit | 1 e 2 bits, bf16/fp16/fp32, 3072x3072: iguais | confirmada |
| P3 | MODEL real, camadas no layout, nada sobrando | 80 camadas `lowbit_affine` (100 diffusers, q/k/v fundidos), 0 erros em 40 renders | confirmada |
| P4 | render = BF16, MS-SSIM >= 0,999 | ternário MLX, gemlite e unpacked: 4/4 imagens **idênticas byte a byte** ao BF16; binário: os 3 packs idênticos entre si | confirmada |
| P5 | DiT <= 45% (tern.) / <= 35% (bin.) do BF16 | 1359 MB (18,4%) / 920 MB (12,5%) vs 7392 MB; gemlite 1468 / 1030 (escalas fp32) | confirmada |
| P6 | 0,85-1,0x do BF16, residente | `--disable-dynamic-vram`: BF16 2,52, ternário 2,42 (0,96x), binário 2,43 it/s | confirmada |
| P7 | offload completa, imagem idêntica à P4 | `--novram`: completa; ternário idêntico ao BF16 **no mesmo modo** (4/4); contra a fase principal 0/4 idênticas, MS-SSIM 0,9974, **igual para o BF16** -> efeito do modo, não do loader | confirmada com a ressalva |
| P8 | caminho CPU bate | P1 inteira rodou no caminho torch em CPU | confirmada (sem render em CPU) |

## Velocidade por modo (it/s)

| modo | BF16 | ternário | binário |
|---|---|---|---|
| padrão do .bat (dynamic VRAM, TE na mesma placa) | 2,01 | 2,47 (1,23x) | 2,47 |
| tudo residente (`--disable-dynamic-vram`) | 2,52 | 2,42 (0,96x) | 2,43 |
| offload forçado (`--novram`) | 0,93 | 1,59 (1,70x) | 1,65 (1,77x) |
| DiT na 3080 Ti (cuda:1), TE na 3090 | - | 2,07 | - |

Hipótese (não medida à parte): no modo padrão o dynamic VRAM não mantém o BF16 inteiro residente
enquanto o text encoder (7,7 GB) ocupa a placa; o ternário cabe. No offload, a banda de PCIe move 1-2 bit.

## Não coberto

- Uma resolução, 4 passos, 2 prompts. Render em CPU. LoRA. Arc A770 (sem placa).
- Log do ComfyUI mostra erro de disco em `L:\` (WinError 1393) ao listar caminhos extras: não é deste trabalho.
