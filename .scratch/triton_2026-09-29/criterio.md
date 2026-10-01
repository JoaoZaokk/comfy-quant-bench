# Critério: `--enable-triton-backend` nos launchers + validação GPU da revisão (2026-09-29)

Escrito antes de medir. Pedido do dono: "mede na GPU e coloca a flag nos launchers, verifica os que quebram com triton e arruma".

## Contexto lido (não medido)

- ComfyUI desliga o backend triton do comfy-kitchen sem `--enable-triton-backend` (`comfy/quant_ops.py:34-42`).
- Prioridade do registry: cuda > triton > eager. O backend cuda implementa todas as ops do triton exceto
  `dequantize_awq_w4a16`; com a flag, o triton só entra no AWQ e onde as constraints do cuda recusam a chamada.
- A revisão tirou o import direto do triton no patch AWQ do kitchen: sem a flag, o AWQ Q4_1 cai no eager (lento).

## Execução

- Placa: RTX 3090 (`CUDA_VISIBLE_DEVICES=0`), lock `comfy:validacao_triton_<flag>` via Assert-GpuLock, bloco de 30 min por flag.
  3080 Ti (em uso por outro processo) não é tocada.
- Servidor: launcher do dono (`--windows-standalone-build --use-sage-attention`, dynamic VRAM ligado) + `--disable-pinned-memory` nos dois modos (Qwen vem do NAS; o abort de 27/09 prende a placa); Q2 BF16 fora, porta 8199,
  com `COMFYUI_MGPU_DISABLED=1` (o multigpu-orchestrator mandaria o prompt a um worker sem as flags), via `comfy_contador.py`, que conta (op, backend) escolhido pelo registry do kitchen durante os renders.
- Mesmos grafos com a flag desligada (OFF) e ligada (ON). Cada grafo roda frio (seed de referência) e quente (seed+1).

Grafos:
- Q1 Qwen 2.1 Q4_1 AWQ nativo (`w4a16_q4_1_nativo` p0 s42), TE qwen3vl_8b_w4a8 (caminho novo do TE quantizado).
- Q2 Qwen 2.1 BF16; Q3 nosso_w4a8; Q4 nosso_w4a4; Q5 nosso_int8 (p0 s42).
- K1 klein BF16 ternário (UNETLoader, rope flux2); K2 klein ternário MLX (LowBitDiffusionLoader, costura nova).
- Z1 Z-Image turbo BF16; Z2 Z-Image turbo W4A4 (rms_rope lumina). Sem referência anterior: só ON × OFF.
- Testes unitários de GPU: `patches/tests/test_comfy_kitchen_awq.py` (triton × eager) e `test_lowbit.py` com opt-in.

## Critérios

C1 (não quebra) ON: todos os grafos terminam com status success, 0 erro de compilação triton, 0 exceção no log.
C2 (fidelidade ON × OFF): imagem ON idêntica à OFF por sha256 dos pixels, ou MS-SSIM ≥ 0,99 onde o contador mostrar
    op desviada para triton (explicar cada caso).
C3 (regressão da revisão): OFF de Q1-Q5 e K1-K2 contra as referências anteriores à revisão (qwen21_bateria/<braço>/p0_s42,
    lowbit_2026-09-27/principal/<braço>/p0_s11): idêntica ou MS-SSIM ≥ 0,99; abaixo disso é regressão do core.
C4 (desempenho): Q1 ON quente com it/s ≥ 1,5× o OFF (esperado: kernel triton de 27/09 contra eager); nos demais,
    tempo de amostragem ON dentro de ±5% do OFF.
C5 (unitários GPU): paridade triton × eager do AWQ e lowbit bit a bit passam.

Se um grafo falhar só com ON: isolar a op pelo contador, corrigir (constraint/registro) ou, se não houver correção segura,
tirar a flag só do launcher daquele uso e registrar.
