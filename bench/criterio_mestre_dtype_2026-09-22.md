# Critério — o peso mestre do ajuste do braço 1: bf16 contra fp32 contra bf16 + arredondamento estocástico

Escrito 2026-09-22, **antes de rodar qualquer ajuste novo**. Não se altera depois.

## O achado que motiva

Pergunta dele ("por que não bf16/bf16?") levou à medição: no braço 1, ajustado em bf16 puro com
Adam do PyTorch (estado também bf16), lr 1e-5, 64 exemplos × 30 épocas = 1.920 passos, **63,4% dos
195.035.136 elementos treináveis terminaram byte a byte iguais ao braço 0**. A fração congelada
cresce com |w| (`context_embedder` |w| mediana 1,15e-2 → mudou 16,4%; `norm_out.linear` 1,2e-3 →
79,7%), que é o padrão de arredondamento: o passo do Adam (~lr) fica abaixo de meio-ulp do bf16.
O mecanismo é [JULGAMENTO] até este A/B.

## Um eixo

`tools/ajusta_denso_diffusers.py --mestre {fp32, bf16-sr}`. Tudo o mais IDÊNTICO ao braço 1:
aluno `klein4b_ternario_ingenuo`, professor BF16, 4 prompts × sementes 1 2 × 8 passos a 512 px,
30 épocas, lr 1e-5, mesmos 9 tensores. O braço bf16 é o braço 1 já no disco (não re-rodado).

    b1      mestre bf16        (existente)  klein4b_braco1_compensado
    b1f     mestre fp32 + autocast bf16     klein4b_braco1f_mestre_fp32
    b1s     bf16 + SR (torchao _AdamW)      klein4b_braco1s_bf16_sr

## Previsões

- **M1 (mecanismo).** b1f e b1s mudam **≥ 95%** dos elementos no bf16 gravado. Refuta o mecanismo
  se ficarem perto dos 36,6% do b1.
- **M2 (a pergunta).** Epsilon fora da amostra (protocolo E do `criterio_render_braco1`: prompts
  F0-F3, sementes 11 12, 1024 px, trajetória do BF16 imposta): b1f **≥ 10% menor** que b1.
  Refutada se não melhorar.
- **M3 (o caminho local do QAT).** b1s dentro de **10%** do b1f no mesmo epsilon. Se passar,
  arredondamento estocástico basta e bf16+SR é candidato para o QAT inteiro na 3090.
- **M4.** Perda final do ajuste: b1f < 0,3688 (a do b1).
- **M5 (render).** Nos 8 renders fora da amostra, b1f com mais reconhecíveis que o b1 (1/8).
  [JULGAMENTO] prevejo que ainda fica abaixo do Bonsai (8/8): isto testa o dtype, não o QAT.

## Não coberto, já sabido

Um ajuste por braço, sem repetição de semente do otimizador; ajuste a 512 px medido a 1024 px
(o mesmo defeito do b1, mantido de propósito para variar um eixo só).
