# Critério — QAT ternário do klein-4B, corpo treinável, destilação contra o BF16 (noite de 2026-09-22)

Escrito **antes do smoke e antes de qualquer passo de QAT**. Não se altera depois; desvios vão
numa seção com a hora. Autorizado pelo dono ao sair: "vai fazendo por enquanto aqui na placa (...)
se quiser alargar os batches, fica a vontade".

## O que já se sabe (e motiva)

- Braço 1 (só 9 densos, corpo congelado): fecha 54% da distância b0→b2 no epsilon fora da amostra,
  **render não presta** (1/8 reconhecível fora da amostra).
- Mestre bf16 sem SR congelava 63% das atualizações; com fp32 ou bf16+SR muda ~96-97% e a perda do
  ajuste cai 1,84x — mas o epsilon cai pouco (primeira corrida: 0,663 → 0,613), sinal de que 9
  tensores estão perto do teto deles.
- O Bonsai treinou o corpo (2,7 M de sinais invertidos).

## Configuração (escolhas MINHAS, registradas como tal)

`tools/qat_ternario_klein.py`, aluno inicializado do BF16 (sugestão do GPT via dono, aceita),
corpo ternário por STE (absmean g128 eixo K, escala L2 recalculada — o mesmo quantizador do b0 e
do Bonsai), denso em precisão normal, `--otim adamw8bit-sr` (peso bf16 + AdamW8bit do torchao com
arredondamento estocástico), lr 1e-5 (a do ajuste, por continuidade), 124 prompts de treino × 2
sementes × 8 passos a 1024 px = 1.984 exemplos; holdout 12 prompts × 2 sementes (inclui F0-F3).
Lote decidido pelo smoke.

## S — smoke (20 passos, 2 prompts)

- **S1.** Cabe na 3090 sem paginar: `vram_pico_gib` ≤ 22,5 e s/passo estável entre o 1º e o 2º
  bloco de log (razão < 1,5x). Refutada → o braço inteiro vai para o Colab.
- **S2.** Perda finita e holdout do passo 0 no nível do b0 (é o mesmo modelo: ternário do BF16).

## Q — a noite

- **Q1 (controle interno).** A perda de holdout desce. Se subir por 3 medições seguidas enquanto a
  de treino desce, é sobreajuste e o checkpoint anterior é o que vale.
- **Q2 (o corpo se move).** Fração de códigos diferentes do inicial > 0 e crescente.
  [JULGAMENTO] prevejo entre 0,1% e 10% no fim. Se ficar em ~0, o lr é baixo demais para o corpo e
  o resultado vira "o ajuste denso de novo".
- **Q3 (a pergunta, epsilon).** Epsilon fora da amostra no protocolo do ComfyUI (F0-F3, sementes 11
  12, 1024 px, trajetória do BF16 imposta): o QAT fecha **≥ 75%** da distância b0→b2. Refutada se
  ≤ 65% (não melhor que o ajuste denso com mestre corrigido).
- **Q4 (a pergunta, imagem).** Render fora da amostra: **≥ 5/8 reconhecíveis**. Refutada se ≤ 2/8.

Não coberto, já sabido: uma noite, uma configuração, um lr, sem repetição; o professor grava na
trajetória dele, não na do aluno; BF16 é o alvo, não a verdade.

## Desvios, com a hora

- **22:08 — S1 REFUTADA para `adamw8bit-sr`, lote 1.** Pico alocado 23,47 GiB; a placa em 24.312/24.576
  MiB a 190-230 W e 20-26 s/passo: WDDM paginando (memória `reference-windows-sysmem-fallback` do
  frankestein: estourar a VRAM aqui não dá OOM, dá lentidão). Pelo critério, este braço vai para a
  A100 do Colab amanhã. Perda de treino no passo 5: 0,824 (holdout do passo 0: 1,358).
- **22:12 — braço NOVO testado localmente, fora da configuração acima:** `adamw4bit-sr` (AdamW4bit do
  torchao com arredondamento estocástico), ~3,6 GiB a menos. É o eixo 8×4 bits que o dono perguntou
  mais cedo. Mesmos S1/S2 e, se couber, mesmos Q1-Q4 — mas comparado ao 8-bit da A100 só depois, e
  não é a configuração que o critério escreveu.
