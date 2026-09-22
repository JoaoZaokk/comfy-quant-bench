# Critério — render do braço 1, epsilon FORA da amostra e o cruzamento diffusers × ComfyUI

Escrito 2026-09-22, **antes de qualquer número desta rodada**. Não se altera depois de medir; o que
mudar vai numa seção "desvios" com a hora.

## Por que esta rodada existe — dois furos achados ao preparar o render

1. **O 60,06% foi medido DENTRO da amostra.** O probe de epsilon usou o prompt default
   `"a red apple on a weathered wooden table, soft window light"`, que é o prompt 0 dos QUATRO do
   ajuste (`tools/ajusta_denso_diffusers.py:57`). O ruído inicial difere (ComfyUI e diffusers geram
   ruído diferente para a mesma semente), mas o condicionamento é o mesmo que o ajuste viu. O
   resultado publicado não diz isso. Nenhum número fora da amostra existe.
2. **O "512 px" era 1024 px.** `probe_epsilon_per_step.py` fazia `SIDE // 8`; o FLUX.2 reduz 16x
   (`latent_formats.Flux2.spacial_downscale_ratio = 16`). Com `--size 512` o latente saiu 64x64,
   ou seja **imagem de 1024 px** — o log mostra `saida [1, 128, 64, 64]`. O ajuste rodou a
   `--size 512` no diffusers, que É 512 px. **Ajuste a 512, medição a 1024**: resolução fora da
   distribuição do ajuste, o que corre CONTRA o resultado, e muda o que o número significa.
   Corrigido nas duas ferramentas (`probe_epsilon_per_step.py`, `quality_ladder.py`); para
   reproduzir o latente 64x64 de antes, a chamada agora é `--size 1024`.

## Braços

    braco3  klein4b_braco3_bf16_original_bfl      referência (o alvo)
    braco0  klein4b_braco0_ternario_ingenuo_bfl   PTQ ingênuo, 1,58 bit
    braco1  klein4b_braco1_compensado_bfl         corpo ternário + 9 densos ajustados
    braco2  klein4b_braco2_bonsai_ternario_bfl    Bonsai treinado (o teto)

O controle de ruído fica de fora: já respondeu o que tinha de responder (8/8 sementes).

Arquivos lidos de cópia local em `ComfyUI/models/diffusion_models` (F:, mesmo nome, tamanho
conferido) em vez do NAS — decisão de custo, não de medida.

## Prompts

Fora do ajuste (nenhum dos quatro aparece em `PROMPTS` do ajuste):

    F0  a golden retriever puppy sleeping on a blue knitted blanket
    F1  a hand-painted wooden sign reading OPEN above a bakery door, morning sun
    F2  a snowy mountain village at dusk, warm lights in the windows
    F3  a glass teapot with green tea leaves unfurling, studio photo, white background

Dentro do ajuste, como CONTROLE da rodada:

    D0  a red apple on a weathered wooden table, soft window light

Sementes **11 e 12**, que nenhuma medição anterior usou. 8 passos, cfg 1.0, euler/simple,
`qwen_3_4b` + `--clip-type flux2`, **1024 px** (latente 64x64, o mesmo das medições anteriores).

## E — epsilon, trajetória do BF16 imposta (`probe_epsilon_ckpt_ab`, agregado pelo `agrega_epsilon_sementes`)

5 prompts × 2 sementes = 10 corridas pareadas.

- **E1 (a pergunta).** Nos prompts FORA (F0-F3, 8 corridas), o braço 1 fecha **≥ 25%** da distância
  braço0 → braço2. Mesmo limiar da P2. Refutada se < 25%.
- **E2 (generalização).** Fração fora ≥ **0,6 ×** fração dentro (D0, medida na mesma rodada).
  Refutada se abaixo: o ajuste estaria decorando o condicionamento dos 4 prompts.
- **E3 (controle da rodada).** D0 nas sementes novas reproduz a faixa já medida: fração entre
  **50% e 70%**. Fora disso a rodada não é comparável com a anterior e E1/E2 não se leem.
- **E4.** Braço 2 continua mais fiel que o braço 1 em todas as 10 corridas.

Previsão numérica [JULGAMENTO], para ficar registrada: fora da amostra ~45-55%.

## R — render livre pelo ComfyUI (`quality_ladder`, VAE diffusers do klein)

Mesmos 5 prompts × 2 sementes, 4 braços = 40 imagens. Julgamento por olho numa grade, critério
"reconhecível" = o objeto principal do prompt identificável por quem não leu o prompt.

- **R1 (controle que tem de passar).** Braço 3 BF16: 10/10 coerentes e fiéis ao prompt. Se falhar,
  a VAE ou o caminho estão errados e **nada abaixo se lê**.
- **R2 (controle que deve falhar).** Braço 0 (rel-RMSE ~1): ≤ 2/10 reconhecíveis. Se o braço 0
  renderizar bem, o epsilon deste modelo não acompanha a imagem e o 60% mede algo que o olho não
  vê — isso seria o achado da rodada.
- **R3.** Braço 2 Bonsai: ≥ 9/10 reconhecíveis.
- **R4 (a pergunta).** Braço 1: **≥ 6/10 reconhecíveis** nos 10, e ≥ 5/8 nos prompts FORA.
  Refutada se ≤ 3/10.
- **R5.** Distância ao BF16 na mesma semente (latente rel-L2 e SSIM/PSNR da imagem), média:
  braço2 < braço1 < braço0. Ressalva registrada antes: trajetória livre a 8 passos é caótica e
  este repo já mostrou que ela não ordena dois quants PARECIDOS; aqui os braços distam 1,75x-3,5x
  no epsilon, então a ordem deve aparecer — se não aparecer, é a ressalva, não o braço.

## X — cruzamento: o mesmo peso no diffusers e no ComfyUI

O braço 1 foi AJUSTADO no diffusers e MEDIDO no ComfyUI. Render dos braços 3, 1 e 0 no
`Flux2KleinPipeline` do diffusers (nomenclatura diffusers, `P:/ComfyBench/originais/…`), mesmos
prompts, sementes, passos, 1024 px.

- **X1 (controle).** Braço 3 no diffusers: 10/10 coerentes.
- **X2.** Dentro do diffusers a ordem se mantém: d(braço1, BF16) < d(braço0, BF16) na média de
  SSIM da imagem. Refutada se inverter.
- **X3.** Razão d(braço0)/d(braço1) no diffusers dentro de **1,5x** da mesma razão no ComfyUI.
  Refutada fora disso: os dois caminhos veriam modelos diferentes.

O ruído inicial NÃO é o mesmo entre os dois runtimes (cada um gera o seu), então X compara
distâncias DENTRO de cada runtime, nunca imagem do diffusers contra imagem do ComfyUI.

## Não coberto, já sabido antes de rodar

Uma placa, 8 passos, 2 sementes por prompt, 5 prompts; nenhuma métrica perceptual treinada
(LPIPS não baixado); o julgamento "reconhecível" é meu e da grade que vai para ele. O BF16 é o
alvo, não a verdade.
