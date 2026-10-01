# HANDOFF 2026-09-22 — o braço 1 fechou, e o que vem depois

Estado no fim da janela: **lock livre**, 3090 em 0 MiB / 0%, três commits novos
(`e72b234`, `532abd2`, `30df7a1`), nada empurrado para remoto (nunca foi pedido).

## 1. O que fechou

**`bench/fila2_braco1_resultado_2026-09-22.md`** — leia este primeiro. Placar do critério, 8 sementes:

    P1  eps(braco0) >= 3x eps(braco2)          CONFIRMADA   3,4946x
    P2  braco1 fecha >= 25% da distancia       CONFIRMADA  60,06%   (min 58,31 max 62,06)
    P3  8 sementes, pareado, ep, placar        HONRADA
    P4  o ganho se concentra em sigma ALTO     REFUTADA    56,84% alto contra 63,23% baixo
    P5  bracos 0 e 1 diferem SO no denso       CONFIRMADA  140/149 byte a byte
    controle zero                              PASSOU      169/169 byte a byte
    controle ruido  "tem de PIORAR"            REFUTADA    melhorou 0,39%

A hipótese do estágio de compensação **passou**: corpo ternário congelado, só 9 tensores densos
(5,03% do modelo) ajustados contra a saída do modelo denso, 64 exemplos, 30 épocas → recupera 60%
da distância entre PTQ ingênuo a 1,58 bit e o Bonsai treinado. O braço 2 ainda ganha 64/64 passos
e é 1,996x mais fiel: 40% da distância não fecha.

O bloqueio da fila 1 (somas in-place no flux do ComfyUI) dissolveu porque **ele** perguntou pelo
runtime do Bonsai: eles usam diffusers, que é diferenciável. Nenhum patch foi necessário.

## 2. Ferramentas novas

    tools/ajusta_denso_diffusers.py       braco 1; modos ajuste|zero|ruido, --casar-com obrigatorio no ruido
    tools/compara_checkpoints_byte.py     igualdade em 3 niveis (arquivo / tensor / estrutura)
    tools/agrega_epsilon_sementes.py      o que P3 exige: pareado, erro-padrao, placar por semente

## 3. Artefatos em disco — 64,97 GiB, e é decisão dele

    P:\ComfyBench\originais\          (nomenclatura diffusers)
      klein4b_ternario_ingenuo            7,75 GiB   braco 0, a FONTE dos outros tres
      klein4b_braco1_compensado           7,75 GiB   o resultado
      klein4b_controle_zero               7,75 GiB   DESCARTAVEL: identico ao braco 0, ja verificado
      klein4b_controle_ruido              7,75 GiB   guardar so se for repetir o controle

    P:\ComfyBench\diffusion_models\   (nomenclatura BFL/ComfyUI, o que o ComfyUI carrega)
      klein4b_braco0_ternario_ingenuo_bfl  7,75 GiB
      klein4b_braco1_compensado_bfl        7,75 GiB
      klein4b_braco2_bonsai_ternario_bfl   7,75 GiB
      klein4b_braco3_bf16_original_bfl     7,75 GiB
      klein4b_controle_ruido_bfl           7,75 GiB

**Nada aqui é modelo original** — os originais estão em `F:\bonsai-re\` e
`P:\ComfyBench\originais\FLUX.2-klein-4B\`, intocados. O `controle_zero` é o único
comprovadamente redundante (169/169 byte a byte idêntico ao braço 0, já registrado no commit).
Os demais são reconstrutíveis, mas cada reconstrução custa GPU.

## 4. Próximos passos, em ordem de valor

### A. Render — o buraco que o resultado inteiro tem (GPU, ~30 min)

**Nenhuma imagem foi gerada em nenhum braço.** Tudo é previsão de epsilon. O próprio repo registra
que a imagem livre não compara dois quants do mesmo modelo — mas aqui a pergunta é outra e válida:
*o braço 1 rende uma imagem que presta?* Um PTQ a rel-RMSE 0,56 pode render lixo coerente.

E existe um cruzamento não conferido: **o braço 1 foi ajustado pelo diffusers e medido pelo
ComfyUI.** Corre contra o resultado, não a favor, mas ninguém confirmou que o mesmo peso se
comporta igual nos dois caminhos.

Sugestão: grade de 4 braços × 3 prompts × 1 semente pelo ComfyUI, mais o mesmo pelo pipeline do
diffusers em um braço, para fechar o cruzamento. Critério escrito antes.

### B. O ajuste está subdeterminado e ninguém sabe onde satura (GPU, ~1 h)

64 exemplos para 195 M parâmetros. A perda ainda caía na época 30 (3,835e-01 → 3,689e-01 nas
últimas 5). Varrer **um eixo por vez**: épocas (30 → 100), exemplos (64 → 256 com mais prompts),
`lr`. Se 60% virar 75%, a hipótese fica muito mais forte; se saturar em 62%, o teto do método está
medido e isso também vale.

### C. Replicar fora do klein (GPU + CPU)

`bench/replicabilidade_fora_do_flux_2026-09-22.md` já mediu que a regra estrutural generaliza para
Qwen-Image e Z-Image e degrada em vídeo. O braço 1 existe agora como ferramenta; aplicar em
Qwen-Image responderia se o estágio de compensação é do klein ou do método. **Qwen-Image-2.1 fica
fora**: ComfyUI 0.33 não o carrega (`QwenImage21Transformer2DModel`, detector devolve `None`).

### D. Fechar a P4 com o mecanismo que eu só julguei

[JULGAMENTO] a P4 caiu duas vezes porque em sigma alto o erro do PTQ **satura** (rel-RMSE ~1,07, a
previsão tão longe do BF16 quanto o sinal), e não há o que compensar num regime saturado. Testável:
um PTQ menos agressivo (4 bits em vez de 1,58) onde sigma alto não sature — se o ganho migrar para
sigma alto, o mecanismo está certo.

## 5. Dele, carregado das janelas anteriores

- **Renderizar o 10Eros W4A8** e julgar (`ComfyUI/user/default/workflows/10Eros_v1.5_W4A8_I2V_DMD.json`).
- **Decidir a licença LTX-2 Community** antes de qualquer publicação.
- **Decidir se atualiza o ComfyUI para v0.37**: 5 pacotes, Torch intocado, `comfy-kitchen 0.2.35` é o
  risco, `comfy-aimdo` é o bloqueador, 2 de 5 patches locais precisam ser reaplicados à mão. Backup
  em `F:\COMFY_PORTABLE_BACKUP_2026-09-22_pre_v0.37`.
- **Topologia**: a estrofe morta `ltx23_c` no yaml; `ltx25_w` apontando para a raiz do `W:`.
- **Carry-over**: NVIDIA-Workbench 3,55 GiB; docker 157,4 GB recuperáveis (**podar antes de
  compactar**, e volume por volume — um deles pode ter o rollback do MacroLog); camada 3 do `W:`
  98,23 GiB.

## 6. Aberto, não começado

- **gemlite não está instalado** e não foi lido. Sem ele, nenhum número de tempo de 1,58 bit existe
  — o corpo ternário roda desempacotado em bf16 em tudo que foi medido.
- **GenEval / HPSv3 / DPG do Bonsai nunca replicados.**
- **A armadilha de Hadamard no `Ternary-Bonsai-2-27B-mlx-2bit`** (`_class_name:
  prism_hadamard_qwen35`) continua de pé: **não** apontar o probe de sinal para ele antes de desfazer
  a rotação.
- **Os commits desta sessão não foram empurrados.**
