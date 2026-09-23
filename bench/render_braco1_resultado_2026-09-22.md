# Render do braço 1: generaliza no epsilon, não vira imagem. E o 63% congelado que explica parte

Critério em `bench/criterio_render_braco1_2026-09-22.md`, escrito antes de qualquer número e não
alterado (os desvios estão na seção 5). 3090, lock por ferramenta, `CUDA_VISIBLE_DEVICES=0`.
5 prompts (F0-F3 fora do ajuste, D0 dentro) × sementes 11 12, 8 passos, cfg 1.0, euler/simple,
1024 px (latente 64x64), `qwen_3_4b` + `--clip-type flux2`, VAE diffusers do klein
(`flux2_klein_vae_diffusers.safetensors`, cópia do original; o ComfyUI a carrega: 128 canais, /16,
`bn` presente). Grades: `render_braco1_2026-09-22/grade_comfy.png`, `grade_diffusers.png`.

## 1. Placar

    R1  BF16 10/10 coerente (controle que tem de passar)      PASSOU      10/10
    R2  b0 <= 2/10 reconhecivel (controle que tem de falhar)   PASSOU      0/10, bege liso
    R3  b2 Bonsai >= 9/10                                       CONFIRMADA  10/10
    R4  b1 >= 6/10, e >= 5/8 fora                               REFUTADA    ~3/10; 1/8 fora
    R5  latente: b2 < b1 < b0                                   CONFIRMADA  0,751 < 0,905 < 1,056, pareado 10/10
    E1  fora da amostra >= 25% da distancia                     CONFIRMADA  54,17%  (min 49,97 max 57,90)
    E2  fora >= 0,6 x dentro                                    CONFIRMADA  0,90
    E3  controle dentro 50-70%                                  PASSOU      59,90% (reproduz 60,06%)
    E4  b2 > b1 em 10/10                                        CONFIRMADA  faixas nao se sobrepoem
    X1  BF16 no diffusers 10/10                                 PASSOU
    X2  ordem b1 < b0 no diffusers                              CONFIRMADA  SSIM 0,554 contra 0,495
    X3  razao d0/d1 dentro de 1,5x entre runtimes               CONFIRMADA  1,156 ComfyUI, 1,132 diffusers

## 2. O que o render diz

O braço 1 **não gera imagem utilizável**: forma borrada na direção certa (vulto do filhote, silhueta
da vila, maçã como oval rosado), nada que alguém reconheça sem ler o prompt fora da amostra. O
Bonsai gera imagem quase igual à do BF16 **na mesma composição da mesma semente** — filhote, placa,
vila, bule. [JULGAMENTO] isso sugere treino por destilação contra o próprio klein.

**O "60% da distância" é distância no epsilon, e o epsilon não é linear na imagem.** b1 fora da
amostra fica em rel-RMSE 0,68; o Bonsai em 0,35 já rende como o BF16. O limiar em que a imagem
presta está entre esses dois números, e a imagem precisa de quase todo o resto.

## 3. Epsilon fora da amostra (o número que faltava)

    fora (F0-F3, 8 corridas)   b0 1,0592   b1 0,6771   b2 0,3546   fracao 54,17%  (sigma alto 48,02%, baixo 59,98%)
    dentro (D0, 2 corridas)    b0 0,9873   b1 0,5650   b2 0,2824   fracao 59,90%

O ajuste **não decorou** os 4 prompts: perde ~6 pontos fora deles. O prompt da maçã é mais fácil
para todos os braços. A P4 (ganho concentrado em sigma alto) cai pela terceira vez na mesma direção.

## 4. Cruzamento diffusers × ComfyUI — fechado, e mais forte do que o critério pedia

O critério supunha ruído inicial diferente entre runtimes e só comparava distâncias dentro de cada
um. **Suposição falsa**: o mesmo braço nos dois runtimes, mesma semente, dá

    BF16  SSIM 0,862 (min 0,761)  PSNR 20,9 dB
    b1    SSIM 0,983 (min 0,957)  PSNR 33,7 dB
    b0    SSIM 0,988              PSNR 41,1 dB

**O b1 é o mesmo modelo nos dois runtimes.** Ajustar num e medir no outro não contaminou nada. O
BF16 concorda menos que o b1 porque imagem nítida amplifica a diferença numérica mínima entre as
implementações; imagem borrada a absorve.

## 5. Desvios e defeitos, com a hora

1. **Furo da amostra** (antes de rodar): o epsilon de 60% usava o prompt 0 do ajuste. Medido fora
   agora (seção 3).
2. **"512 px" era 1024 px** (antes de rodar): `SIDE // 8` num formato /16. Corrigido em
   `probe_epsilon_per_step.py` e `quality_ladder.py`. O ajuste rodou a 512 px e a medição a 1024 px.
3. **Decode no mesmo processo do ladder morreu** (`hostbuf_allocate`, defeito conhecido); decodificado
   com `decode_latents.py` na 3080 Ti.
4. **Vazamento de VRAM na troca de transformer** do `render_klein_diffusers.py`: 24,27/24,58 GB, o WDDM
   paginando a 46-75 s/passo. Processo morto (meu), lock morto apagado à mão, ferramenta corrigida
   (`.to("cpu")` antes de trocar; 14,93 GiB alocados depois). Braço 0 re-renderizado.
5. **Afirmei sem medir que o ruído diferia entre runtimes** (critério e duas docstrings). Corrigido nas
   docstrings; o critério fica como estava, com este desvio registrado.

## 6. O que isto abriu

Ao explicar por que bf16 como peso mestre é arriscado, medi o próprio braço 1: **63,4% dos
195.035.136 elementos treináveis nunca mudaram** em 1.920 passos de Adam (bf16 puro, lr 1e-5), e a
fração congelada cresce com |w| (16,4% mudou em `context_embedder`, 79,7% em `norm_out.linear`).
Critério do A/B em `bench/criterio_mestre_dtype_2026-09-22.md`.

## Não coberto

Uma placa; 2 sementes por prompt; julgamento de "reconhecível" meu; LPIPS não medido (VGG fora do
cache); nenhuma métrica perceptual validada nesta bancada.

## 7. A/B do peso mestre (critério `criterio_mestre_dtype_2026-09-22.md`), mesma noite

Um eixo: o dtype do mestre no ajuste do braço 1. Dados, épocas, lr e tensores idênticos ao b1.

    M1  elementos mudados >= 95%              CONFIRMADA  fp32 96,13%  sr 97,04%  (b1: 36,59%)
    M2  b1f >= 10% abaixo do b1 (epsilon)     REFUTADA    6,1%  (0,677 -> 0,636), mas 8/8 e 15,6x o ep
    M3  sr dentro de 10% do fp32              CONFIRMADA  0,5%  (0,639 contra 0,636)
    M4  perda final < 0,3688                  CONFIRMADA  fp32 0,1996  sr 0,2011
    M5  render fora: mais reconheciveis que b1 CONFIRMADA ~4/8 contra 1/8

Fração da distância b0→b2 fora da amostra: b1 54,17% → b1f 60,04% / b1s 59,57%.
Grade: `render_braco1_2026-09-22/grade_mestre.png`.

**O mecanismo era o arredondamento**, e bf16 + arredondamento estocástico vale o fp32 (perda 0,7%,
epsilon 0,5%). Consequência prática: o QAT inteiro cabe em ~21,7 GiB de peso+grad+estado.

**A imagem mudou muito mais que o epsilon.** O b1 era um borrão cinza; b1f/b1s são nítidos, com cor
e estrutura: vila ao anoitecer com janelas acesas nas duas sementes, placa de madeira com letras
ilegíveis, maçã vermelha nítida. Filhote e bule ainda não: fragmentos laranja, folhas soltas.

**E a divergência de latente PIOROU enquanto a imagem melhorou** (0,905 → 0,926, decisão dividida
2/10). Borrão é mudança pequena de latente: a métrica elogia a falha macia. É a regra do Wan 2.2
deste repo, agora vista pelo outro lado — um build nítido e errado é penalizado contra um borrado.

A perda do ajuste caiu 1,84x e o epsilon só 6%: os 9 densos estão perto do teto deles. O que falta
está no corpo — QAT, `bench/criterio_qat_klein_2026-09-22.md`.
