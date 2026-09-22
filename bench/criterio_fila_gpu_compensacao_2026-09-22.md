# Criterio da FILA DE GPU: o estagio de compensacao e o mecanismo do Bonsai?

Escrito **2026-09-22, antes de rodar qualquer coisa na GPU**, enquanto as duas placas estao em uso
dele. Quando liberarem, roda tudo de uma vez e compara com este arquivo sem mexer nele.

## A hipotese, e por que ela e a unica parte replicavel aqui

O Bonsai **treinou** -- isso esta medido (`bonsai_image_engenharia_reversa.md`) e reforcado pelo
controle positivo de 2026-09-22: um PTQ de verdade do mesmo original deixa o conjunto denso
**66/69 intacto**, o Bonsai deixa **1/69**. Eles reescreveram 68 dos 69 densos.

Treinar 3,68 bilhoes de pesos a 1,58 bit nao cabe nesta maquina, e eles nao publicam passo, dataset
nem GPU-hora. O que cabe e a hipotese que eu levantei lendo isso:

> **Nao e preciso retreinar o corpo. Basta esmagar o corpo para ternario e ajustar SO o conjunto
> denso contra a saida do modelo denso.** No klein-4B isso e 195.042.816 params treinaveis, 5,03% do
> modelo -- cabe numa 3090 com folga.

Se isso for verdade, e o achado pratico mais util de toda esta investigacao, porque **nao exige
treino do zero** e se aplica a qualquer PTQ desta bancada. Se for falso, o Bonsai fez algo maior e
eu paro de sugerir estagio de compensacao.

## Os bracos, e o gabarito

Alvo **klein-4B primeiro**, porque e o unico caso com resposta publicada para conferir.

    braco 0   PTQ ternario ingenuo (absmean, g128 no eixo K), denso intacto     controle INFERIOR
    braco 1   o mesmo PTQ, e depois SO o conjunto denso ajustado contra a
              saida do bf16 num conjunto pequeno de calibracao                  a HIPOTESE
    braco 2   o Bonsai ternario publicado                                       controle SUPERIOR
    braco 3   o bf16 original                                                   REFERENCIA

Sem o braco 0 nao ha o que o braco 1 melhore; sem o braco 2 nao ha teto; sem o braco 3 nao ha eixo.
**Se qualquer um dos quatro nao rodar, o resultado nao se reporta como numero, se reporta como
bloqueio.**

## A metrica, e a que NAO se usa

**Epsilon por passo com trajetoria imposta** (`tools/probe_epsilon_per_step.py`): o braco 3 grava todo
`(x, timestep)` com que foi chamado e os outros tres **repetem exatamente essas entradas**, entao a
divergencia de trajetoria nao existe por construcao.

Proibido como criterio de aceitacao, e este repo ja mediu por que:

- **imagem livre** -- a 8 passos uma perturbacao minima manda o sampler para outro lugar que tambem
  presta; mede caos, nao fidelidade.
- **divergencia de latente sozinha** -- e **enviesada para falha macia**: borrao e uma mudanca
  PEQUENA de latente, entao um braco que suaviza se elogia nessa metrica.
- **erro por camada em espaco de peso** -- ordena formatos e **nao localiza penhasco** (medido: 0,1080
  produziu lixo enquanto 0,1254 funcionou).

## Previsoes numericas, escritas antes

**P1 -- o braco 0 e claramente pior que o braco 2.** Um PTQ ingenuo a 1,58 bit contra um modelo
treinado na grade. Previsao: epsilon medio do braco 0 **>= 3x** o do braco 2.
*Refuta se* ficar abaixo de 1,5x -- e ai o treino deles compra menos do que eu suponho e a pergunta
toda muda.

**P2 (a hipotese) -- o braco 1 fecha pelo menos 25% da distancia entre o braco 0 e o braco 2**, medida
como `(eps0 - eps1) / (eps0 - eps2)`.
*Refuta se* fechar menos de 25%. E **morre** se fechar <= 0, ou se a diferenca entre os bracos 0 e 1
ficar dentro do espalhamento entre sementes do proprio braco 0 -- que e o erro que eu ja cometi neste
repo ao ler um efeito de 3,4% contra um espalhamento de 1,39x como se fosse resultado.

**P3 -- 8 sementes, nao 3.** O espalhamento entre sementes DENTRO de um braco vai a 1,39x nesta
bancada; com 3 sementes o ranking entre bracos vizinhos troca. O relatorio reporta a diferenca
pareada, o erro-padrao e o placar de vitorias por semente, e **nao** reporta uma media sem o
espalhamento ao lado.

**P4 -- o erro se concentra em sigma alto.** Medido antes aqui: 6,17e-1 em sigma 1,000 contra 5,31e-2
em sigma 0,300. Previsao: o ganho do braco 1 tambem se concentra em sigma alto, porque e la que a
estrutura e decidida e onde o denso (modulacao, embedders) manda mais.
*Refuta se* o ganho for plano em sigma, ou concentrado em sigma baixo.

**P5 -- orcamento casado, senao nao vale.** Os bracos 0 e 1 tem de ter **exatamente os mesmos pesos
ternarios**, byte a byte, e diferir SO no conjunto denso. Isto e verificavel e vai verificado com
`torch.equal` antes de qualquer medicao.
*Refuta o experimento inteiro se* diferirem em qualquer peso quantizado -- seria o
`ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` outra vez, e este repo ja o cometeu comparando um
build de 56 promocoes com um de 53.

## O controle que TEM de falhar

**Braco 1-zero: o mesmo ajuste, com o gradiente desligado (zero passos).** Tem de sair
**byte a byte identico ao braco 0**. Se sair diferente, o arnes esta tocando pesos que nao devia e
nenhum numero de P2 vale.

Segundo controle: **braco 1-aleatorio** -- perturbar o conjunto denso com ruido do mesmo tamanho do
ajuste, em vez de ajustar. Tem de **piorar**. Se melhorar, o que o braco 1 mede e regularizacao por
acidente, nao compensacao. E o mesmo controle de orcamento casado que matou a minha hipotese de
canal em 2026-09-21.

## Pre-requisitos, e o que ja esta resolvido

    [ok]     klein-4B em diffusers (original)          F:\bonsai-re\FLUX.2-klein-4B\transformer
    [ok]     Bonsai ternario e binario desempacotados   F:\bonsai-re\bonsai-image-*-unpacked
    [ok]     klein-4B em nomenclatura ComfyUI  7.751.105.712 B, detectado como `Flux2`
    [ok]     VAE do klein                      168.120.878 B
    [ok]     text encoder: klein usa Qwen3ForCausalLM hidden 2560, 36 camadas = Qwen3-4B, e
             `qwen_3_4b.safetensors` esta no disco  (CONFERIR hidden e camadas antes de usar)
    [ok]     **o braco 2 esta DESBLOQUEADO.** O remap diffusers -> BFL foi derivado por sha256 dos
             bytes entre as duas publicacoes do mesmo klein (`tools/deriva_mapa_diffusers_bfl.py`):
             138 um-para-um, 10 fundidos-3 (qkv), 1 permutado (metades de `norm_out.linear`
             trocadas). 149 de 149, 169 de 169, nada nao resolvido. Aplicado ao original ele
             reproduz o `flux-2-klein-4b.safetensors` oficial **byte a byte**, sha256 `ec3d4e73`.
    [ok]     braco 0 e braco 2 escritos em nomenclatura BFL, 7.751.105.712 B cada

Sobra **so o braco 1**, que e o experimento. Tudo que era engenharia de arquivo esta feito.

## O que esta fila nao responde, de nenhum jeito

- Nao replica o treino deles nem as metricas publicadas (GenEval, HPSv3, DPG).
- Se P2 confirmar, mostra que **um** estagio de compensacao curto ganha; nao mostra que e **o** que o
  Bonsai fez.
- Nada aqui mede velocidade: gemlite nao esta instalado e o braco 0/1 nao tem kernel de 1,58 bit.
  O corpo ternario roda **desempacotado** para bf16, entao **nenhum numero de tempo desta fila vale
  como ganho de inferencia.**

---

## Apendice: o braco 0 foi construido e conferido, ainda em CPU

Acrescentado 2026-09-22 depois de construir, **sem mudar nenhuma previsao acima**. Sao fatos do
arquivo, nao resultado do experimento.

`P:\ComfyBench\originais\klein4b_ternario_ingenuo.safetensors`, 7.751.109.744 B -- **a mesma contagem
de bytes do original**, porque os pesos saem desempacotados nos mesmos shapes e dtypes.

    camadas alvo MUDADAS               100/100    como tem de ser
    tensores fora do alvo IDENTICOS      69/69    o conjunto denso esta byte a byte intacto
    niveis distintos por grupo de 128         3   ternario de verdade, em 8 de 8 amostradas

E um numero que enquadra a fila inteira, mediana de 8 camadas:

    braco 0  PTQ ternario ingenuo   rel-L2 ao original  0,46626
    braco 2  Bonsai publicado       rel-L2 ao original  0,53474    1,1469x MAIS LONGE

**O braco 0 esta MAIS PERTO do original em espaco de peso que o Bonsai.** Isso reproduz, por uma
construcao independente, o que foi medido em 2026-09-21 (o Bonsai perde de um PTQ BitNet ingenuo em
100 de 100 camadas na propria metrica de peso).

E deixa a pergunta da GPU na forma mais limpa possivel: **o braco 0 e mais fiel no peso e deve ser
pior na saida.** Se for, e a demonstracao mais direta que esta bancada consegue de que o treino deles
comprou algo que distancia de peso nao ve -- e e o mesmo padrao que este repo ja registra para erro
por camada, que ordena formatos e nao localiza penhasco.

Se o braco 0 for melhor na saida tambem, entao nada aqui aponta para treino e eu tenho um problema
maior que a hipotese de compensacao.

---

## Apendice 2: os tres bracos estao em disco e o ComfyUI detecta os tres

Medido 2026-09-22, ainda em CPU, **sem mudar previsao nenhuma**.

    arquivo                                          tensores   deteccao
    klein4b_braco0_ternario_ingenuo_bfl              149        Flux2
    klein4b_braco2_bonsai_ternario_bfl               149        Flux2
    flux-2-klein-4b.safetensors  (braco 3)           149        Flux2

`comfy.model_detection.model_config_from_unet` em tensores `device='meta'`. Os tres em
`P:\ComfyBench\diffusion_models\`, 7.751.105.712 B cada, **nomes identicos nos tres**.

Corrigido no caminho: eu os escrevi primeiro em `checkpoints/`, e eles sao **so o transformer**. Um
`CheckpointLoaderSimple` procura VAE e CLIP dentro e falharia de um jeito que nao aponta a causa.
Movidos para `diffusion_models/` e conferidos pelo `folder_paths` -- e conferido tambem que **nao
sobrou copia em `checkpoints`**, que e o erro simetrico.

### P5 (orcamento casado) conferida ANTES de qualquer medicao

    69 tensores fora dos blocos, 80 dentro   (80 e nao 100 porque a nomenclatura BFL funde o qkv)

    denso identico ao original:   braco 0  69/69      braco 2   1/69
    dentro dos blocos MUDADO:     braco 0  80/80      braco 2  80/80
    braco 0 != braco 2 nos blocos:         80/80

O braco 0 tem o **conjunto denso byte a byte intacto** e todas as 80 esmagadas -- que e exatamente o
que P5 exige dele. E o braco 2 volta a dar **1 de 69**: a assinatura de treino do Bonsai sobrevive ao
remap, que e a **quarta** confirmacao independente dela, agora atravessando uma transformacao de nome
que reproduz o arquivo oficial byte a byte.

Falta **so o braco 1**. Toda a engenharia de arquivo esta feita; o que resta e o experimento.
