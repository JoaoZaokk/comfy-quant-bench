# A receita sai do Flux? Sim, e o Qwen-Image tem DUAS respostas opostas

Criterio em `bench/criterio_replicabilidade_fora_do_flux_2026-09-22.md`, escrito antes dos numeros.
Cinco previsoes: **3 confirmadas, 2 divididas** -- e as duas se dividem porque `Qwen-Image-2.1` e
`Qwen-Image-2512` sao arquiteturas diferentes no eixo exato que decide a conta. O controle **nao deu o
resultado que eu pedi**: ele achou um defeito confiante na minha ferramenta, que era a razao de ele
existir.

Nada foi baixado para responder a parte matematica. O `Qwen-Image-2.1` foi projetado lendo **so o
header por Range HTTP, zero byte de peso**.

---

## 1. A tabela: a receita deles aplicada a tudo que esta no disco

`tools/projeta_receita_bonsai.py`, calibrado contra o pack medido (no klein-4B **tem** de dar 100
camadas e 195.042.816 params densos, e da, exato).

    modelo                     params   camadas   denso     ternario  bit/peso   binario  piso do
                                                  (params)            medio               arquivo
    Z-Image v2 (nosso)         6,15 B      273    0,04%      1,80 GiB  2,5055   0,90 GiB   0,26%
    Qwen-Image-2512           20,38 B      840    0,23%      6,02 GiB  2,5305   3,05 GiB   1,43%
    Qwen-Image-Edit-2511      20,38 B      840    0,23%      6,02 GiB  2,5305   3,05 GiB   1,43%
    HunyuanVideo 1.5           8,33 B      556    0,51%      2,49 GiB  2,5686   1,28 GiB   3,17%
    FLUX.1-dev                11,90 B      304    0,56%      3,57 GiB  2,5762   2,19 GiB   3,51%
    Wan 2.2 animate 14B       17,27 B      515    1,18%      5,35 GiB  2,6594   2,86 GiB   7,10%
    Krea2 Turbo               12,82 B      260    1,78%      4,09 GiB  2,7409   2,26 GiB  10,42%
    Qwen-Image-2.1             7,12 B      224    1,91%      2,28 GiB  2,7577   1,47 GiB  11,07%
    LTX 2.5 22B               21,00 B     2032    2,07%      6,80 GiB  2,7790   3,80 GiB  11,90%
    FLUX.2-klein-4B (o deles)  3,88 B      100    5,03%      1,43 GiB  3,1794   1,01 GiB  25,33%

**Ler a coluna do bit/peso medio, nao a do tamanho.** A receita nominal e 2,50 bit. Quanto mais perto
disso, melhor o modelo aceita a receita SEM o piso estragar a conta -- e o klein-4B, que e o que eles
escolheram, e **o pior da lista**.

**O melhor alvo da lista e o nosso Z-Image v2: 2,5055 bit/peso, piso de 0,26% do arquivo.** Praticamente
nao ha peso que nao encolha. E o modelo sobre o qual esta bancada tem mais infraestrutura montada
(calibracao por ativacao real, epsilon por passo com trajetoria imposta, quatro builds no disco).

## 2. O mecanismo e um so, e e onde mora a modulacao

    tensores 2-D de modulacao, e onde eles ficam
      Qwen-Image-2512     120 tensores,  120 DENTRO dos blocos   -> quantizam
      FLUX.1-dev           77 tensores,   76 DENTRO              -> quantizam
      Qwen-Image-2.1        1 tensor,       0 dentro             -> denso
      FLUX.2-klein-4B       4 tensores,     0 dentro             -> densos, 141.557.760 params

Arquitetura que mantem uma modulacao **por bloco** entrega tudo para a receita. Arquitetura que
**puxa a modulacao para fora e compartilha entre os blocos** cria um piso que nenhuma receita de bit
encolhe. Esse e o eixo, e ele explica a tabela inteira de cima a baixo.

### Q1 e Q4 se dividiram, e a razao e que "Qwen-Image" nao e um modelo

Eu previ que a modulacao do Qwen ficaria **dentro** dos blocos (Q1) e que o bit/peso ficaria **abaixo
de 2,70** (Q4). As duas coisas valem para o 2512 e **falham para o 2.1**:

    Qwen-Image-2512   20,38 B   840 camadas   modulacao dentro   2,5305   Q1 ok   Q4 ok
    Qwen-Image-2.1     7,12 B   224 camadas   modulacao fora     2,7577   Q1 NAO  Q4 NAO (por 0,06)

Eles nao sao versoes do mesmo desenho: **2.1 tem 7,12 B e 28 blocos, o 2512 tem 20,38 B e 60**. O 2.1
e quase 3x menor e voltou a deixar a modulacao fora dos blocos, como o klein. **Pedir "faz no Qwen-Image 2.1"
e pedir o caso mais parecido com o do Bonsai, nao o mais favoravel** -- o mais favoravel da familia
Qwen e o 2512, e o mais favoravel do disco e o Z-Image.

## 3. A regra de selecao vale fora do Flux: medida, nao assumida

`tools/probe_conjunto_denso.py`, calibrado nos dois bracos do klein-4B antes de sair de la.

    release                              regra vale?   concordancia   densos identicos ao original
    SVDQuant int4 klein-4B  (PTQ)            SIM          69/69            66/69   95,7%
    Bonsai ternario klein-4B (treino)        SIM          69/69             1/69    1,4%
    SVDQuant int4 Qwen-2512  (PTQ)           SIM        247/247          247/247  100,0%

"Regra vale" = **zero** violacoes nas duas direcoes: nenhum 2-D dentro de bloco ficou denso, nenhuma
camada de fora foi quantizada. Vale em duas familias, com dois quantizadores diferentes e com um
treino. **Q2 confirmada.**

E **Q3 confirmada com folga**: eu previ >= 90% de densos intactos no PTQ do Qwen e deu **100,0%, 247 de
247**. Nem uma dobra de smooth factor -- coerente com Q1, porque no 2512 nao ha modulacao fora dos
blocos para dobrar nada dentro.

**O contraste e a medida inteira:** o mesmo instrumento, no mesmo modelo, da 95,7% para um PTQ e 1,4%
para o Bonsai. A assinatura "quem nao quantizou a camada nao a reescreve" agora esta medida em duas
familias, e e barata: nao precisa de layout, de escala nem de desempacotamento, so de `torch.equal`.

### Q5 confirmada: a regra NAO se comporta igual em todas as familias

Nas pilhas puras de transformer (Qwen, Z-Image, Flux, Hunyuan) ela cobre quase tudo. Fora delas
aparece maquinario que ela deixa denso e que cresce o piso:

    Wan 2.2 animate   44 tensores 3-D, 39 4-D, 2 5-D (convolucao), e 14 2-D com K=4
    LTX 2.5           33 tensores 2-D fora de qualquer bloco, 2284 1-D
    Krea2             piso de 10,42%, com txtfusion e tmlp/txtmlp separados

Eu nao ia fazer a quarta extrapolacao entre arquiteturas deste repo sem olhar, e foi bom: a regra e
boa para transformer de imagem e **perde cobertura em video e em modelos com fusao de texto separada**.

## 4. O controle nao deu zero -- ele achou um veredito confiante e ERRADO meu

Eu escrevi que comparar o SVDQuant do **Qwen** contra o original do **klein-4B** tinha de dar zero
nomes em comum. Deu **21 de 247**, porque `norm_out.linear.weight`, `proj_out.weight`, `img_in`,
`txt_in` e `time_text_embed.*` existem com o **mesmo nome e o mesmo shape** nas duas arquiteturas. Os
21 diferiam, e a ferramenta imprimiu:

    LEITURA: o denso foi REESCRITO (21 de 21). Um PTQ nao faz isso.

**Veredito de treino sobre dois modelos sem nenhum parentesco.** Se eu tivesse apontado a ferramenta
para o par errado em qualquer ponto desta investigacao, ela teria concordado comigo com numeros.

Conserto: **guarda de arquitetura**. A ferramenta agora imprime a *concordancia* (fracao de densos com
par de nome E shape no original) em toda execucao, e abaixo de 90% recusa com `SEM VEREDITO` em vez de
opinar. Nos tres bracos reais a concordancia e 100%; no par trocado e **8,5%** e a ferramenta recusa.

Terceiro round seguido em que o defeito estava no meu instrumento, e o quarto em que **a calibracao
contra um numero que eu ja media** foi o que pegou. Sem gabarito, as versoes erradas produziam tabelas
plausiveis.

---

## 5. O que vai para a GPU, e por que nesta ordem

A parte (C) do criterio -- **o treino** -- nao e replicavel aqui em escala: o Bonsai nao publica passo,
dataset nem GPU-hora, e 3,68 bilhoes de pesos a 1,58 bit nao cabem nesta maquina. O que E testavel e a
hipotese que eu mesmo levantei no relatorio deles: **estagio de compensacao** -- congelar o corpo
ternario e ajustar SO o conjunto denso contra a saida do modelo denso.

Ordem, e cada item tem gabarito ou nao entra:

**Fila 1 -- klein-4B, porque existe resposta publicada para conferir.**

    braco 0  PTQ ternario ingenuo (absmean g128 em K), denso intacto     controle inferior
    braco 1  o mesmo PTQ, e so o conjunto denso (195.042.816 params)
             ajustado contra a saida bf16 num conjunto pequeno            a hipotese
    braco 2  o Bonsai ternario publicado                                  controle superior
    braco 3  o bf16 original                                              referencia

Metrica: **epsilon por passo com trajetoria imposta** (`tools/probe_epsilon_per_step.py`), que e o
instrumento que esta bancada validou. **Nao** imagem livre -- este repo ja mediu que imagem livre com
trajetoria solta mede caos, e que divergencia de latente e enviesada para falha macia.

Previsao numerica e criterio de parada **a escrever antes de rodar**, nao agora. Forma: se o braco 1
nao se separar do braco 0 alem do espalhamento entre sementes, a hipotese de compensacao morre.

**Fila 2 -- Z-Image v2, porque e o melhor alvo da tabela e a infraestrutura ja existe.** Sem gabarito
publicado, entao so entra se a Fila 1 der sinal.

**Fila 3 -- Qwen-Image-2.1.** O transformer esta sendo baixado (13,57 GiB, 2 shards, para
`P:\ComfyBench\originais\Qwen-Image-2.1`) justamente porque ele e o que ele pediu. Mas a tabela diz que
ele e o caso mais parecido com o do Bonsai, nao o mais favoravel, e com 224 camadas contra 100 do
klein e 7,12 B contra 3,88 B o custo do braco 1 sobe. **Terceiro, nao primeiro.**

---

## Nao coberto

- **Nada executado em nenhum modelo. Nenhuma imagem, nenhuma ativacao, nenhum kernel. GPU nao tocada
  em passo nenhum** -- as duas placas estao em uso dele.
- A secao 1 e **aritmetica**: tamanho de arquivo e piso, **nunca** qualidade. Nao diz que o treino a
  1,58 bit funcionaria em nenhum destes modelos.
- A secao 3 e **identidade de bytes e topologia de nome**. Um denso identico nao diz que a camada
  presta; um denso mudado nao diz COMO mudou.
- `Qwen-Image-2.1` foi projetado por header remoto: os **parametros** estao lidos do arquivo, mas
  nada do conteudo foi verificado.
- A regra de selecao foi calibrada no klein-4B e conferida no Qwen-2512. Nas outras oito linhas da
  tabela ela e **extrapolacao**, e a Q5 mostra que a cobertura cai em video.
- O text encoder do Qwen-Image-2.1 (16,7 GiB, Qwen2.5-VL) **nao** foi baixado; sem ele nao ha render.

---

## 6. Apendice 2026-09-22: os pesos chegaram, e o Qwen-Image-2.1 NAO RODA nesta pilha

Baixados e conferidos byte a byte contra o que o HF declara:

    P:\ComfyBench\originais\Qwen-Image-2.1\transformer\    14.230.284.408 B em 2 shards
    P:\ComfyBench\originais\FLUX.2-klein-4B\flux-2-klein-4b.safetensors   7.751.105.712 B
    P:\ComfyBench\originais\FLUX.2-klein-4B\vae\...              168.120.878 B

O arquivo no disco **confirma a projecao feita por Range HTTP**: 297 tensores, **7.115.124.736
parametros e 224 camadas quantizaveis**, os mesmos numeros que sairam do header remoto. A tecnica de
projetar sem baixar esta validada contra o arquivo real.

**E aqui esta o bloqueio que nao e de peso nem de matematica.** O config do 2.1 diz
`_class_name: QwenImage21Transformer2DModel`, com `axes_dims_rope [16, 56, 56]`, `context_in_dim 4096`
e `num_layers 32` -- **outra classe**, nao a do 2512. Testado por EXECUCAO, nao por grep, chamando o
detector do proprio ComfyUI sobre os tensores em `device='meta'`:

    comfy.model_detection.model_config_from_unet, ComfyUI 0.33.0
      Qwen-Image-2.1,  prefixo ''                        -> None
      Qwen-Image-2.1,  prefixo 'model.diffusion_model.'  -> None
      Qwen-Image-2512  (controle, que ESTA rodando aqui)  -> QwenImage

O controle detecta, o 2.1 nao. **O ComfyUI 0.33.0 desta bancada nao carrega o Qwen-Image-2.1**, e
subir ComfyUI e decisao dele e proibida a mim pela regra de nao fazer upgrade em massa. Os pesos ficam
no disco para o dia em que houver suporte, e toda a parte matematica ja esta feita sem eles.

**Consequencia pratica, e ela e boa:** se ele quiser o experimento de GPU na familia Qwen, o alvo e o
**2512**, que o ComfyUI carrega hoje -- e a tabela da secao 1 ja dizia que o 2512 e o melhor alvo da
familia de qualquer forma (2,5305 bit/peso contra 2,7577 do 2.1). O modelo que ele pediu e, ao mesmo
tempo, o pior da familia para a receita e o unico que nao roda.
