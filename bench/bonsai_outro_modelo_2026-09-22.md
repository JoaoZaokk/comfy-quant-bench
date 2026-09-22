# Qual outro modelo daria, como ficaria, e o controle positivo que caiu no colo

Tres perguntas dele: (1) da para aplicar a tecnica noutro modelo, temos outro Flux? (2) como ficaria,
matematicamente? (3) eles juntam tudo num arquivo so de runtime?

A resposta da (1) veio melhor do que eu esperava: **o outro derivado do MESMO original ja estava no
disco**, e ele fornece o controle positivo que o meu metodo nunca teve.

---

## 1. O controle positivo, de graca: 68 contra 3

`svdq-int4_r32-flux.2-klein-4b.safetensors` (2,28 GiB, Nunchaku SVDQuant) e um **PTQ do mesmo
FLUX.2-klein-4B** que o Bonsai treinou. Dois derivados, um original, metodos que nao tem nada em
comum.

Primeiro, o que os dois times concordam -- e a concordancia e total:

    tensores densos BF16          SVDQuant  69     Bonsai  69     intersecao  69/69

**Zero diferenca.** Os mesmos 69: 60 `norm_q`/`norm_k`/`norm_added_*` (1-D), mais 9 matrizes que
ficam FORA dos blocos (as tres modulacoes, `context_embedder`, `norm_out.linear`, os dois
`timestep_embedder`, `proj_out`, `x_embedder`).

Agora a diferenca, que e o controle:

    os MESMOS 69 densos, comparados byte a byte com o klein-4B original
      SVDQuant identicos ao original     66/69
      Bonsai   identicos ao original      1/69

**Um quantizador de verdade, do mesmo modelo, mudou 3. O Bonsai mudou 68.**

Ontem eu argumentei que as camadas "puladas" do Bonsai sao capacidade de adaptacao porque **8 das 9
mudaram**. A fraqueza era que eu nao tinha linha de base: quantos um quantizador mudaria? Agora tem,
e nao e uma linha de base teorica -- e um arquivo publicado, do mesmo original, medido aqui.

E as 3 que o SVDQuant mudou tem mecanismo conhecido, nao treino:

    double_stream_modulation_img.linear.weight   rel-L2 0,12123
    double_stream_modulation_txt.linear.weight   rel-L2 0,10808
    single_stream_modulation.linear.weight       rel-L2 0,14828

Sao **exatamente as tres modulacoes**, e sao exatamente o que o SmoothQuant tem de reescrever: o
`smooth_factor` [3072] por canal que o arquivo carrega e dobrado de volta na camada que alimenta as
camadas suavizadas. Este repo ja documenta esse mecanismo no proprio `quant_w4a4_smooth.py`, onde
"preservado == identico" **inverte** para as normas por construcao. Mudanca mecanica, com causa
rastreavel, em 3 tensores. Nao 68.

**Reformulacao que isso obriga, e ela e mais forte que a de ontem:** o conjunto denso **nao e escolha
de receita, e propriedade da arquitetura** -- dois times chegam aos mesmos 69. O que distingue o
Bonsai nao e QUAIS camadas ele deixou densas, e que ele **as reescreveu**.

### E um aviso, porque o teste de sinal nao se aplica cru aqui

O SVDQuant guarda `proj_down`/`proj_up` (rank 32) e `smooth_factor`: o que foi quantizado e o
**residuo depois de tirar as 32 componentes principais**, nao o peso. Um teste de sinal ingenuo
`qweight` contra `W` daria ~0,5 e chamaria um PTQ de treinado -- a mesma armadilha do Hadamard que eu
registrei para o `Ternary-Bonsai-2-27B`. A boa noticia e que o `smooth_factor` e **diagonal**, nao
rotacao, entao preserva sinal e o teste E aplicavel *desde que aplicado ao residuo*. Nao rodei: o
controle de 68-contra-3 responde a pergunta sem precisar dele, e vale mais porque nao depende de eu
reconstruir a receita deles.

---

## 2. Matematicamente, noutro modelo: o Flux.1-dev e alvo MUITO melhor

`tools/projeta_receita_bonsai.py`. A regra de classificacao nao e chute -- e **calibrada contra o pack
medido**: no klein-4B ela tem de dar 100 camadas e 195.042.816 params densos, e da, exato.

    modelo                  params      camadas   denso     ternario   bit/peso   binario
                                                                        medio
    FLUX.2-klein-4B          3,88 B       100     5,03%     1,43 GiB    3,1794    1,01 GiB
    FLUX.2-klein-base-4b     3,88 B        80*    5,03%     1,43 GiB    3,1794    1,01 GiB
    FLUX.1-dev              11,90 B       304     0,56%     3,57 GiB    2,5762    2,19 GiB

`*` 80 e nao 100 porque a nomenclatura BFL funde o qkv; o **numero de parametros quantizaveis e
identico ao do klein-4B em diffusers, 3.680.501.760**, e o denso difere em 156 -- que sao os escalares
de escala do fp8. Duas nomenclaturas, mesmo modelo, mesma conta: a regra nao depende de nome.

**O achado e a coluna do bit/peso medio, nao a do tamanho.** No klein-4B a receita nominal de 2,50
bit sai a **3,18 bit/peso** no arquivo; no Flux.1-dev sai a **2,58**. A diferenca e o piso, e o piso
tem uma causa arquitetural exata, medida:

    tensores 2-D de modulacao, e onde eles moram
      FLUX.1-dev     77 tensores,  76 DENTRO dos blocos   -> quantizaveis
      FLUX.2-klein    4 tensores,   0 dentro dos blocos   -> densos, 141.557.760 params de piso

O klein **puxou a modulacao para fora dos blocos** e a compartilha entre os blocos; o Flux.1-dev mantem uma por
bloco. Sob a regra que os dois times usaram -- quantiza o que esta dentro do bloco -- isso e a
diferenca entre 25,33% e 3,51% do arquivo sendo peso que nao encolhe.

Consequencia pratica: **de 23,8 GiB bf16, o Flux.1-dev sairia a 3,57 GiB ternario (6,7x) ou 2,19 GiB
binario (10,9x)**, e seria o primeiro modelo desta familia onde o bit/peso do arquivo fica perto do
bit/peso do formato. O klein-4B, que e o que eles escolheram, e o caso em que o piso mais atrapalha.

[JULGAMENTO] se a Prism quisesse mostrar o metodo no melhor caso de TAMANHO, o Flux.1-dev era a
escolha. Escolheram o klein-4B, que e 3x menor e tem o pior piso da familia. O que me faria mudar de
ideia: um custo de treino que cresce com o modelo -- 12 B a 1,58 bit e mais GPU-hora do que 4 B, e
eles nao publicam nenhum numero de treino, entao "eles escolheram o mais facil de treinar" e tao
compativel com os fatos quanto qualquer outra leitura.

---

## 3. "Eles juntam tudo num arquivo so de runtime?" NAO. Medido.

O Bonsai ship **tres componentes em tres pastas**, layout diffusers:

    transformer-gemlite-int2/state_dict.pt        1469,09 MiB
    text_encoder-hqq-4bit/qmodel.pt               2691,59 MiB
    vae/diffusion_pytorch_model.safetensors        160,33 MiB

Nada de arquivo unico. Mas a pergunta acertou um alvo de lado, e vale a correcao: **o original da BFL
publica DUAS versoes do transformer**, e isso engana:

    flux-2-klein-4b.safetensors                        7.751.105.712 B   149 tensores
    transformer/diffusion_pytorch_model.safetensors    7.751.109.744 B   169 tensores

Tamanhos diferentes (4.032 B), sha256 diferentes, e **zero nomes em comum** -- lido por Range HTTP,
sem baixar os 7,4 GiB. O da raiz usa a nomenclatura BFL (`double_blocks.0.img_attn.proj.weight`), o
da pasta usa diffusers (`transformer_blocks.*`, `context_embedder`). 149 contra 169 porque a
nomenclatura BFL funde tensores. **Nao e "tudo num arquivo": e o MESMO transformer em duas
nomenclaturas**, 7,39 GiB cada, sem text encoder nem VAE dentro de nenhum dos dois.

Quem de fato empacota tudo num arquivo e o **ecossistema LTX/ComfyUI**, nao o Bonsai: o nosso
`10Eros_v1.5_bf16_w4a8.safetensors` carrega, num safetensors so, `model.diffusion_model` (8764
tensores), `vocoder` (1227), `vae` (170), `audio_vae` (102) e `text_embedding_projection` (4) -- e
**nem esse** traz o text encoder, que continua sendo um arquivo separado em `text_encoders/`. Entao a
regra geral e: **o transformer e um arquivo; o text encoder quase nunca vem junto.**

---

## 4. A minha ferramenta errou duas vezes nesta rodada

1. **Classificador por lista de nomes.** A primeira versao tinha `norm|proj_out|modulation|...`. A
   calibracao contra o klein-4B reprovou: **105 camadas em vez de 100** e 52.698.624 params densos em
   vez de 195.042.816. O que a lista nao pegava: `double_stream_modulation_img` (sufixo depois de
   `modulation`), `norm_out.linear` (sufixo depois de `norm`) e `time_guidance_embed` -- eu tinha
   escrito `time_text_embed`, que e o nome noutra arquitetura. O conserto **apagou a lista**: a regra
   verdadeira e "2-D dentro de bloco quantiza, o resto e denso", que nao tem nome nenhum dentro e
   bate exato nos dois lados.
2. **Detector de pilha contava pilha de um.** `final_layer.adaLN_modulation.1.weight` casa `.<n>.`
   porque o `1` e indice de `nn.Sequential`, nao de bloco, e isso promovia **18.874.368 params de
   modulacao para quantizavel** nos dois Flux -- o oposto da regra medida. Conserto: uma pilha exige
   **>= 2 indices distintos**. Sem ele o Flux.1-dev aparecia com 305 camadas e 0,41% de denso em vez
   de 304 e 0,56%.

**A calibracao pegou as duas.** Sem um alvo cujo numero eu ja media, as duas versoes erradas teriam
produzido tabelas plausiveis para os outros modelos e eu nao teria motivo para desconfiar.

---

## Nao coberto

- **Nada executado no modelo. Nenhuma imagem, nenhuma ativacao, nenhum kernel. GPU nao tocada.**
- A secao 2 e **aritmetica**, nao medicao: diz o tamanho do arquivo e onde fica o piso, **nao** diz se
  o treino a 1,58 bit funcionaria no Flux.1-dev. A regra foi calibrada no klein-4B e este repo ja
  registra **tres** extrapolacoes entre arquiteturas que morreram na medicao.
- O teste de sinal **nao foi rodado no SVDQuant**: exigiria reconstruir o residuo com a low-rank e o
  smooth deles, e o controle de 68-contra-3 ja responde sem isso.
- As 3 mudancas do SVDQuant foram **atribuidas** ao dobramento do `smooth_factor` pelo mecanismo e
  pelas camadas envolvidas; nao reconstrui o fator para confirmar numericamente.
- `flux1-dev` e `klein-base` no disco sao **fp8**, nao bf16: a coluna de parametros vale, a
  comparacao com "arquivo atual" nao e contra um bf16.
- Nao ha original bf16 do Flux.1-dev nesta maquina, entao o par original-derivado que existe de fato
  e so o do klein-4B.
