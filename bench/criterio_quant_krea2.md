# Criterio: quantizar o Krea2 Turbo, e onde ele cai na tabela de tolerancia

Escrito em 2026-09-12 **antes** de calibrar, converter ou renderizar qualquer coisa. A janela de
GPU foi aberta pelo dono com "placa e sua, manda brasa", e o alvo do `/goal` e
*"faz os quants, verifique a qualidade, 4 prompts para cada"*.

## A pergunta

Esta bancada tem **tres** pontos na tabela que relaciona tamanho do modelo com o erro por camada
que ele aguenta em W4A4. Tres pontos nao sao uma lei, e o proprio `CLAUDE.md` diz isso com todas
as letras. O Krea2 Turbo seria o **quarto**, e ele cai exatamente em cima da linha mais alta:

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1421        0,1848
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147
    Krea2 Turbo            12,82 B        ?             ?

**12.820.073.036 parametros, medido**, somando `prod(shape)` de todos os 430 tensores do
`krea2_turbo_bf16.safetensors` (26.283.332.608 bytes). Nao e estimativa por tamanho de arquivo.

## O que o Krea2 e, e por que nenhuma ferramenta daqui o alcanca hoje

`SingleStreamDiT` (`ComfyUI/comfy/ldm/krea2/model.py`): **28 blocos**, cada um com

    attn.wq  attn.wk  attn.wv  attn.gate  attn.wo      mlp.gate  mlp.up  mlp.down

oito `Linear` por bloco, **224 no total**, e **94,8% dos parametros do modelo** estao nelas.
Todas tem `shape[1] % 256 == 0` (medido: zero excecoes), entao o `convrot_groupsize` 256 -- o
unico que o caminho W4A8 aceita nesta arvore -- serve em todas.

`PROFILE_PATTERNS` em `tools/calibrate_activations.py` e uma **allowlist estrita** e so tem
`zimage`, `ltx_2_5`, `hunyuan_video_15` e `wan_2_1`. Nenhuma casa `blocks.N.attn.wq`. Entao o
primeiro passo deste trabalho e **escrever um perfil `krea2`**, e o perfil e a primeira coisa que
pode estar errada -- foi assim que o `hunyuan_video_15` nasceu casando **zero** camadas, porque
tinha sido escrito lendo os nomes do arquivo sem nunca ser rodado contra um modelo carregado.

### O que fica de fora, de proposito

- `blocks.N.mod.lin` e `last.modulation.lin` -- o caminho de **modulacao**. Nao sao `Linear`:
  sao `nn.Parameter` (`model.py:110` e `:121`), entao nao ha o que quantizar mesmo querendo.
- `txtfusion.*` (4 blocos, 32 `Linear`, 2,7% dos parametros) -- a **entrada** do condicionamento
  de texto. E a regra "manter as pontas mais altas" que este repo ja aplica em Z-Image, Hunyuan e
  Wan.
- `first`, `last.linear`, `tmlp.*`, `tproj.*`, `txtmlp.*` -- patchify/unpatchify e embeddings.

### A verificacao independente de escopo, que ja passou

O `krea2_turbo_int8_convrot.safetensors` publico carrega **224 tensores `comfy_quant`**, lidos do
header. O perfil que eu escrevi casa **224 pesos**. Dois caminhos independentes -- o meu regex
sobre o BF16 e o conteudo do arquivo de outra pessoa -- concordam no escopo. Se tivessem
discordado, o perfil estaria errado antes de qualquer medicao.

## Os bracos

    A  krea2_turbo_bf16              24,48 GiB   REFERENCIA e CONTROLE POSITIVO
    B  krea2_turbo_int8_convrot      12,57 GiB   o build publico, int8_tensorwise, 224 camadas
    C  o nosso W4A4/misto            ~7,5 GiB    o que este trabalho produz

O braco A **nao e enfeite**. Se ele sair ruim, nada nas outras linhas vale -- e a licao que o Wan
custou quatro renders (`bench/criterio_guarda_referencia.md`), e ela custou porque o braco de
referencia estava quebrado e ninguem olhou primeiro.

O braco A tem um problema pratico proprio: **24,48 GiB nao cabem numa 3090 de 24 GiB**. Pelo
caminho default, `lowvram_model_memory = max(0, livre - (tamanho + reserva))` da **zero**, e o
modelo roda inteiro em streaming. O plano e `--distorch 'cuda:0,15gb;cuda:1,8gb;cpu,*'`, doando
para a segunda placa antes de doar para RAM pageavel. **Isso muda a coluna `s/step` do braco A e
nao pode ser lido como velocidade do BF16.**

## Previsoes, escritas antes de medir

1. **O perfil casa exatamente 224 `Linear`** no modelo carregado, nao so no arquivo. Se casar
   outro numero, paro: o perfil esta errado e tudo depois dele descreve outra coisa.
2. **A mediana de `err_w4a4` cai entre 0,15 e 0,22.** E a previsao da hipotese do tamanho: o
   Krea2 tem 12,82 B e o Hunyuan (~13 B) tolera 0,1837 e quebra em 0,2147. **Abaixo de 0,10 ou
   acima de 0,30 a hipotese do tamanho fica em serio apuro** e isso vai escrito como refutacao,
   nao como ruido.
3. **`err_w4a8 < err_w4a4` em praticamente toda camada.** Vale em todas as familias medidas aqui
   ate hoje. Se inverter em mais de 10% das camadas, ha bug na medicao, nao achado.
4. **O int8 (braco B) fica MAIS FIEL que o nosso W4A4 (braco C)** contra o BF16. Tres medicoes
   independentes desta bancada apontam nessa direcao (1,49x no Z-Image com ativacao real, 1,33x
   no epsilon por passo, 1,40x no Winnougan). Se o W4A4 ganhar, e um resultado contra tres, e
   precisa de replica antes de virar frase.
5. **O braco C e mais rapido por passo que o A, e provavelmente que o B.** 1024^2 dao ~4096
   tokens de latente, que e o regime M grande onde o MMA int4 nativo ganha. **Nao e garantia:**
   no HunyuanVideo 1.5 o W4A4 saiu 1,055x **mais lento** que o fp16 com a mesma ferramenta.
6. **A guarda de referencia (`avaliar_referencia.py`) da `d_prompt/d_semente` > 0,8 no braco A.**

## O que decide "quebrou"

Nada automatico. **`APROVADO` nao existe em nenhuma camada do avaliador deste repo**, e a razao e
medida: 0,1837 correto contra 0,2147 destruido, 0,7173 bom contra 0,8255 destruido -- nenhum corte
em nenhum dos dois eixos separa usavel de inusavel. A divergencia de latente entra no relatorio
como **distancia**, nunca como nota. Quem decide se a imagem presta e alguem olhando as 4 imagens
por braco.

## Condicoes de refutacao

1. **Braco A sai ruim** -> nao publicar nenhuma linha. O problema esta fora da quantizacao (VAE,
   encoder, sampler, shift), e a tabela continua com tres pontos.
2. **O perfil casa != 224 camadas** -> paro antes de calibrar.
3. **A calibracao acusa camada que nunca rodou** (`never_ran` nao vazio) -> o perfil casou algo
   fora do caminho amostrado; a analise dessas camadas e chute.
4. **O `--promote-error 0.15` e importado do Z-Image e NAO e default seguro.** No Wan ele escreve
   um arquivo que carrega, despacha nativo, passa toda checagem estrutural e desenha borrao. Aqui
   ele so entra depois de eu ver a distribuicao do `err_w4a4`, nunca antes.
5. **Amostras em bfloat16, sempre.** Em fp16 uma ativacao acima de 65504 vira `inf`, todo erro
   daquela camada vira `nan`, e `nan` nao e maior que nenhum limiar -- a pior camada do modelo
   sai roteada para o formato mais barato, em silencio.

## Nao coberto por este experimento, escrito antes de saber o resultado

- Uma placa (sm86), um tamanho (1024^2), um sampler, um scheduler, uma calibracao.
- **Nenhuma metrica perceptual.** Divergencia de latente nao e qualidade, e imagem gerada em
  trajetoria livre nao serve para comparar duas quantizacoes do mesmo modelo -- esta bancada ja
  mediu que em 8 passos uma perturbacao minima reroteia o sampler e o destino continua bom.
- O **Krea2 Edit** nao entra aqui. A edicao passa pelo mesmo transformer, mas com condicionamento
  duplo (latente da origem + Qwen3-VL), e nada neste criterio diz o que a quantizacao faz com a
  preservacao de identidade.
- O `krea2_raw` (o modelo base, nao-turbo) fica de fora. Ele esta no disco, e byte a byte
  diferente do turbo (medido: 64 de 64 amostras de 1 MiB diferem), e nao foi medido.
- O **encoder de texto continua BF16** e travado pelos dois cadeados do ComfyUI. Nada aqui mede
  o que quantizar o Qwen3-VL-4B faria.

---

# RESULTADO (anexado 2026-09-12, depois de medir)

## Previsao 1: CONFIRMADA

    hooking 224 Linear layers matching profile 'krea2'

224, contra o modelo **carregado**, nao contra o header. Zero camadas em `never_ran`. A
calibracao levou 83,2 s -- tres prompts, uma semente, 10 passos -- e escreveu 412,7 MiB.

## Previsao 2: REFUTADA

Eu previ que a mediana de `err_w4a4` cairia **entre 0,15 e 0,22**, porque e ai que o
HunyuanVideo 1.5 (~13 B) esta e o Krea2 tem 12,82 B. Medido sobre as 224 camadas, todas
calibradas, `convrot_groupsize` 256, amostras em bfloat16:

    err_bf16   mediana 0,0019   min 0,0016   p25 0,0017   p75 0,0021   max 0,0025
    err_w4a4   mediana 0,1199   min 0,0113   p25 0,0822   p75 0,1481   max 0,2942
    err_w4a8   mediana 0,0373   min 0,0063   p25 0,0264   p75 0,0467   max 0,0634

**0,1199.** Fora da faixa que escrevi, por baixo. Um modelo de 12,82 B mede **menos** erro por
camada que o Z-Image de ~6 B (0,1421 tolerado) e bem menos que o Hunyuan de ~13 B (0,1837
tolerado). Nao chegou a cruzar o 0,10 que eu tinha marcado como "a hipotese do tamanho fica em
serio apuro", entao o que morre aqui e **a previsao**, nao a tabela -- mas a tabela perde o
unico argumento que eu tinha para extrapolar dela.

Cuidado com o que isto NAO diz: a coluna `tolerado` registra o **maior erro que ja se viu
funcionar**, e 0,1199 funcionar nao impede o Krea2 de tolerar 0,18. O teto do Krea2 continua
tao nao-medido quanto o do Z-Image.

## Previsao 3: CONFIRMADA, e com folga

`err_w4a8 < err_w4a4` em **224 de 224** camadas. Mediana da razao por camada: **3,12x**, faixa
1,57x a 4,91x. Nenhuma inversao, entao nao ha sinal de bug na medicao por esse lado.

## O que a distribuicao por familia mostra, e nao estava previsto

    familia          n    mediana    min      max
    attn.wo         28    0,2181   0,0934   0,2942
    mlp.down        28    0,2029   0,0222   0,2354
    mlp.up          28    0,1370   0,0423   0,1488
    attn.wv         28    0,1283   0,0320   0,1570
    mlp.gate        28    0,1181   0,0342   0,1443
    attn.gate       28    0,0979   0,0255   0,1271
    attn.wq         28    0,0813   0,0128   0,0985
    attn.wk         28    0,0749   0,0113   0,0902

As duas piores familias sao **exatamente as duas projecoes de saida** -- `attn.wo`, que le a
saida da atencao, e `mlp.down`, que le a saida do SwiGLU. As duas mais baratas sao `wq` e `wk`,
que leem o residual normalizado. **2,9x separa a pior familia da melhor**, e a ordem e a mesma
em todos os 28 blocos.

Isso e uma observacao, nao um mecanismo demonstrado: a leitura obvia e que entrada
pos-ativacao carrega os outliers que a rotacao existe para suprimir, mas nada aqui isolou esse
eixo -- `crest_p99` na mesma tabela vai de 18 a 99 dentro da mesma familia, entao crest
tambem nao explica sozinho (e esta bancada ja mediu Spearman +0,10 entre crest e `err_w4a4`
em 170 camadas do Z-Image).

## Os dois bracos construidos, e por que estes dois

Com a distribuicao na mao -- que era a condicao que o criterio impos antes de deixar
`--promote-error` entrar:

    acima de 0,10:  132 de 224   promover custaria 59% do modelo em 8 bits
    acima de 0,15:   51 de 224   promover custaria 23%
    acima de 0,18:   46 de 224   promover custaria 21%
    acima de 0,2147: 24 de 224   promover custaria 11%

    C  krea2_turbo_w4a4    --promote-error 10.0   224/224 em W4A4, mediana 0,1199
    D  krea2_turbo_mixed   --promote-error 0.15   173 W4A4 + 51 W4A8

O 0,15 nao foi importado do Z-Image: ele foi escolhido **depois** de ver que naquele ponto a
curva promove 23% dos parametros, um orcamento parecido com o do par ja publicado do Z-Image
(w4a4 0,1241 contra misto 0,0774). Ter os dois bracos com o mesmo par de valores torna os dois
modelos comparaveis; ter so um nao tornaria.
