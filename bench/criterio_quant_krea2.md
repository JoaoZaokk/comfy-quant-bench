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

## Previsao 4: CONFIRMADA, e essa e a quarta medicao independente na mesma direcao

O ladder rodou **5 prompts x 2 sementes x 4 bracos = 40 renderizacoes**, 10 passos, 1024^2,
euler/simple, cfg 1,0, tudo na 3090 com `CUDA_VISIBLE_DEVICES=0`:

    checkpoint                   divergencia  espalhamento  s/passo    GiB  runs
    krea2_turbo_bf16                    -           -        2,239   24,48   10
    krea2_turbo_int8_convrot         0,2435      0,6512      1,233   12,57   10
    krea2_turbo_w4a4                 0,5843      0,7005      0,839    7,50   10
    krea2_turbo_mixed                0,4972      0,7257      1,000    7,70   10

    pareado contra o int8, corrida por corrida:
      krea2_turbo_w4a4    +0,3408 de media   pior +0,6593   melhor +0,1429   vence  0/10
      krea2_turbo_mixed   +0,2537 de media   pior +0,6020   melhor +0,0461   vence  0/10

**O int8 publico e 2,40x mais fiel ao BF16 que o nosso W4A4** (0,2435 contra 0,5843), e o
pareamento nao deixa duvida: **0 de 10** corridas em que qualquer dos nossos dois chega mais
perto. Isso e a quarta medicao independente na mesma direcao nesta bancada -- 1,49x no Z-Image
com ativacao real, 1,33x no epsilon por passo, 1,40x no Winnougan, 2,40x aqui -- agora numa
quarta familia de modelo.

Repare no **espalhamento**: 0,65 a 0,73 contra medias de 0,24 a 0,58. A divergencia de imagem
livre e ruidosa por construcao, e por isso a linha que carrega o resultado e a **pareada**, onde
os bracos compartilham prompt e semente.

## Previsao 5: CONFIRMADA

    W4A4  0,839 s/passo   contra int8  1,233   ->  1,47x mais rapido
    misto 1,000 s/passo   contra int8  1,233   ->  1,23x mais rapido

O `s/passo` do BF16 (2,239) **nao entra nessa comparacao**: ele nao cabe na placa e roda
descarregando, entao mede a politica de memoria, nao o kernel.

O braco misto e a troca explicita: promover 51 de 224 camadas comprou **15% menos divergencia**
(0,4972 contra 0,5843) por **19% mais tempo** (1,000 contra 0,839) e 0,20 GiB de arquivo.

## E as imagens? As 40 prestam. Nenhuma quebrou.

`bench/krea2_qualidade_s1.png` e `_s2.png`, 4 bracos x 5 prompts por semente. Julgamento de quem
olha, nao de metrica:

- **maca** (controle positivo): as quatro boas nas duas sementes.
- **rosto do pescador**: as quatro sao rostos coerentes, com ruga, poro e barba por fazer. O W4A4
  muda a expressao e o enquadramento; nao degrada a pele.
- **placa "OPEN"**: **legivel e bem formada em 8 de 8 celulas**. A cor da placa muda -- branca
  aqui, vermelha ali -- mas **muda com a semente tambem**: na semente 1 o BF16 fez branca e na 2
  fez vermelha. Se a cor acompanhasse a quantizacao seria dano; acompanhando a semente, e o
  sampler.
- **mercado noturno**: as quatro sao cenas densas e coerentes, com lanterna, banca, gente e
  reflexo no chao molhado. Os ideogramas sao plausiveis-e-falsos **inclusive no BF16**.
- **cristais de gelo**: as quatro tem estrutura cristalina fina.

Entao **0,5843 de divergencia nao e uma imagem destruida** -- e a mesma leitura que o Z-Image ja
tinha dado, agora numa arquitetura diferente. O que 0,58 mede e que o sampler foi para outro
lugar, e o outro lugar tambem e bom.

**Consequencia para a tabela de tolerancia:** o Krea2 entra com **tolerado 0,1199** (mediana de
`err_w4a4` do build que renderiza bem), e a coluna `NAO tolerado` fica **vazia** -- nada foi
medido acima disso. A monotonia por tamanho **nao sobrevive**: 12,82 B tolerando 0,1199 senta
abaixo do Z-Image de ~6 B, que tolera 0,1421. Tres pontos ja eram poucos; agora sao quatro e a
ordem quebrou.

## Camada 2: a conta quantizada roda MESMO, hoje

`tools/probe_quant_dispatch.py --mode diffusion --forward-only`, com os pesos na placa e a
contagem instrumentada **depois** da carga:

    checkpoint                  modulos  quant_format                      forwards  dequantize
    krea2_turbo_int8_convrot        224  {int8_tensorwise: 224}               8 / 8           0
    krea2_turbo_w4a4                224  {convrot_w4a4: 224}                  8 / 8           0
    krea2_turbo_mixed               224  {convrot_w4a4: 173, w4a8: 51}        8 / 8           0

    impl resolvido:  int8_linear / convrot_w4a4_linear / w4a8_int8_linear
                     -> comfy_kitchen.backends.cuda  em todos
    `convrot_linear_dtype=int4` nos nossos dois -- e o MMA int4 nativo, nao o ramo INT8
    224 pesos quantizados em cuda:0, `comfy_force_cast_weights=False`, `full_precision_mm=False`

Zero `dequantize` nos tres. O `int8_tensorwise` do build publico confirma o que a nota do
workflow ja dizia: **o nome do arquivo diz "convrot" e o formato por camada e int8**.

**`DESPACHA` nao e aprovacao** e nunca foi: responde se o kernel foi chamado, mais nada.

## Armadilha em que EU cai, de novo, e ela ja estava escrita

A primeira tentativa desta medicao levou `Assert-GpuLock -Owner 'bench:krea2_despacho_e_guarda'`
por fora e depois chamou `avaliar_referencia.py`, que entra no `BenchGuard` sozinho:

    F:\GPU_BENCH.lock is held: owner=bench:krea2_despacho_e_guarda pid=71944 alive=False
    Not reclaiming it.

A ferramenta recusou a propria corrida. O `CLAUDE.md` documenta exatamente isto -- *"do NOT take
the lock before a benchmark"* -- e a memoria `lock-gpu-compartilhado` tambem. Tomar o lock por
fora vale para conversor e sonda ad-hoc; **nao vale para nada que passe por `_timing.compare()`
ou `BenchGuard`**.

## Previsao 6: CONFIRMADA

`tools/avaliar_referencia.py` sobre o BF16, dois prompts nao relacionados x duas sementes:

    resposta (d_prompt / d_semente)   1,7686   por semente [0,9441, 2,5932]   espalhamento 2,747x
    |latente|  545,8 / 565,1 / 552,7 / 559,9
    veredito   SEM VEREDITO -- o braco responde ao prompt

Previsto > 0,8, medido 1,7686. O espalhamento entre sementes e grande (2,7x), como ja estava
documentado para esta guarda, entao o que conta e a distancia do limiar, nao a segunda casa.

## Balanco das seis previsoes

    1  perfil casa 224 Linear                    CONFIRMADA
    2  mediana err_w4a4 entre 0,15 e 0,22        REFUTADA -- 0,1199
    3  err_w4a8 < err_w4a4 quase sempre          CONFIRMADA -- 224/224, 3,12x
    4  int8 mais fiel que o nosso W4A4           CONFIRMADA -- 2,40x, 10/10 pareado
    5  W4A4 mais rapido por passo                CONFIRMADA -- 1,47x sobre o int8
    6  guarda de referencia > 0,8 no BF16        CONFIRMADA -- 1,7686

Uma refutada de seis, e e a que importa: **a hipotese do tamanho nao previu o erro**, e ela era o
unico argumento para extrapolar da tabela de tolerancia.

## Como refazer

    calibrar   tools/calibrate_activations.py --model krea2_turbo_bf16.safetensors --profile krea2
               --clip qwen3vl_4b_bf16.safetensors --clip-type krea2 --steps 10 --size 1024
    analisar   tools/quant_mixed.py --promote-error 10.0 --save-analysis ... --dry-run
    converter  tools/quant_mixed.py --analysis calib/krea2_turbo.analysis.json --profile krea2
    qualidade  tools/quality_ladder.py --clip-type krea2 --vae qwen_image_vae.safetensors
               (CUDA_VISIBLE_DEVICES=0; o cortex segura 2,4 GiB na 3080 Ti e o BenchGuard recusa)
    imagens    tools/decode_latents.py bench/quality_ladder_krea2/latents --vae qwen_image_vae.safetensors
    despacho   tools/probe_quant_dispatch.py <arq> --mode diffusion --forward-only
    guarda     tools/avaliar_referencia.py ...  **sem** Assert-GpuLock por fora

## O que continua nao coberto, depois de tudo

- **Uma placa (sm86), um tamanho (1024^2), um sampler, uma calibracao, duas sementes.**
- **Nenhuma metrica perceptual.** Quem disse que as 40 imagens prestam fui eu, olhando.
- **O teto do Krea2 nao foi medido.** 0,1199 funciona; nada acima disso foi tentado.
- **O `krea2_raw`** (base, nao-turbo) ficou de fora inteiro.
- **O Krea2 Edit nao foi testado com os nossos builds** -- a edicao usa condicionamento duplo e
  nada aqui diz o que a quantizacao faz com preservacao de identidade.
- **O encoder de texto continua BF16 e travado** pelos dois cadeados do ComfyUI.
- O `s/passo` do BF16 mede descarregamento, nao kernel, e por isso nao entra em nenhuma razao.

## O epsilon pareado NAO foi medido no Krea2, e a razao fica escrita

`tools/probe_epsilon_ckpt_ab.py` impoe a trajetoria do BF16 a todos os bracos, entao cada passo e
uma comparacao casada e divergencia de trajetoria nao existe por construcao. **E o instrumento
certo para comparar duas quantizacoes do mesmo modelo** -- o `CLAUDE.md` diz isso com todas as
letras, e o espalhamento de 0,65-0,73 do ladder acima e exatamente o motivo.

Tentado em 2026-09-12 e **nao concluido**. Duas paredes, nesta ordem:

1. **Sem DynamicVRAM, o braco BF16 pagina.** 49,4 GiB de pagefile para um modelo de 24,5 GiB --
   o dobro-comprometimento que `_dynamic_vram` existe para evitar -- com 4 por cento de CPU.
   Nao ia terminar.
2. **Com DynamicVRAM, o subprocesso morre em `AttributeError: 'NoneType' object has no attribute
   'hostbuf_allocate'`.** `comfy_aimdo/host_buffer.py:6` faz `lib = control.lib` no import, e
   `comfy/memory_management.py:7` importa esse modulo quando o ComfyUI e importado; ligar o
   aimdo depois disso nao desfaz a ligacao congelada. O `quality_ladder` escapa porque o
   bootstrap dele acontece antes, no topo do arquivo.

A edicao que eu tinha feito nos dois templates foi **revertida**: ela nao funcionou e mudaria o
caminho de carga para todo modelo, incluindo o Z-Image, onde a sonda funciona hoje. O
`--clip-type` fica (esse foi testado e e o que faltava para a sonda alcancar o Krea2).

**Consequencia honesta:** a afirmacao "o int8 e mais fiel que o nosso W4A4 no Krea2" repousa
sobre a comparacao de imagem livre, que e ruidosa -- mas na forma **pareada**, 10 de 10 corridas,
onde os bracos compartilham prompt e semente. Nao repousa sobre a media de divergencia sozinha.
As tres medicoes anteriores desta bancada, essas sim com entrada casada, apontam na mesma
direcao em outros tres modelos.
