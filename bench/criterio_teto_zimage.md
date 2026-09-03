# Criterio: onde o Z-Image quebra? A celula vazia da tabela de tolerancia

Escrito em 2026-09-01 **antes** de rodar, numa janela de GPU que o dono abriu para trabalho
autonomo.

## A celula esta vazia, e o motivo pelo qual ela ficou vazia estava errado

A tabela publicada (README do HF, `CLAUDE.md`) diz:

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1241     NAO MEDIDO
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147

Eu ia preencher subindo o `--promote-error` do `quant_mixed`. **Isso nao funciona, e perceber o
porque e o achado que abre este trabalho:** o `zimage-v2-w4a4` ja e 170 de 170 camadas em 4 bits
(`promote_error 10.0`), mede 0,1241 e a imagem presta. Nao existe build mais agressivo nesse eixo --
promover mais nao ha o que promover.

## A alavanca e o `convrot_groupsize`, e ela anda nos DOIS sentidos

A rotacao de Hadamard e aplicada em grupos de `convrot_groupsize` colunas. Grupo **maior** = rotacao
mais grossa = menos capacidade de espalhar outlier = **mais erro**. Grupo menor = o contrario.

O tamanho tem de ser potencia de **4** -- medido nesta bancada em 2026-09-01, quando 128 levantou
`Regular Hadamard size must be a power of 4` em `comfy_kitchen/tensor/int8_utils.py:22`. Entao os
degraus disponiveis em volta do 256 usado hoje sao **64** e **1024**.

    cg 64     mais fino    esperado: erro MENOR que 0,1241
    cg 256    o de hoje    0,1241, imagem boa
    cg 1024   mais grosso  esperado: erro MAIOR -- o candidato a quebrar

## Previsoes, escritas antes

1. `cg 1024` mede mediana **acima** de 0,1241. Se nao medir, a alavanca nao e o que eu penso e o
   resto deste experimento nao vale nada.
2. `cg 64` mede **abaixo** de 0,1241. **Este e o controle na direcao oposta**: sem ele, um `cg 1024`
   que sobe poderia estar subindo por qualquer motivo (um bug meu de conversao, por exemplo) e eu
   leria como confirmacao. Uma alavanca que so anda para um lado nao foi demonstrada.
3. Nao sei se `cg 1024` **quebra a imagem**. E a pergunta. Erro maior nao implica imagem ruim: esta
   bancada ja mediu 0,1837 correta contra 0,2147 destruida no Hunyuan, e nao ha corte que separe.

## O braco de referencia e um controle, nao um enfeite

O BF16 (`beyond-reality-zimage-v2_native`) entra na mesma corrida. Se ELE sair ruim, o problema e do
VAE / do encoder / dos parametros e **nenhuma outra linha da tabela vale** -- e a licao que o Wan
custou quatro renders para ensinar (`bench/criterio_guarda_referencia.md`). O VAE do Z-Image nao
esta confirmado nesta bancada: o ladder anterior rodou **sem VAE nenhum** e por isso nao escreveu
imagem. Se o braco BF16 sair errado, o veredito e "VAE errado", nao "quantizacao quebrou".

## Condicoes de refutacao

1. **BF16 sai ruim** -> nao publicar nenhuma linha; o problema esta fora da quantizacao.
2. **`cg 64` nao melhora** -> a alavanca nao foi demonstrada; `cg 1024` piorar nao prova nada.
3. **`cg 1024` nao piora** -> a hipotese morre; a celula continua vazia e isso vai escrito.

## Nao coberto

Um prompt, poucas sementes, um tamanho, uma placa. A mediana do erro por camada sai do
`tools/avaliar.py`, offline, e ela mede o **erro contra float32 nas ativacoes calibradas**, nao a
imagem. Nenhum corte desta bancada separa "usavel" de "inutilizavel" nesse eixo -- por isso a
imagem entra junto e por isso quem decide se ela presta e uma pessoa olhando, nao este arquivo.

E `cg` nao e o unico eixo possivel de agressividade (group_size do w4a8, formato, calibragem);
e o unico testado aqui.

---

# RESULTADO -- 2026-09-01. A celula continua VAZIA, e agora o motivo e preciso.

Criterio acima escrito antes. Nenhuma previsao foi ajustada depois de ver numero.

## A condicao de refutacao 2 disparou: a alavanca nao pode ser acionada

    cg 1024   FALHOU   RuntimeError: convrot rotate kernel only supports group_size 256
    cg  256   OK       mediana err_w4a4 = 0,1266
    cg   64   FALHOU   mesma mensagem

O controle na direcao oposta (`cg 64`) era a previsao 2, e ele **nao rodou**. Pela regra escrita
antes, isso encerra o experimento: uma alavanca que so anda para um lado nao foi demonstrada -- e
aqui ela nao andou para lado nenhum.

O braco `cg 256` serve como controle de sanidade e passa: 0,1266 contra os 0,1241 publicados para o
`zimage-v2-w4a4` e +2,0%, dentro dos 2% a 6% que a semente da calibragem move nesta bancada.

## A mensagem de erro me levou a conclusao errada, e o probe derrubou

Ao ler `convrot rotate kernel only supports group_size 256` eu ia registrar que **256 e o unico
valor utilizavel** e que `--convrot-groupsize` tem um valor so. Escrevi
`tools/probe_convrot_groupsize.py` para medir o alcance antes de publicar, e ele refutou:

    cg     quantize_convrot_w4a4_weight   convrot_w4a4_linear   quantize_w4a8_int8_weight
    16     ok                             ok                    RuntimeError
    64     ok                             ok                    RuntimeError
    128    ValueError (potencia de 4)     --                    RuntimeError
    256    ok                             ok                    ok
    512    ValueError (potencia de 4)     --                    RuntimeError
    1024   ok                             ok                    RuntimeError

**O limite nao e do kernel ConvRot: e do caminho W4A8.** O W4A4 aceita 16, 64, 256 e 1024 de ponta
a ponta -- quantizar E executar. O W4A8 aceita 256 e mais nada. A mensagem diz "convrot rotate
kernel" sem dizer de qual formato, e e por isso que ela induz ao erro.

`K = 2048` em todas as linhas, divisivel por todos os valores testados, entao nenhuma falha acima e
de divisibilidade.

## Por que isso trava ESTE experimento, especificamente

O unico conversor com perfil `zimage` e o `quant_mixed`, e ele mede os DOIS formatos por camada
para escolher entre eles -- entao ele toca o caminho W4A8 mesmo quando o resultado sera 170/170 em
W4A4. Logo esta pinado em cg 256.

O `quant_w4a4`, que so toca o W4A4 e aceitaria 1024, **nao tem perfil zimage**: os dele sao `gemma`,
`qwen` e `hunyuan_video_15`.

Entao o caminho para preencher a celula existe e custa uma coisa: um perfil `zimage` no
`quant_w4a4`. O `CLAUDE.md` proibe estender perfil para uma arquitetura nova sem confirmar o loader
e a configuracao de camada dela, e as duas tabelas `PROFILE_PATTERNS` da arvore tem significados
INCOMPATIVEIS (nome de tensor contra caminho de modulo), entao copiar a do `quant_mixed` e
exatamente a armadilha ja catalogada. Nao fiz numa janela autonoma.

## O que fica

**A celula do Z-Image continua vazia**, e a diferenca e que antes ela estava vazia por ninguem ter
tentado, e agora esta vazia com o obstaculo nomeado e medido. Um passo util foi dado no sentido
errado do que eu esperava: descobrir que o eixo existe, funciona no W4A4 e esta bloqueado pelo
conversor -- nao pelo kernel.

## Nao coberto

Uma versao de comfy-kitchen (0.2.31), uma placa, uma forma de peso por chamada no probe. Nao foi
medido o que acontece com um checkpoint JA escrito em cg 1024 -- se carrega e quebra no forward, ou
se nem carrega. E nada aqui mede QUALIDADE: 1024 aceitar nao diz que a imagem presta, que era a
pergunta original.


---

## 2026-09-03: o bloqueio caiu, e a hipotese mecanica estava INVERTIDA

### O bloqueio

Era o caminho W4A8, nao o ConvRot. `quant_mixed` mede os DOIS formatos por camada para escolher
entre eles, e o W4A8 so aceita `convrot_groupsize` 256, entao o conversor inteiro ficava pinado.
Resolvido com `--somente-w4a4`, que nao mede nem escreve W4A8 e por isso nao preflighta aqueles
dois ops. Ele **recusa** `--keep-bf16-error` e `--uncalibrated w4a8` em vez de escrever em silencio
um formato cujo criterio de selecao nao existiu.

Nao foi escrito perfil `zimage` no `quant_w4a4`: as duas tabelas `PROFILE_PATTERNS` da arvore tem
significados incompativeis (nome de tensor contra caminho de modulo) e copiar uma para a outra e a
armadilha que o `CLAUDE.md` ja cataloga. Reusar a selecao de camada que ja esta provada nesta
arquitetura custa menos codigo e nenhum perfil novo.

### A armadilha do conjunto de camadas, que quase produziu um numero sem sentido

Uma camada so entra se `shape[1] % convrot_groupsize == 0`. Rodando o conversor nos tres valores:
**170** camadas em cg 64, **170** em cg 256, **34** em cg 1024. Comparar a mediana de 170 com a
mediana de 34 OUTRAS nao mede granularidade -- mede qual subconjunto calhou de ser divisivel.
`tools/probe_teto_groupsize.py` mede sobre a **intersecao**, mesmas ativacoes, mesma referencia.

### O resultado: a alavanca existe, e anda para o outro lado

Sobre as 34 camadas que os tres valores aceitam:

    cg 64      0,1682     MAIS erro
    cg 256     0,1400
    cg 1024    0,1327     MENOS erro

Sobre as 150 camadas calibradas que 16, 64 e 256 aceitam:

    cg 16      0,1926     1,467x o cg256
    cg 64      0,1516     1,155x o cg256
    cg 256     0,1312

Quatro pontos, monotonicos, na direcao **oposta** a previsao 1 deste documento. O
`CONTROLE FALHOU` disparou exatamente como escrito: *"grupo menor NAO reduz o erro"*.

**O mecanismo que este documento afirmava esta invertido.** Ele dizia: grupo maior = rotacao mais
grossa = menos capacidade de espalhar outlier = mais erro. Uma rotacao de Hadamard de tamanho N
espalha cada outlier por N canais, entao N **maior** mistura MAIS, nao menos -- a incoerencia
melhora com o tamanho do grupo e o pico por grupo cai. "Mais grosso" descrevia a intuicao de um
quantizador por grupo, onde grupo maior significa uma escala para mais valores; a rotacao nao e
isso. O erro foi tratar duas coisas diferentes com a mesma palavra.

Isso nao invalida o experimento: **inverte qual braco e o candidato a quebrar.** Nao e o 1024, que
alem de melhor so alcanca 34 camadas. E o **cg 16**, que alcanca todas e mede 1,467x.

### Previsoes novas, escritas ANTES do render

1. `cg 16` (mediana 0,1926 na intersecao) **quebra a imagem**. E o unico braco que passa do topo do
   Hunyuan tolerado (0,1837) e chega perto do quebrado dele (0,2147).
2. `cg 64` (0,1516) **nao quebra**, no maximo degrada. Este e o controle intermediario: se ele
   quebrar junto com o 16, o corte fica entre 0,1312 e 0,1516 e a celula ganha um valor bem mais
   apertado -- ainda e resposta, so que outra.
3. O braco **BF16 tem de sair bom**. Se sair ruim, o problema esta no VAE, no encoder ou nos
   parametros, e nenhuma outra linha vale. E a licao que o Wan custou quatro renders.
4. Nao sei se a mediana por camada prediz a imagem. Esta bancada ja mediu que nao prediz: 0,1837
   correto contra 0,2147 destruido, e 0,1391 destruido contra 0,0956 bom. **Se os tres bracos
   sairem bons**, a resposta e que o eixo do groupsize nao alcanca o teto do Z-Image, e a celula
   continua vazia -- com um segundo obstaculo nomeado em vez de nenhum.

### Condicoes de refutacao, ainda validas

1. BF16 sai ruim -> nao publicar nenhuma linha.
2. cg 16 nao quebra -> o eixo nao alcanca o teto; a celula continua vazia, e isso vai escrito.

### Nao coberto, atualizado

A mediana medida sobre a intersecao **nao e comparavel** com o 0,1241 publicado sobre as 170 -- ela
sai sobre 150 ou 34 camadas conforme o conjunto. As comparacoes entre grupos sao validas porque
compartilham o conjunto; a comparacao com a tabela publicada nao e. `cg 1024` nao foi construido
como checkpoint: com 34 de 170 camadas quantizadas ele nao e o mesmo tipo de arquivo que os outros.


---

## RESULTADO, 2026-09-03: a celula esta preenchida, e as tres previsoes bateram

Render de quatro bracos, tres sementes, 8 passos a 1024px, mesmo prompt, mesmo VAE
(`bench/teto_zimage/imagens/folha_wan.png`):

    braco     mediana (170 camadas)   divergencia   imagem
    BF16                          -             -   boa            <- o controle, passou
    cg 256                   0,1216        0,5557   boa
    cg  64                   0,1421        0,6095   boa
    cg  16                   0,1848        0,7280   DESTRUIDA 3/3

As tres previsoes escritas antes bateram: `cg 16` quebra, `cg 64` nao quebra, BF16 sai bom. A
divergencia acompanhou a ordem, o que nem sempre acontece nesta bancada -- na esparsidade ela subiu
enquanto a imagem melhorava.

**A tabela de tolerancia agora tem tres linhas cheias e e monotonica nas DUAS colunas:**

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1421        0,1848
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147

E o detalhe que aperta a leitura: **0,1848 destroi o Z-Image de ~6 B e 0,1837 e tolerado no
HunyuanVideo de ~13 B.** As duas faixas nao se sobrepoem, e a distancia entre elas e 0,6% -- que e
exatamente o que se esperaria se o tamanho do modelo fosse o eixo. Tres pontos ainda sao tres
pontos; isso e hipotese com mais apoio, nao lei.

## O achado que vale mais que a celula: o avaliador era cego a este eixo

`tools/avaliar.py` casava o `.analysis.json` pelo **sha da fonte** e lia `err_w4a4` de la, sem
nunca olhar o `convrot_groupsize`. Como os tres builds tem a mesma fonte, os tres recebiam a
**mesma mediana 0,1216** -- o que produz uma imagem linda e o que produz lixo, com o mesmo numero e
o mesmo veredito. O erro por camada foi medido *em* um groupsize e nao vale nos outros.

Corrigido na ferramenta: `mediana_efetiva` descarta camadas cujo groupsize nao case com o da
analise, e quando nenhuma analise casa o arquivo ganha o achado `analise_de_outro_groupsize`
dizendo em que valores existem analises e em qual ele esta -- em vez de sair sem veredito como se
ninguem tivesse calibrado a fonte. Com as analises certas em disco, `cg 16` agora sai **REPROVADO**
por `erro_acima_da_quebra` e `cg 64` sai limpo.

## Uma validacao que veio de graca

`--somente-w4a4` em `convrot_groupsize` 256 produziu um arquivo **byte a byte identico** ao
`zimage-v2-w4a4` publicado -- mesmo tamanho, mesmo sha256. A flag nova nao mexeu no caminho em que
nao devia mexer.

## Nao coberto

Um prompt, tres sementes, um tamanho, uma placa, um modelo. O `cg 1024` nao entrou no render: com
34 de 170 camadas quantizadas ele nao e o mesmo tipo de arquivo. E o corte fica entre 0,1421 e
0,1848 -- um intervalo largo; nada foi medido no meio, entao "0,1848 quebra" e "0,1421 nao quebra"
sao os dois fatos, e o ponto exato da virada nao e um deles.
