# Criterio da guarda do braco de referencia

**Escrito em 2026-09-01, ANTES de rodar qualquer medicao.** A regra desta bancada e que so se
fecha uma divida sem consertar quando o criterio estava escrito antes de alguem olhar o resultado.
Este arquivo existe para que a proxima sessao saiba se a guarda funcionou ou se eu movi a trave.

## O problema que ela existe para pegar

Em 2026-09-01 o Wan 2.1 VACE 1.3B custou **quatro renderizacoes** porque o braco de referencia
estava quebrado e ninguem olhou. O FP16, sem quantizacao nenhuma, saiu como trama tecida em toda
configuracao tentada. A primeira leitura foi "a quantizacao destruiu o modelo, divergencia 1,2365"
-- um numero que nao media nada, porque os dois lados estavam quebrados.

Causa: `WAN21_Vace.extra_conds` preenche `vace_frames` com zeros quando nao ha no de controle,
passa por `process_latent_in` (zero vira **nao** nulo), concatena mascara de uns e aplica em
`vace_strength = 1.0`. **Nao e "sem controle": e um controle constante em forca total.** Sem erro,
sem aviso.

A camada 1 do `avaliar.py` ja marca estaticamente qualquer checkpoint com `vace_blocks`. Isso pega
**esta** armadilha porque ela ja e conhecida. A camada 3 tem que pegar a **proxima**, que ninguem
viu ainda.

## A hipotese

Um braco de referencia sao **responde ao proprio condicionamento**. Um braco dominado por um
controle constante nao responde: a saida e a mesma trama independentemente do que se pede.

Entao a guarda nao pergunta "esta imagem esta boa" -- essa pergunta esta proibida nesta bancada,
porque nenhum corte medido aqui separa usavel de inutilizavel. Ela pergunta **"este modelo esta
ouvindo?"**, que e uma pergunta com resposta.

## A medicao

Tres amostragens do modelo NAO quantizado, tudo igual menos o eixo variado:

```
A   prompt P1, semente S1
B   prompt P2, semente S1      (P2 bem diferente de P1)
C   prompt P1, semente S2
```

```
d_prompt = |lat_B - lat_A| / |lat_A|      quanto o PROMPT move a saida
d_semente = |lat_C - lat_A| / |lat_A|     quanto a SEMENTE move a saida
resposta = d_prompt / d_semente
```

`d_semente` esta ali como **controle negativo**, e e ele que torna a medida interpretavel: um
`d_prompt` pequeno sozinho pode significar so que o modelo e estavel. Pequeno *em relacao ao que a
semente move* significa que o condicionamento nao esta chegando.

## A previsao, escrita antes

Medindo o par de verdade conhecida do Wan -- `vace_strength 1.0` (destruido, verificado por quem
olhou) contra `0.0` (bom, verificado por quem olhou):

| braco | previsao |
|---|---|
| `vace_strength 1.0`, destruido | `resposta < 0,5` |
| `vace_strength 0.0`, bom | `resposta > 0,8` |

E o Z-Image v2 BF16, um segundo braco sao de outra familia, tambem `resposta > 0,8`.

Limiar operacional proposto: **`resposta < 0,5` reprova o braco de referencia.**

## O que REFUTA esta guarda

Escrito antes de propria medicao, para nao ser reescrito depois:

1. **Se os dois bracos do Wan derem `resposta` parecida** -- digamos, dentro de 30% um do outro --
   a guarda nao separa quebrado de bom e nao serve. Isso e plausivel: a alto sigma a semente domina
   tudo, e o denominador pode engolir o sinal nas duas pontas.
2. **Se o braco BOM do Wan der `resposta < 0,5`**, o limiar reprova arquivo bom e esta errado.
3. **Se o braco destruido der `resposta > 0,8`**, a hipotese esta errada: o modelo estava ouvindo o
   prompt e mesmo assim produziu trama.

Qualquer um dos tres e um resultado negativo e vai ser registrado como tal, nao consertado com um
limiar novo. Um limiar escolhido depois de ver os numeros nao e uma guarda, e uma descricao.

## O que esta guarda NAO faz, mesmo se funcionar

- **Nao diz que a imagem esta boa.** Um modelo pode ouvir o prompt perfeitamente e ainda assim
  gerar lixo. Ela detecta uma classe especifica de quebra: condicionamento nao chegando.
- **Nao substitui olhar.** Continua sendo verdade que so uma renderizacao vista por alguem decide.
- Nao mede o braco quantizado -- e sobre a **referencia**, que e o que ninguem confere.
- Uma placa, um scheduler, um tamanho, um numero de passos.

---

# Resultado, 2026-09-01

Medido na 3090 com `CUDA_VISIBLE_DEVICES=0`, `tools/avaliar_referencia.py`, laudos em
`bench/guarda_referencia.json`.

| braco | resposta | d_prompt | d_semente | veredito | previsao | acertou |
|---|---|---|---|---|---|---|
| Wan VACE `strength 1.0` — **destruido**, verdade conhecida | **0,2477** | 0,075 / 0,055 | 0,250 / 0,285 | REPROVADO | `< 0,5` | **sim** |
| Wan VACE `strength 0.0` — **bom**, verdade conhecida | **0,7911** | 0,905 / 0,755 | 1,098 / 0,996 | OLHAR | `> 0,8` | **nao**, por 1,1% |
| Z-Image v2 BF16 — bom, segunda familia | **1,3459** | 1,064 / 1,150 | 0,862 / 0,789 | SEM VEREDITO | `> 0,8` | sim |
| HunyuanVideo 1.5 FP16 — bom, terceira familia | **2,8603** | 1,584 / 1,481 | 0,846 / 0,385 | SEM VEREDITO | `> 0,8` | sim |

## A guarda funciona, e as tres condicoes de refutacao ficaram todas em silencio

1. Os dois bracos do Wan **nao** deram resposta parecida: **3,19x** de distancia entre 0,2477 e
   0,7911. Contra a mediana dos bracos sadios, 5,4x.
2. O braco **bom** do Wan **nao** ficou abaixo de 0,5. O limiar de reprova nao rejeita arquivo bom
   em nenhum dos tres sadios medidos.
3. O braco **destruido** **nao** ficou acima de 0,8.

**Ela teria abortado a rodada do Wan na primeira imagem em vez da quarta.** E o motivo dela existir.

E o mecanismo aparece cru no numero, nao so na razao: no braco destruido o prompt move o latente
**0,075**, enquanto a semente move **0,250**. O modelo esta gerando a partir do ruido e ignorando
o que se pede. Nos tres sadios o prompt move de 0,76 a 1,58.

## A previsao do limite SAO estava errada, e ele nao vai ser mexido

O braco bom do Wan deu **0,7911** contra os `> 0,8` previstos. Erro de 1,1%, e ele cai na faixa
`OLHAR`.

**O limiar continua 0,8.** Mudar um limiar depois de ver o numero que ele deveria ter classificado
nao e calibrar: e descrever. O registro fica assim -- previsao errada, limiar intacto -- e quem
quiser um limite superior defensavel precisa de bracos sadios novos, medidos depois, nao dos
mesmos que testaram este.

Na pratica o custo dessa previsao errada e pequeno: um braco sao recebe "olhe a imagem", que e um
conselho sempre verdadeiro nesta bancada. O erro caro seria o outro -- reprovar arquivo bom -- e
esse nao aconteceu.

## O que este resultado NAO estabelece

O lado da **reprova** se apoia em **um** braco quebrado de verdade. Um so. Os tres sadios sao de
tres familias, mas o quebrado e um caso, com uma causa, num modelo. A guarda pode nao reconhecer
uma quebra de condicionamento com outro mecanismo.

A razao tambem e **barulhenta em valor absoluto**: no Hunyuan as duas sementes deram 1,87 e 3,85,
espalhamento de 2,06x. A direcao e estavel, o numero exato nao e -- entao a distancia ao limiar
importa, e nao a segunda casa decimal.

E `d_semente` varia de 0,38 a 1,10 entre familias e ajustes, o que e exatamente por que a
estatistica e a **razao** e nao o `d_prompt` sozinho.

Nada aqui decodifica imagem: a medida e no latente. Um modelo que legitimamente responde pouco ao
prompt -- um refinador, um upscaler, um modelo de controle -- reprovaria sem estar quebrado.
