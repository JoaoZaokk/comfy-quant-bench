# Critério — restaurar um grupo do corpo ternário ao BF16 (b6) — escrito 25/09, antes de montar

Pergunta do dono: o defeito do ternário (zebra vira leão+javali, "preto e branco" ignorado, braço com
duas pontas) está no que "lê" o prompt ou no que "desenha"? O text encoder (Qwen3-4B BF16) é o mesmo
arquivo em todos os braços e no BF16, então o defeito está no transformer.

Base: `klein4b_qat_b6_final_bfl` (resto já idêntico ao BF16, K0). Cada híbrido devolve ao BF16 original
um grupo de matrizes do corpo e mantém o resto ternário do b6:

| híbrido | grupo devolvido ao BF16 | matrizes | params |
|---|---|---|---|
| H1 | double blocks, fluxo txt (`txt_attn.qkv/proj`, `txt_mlp.0/2`) | 20 | 613 M |
| H2 | double blocks, fluxo img (`img_attn.qkv/proj`, `img_mlp.0/2`) | 20 | 613 M |
| H3 | single blocks 0–9 (`linear1`, `linear2`) | 20 | 1.227 M |
| H4 | single blocks 10–19 (`linear1`, `linear2`) | 20 | 1.227 M |

Render: avaliação fixa (10 prompts, semente 11) e grade principal (5 prompts x 11 12), mesmo protocolo
(8 passos, cfg 1, 1024 px), com b6 e BF16 nas mesmas folhas.

Julgado por mim nas folhas, por célula: **identidade** (o objeto pedido é o objeto: zebra é zebra,
garça é garça), **atributo** (preto e branco, aquarela, low-poly respeitados), **anatomia** (membros e
montagem coerentes) e **atrator** (forma branca curva / tecido / tigela).

## Previsões

- **P0 (nulo):** nenhum grupo sozinho conserta; a melhora é proporcional aos params devolvidos
  (H3 ≈ H4 > H1 ≈ H2). Refutada se um grupo de 613 M consertar mais que um de 1.227 M.
- **P1 [hipótese da literatura de Flux]:** identidade e atributo voltam mais com os double blocks
  (H1 ou H2) do que com os single; anatomia volta mais com os single blocks finais (H4).
- **P2:** o atrator mora onde a identidade é decidida: some no mesmo híbrido que devolve a identidade.

Não coberto: um grupo por vez (interação entre grupos não medida); single blocks não separam atenção
de MLP (`linear1` funde qkv e entrada do MLP); uma semente na fixa, duas na grade; julgamento meu.

## Resultado — renderizado 25/09 10:50-11:06 (`.scratch/restauracao_grupos.sh`)

Folhas: `bench/qat_klein/avaliacao_fixa/render_rest/folha_fixa.png` e
`bench/qat_klein/render_rest/folha_grade.png` (BF16 | b6 | H1 | H2 | H3 | H4). Os quatro híbridos
conferidos na montagem: 20 tensores do BF16, 129 do b6 cada. Julgamento meu, célula a célula:

| célula (fixa) | b6 | H1 txt | H2 img | H3 single 0-9 | H4 single 10-19 |
|---|---|---|---|---|---|
| zebra | leão+javali | **bicho listrado bebendo** | onça/zebra misturadas | forma listrada p&b | guepardo (errado) |
| navio a lápis (p&b) | colorido | **cinza, quase monocromático** | colorido | colorido | colorido |
| pavão | coruja-pássaro | **pássaro azul-esverdeado** | pássaro emplumado | pássaro estranho | pássaro ornamentado |
| moinho low-poly | bolha | **cruz de pás** | bolha | cruz de pás | bolha colorida |
| "SALE" | LAIT | SAAL | SALIS | SALE torto | **SALE SALE legível** |
| bonde | ausente | ausente | ônibus vago | bonde de frente | **bonde nítido** |
| guitarrista | torto | igual ao b6 | instrumento estranho | guitarra, melhor | **guitarra, melhor** |
| sopa | xícara estranha | igual ao b6 | tigela | tigela | **tigela limpa** |

- **P0 (nulo, proporcional aos params) refutada em parte.** H1 (613 M) devolve identidade e atributo
  que nem H3 nem H4 (1.227 M cada) devolvem (listras da zebra, navio monocromático, pavão, pás do
  moinho). H2 (613 M, fluxo img) quase não muda nada em relação ao b6.
- **P1 confirmada pela metade, e com endereço mais fino.** Identidade/atributo voltam com os double
  blocks, mas SÓ com o fluxo de TEXTO (H1), não com o de imagem (H2). Desenho, anatomia e texto
  renderizado voltam com os single blocks finais (H4, e parcialmente H3).
- **P2 refutada.** O atrator (forma branca curva, respingo branco) continua no H1, que devolve
  identidade, e diminui no H4 (navio, bonde, sopa sem o respingo), que não devolve identidade. O
  atrator anda com o DESENHO, não com a leitura.

**Resposta à pergunta do dono:** os dois quebraram, em lugares diferentes. "O que é" (qual objeto,
qual atributo vale para quem) passa pelo fluxo de texto dos double blocks; "como desenhar" (anatomia,
montagem, letras) e o atrator passam pelos single blocks finais. Nenhum grupo sozinho conserta tudo.

Consequência para o QAT: candidato a precisão mista é fluxo txt dos double (613 M, 17% do corpo) +
single 10-19 (1.227 M, 33%). Não medido: H1+H4 juntos, e esses grupos em 4 bits em vez de BF16.

Não coberto: uma semente na fixa; julgamento meu; um grupo por vez; attn x MLP não separados.

## Rodada 2 — H1+H4 juntos, em BF16 e em 4 bits (escrito 25/09, antes de montar)

- **H14:** b6 com fluxo txt dos double + single 10-19 devolvidos ao BF16 (40 matrizes, 1.840 M, ~50% do corpo).
- **H14q4:** os mesmos 40 grupos em 4 bits RTN simulado: absmax simétrico, grupo 32 no eixo K, níveis
  -7..7, dequantizado para BF16 no arquivo (mede qualidade, não tamanho; sem calibração, sem treino).
  O resto continua o ternário do b6.

Previsões:
- **Q1:** H14 junta os ganhos de H1 (identidade/atributo) e H4 (desenho, letras, menos atrator):
  zebra listrada E coerente, navio monocromático E sem respingo. Refutada se H14 ficar igual ao
  melhor dos dois sozinho.
- **Q2:** H14q4 fica perto de H14 (4 bits RTN g32 costuma custar pouco em peso de difusão). Refutada
  se H14q4 perder a identidade/atributo que H14 ganhou (volta a leão/colorido) em ≥ 3 das 4 células
  marcadas (zebra, navio, pavão, moinho).

Tamanho se valer, estimado: 40 grupos a 4,5 bits efetivos ~1,0 GB + 60 ternários a ~2 bits ~0,5 GB +
resto BF16, contra 7,75 GB do BF16 do transformer. Estimativa, não medida.

### Resultado rodada 2 — renderizado 25/09 11:09-11:18 (`.scratch/restauracao_r2.sh`)

Folhas: `bench/qat_klein/avaliacao_fixa/render_rest2/folha_fixa.png` e `bench/qat_klein/render_rest2/folha_grade.png`
(BF16 | b6 | H1 | H4 | H14 | H14q4).

- **Q1 confirmada, e acima da soma.** H14 não é "H1 + H4": é outro patamar. Garça é garça, sopa de tomate
  vista de cima com manjericão (a composição do BF16), guitarrista com guitarra coerente, raposa em
  aquarela limpa, navio a lápis monocromático e sem respingo, moinho low-poly com pás, bonde na rua,
  filhote e bule de vidro na grade, maçã vermelha. O que sobra: zebra listrada mas ainda com uma forma
  branca atrás (atrator residual), letras quase certas ("SAALE", "ORANK"), pavão branco, maçã duplicada
  numa semente.
- **Q2 confirmada.** H14q4 (40 matrizes em 4 bits RTN g32, sem calibração) é praticamente indistinguível
  do H14 em todas as 20 células: nenhuma célula perde identidade ou atributo.

Tamanho estimado do misto (não medido, não empacotado): 1.840 M em 4 bits a ~4,5 bits efetivos ≈ 1,03 GB;
1.840 M ternários a ~2,1 bits empacotados ≈ 0,49 GB; resto BF16 (~0,19 B params) ≈ 0,39 GB; **≈ 1,9 GB**
contra 7,75 GB do BF16. Um corpo inteiro em 4 bits RTN daria ≈ 2,5 GB -- o ternário da metade economiza
~0,55 GB a mais, e ainda não se sabe se o corpo todo em 4 bits já fica igual (controle C1 pendente).

Controles pendentes para a conclusão valer: **C1** corpo inteiro em 4 bits RTN g32 (sem ternário);
**C2** H14q4 com a metade ternária vinda do PTQ ingênuo (braco0) em vez do b6 -- diz se o QAT da
metade ternária importa.

## Controles C1 e C2 (escrito 25/09, antes de montar)

- **C1:** BF16 original com as 80 matrizes do corpo em 4 bits RTN g32 (mesmo quantizador do H14q4), sem ternário.
- **C2:** PTQ ternário ingênuo (braco0) com os 40 grupos do H14 em 4 bits RTN g32. Igual ao H14q4, só que a
  metade ternária é a do PTQ, não a do b6 treinado.

Previsões:
- **C1-a:** C1 fica pelo menos tão bom quanto o H14q4 (4 bits no corpo todo é menos agressivo que meio
  corpo ternário). Se C1 ficar pior, o QAT da metade ternária compensa algo que o RTN não compensa.
- **C2-a:** C2 fica claramente pior que o H14q4 -- a metade ternária do PTQ é ruído (braco0 tem epsilon
  1,06 contra 0,55 do b6). Refutada se C2 ≈ H14q4: aí o treino QAT da metade ternária não importaria.

### Resultado C1/C2 — renderizado 25/09 11:20-11:31 (`.scratch/restauracao_c1c2.sh`)

Folhas: `bench/qat_klein/avaliacao_fixa/render_c1c2/folha_fixa.png`, `bench/qat_klein/render_c1c2/folha_grade.png`
(BF16 | H14q4 | C1 | C2).

- **C1-a confirmada, com folga.** O corpo inteiro em 4 bits RTN g32 (sem calibração, sem treino) sai
  praticamente idêntico ao BF16 nas 20 células -- mesma composição, mesmo detalhe, letras certas
  ("SALE", "OPEN"), zebra, bonde, pavão azul, navio a lápis. É muito melhor que o H14q4, que ainda
  carrega a metade ternária do b6.
- **C2-a confirmada, ao extremo.** Com a metade ternária do PTQ ingênuo, o H14q4 vira ruído roxo nas 20
  células. O QAT da metade ternária é o que separa imagem de ruído -- mas nem ele chega perto do 4 bits.

**O que isto muda no projeto.** Para o Klein 4B, 4 bits RTN g32 no corpo inteiro já é quase sem perda
(≈ 2,5 GB estimados contra 7,75 GB). O ternário economizaria mais ~0,5-1 GB e, com a receita que temos,
custa qualidade visível mesmo na metade mais favorável. A pergunta útil passa a ser onde fica o
precipício entre 4 bits (limpo) e 1,58 bit (ruído): rodada 3, 3 bits.

## Rodada 3 — onde fica o precipício: 3 bits (escrito 25/09 11:40, antes de montar)

Mesmo RTN simétrico por grupo no eixo K, dequantizado para BF16, sem treino:
- **C3:** corpo inteiro 3 bits g32 (níveis -3..3).
- **C3g16:** corpo inteiro 3 bits g16.
- **M43:** fluxo txt dos double + single 10-19 em 4 bits g32; resto do corpo em 3 bits g32.

Previsões: **R3-a:** C3 perde visivelmente para C1 (3 bits RTN costuma ser o joelho), mas mantém
identidade na maioria das células (≥ 7/10). **R3-b:** g16 recupera parte da perda (C3g16 melhor que C3).
**R3-c:** M43 fica entre C3 e C1, e mais perto de C1 -- os dois grupos que importam em 4 bits.
Gate (erro relativo do quantizador em gaussiano, do teste da ferramenta): 4 bits 0,10; 3 bits ~0,19.

### Resultado rodada 3 — renderizado 25/09 11:40-11:53, epsilon até 12:29 (`.scratch/restauracao_r3.sh`)

Folhas: `bench/qat_klein/avaliacao_fixa/render_r3/folha_fixa.png`, `bench/qat_klein/render_r3/folha_grade.png`
(BF16 | C1 4b g32 | C3 3b g32 | C3g16 3b g16 | M43 | H14q4). Epsilon fora da amostra, 8 sementes
(`bench/qat_klein/agrega_eps_rest.txt`):

```
braço                       epsilon   render (meu julgamento, 20 células)
BF16 (referência)              --     --
C1   corpo 4 bits g32        0,221    quase idêntico ao BF16
M43  4b nos 2 grupos, 3b resto 0,355  identidade 20/20, composição mais perto do BF16 que o C3
C3g16 corpo 3 bits g16       0,378    identidade 20/20; letras escorregam ("SALL", "OPENF")
C3   corpo 3 bits g32        0,428    identidade 20/20; composição e luz derivam, detalhe mais mole
b6   ternário QAT            0,552    atrator, quimeras
H14  b6 + 2 grupos BF16      0,597    identidade quase toda; atrator residual
H14q4 b6 + 2 grupos 4 bits   0,603    idem
braco0 ternário PTQ          1,059    --
```

- **R3-a confirmada:** C3 perde para C1 (deriva de composição, detalhe), mas identidade 20/20.
- **R3-b parcial:** g16 melhora o epsilon (0,378 contra 0,428) e aproxima algumas composições, mas erra
  mais letras; no render não é melhor de forma clara.
- **R3-c parcial:** M43 fica entre C3 e C1 no epsilon (0,355) e visualmente mais perto do C1.
- **O epsilon mente na direção que mais importa:** H14 e H14q4 têm epsilon PIOR que o b6 (0,597 e 0,603
  contra 0,552) e render muito melhor. Nenhuma decisão desta série poderia ter sido tomada pelo epsilon.

**Conclusão até aqui:** para o Klein 4B, RTN sem treino em 3 bits (≈ 3,5 bits efetivos com a escala,
≈ 1,6 GB de corpo + 0,39 GB de resto ≈ 2,0 GB) domina a linha ternária inteira (H14q4 ≈ 1,9 GB,
qualidade visivelmente pior; ternário puro, 6 braços treinados, atrator). O precipício está entre
3 bits e 1,58 bit. Rodada 4 (2 bits RTN, níveis -1..1) localiza.

## Rodada 4 — 2 bits (escrito 25/09 12:35, antes de montar)

- **C2r:** corpo inteiro RTN 2 bits g16 (níveis -1..1 por absmax -- ternário de grupo pequeno, sem treino).
- **M32:** os 2 grupos em 3 bits g32 sobre o C2r.

Previsão: **R4-a:** C2r vira ruído ou quase (o PTQ ternário g128 absmean do braco0 é ruído; g16 absmax
não deve salvar). **R4-b:** M32 recupera imagem reconhecível só se os 2 grupos bastarem sozinhos em
3 bits -- aposta: não (o C2 mostrou ruído com o resto ternário do PTQ).

### Resultado rodada 4 — renderizado 25/09 12:30-12:34 (`.scratch/restauracao_r4.sh`)

Folha: `bench/qat_klein/avaliacao_fixa/render_r4/folha_fixa.png` (BF16 | C3 | C2r | M32 | H14q4).

- **R4-a confirmada:** C2r (corpo 2 bits RTN g16) é ruído liso nas 10 células.
- **R4-b confirmada:** M32 (os 2 grupos em 3 bits sobre o C2r) é ruído com sombra de estrutura. Os dois
  grupos, sozinhos, não seguram a imagem quando o resto cai para 2 bits sem treino.

## Fechamento da série (25/09)

```
corpo                                  treino   tamanho est.   render
BF16                                    --       7,75 GB       referência
4 bits RTN g32 (C1)                     não      ~2,5 GB       quase idêntico
3 bits RTN g32 (C3) / misto 4+3 (M43)   não      ~2,0 GB       identidade 20/20, deriva de composição
2 bits RTN g16 (C2r)                    não      ~1,5 GB       ruído
ternário PTQ (braco0)                   não      ~1,4 GB       ruído
ternário QAT (b6)                       sim      ~1,4 GB       reconhecível, atrator, quimeras
b6 + 2 grupos em 4 bits (H14q4)         sim      ~1,9 GB       bom, atrator residual -- pior que C3
```

O precipício sem treino fica entre 3 e 2 bits. O QAT desta bancada leva o ternário de ruído para
reconhecível, mas não alcança o 3 bits RTN, que não treina nada e tem tamanho parecido com o misto
H14q4. Para o ternário valer a pena, o QAT tem de bater o C3 com ~0,6 GB a menos; com a receita atual,
não bate. Caminhos que ainda fariam sentido (decisão do dono): QAT misto (os 2 grupos em 3-4 bits,
resto ternário, treinados juntos) ou QAT em 2 bits de 4 níveis em vez de ternário. Para uso prático
agora: 4 bits (quase sem perda) ou 3 bits (menor, com deriva), ambos sem treino -- falta o formato
nativo empacotado no ComfyUI para o tamanho deixar de ser estimativa.
