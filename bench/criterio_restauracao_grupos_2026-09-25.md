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
