# Criterio: o runtime do Z-Image de ponta a ponta, com 6 prompts

Escrito em 2026-09-12 **antes** de rodar qualquer braco. O dono pediu explicitamente
"pelo menos 4 prompts diferentes para nao cair no erro de gerar a maca, que por acaso,
funciona no modelo" -- e ele esta certo: **todo numero de imagem que esta bancada publicou
sobre o Z-Image saiu de tres prompts**, e um deles e a maca.

## Por que a maca continua na lista

Ela vira **controle positivo**, nao amostra. A maca e o unico prompt sobre o qual esta
bancada tem afirmacao publicada e verificada: `zimage-v2-w4a4` renderiza bem, 3 sementes,
no card do HuggingFace. Se a maca sair quebrada nesta rodada, o problema e a rodada -- stack
mudou, arquivo baixado errado, parametro trocado -- e nao o modelo. Sem ela, um resultado
ruim nos cinco prompts novos seria ambiguo entre "o modelo falha nisso" e "eu montei errado".

## Os seis prompts, e o que cada um esta testando

    1  maca            CONTROLE. Objeto unico, luz suave. Sabidamente bom.
    2  rosto idoso     Rosto humano. Onde a quantizacao costuma quebrar primeiro.
    3  maos no barro   Maos. Falham antes do resto em quase todo modelo.
    4  placa "OPEN"    Texto DENTRO da imagem. Folclore diz que e o primeiro a sumir.
    5  mercado noturno Cena com muitos objetos, oclusao, gente. Testa composicao.
    6  cristais gelo   Textura fina de alta frequencia. Onde granulado apareceria.

Seis, nao quatro, porque tres eixos distintos (rosto, maos, texto) nao cabem em quatro
prompts junto com o controle e uma cena.

## Os bracos

    A  beyond-reality-zimage-v2_native.safetensors   BF16, a referencia
    B  zimage-v2-w4a4.safetensors                    170 camadas convrot_w4a4 puro
    C  zimage-v2-mixed.safetensors                   115 x 4 bits / 55 x 8 bits

B e C foram **apagados do disco em 2026-09-12 e rebaixados do HuggingFace** para esta
rodada. `zimage-v2-w4a4` voltou com 3290988448 bytes, identico ao que saiu. Isso e, de
quebra, o teste da alegacao "esta no HF, volta por download" que a limpeza usou como
justificativa.

## Previsoes, escritas antes

1. **A maca passa nos tres bracos.** Se nao passar, a rodada nao vale e nada abaixo se le.
2. **Despacho: B da 340 forwards quantizados e 0 `dequantize`.** E o que o `CLAUDE.md`
   registra de 2026-08-31. Se vier `dequantize`, os bracos B e C sao duas dequantizacoes
   comparadas entre si e a rodada inteira nao mede quantizacao.
3. **O braco de referencia responde ao proprio condicionamento**, `d_prompt / d_semente`
   acima de 0,8. Medido 1,3459 em 2026-09-01.
4. **Rosto e maos degradam mais que a maca.** [JULGAMENTO] -- e a intuicao corrente, e esta
   bancada **nunca mediu isso**. E a previsao que eu mais espero ver falhar, porque o
   `adaLN_modulation` ja foi um caso em que "todo mundo sabe" estava errado.
5. **O texto da placa degrada mais que tudo.** [JULGAMENTO], mesmo status.
6. **C (misto) fica mais perto de A que B**, porque tem erro por camada menor (0,0774
   contra 0,1241). **Ressalva que vale mais que a previsao**: esta bancada ja mediu que erro
   por camada NAO preve a imagem livre -- em 8 passos uma perturbacao minima desvia o
   sampler e o destino continua sendo uma imagem boa. Entao espero que esta previsao seja a
   menos confiavel das seis.

## Condicoes de refutacao -- se qualquer uma disparar, a leitura muda ou para

- **A maca quebra em A.** A referencia esta quebrada. Para tudo; foi o que custou quatro
  renders no Wan em 2026-09-01.
- **Despacho acusa `dequantize` em B ou C.** A comparacao vira BF16 contra BF16.
- **A e B saem identicos.** Os bracos nao diferiram; algum arquivo foi carregado duas vezes.
- **Os seis prompts dao o mesmo veredito.** Se tudo passa ou tudo quebra igual, os prompts
  nao separaram nada e o desenho nao serviu -- o que seria informacao sobre o desenho, nao
  sobre o modelo.

## O que esta rodada NAO vai cobrir, dito antes

Uma placa (sm86), um sampler, um tamanho, um numero de passos, sem metrica perceptual --
quem decide se a imagem presta e alguem olhando. Seis prompts sao seis amostras. E o
`avaliar.py` ja recusa a palavra APROVADO de proposito: nenhum corte nesta bancada separa
usavel de inutilizavel.

---

# RESULTADO -- medido 2026-09-12, depois do criterio acima

## Controles, primeiro

    despacho (zimage-v2-w4a4)   170 modulos convrot_w4a4, 8/8 forwards quantizados,
                                0 dequantize, linear_dtype=int4,
                                impl comfy_kitchen.backends.cuda, pesos 170/170 em cuda:0
    braco de referencia         resposta 1.3459 (d_prompt 1.0642/1.1504 contra
                                d_semente 0.8623/0.7892) -- o MESMO valor de 2026-09-01,
                                quatro casas decimais, com o arquivo recarregado do disco
    controle da maca            passa nos tres bracos, nas duas sementes

Nenhuma condicao de refutacao disparou. A rodada vale.

**Previsao 2 estava mal escrita e isso vale registrar:** eu previ "340 forwards" citando o
`CLAUDE.md`, mas aquele numero veio de uma passada COM amostragem; `--forward-only` chama 8.
Modo diferente, mesma conclusao -- so que a previsao, do jeito que estava escrita, nao
poderia bater. Previsao tem de nomear o modo de medicao.

## Os numeros

    braco                            divergencia  espalhamento  s/step    GiB  runs
    beyond-reality-zimage-v2_native            -             -   0.898  11.46    12
    zimage-v2-w4a4                        0.6615        0.4470   0.348   3.06    12
    zimage-v2-mixed                       0.5953        0.6881   0.421   3.18    12

W4A4 puro: **2.58x mais rapido por passo e 3.74x menor** que o BF16.

## As duas previsoes que CAIRAM, e e esse o achado

**Previsao 4 (rosto degrada mais que a maca): FALSA.** Seis renders de rosto -- duas
sementes, tres bracos -- e todos saem com pele, rugas, barba e olhar coerentes. Nao ha
degradacao visivel de rosto em nenhum braco quantizado.

**Previsao 5 (texto na imagem degrada mais que tudo): FALSA, e a mais limpa das duas.** A
placa `OPEN` sai **legivel e bem formada em 6/6 renders quantizados**. No W4A4 da semente 1
ela vem ate com letra vermelha e desgaste de esmalte, ou seja, o modelo nao so escreveu como
estilizou. "Texto e o primeiro a sumir" e folclore que este par de folhas nao sustenta.

As duas eram [JULGAMENTO] declarado, e a previsao 4 dizia explicitamente "e a que eu mais
espero ver falhar, porque o `adaLN_modulation` ja foi um caso em que todo mundo sabia e
estava errado". Foi.

## O que muda entre os bracos NAO e qualidade, e composicao

Enquadramento diferente, pose diferente, lanterna em outro lugar, cor da placa diferente.
Nenhum braco produz artefato, borrao ou estrutura quebrada. Isso reproduz, em imagem, o que
esta bancada ja tinha medido no epsilon: **em 8 passos uma perturbacao minima desvia o
sampler e o destino continua sendo uma imagem boa.** Por isso a divergencia de 0.6615 nao
significa "pior", significa "outra" -- e por isso o proprio `CLAUDE.md` proibe usar imagem
livre para comparar duas quantizacoes do mesmo modelo.

## Previsao 6 (misto mais perto que o puro): direcao certa, forca insuficiente

0.5953 contra 0.6615, e o misto vence 9/12 pareado -- mas a ferramenta carimba **"nao
separado em 12 runs"**. Continua sendo hint, nao resultado, que foi exatamente o status que
o criterio previu para ela.

## Dois defeitos de FERRAMENTA achados por esta rodada

1. **`quality_ladder.py --prompt-file` colava o arquivo INTEIRO num unico prompt**
   (`prompts.append(read_text())`). Seis prompts viravam um prompt de seis linhas e o
   cabecalho anunciava "1 prompt(s) x 2 seed(s)". Nao falhava -- rodava, e rodava errado.
   So foi visto porque o dono pediu explicitamente varios prompts. Consertado: uma linha um
   prompt, `#` e linha vazia ignoradas, e agora imprime os prompts lidos.
2. **`monta_grade.py` so conhecia `<braco>_s<semente>.png`.** Apontada para uma pasta de seis
   prompts ela montava uma folha 3x2 vazia com "6 celulas ausentes" em vez de erro. Ganhou
   `--prompts`.

Os dois tem a mesma forma: **a ferramenta feita para um prompt so, silenciosamente errada
quando recebe varios** -- que e precisamente o vies que o dono mandou corrigir.

## Nao coberto

Seis prompts, duas sementes, um tamanho, 8 passos, um sampler, uma placa sm86, sem metrica
perceptual -- quem julga a imagem e alguem olhando, e quem olhou fui eu. O decode falhou na
primeira tentativa (`comfy_aimdo` / `hostbuf_allocate`) e as imagens sairam de
`tools/decode_latents.py` sobre os latentes gravados; o proprio decodificador avisa que nada
ali confere que o latente veio do modelo que o nome diz.
