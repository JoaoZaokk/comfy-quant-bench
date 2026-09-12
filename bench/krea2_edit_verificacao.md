# Krea2 Edit funciona: 4 de 4, e cada uma mudou so o que foi pedido

Executado 2026-09-12 na 3090. Nao e leitura de codigo: sao quatro renderizacoes pelo servidor
real, com o workflow oficial `krea2_identity_edit`, e as imagens estao em
`bench/krea2_edit_4prompts.png`.

## O desenho do teste

Quatro **categorias diferentes** de instrucao, nao quatro variacoes da mesma -- se as quatro
saissem iguais, ou as quatro ignorassem a instrucao, isso seria informacao sobre o teste, nao
sobre o modelo.

    braco      instrucao                                            semente
    roupa      "Change her outfit to a red raincoat."                1000
    noite      "Make it night time, with a full moon and stars."     1001
    insercao   "Add a small brown dog sitting on the grass."         1002
    fundo      "Replace the grassy hill with a sandy beach..."       1003

**O controle e a propria imagem de origem.** Edicao com preservacao de identidade nao tem braco
BF16 de referencia: o veredito e "mudou o que foi pedido e so isso", e isso so se le lado a lado.
Por isso a origem ocupa a primeira celula da folha de contato.

## Resultado

    braco      status    tempo     arquivo
    roupa      success   181,3 s   krea2_identity_edit_00002_.png
    noite      success   ~180 s    krea2_identity_edit_00003_.png
    insercao   success   124,9 s   krea2_identity_edit_00004_.png
    fundo      success   182,9 s   krea2_identity_edit_00005_.png

Olhando as cinco lado a lado -- e este julgamento e de quem olha, nao de nenhuma metrica:

- **roupa**: vestido rosa -> capa de chuva vermelha com capuz, cordao, botoes e bolsos. Rosto,
  cabelo, pose, bracos e fundo preservados. Os bracos passam a estar dentro das mangas.
- **noite**: ceu azul -> noite com lua detalhada e estrelas, grama escurecida, nuvens mantidas e
  repintadas como iluminadas pela lua. **O sujeito nao foi tocado** -- vestido ainda rosa, mesma
  pose.
- **insercao**: cachorro marrom acrescentado na grama, no **mesmo estilo de desenho** da origem
  (preenchimento chapado, mesmo peso de contorno). Sujeito intacto.
- **fundo**: grama -> praia com areia, mar, espuma e horizonte, com pegadas na areia. Sujeito
  intacto, mesma pose, mesmo lugar no quadro.

Nas quatro, o **estilo** da origem sobreviveu: continua o mesmo desenho infantil chapado, nao
virou foto nem ilustracao de outro tipo.

## O defeito que este teste encontrou, e que nao era do modelo

A primeira versao do script consultava **duas portas fixas** (o pai 8190 e o worker 27715). A
segunda edicao caiu no worker **27716** -- `ComfyUI-MultiGPU` sobe um processo por device e o
`/history` do resultado e o **daquele worker**. O script ficou pendurado ate o timeout enquanto a
imagem ja estava gravada em disco, e teria reportado `TIMEOUT`, que le como "o modelo nao gerou".

Duas das quatro edicoes (`noite` e `fundo`) sairam do 27716. Ou seja: **metade deste resultado
era invisivel para o cliente anterior.** Corrigido em `tools/comfy_run_workflow.py:bases_irmas`,
que descobre os servidores lendo a linha de comando dos processos em vez de assumir portas, e
dentro do laco de poll, porque o worker nasce sob demanda.

## Nao coberto

- **Uma imagem de origem, e ela e um desenho infantil chapado, nao uma foto.** Nada aqui diz o
  que o modelo faz com um rosto real, com pele, com textura fina ou com um fundo fotografico.
- **Uma semente por instrucao.** Nao ha replica.
- **Ninguem mediu identidade.** Nao ha metrica de rosto, de pose ou de estilo; quem diz que a
  pessoa continua a mesma e alguem olhando.
- Tudo isto rodou sobre o `krea2_turbo_int8_convrot` (o build int8 publico) com o LoRA
  `krea2_identity_edit_v1_2`. **Nada aqui diz o que a nossa quantizacao faz com a edicao** -- o
  criterio `bench/criterio_quant_krea2.md` deixa isso explicitamente de fora.

---

# O Edit sobre o NOSSO W4A4 (2026-09-12, depois dos quants)

Ate aqui a edicao so tinha sido verificada sobre o `krea2_turbo_int8_convrot` **publico** -- o
arquivo que o dono vai rodar e o nosso. Um eixo variado: o checkpoint de difusao. Mesma imagem
de origem, mesmas quatro instrucoes, **mesmas sementes** (1000-1003), mesmo LoRA de identidade.

    braco      status    tempo    | int8 (medido antes)
    roupa      success   161,5 s  | 181,3 s
    noite      success   177,3 s  | ~180 s
    insercao   success    88,3 s  | 124,9 s
    fundo      success   ~180 s   | 182,9 s

**4/4, e a identidade sobrevive**: rosto, formato do cabelo, pose, bracos e vestido intactos nas
quatro, no mesmo estilo chapado da origem. `bench/krea2_edit_int8_vs_w4a4.png`.

O **LoRA de identidade carrega e faz efeito sobre o W4A4 quantizado** -- nao houve erro nem
recusa, e as edicoes respondem a instrucao, o que nao aconteceria com o LoRA inerte.

## O desvio, e o controle que ele exigiu

Na semente 1002 o int8 pos **um** cachorro e o nosso W4A4 pos **dois**, onde a instrucao diz
*"a small brown dog"*, singular. Com uma celula so nao da para distinguir dano da quantizacao de
sorte do sampler, entao rodei o controle: a mesma instrucao, nos dois checkpoints, em duas
sementes novas (`bench/krea2_edit_controle_cachorro.png`).

    braco   s1002        s2002        s3002       cachorros corretos
    int8    1 cachorro   1 cachorro   1 cachorro        3 de 3
    W4A4    2 cachorros  1 cachorro   1 cachorro        2 de 3

**1 de 3 contra 0 de 3.** Isso nomeia o desvio sem fechar a questao: uma ocorrencia em tres nao
separa "o W4A4 duplica as vezes" de "esta instrucao duplica as vezes e o int8 teve sorte tres
vezes". Esta bancada ja registrou exatamente esse erro -- o criterio ponderado por sigma "venceu
3/3" e a vitoria dissolveu em oito sementes. **O que fecharia:** oito sementes por braco, ou uma
segunda instrucao com contagem explicita ("two dogs") para ver se o erro e de contagem ou de
duplicacao.

Um segundo desvio na mesma tabela, menor e nao controlado: na semente 2002 o fundo do W4A4 saiu
bem mais escuro e esverdeado que a origem, enquanto as tres celulas do int8 mantiveram o ceu
azul. Uma ocorrencia, sem replica.

## Leitura

A edicao **funciona** sobre o nosso W4A4 e e ~1,2x mais rapida que sobre o int8, com 1,68x menos
disco. O que nao se pode dizer e que a fidelidade a instrucao e igual: em 6 celulas comparaveis o
unico erro de contagem apareceu no W4A4, e a direcao concorda com o que o ladder ja tinha medido
(int8 2,40x mais fiel ao BF16, vencendo 10 de 10 pareado). **Nao e prova; e um segundo sinal na
mesma direcao, com n pequeno demais e dito assim.**

## Nao coberto

Uma imagem de origem, e ela e um desenho chapado -- nada aqui diz o que o W4A4 faz com rosto
real numa edicao. Tres sementes so na instrucao do cachorro; as outras tres instrucoes tem uma.
Contagem de cachorro e olho humano, nao metrica. O braco `misto` nao foi testado na edicao.

---

# FECHAMENTO do desvio do cachorro -- 2026-09-12, 51 renderizações

O desvio ficou aberto em **1 de 3 contra 0 de 3**, que nomeia e não fecha. Fechado agora, e o
resultado é o contrário do que a primeira leitura sugeria: **não era sorte do sampler.**

## O que eu quase concluí errado, e por quê

A primeira varredura foram 8 sementes (2002..9002) x 3 braços x 2 instruções = 48 renderizações,
**todas com exatamente um cachorro**. Isso lê como "o desvio não reproduz" -- e estaria errado,
porque **nenhuma das 48 incluía a semente 1002**, que é justamente onde ele tinha acontecido.
Varrer ao redor do ponto não é medir o ponto.

E havia um segundo defeito no meu desenho: nas primeiras 24 eu variei **dois eixos de uma vez**,
o número de sementes E a instrução (pus "exactly one"). Com os dois andando juntos, nada
distinguiria "sumiu por causa das sementes" de "sumiu por causa do numeral". É
`teste-varia-o-eixo-errado`, cometido por mim no teste escrito para fechar outro erro meu.

## O 2×2 na semente que importa

Três renderizações por célula não: uma por célula, porque a edição é determinística na semente.

    semente 1002              int8 publico   nosso W4A4   nosso misto
    instrucao ORIGINAL          1 cachorro    2 CACHORROS   1 cachorro
    com "exactly one"           1 cachorro    1 cachorro    1 cachorro

`bench/krea2_edit_cao_semente1002.png`. Duas leituras saem daí, e as duas importam:

1. **O desvio é do W4A4, e reproduz.** Não é ruído: na mesma semente, mesma instrução, mesmo
   LoRA, mesma origem, o int8 e o misto põem um cachorro e o W4A4 põe dois.
2. **Nomear a contagem na instrução corrige.** A mesma semente, mesmo braço, com "exactly one"
   volta para um cachorro.

## A contagem completa

    braco          instrucao original    com "exactly one"
    int8 publico    0 de 9 duplicaram     0 de 8
    nosso W4A4      1 de 9 duplicaram     0 de 8
    nosso misto     0 de 9 duplicaram     0 de 8

Nove sementes: 1002 e 2002..9002. Folhas: `krea2_edit_cao_8sementes.png` (as oito, três braços) e
`krea2_edit_cao_semente1002.png` (o 2x2).

**O braço `misto` entra na edição aqui pela primeira vez** e não duplica em nenhuma das nove. Ele
promove a 8 bits as 51 camadas de maior erro -- `attn.wo` e `mlp.down`, as duas projeções de
saída -- e custa 0,20 GiB a mais que o W4A4 puro. Isso é consistente com o desvio acompanhar o
erro por camada, mas **é uma coincidência de duas medições, não uma causa demonstrada**: ninguém
variou a promoção camada a camada para ver o desvio aparecer e sumir.

## O que continua NÃO coberto

Uma imagem de origem, e ela é um desenho infantil chapado. Uma instrução de inserção, um objeto,
um LoRA de identidade. Nove sementes: **1 em 9 não se distingue de 0 em 9 com confiança**, o que
esta linha afirma é que o evento existe e reproduz naquela semente, não a sua frequência. Contar
cachorro é olho humano, não métrica. E nada aqui testa se o mesmo acontece com outra contagem
("dois cachorros") ou com outro objeto.
