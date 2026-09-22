# Criterio: a receita do Bonsai sai do Flux? E da para tentar no Qwen-Image?

Escrito **2026-09-22, antes de qualquer numero** deste round. Pergunta dele: "seria possivel replicar
uma semelhanca disso no Qwen-Image 2.1? Nao sei a replicabilidade para fora do Flux."

Separo tres coisas que costumam virar uma so, porque so a terceira precisa de GPU:

**(A) A REGRA DE SELECAO** -- quais camadas ficam densas. Medida no klein-4B por dois times
independentes (Prism/treino e Nunchaku/PTQ), intersecao 69 de 69: e 1-D + todo 2-D fora dos blocos.
**Testavel agora, em CPU, sem baixar nada.**

**(B) A CONTA DE TAMANHO** -- que arquivo sairia. Aritmetica sobre o header.
**Testavel agora, em CPU, e de graca por Range HTTP para modelos que nem estao no disco.**

**(C) O TREINO** -- a parte que faz o Bonsai funcionar. **Nao replicavel aqui em escala.** O que E
replicavel e a hipotese que eu mesmo levantei no relatorio deles: um **estagio de compensacao** --
congelar o corpo ternario e ajustar SO o conjunto denso contra a saida do modelo denso. No klein-4B
isso e 195.042.816 params treinaveis, nao 3,68 bilhoes. Isso cabe numa 3090. **Fila de GPU.**

---

## Previsoes refutaveis, parte A e B (CPU, hoje)

**Q1 -- no Qwen-Image a modulacao mora DENTRO dos blocos, como no Flux.1-dev, nao fora como no klein.**
E o eixo que decide o piso: no klein as 4 modulacoes de fora sao 141.557.760 params que nunca
encolhem (25,33% do arquivo), no Flux.1-dev 76 de 77 ficam dentro.
*Refuta se* a modulacao do Qwen estiver fora dos blocos.

**Q2 -- a regra de selecao (A) vale no Qwen: o conjunto denso do SVDQuant do Qwen e exatamente
"1-D + 2-D fora dos blocos".**
Temos o par no disco: `qwen_image_2512_bf16` (38,05 GiB) e `svdq-int4-qwen-image-2512-balance`.
*Refuta se* o SVDQuant do Qwen deixar densa alguma camada DENTRO de bloco, ou quantizar alguma de
fora. Qualquer um dos dois quebra a generalidade da regra e a conta de (B) passa a ser chute.

**Q3 -- no Qwen o SVDQuant tambem quase nao toca o conjunto denso: >= 90% byte a byte identicos ao
bf16, e os que mudam sao os de modulacao.**
E o mesmo controle positivo que no klein deu 66/69 contra 1/69 do Bonsai. Se der aqui tambem, a
assinatura "PTQ nao reescreve o denso" e propriedade do metodo, nao do modelo.
*Refuta se* ficar abaixo de 90%, ou se os que mudam nao forem os de modulacao.

**Q4 -- o bit/peso medio projetado do Qwen-Image fica mais perto do Flux.1-dev (2,58) que do
klein-4B (3,18).**
Mecanismo: modelo de 20 B dilui o piso. Previsao concreta: **abaixo de 2,70**.
*Refuta se* passar de 2,70.

**Q5 -- a regra (A) NAO vale em todas as familias.** Previsao de que ela quebra em pelo menos uma das
outras do disco (LTX, Wan, Hunyuan, Z-Image, Krea2), porque nem toda arquitetura poe tudo que importa
dentro de uma pilha numerada. Este repo ja registra **tres** extrapolacoes entre arquiteturas que
morreram na medicao, e eu nao vou fazer a quarta sem olhar.
*Refuta se* todas as familias se comportarem igual.

## O controle que TEM de falhar

Rodar o teste de conjunto denso do **SVDQuant do Qwen contra o original do FLUX.2-klein-4B**. Sao
arquiteturas diferentes: tem de dar **zero nomes em comum**. Se der qualquer coisa acima de zero, o
casamento de nomes esta frouxo e Q2/Q3 nao valem.

---

## Parte C: o que vai para a fila da GPU, e com que criterio

O experimento e o menor que ainda falsifica a hipotese, e roda no **klein-4B primeiro, porque ali
existe gabarito**: o Bonsai publicado.

    braco 0  PTQ ternario ingenuo (absmean g128 no K), denso intacto     <- controle inferior
    braco 1  o mesmo PTQ, e depois SO o conjunto denso ajustado contra
             a saida do modelo bf16 num conjunto pequeno de calibracao   <- a hipotese
    braco 2  o Bonsai ternario publicado                                 <- controle superior
    braco 3  o bf16 original                                             <- referencia

Previsao a escrever ANTES de rodar, quando a GPU liberar: se o estagio de compensacao e o mecanismo,
o braco 1 fecha uma fracao mensuravel da distancia entre 0 e 2 na metrica de SAIDA (epsilon por passo
com trajetoria imposta, que e o instrumento que esta bancada ja validou -- **nao** imagem livre, que
mede caos). Se o braco 1 empatar com o braco 0, a hipotese morre e o que o Bonsai faz nao e
compensacao de um conjunto pequeno.

**Nao vou rodar isso antes de escrever a previsao numerica e o criterio de parada**, e a GPU esta em
uso dele agora.

## O que nada disto responde

- (B) e aritmetica: diz tamanho de arquivo, **nunca** qualidade.
- (A) valer no Qwen **nao** diz que o treino a 1,58 bit funcionaria no Qwen.
- Nada aqui replica as metricas publicadas deles, e nada aqui gera imagem.
