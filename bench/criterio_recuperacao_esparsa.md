# Criterio: recuperacao por camada sobre poda 2:4

Escrito em 2026-09-03 **antes** de rodar `tools/recupera_esparso.py`, numa janela de GPU que o dono
abriu para trabalho autonomo. A proposta e dele, de 2026-09-01, e e a unica frente aberta na
esparsidade depois que a foto matou o INT4 por par.

## De onde vem a pergunta

Medido em 2026-09-01: o tensor core esparso da sm_80 executa 2:4 a 1,7x (bf16), 3,2-4,1x (int8) e
5,0-7,6x (int4) sobre o denso, bit-exato, e o obstaculo nunca foi o `cuSPARSELt` nem a placa -- era
um tile dimensionado para a A100.

E o render disse nao. Isolando um eixo por vez, o culpado e a **granularidade** que o INT4 impoe: a
mascara dele e o PAR, e podar em pares destroi a imagem mesmo em bf16, sem quantizar nada. O INT8
aceita por ELEMENTO, preserva o sujeito, e ainda assim esta ruim demais para enviar.

    W4A4 ConvRot (hoje)                 0,0956   imagem boa
    2:4 por elemento, magnitude         0,1849   sujeito arruinado mas presente
    2:4 por par, magnitude              0,2587   ruido
    2:4 por par, Wanda, + int4          0,1391   ruido

Recuperacao e o que pode fechar a distancia entre 0,1849 e 0,0956. Nunca foi testada aqui.

## O metodo, e por que este

Nao e fine-tuning. Com a mascara ja escolhida, achar os pesos sobreviventes que minimizam o erro de
**saida** nas ativacoes reais: `min ||X Ws^T - X W^T||` com `Ws = M (*) Ws`. E minimos quadrados por
linha, e o sistema de cada linha e `H = X^T X` restrita ao suporte. Resolvido por **gradiente
conjugado mascarado**, que trata todas as linhas em paralelo porque cada iteracao e um unico
`[N x K] @ [K x K]`.

`H` em float64 com amortecimento `damp * mean(diag(H))`: com menos linhas amostradas que colunas ela
e singular por construcao, e sem amortecimento o CG anda para o espaco nulo.

## Os controles, e o que cada um derruba

1. **denso** -- a mesma recuperacao com mascara toda de uns. **TEM de dar erro ~0.** Se nao der, o
   solver esta errado e nenhuma linha da tabela significa nada. Sem ele, um solver que nao faz nada
   produz "a recuperacao nao ajudou" e isso passa por resultado.
2. **aleatorio** -- poda 2:4 com criterio aleatorio no lugar do Wanda, recuperada igual. Se a
   recuperacao salvar as duas na mesma medida, ela esta compensando a poda em vez de aproveitar o
   criterio, e o 2,20x que esta bancada atribuiu ao Wanda em 2026-09-01 nao sobrevive.
3. **par** -- a granularidade que o INT4 forca, recuperada. Diz se a recuperacao ressuscita o
   caminho que a foto matou.

## Previsoes, escritas antes

1. O controle **denso** passa (< 1e-3). Se falhar, paro e conserto o solver antes de ler qualquer
   outra coisa.
2. `elem_rec` fica **abaixo** de `elem` por pelo menos 1,5x. Reconstrucao por camada e um metodo
   estabelecido; se nao render nem 1,5x, suspeito da minha implementacao antes da hipotese.
3. `elem_rec` **nao** chega ao 0,0956 do W4A4. Poda joga metade dos pesos fora; recuperar dentro do
   suporte nao inventa capacidade. Se chegar, e a noticia do dia.
4. O criterio **importa**: `alea_rec` fica pior que `elem_rec`. Se forem indistinguiveis, o Wanda e
   decorativo e isso corrige uma conclusao publicada.
5. **Erro por camada nao decide nada.** Esta bancada mediu tres instrumentos numericos apontando
   para o lado errado no mesmo dia. Qualquer braco que passe daqui vai para o render, e e a foto
   que fecha.

## Condicoes de refutacao

1. Controle denso falha -> a tabela inteira e ilegivel; consertar o solver.
2. `elem_rec` nao melhora nada -> reconstrucao por camada independente nao basta; o proximo degrau
   e o sequencial (cada camada ve a entrada ja degradada), que e o que o SparseGPT faz.
3. `alea_rec ~= elem_rec` -> o criterio de poda nao importa depois da recuperacao, e o numero do
   Wanda publicado em 2026-09-01 precisa de ressalva.

## Nao coberto

Erro de saida por camada nas ativacoes calibradas, **nao imagem**. Um modelo, uma calibragem, uma
placa. Reconstrucao por camada **independente**, que e estritamente mais fraca que a sequencial.
Sem treino de verdade em nenhum braco. E o `int8` do braco `--quantiza` e simetrico por linha, nao
e ConvRot -- entao aquela coluna soma dois efeitos e nao isola nenhum.


---

## RESULTADO parcial, 2026-09-03: a medicao fechou, o render NAO

### O que as previsoes fizeram

1. **Controle denso passa.** Bateu, mas so depois de dois consertos (abaixo).
2. **`elem_rec` melhora pelo menos 1,5x.** Bateu com folga: **14,2x**.
3. **`elem_rec` nao chega ao W4A4.** **ERRADA, e por uma margem enorme.** Medidos nas MESMAS
   linhas de teste: W4A4 `0,0907`, 2:4 elemento recuperado `0,0052`. **17,4x mais fiel**, a 5,0
   bits/peso contra 4,0. Eu previ que poda joga capacidade fora e recuperar dentro do suporte nao
   a inventa de volta. O que a medicao diz e que a capacidade jogada fora nao estava sendo usada
   pelas 8192 direcoes de ativacao que o modelo de fato visita.
4. **O criterio importa.** Bateu: `alea_rec 0,0288` contra `elem_rec 0,0052`, 5,5x.
5. **Erro por camada nao decide nada.** Continua valendo, e e por isso que a linha abaixo importa
   mais que todas as de cima.

    formato                          bits/peso     erro
    W4A4 ConvRot (o de hoje)               4,0   0,0907
    2:4 elemento, so podado                9,0   0,0736
    2:4 elemento RECUPERADO                9,0   0,0052
    2:4 elemento RECUPERADO + int8         5,0   0,0052
    2:4 par RECUPERADO + int8              5,0   0,0062
    denso recuperado (controle)              -   piso do int8, exato

### Os dois erros de desenho, ambos pegos por controle

**Media residuo de TREINO.** A primeira versao ajustava e avaliava na mesma matriz X. Com
`X = [128, 3840]` sao 128 equacoes para 1920 incognitas por linha de saida: o CG zera o residuo por
construcao e a tabela dizia `0,0790 -> 0,0007`, um ganho de **110x** que significaria que poda 2:4
sai de graca. Nao sai. Com separacao treino/teste o MESMO run mostrou **32,1x** de distancia entre
os dois numeros. O conserto nao e mais iteracao, e **mais linha de ativacao**: uma calibragem de
8192 linhas leva teste/treino a **1,1x**, e so entao os numeros acima significam alguma coisa.

**O controle denso reprovava o caso correto.** Sob `--quantiza` o braco denso tambem leva int8,
entao ele mede o piso do int8 (3,71e-3) e nao erro de solver -- e o limiar fixo de 1e-3 o
reprovava. O piso agora e MEDIDO (int8 do peso sem poda) e o controle compara contra ele. Um
controle que reprova o caso correto ensina a desligar controles, que e o argumento que este repo ja
faz sobre WARN apoiado em hipotese.

### O render nao fechou, e as tres hipoteses erradas ficam escritas

O probe reportou `CONTROLE FALHOU: a cirurgia nao sobreviveu ate o fim da amostragem` no braco
`so_poda_elem`, e eu errei a causa tres vezes:

1. **"E a pressao de memoria."** A calibragem de 12,9 GiB ficava residente durante a amostragem, e
   a hipotese era que o ComfyUI despejava o modelo e recarregava do disco. Liberei a calibragem
   antes de amostrar. **O controle continuou falhando.**
2. **"E o invariante errado."** `fracao_par24` cobra <=2 PARES vivos por grupo de 8, que e o
   invariante do INT4; um braco podado por ELEMENTO mantem 2 de cada 4 valores e esses dois podem
   cair em pares diferentes, deixando os 4 pares vivos legitimamente. Escrevi `fracao_elem24` e fiz
   o invariante seguir a granularidade do braco. **O controle continuou falhando** -- entao esse
   conserto e correto e nao era a causa.
3. **"E o carregamento preguicoso."** Sem `--disable-dynamic-vram` o ComfyUI 0.33 carrega peso do
   arquivo durante a amostragem e sobrescreve o patch em memoria. Passei a flag. **O processo
   morreu sem stdout nem stderr** -- provavelmente RAM, porque a flag carrega os 11,5 GiB de uma vez
   ao lado dos 12,9 GiB de amostras.

O caminho que sobra, e que a proxima sessao deve tomar: **calcular os pesos recuperados num
processo separado e grava-los**, depois carregar so eles no processo do render. Separa a memoria da
reconstrucao da memoria da amostragem, e de quebra torna o braco reproduzivel sem recalcular.

### Nao coberto

Tudo que esta secao afirma e **erro de saida por camada nas ativacoes calibradas**, em 24 camadas,
com 25% das linhas separadas para avaliar. Nao ha imagem, e esta bancada mediu no mesmo dia tres
instrumentos numericos apontando para o lado errado. **17,4x nao e um resultado ate a foto existir.**
