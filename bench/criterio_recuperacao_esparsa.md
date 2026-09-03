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
