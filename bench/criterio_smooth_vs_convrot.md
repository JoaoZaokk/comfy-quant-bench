# Criterio: suavizar canal paga? `_w4a4_smooth` contra `_w4a4_convrot` no Gemma

Escrito em 2026-09-03 **antes** de medir. Os dois arquivos estao em disco desde 2026-09-01, com
7,417 GiB cada e 32 bytes de diferenca, e ninguem comparou -- a pergunta que o conversor `smooth`
existe para responder esta sem resposta com a evidencia pronta.

## O que muda entre os dois arquivos

Mesma origem (`gemma_3_12B_it_heretic.safetensors`, 23,5 GiB), mesmo `convrot_groupsize` 256,
mesmas 336 camadas quantizadas, mesmo backend CUDA. O `smooth` acrescenta SmoothQuant com
`alpha 0.5`: reescreve cada norm como `norm <- (norm+1)/lambda - 1` e o peso como `W <- W*lambda`,
movendo a magnitude da **ativacao** para o **peso**. O sidecar registra o efeito pretendido:
razao de outlier por canal na ativacao **82,52 -> 9,60**.

`lambda` recuperado da propria formula invertida: min 4,90, max 103,47, media 13,60 sobre 3840
canais.

## A previsao tem um controle embutido, e e por isso que o desenho e este

SmoothQuant existe para tornar a **ativacao** mais facil de quantizar, pagando com um peso mais
dificil. Num text encoder do ComfyUI a ativacao **nao e quantizada** pelo caminho normal: as duas
travas (`comfy_force_cast_weights` de `comfy/sd.py:269` e `full_precision_mm` de
`comfy/sd1_clip.py:114`) dequantizam o peso e rodam a GEMM em BF16.

Logo:

    caminho TRAVADO   a ativacao nunca e quantizada  -> suavizar nao tem o que ajudar,
                                                        e o peso distorcido so pode atrapalhar
    caminho SOLTO     a ativacao vai a 4 bits         -> e aqui que suavizar tem de pagar

**Um mecanismo que so aparece no braco onde ele deveria aparecer e uma evidencia muito mais forte
que uma diferenca media.** Se o `smooth` ganhasse nos dois caminhos por igual, o ganho nao seria de
SmoothQuant -- seria de qualquer perturbacao, e o experimento nao teria distinguido nada.

Medido em 2026-09-03 na mesma bancada, e por isso o braco solto e alcancavel: soltar as travas exige
escrever nos modulos **e** na fonte, porque para um encoder ja residente `ModelPatcher.load` nao
roda de novo e a escrita na fonte nao chega a modulo nenhum.

## Os bracos

    A  bf16          gemma_3_12B_it_heretic.safetensors           23,5 GiB   referencia
    B  convrot TRAV  ..._w4a4_convrot.safetensors                  7,4 GiB
    C  convrot SOLT  ..._w4a4_convrot.safetensors                  7,4 GiB
    D  smooth  TRAV  ..._w4a4_smooth.safetensors                   7,4 GiB
    E  smooth  SOLT  ..._w4a4_smooth.safetensors                   7,4 GiB

Metrica: rel-RMSE do condicionamento contra o braco A, mesmos prompts, mesma placa.

## Controles

1. **Despacho contado.** B e D tem de fazer 0 chamadas do kernel de 4 bits; C e E, mais que 0. Sem
   isso "solto" e um rotulo, nao um estado -- e esta bancada ja publicou uma medicao em que o braco
   destravado nao estava destravado.
2. **B != D.** Os dois arquivos tem de produzir condicionamentos diferentes. Se forem iguais,
   `lambda` foi 1 em toda parte e o `smooth` nao fez nada.
3. **A e o mesmo em todos.** A referencia sai de um unico carregamento.

## Previsoes, escritas antes

1. **C e E sao muito piores que B e D.** Ja medido nesta bancada em outro encoder: soltar as travas
   multiplica o erro por 4,23x no Qwen. Se isso nao aparecer, algo esta errado no braco solto.
2. **D >= B** (o smooth e igual ou pior no caminho travado). E o custo sem o beneficio.
3. **E < C** (o smooth ganha no caminho solto). E a hipotese que o conversor encarna.
4. Nao sei a magnitude. Se `E/C` ficar acima de 0,9 o ganho existe e nao paga a complexidade; abaixo
   de 0,7 e recomendacao de conversor.

## Condicoes de refutacao

1. **B == D** -> o `smooth` nao fez nada; comparar nao mede.
2. **E >= C** -> SmoothQuant nao ajuda nem onde deveria; o conversor `smooth` nao tem caso de uso
   nesta arquitetura, e isso vai escrito.
3. **E < C e D < B na mesma proporcao** -> o ganho nao vem de suavizar a ativacao, porque aparece
   onde a ativacao nem e quantizada. A explicacao teria de ser outra.

## Nao coberto

Um modelo, uma arquitetura, uma placa, `alpha 0.5` unico. **Sem imagem**: o Gemma deste projeto e
usado como encoder do LTX e nenhum render entra aqui, entao isto mede condicionamento e nada mais --
e esta bancada ja mediu que a distancia no condicionamento pode ser alta com a imagem boa e vice
versa. Nao mede tempo. E o `alpha` nao foi varrido, entao um resultado negativo e sobre 0,5, nao
sobre SmoothQuant.
